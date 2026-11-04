"""重复分组交叉验证：估计固定单次划分之外的精度波动。

外层按 split_group 做 GroupKFold（每次重复打乱组到折的分配）；
内层仅在外层训练折里再按组留出验证集，用于岭/PLS 参数和神经网络早停，
与 chlorophyll.train 的选模口径一致。外层测试折只用于计分。

注意：本脚本汇总了 train/val/test 全部数据，是独立于冻结测试集的另一种评估，
不能用其结果重新挑选服务模型后再引用原冻结测试分数。
"""

import argparse
import sys
import time
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

from chlorophyll.common import FEATURES, TARGET, atomic_directory, named_path, read_json, sha256, write_json
from chlorophyll.data import check_spectra
from chlorophyll.metrics import invert_target, scores
from chlorophyll.models import create_model
from chlorophyll.train import fit_network, grouped_scores

MODELS = ["mean_log", "ridge", "pls", "random_forest", "mlp", "cnn"]


def macro_log10(y, p, waterbodies):
    return grouped_scores(y, p, waterbodies)["macro_waterbody_log10_rmse"]


def run_fold(train_part, test_part, seed, args, scratch):
    # 内层组划分：只在外层训练折中留出验证组
    inner = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    fit_idx, val_idx = next(inner.split(train_part, groups=train_part.split_group))
    fit, val = train_part.iloc[fit_idx], train_part.iloc[val_idx]

    x, xv, xt = check_spectra(fit), check_spectra(val), check_spectra(test_part)
    raw_y, val_y = fit[TARGET].to_numpy(float), val[TARGET].to_numpy(float)
    ym, ys = float(np.log10(raw_y).mean()), float(max(np.log10(raw_y).std(), 1e-8))
    xm, xs = x.mean(0), x.std(0).clip(1e-8)
    x, xv, xt = (x - xm) / xs, (xv - xm) / xs, (xt - xm) / xs
    y = (np.log10(raw_y) - ym) / ys
    val_groups = val.waterbody.to_numpy()

    predictions = {"mean_log": np.full(len(test_part), 10**ym)}
    chosen = {}
    for name, choices in [
        ("ridge", [(a, Ridge(alpha=a)) for a in [1.0, 10.0, 100.0, 1000.0]]),
        ("pls", [(n, PLSRegression(n_components=n, scale=False)) for n in [2, 5, 10]]),
    ]:
        best = float("inf")
        for label, estimator in choices:
            estimator.fit(x, y)
            value = macro_log10(val_y, invert_target(estimator.predict(xv).reshape(-1), ym, ys), val_groups)
            if value < best:
                best, chosen[name] = value, label
                predictions[name] = invert_target(estimator.predict(xt).reshape(-1), ym, ys)
    forest = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_features=0.7, n_jobs=2, random_state=seed)
    forest.fit(x, y)
    predictions["random_forest"] = invert_target(forest.predict(xt), ym, ys)

    prep = {"ym": np.array(ym), "ys": np.array(ys)}
    net_args = SimpleNamespace(
        seed=seed, lr=args.lr, batch_size=args.batch_size, epochs=args.epochs, patience=args.patience
    )
    xtt = torch.tensor(xt, dtype=torch.float32).unsqueeze(1)
    for name in ["mlp", "cnn"]:
        _, epoch, _ = fit_network(name, x, y, xv, val_y, prep, net_args, scratch, val_groups)
        model = create_model(name)
        model.load_state_dict(torch.load(scratch / f"{name}.pt", weights_only=True))
        model.eval()
        with torch.no_grad():
            predictions[name] = invert_target(model(xtt).numpy(), ym, ys)
        chosen[f"{name}_epoch"] = epoch

    # 在内层验证集上按 train.py 的口径挑选“本折会被部署”的模型
    val_scores = {"random_forest": macro_log10(val_y, invert_target(forest.predict(xv), ym, ys), val_groups)}
    for name in ["mlp", "cnn"]:
        model = create_model(name)
        model.load_state_dict(torch.load(scratch / f"{name}.pt", weights_only=True))
        model.eval()
        with torch.no_grad():
            pv = invert_target(model(torch.tensor(xv, dtype=torch.float32).unsqueeze(1)).numpy(), ym, ys)
        val_scores[name] = macro_log10(val_y, pv, val_groups)
    chosen["selected"] = min(val_scores, key=val_scores.get)
    return predictions, chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gloria_v2")
    parser.add_argument("--run", default="cv_gloria_v2")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    data_dir = named_path("data/prepared", args.dataset)
    quality = read_json(data_dir / "quality.json")
    frame = pd.concat(
        [pd.read_csv(data_dir / f"{s}.csv", dtype={"sample_id": str}) for s in ["train", "val", "test"]],
        ignore_index=True,
    )
    for split in ["train", "val", "test"]:
        if sha256(data_dir / f"{split}.csv") != quality["hashes"][split]:
            raise ValueError("准备后的数据被修改，请建立新的数据集版本")

    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    start = time.monotonic()
    fold_rows, oof_rows = [], []
    with atomic_directory(named_path("runs", args.run)) as out:
        for repeat in range(args.repeats):
            splitter = GroupKFold(n_splits=args.folds, shuffle=True, random_state=1000 + repeat)
            for fold, (tr, te) in enumerate(splitter.split(frame, groups=frame.split_group)):
                seed = 42 + repeat * 100 + fold
                train_part, test_part = frame.iloc[tr], frame.iloc[te]
                predictions, chosen = run_fold(train_part, test_part, seed, args, out)
                y = test_part[TARGET].to_numpy(float)
                waterbodies = test_part.waterbody.to_numpy()
                for name in MODELS:
                    metric = grouped_scores(y, predictions[name], waterbodies)
                    fold_rows.append(
                        {
                            "repeat": repeat,
                            "fold": fold,
                            "model": name,
                            "test_samples": len(test_part),
                            "test_groups": test_part.split_group.nunique(),
                            "largest_group_share": float(test_part.split_group.value_counts(normalize=True).iloc[0]),
                            "selected_on_inner_val": chosen["selected"] == name,
                            **{k: v for k, v in chosen.items() if k != "selected"},
                            **metric,
                        }
                    )
                part = test_part[["sample_id", "waterbody", "split_group"]].copy()
                part["repeat"], part["fold"], part["observed_ug_l"] = repeat, fold, y
                for name in MODELS:
                    part[name] = predictions[name]
                oof_rows.append(part)
                rf = [r for r in fold_rows[-len(MODELS) :] if r["model"] == "random_forest"][0]
                mlp = [r for r in fold_rows[-len(MODELS) :] if r["model"] == "mlp"][0]
                print(
                    f"repeat={repeat} fold={fold} n={len(test_part)} groups={rf['test_groups']} "
                    f"macro log10: rf={rf['macro_waterbody_log10_rmse']:.3f} mlp={mlp['macro_waterbody_log10_rmse']:.3f} "
                    f"selected={chosen['selected']} elapsed={time.monotonic() - start:.0f}s",
                    flush=True,
                )
        for name in ["mlp.pt", "cnn.pt"]:
            (out / name).unlink(missing_ok=True)

        folds = pd.DataFrame(fold_rows)
        oof = pd.concat(oof_rows, ignore_index=True)
        folds.to_csv(out / "fold_metrics.csv", index=False)
        oof.to_csv(out / "oof_predictions.csv", index=False)

        keys = ["macro_waterbody_log10_rmse", "log10_rmse", "rmse_ug_l", "median_factor_error", "bias_ug_l"]
        summary = {}
        for name, part in folds.groupby("model"):
            summary[name] = {k: {"mean": float(part[k].mean()), "std": float(part[k].std(ddof=1))} for k in keys}
            # 每次重复的汇总折外预测（每条样本恰好被预测一次）
            pooled = [scores(g.observed_ug_l.to_numpy(), g[name].to_numpy()) for _, g in oof.groupby("repeat")]
            summary[name]["pooled_oof"] = {
                k: float(np.mean([p[k] for p in pooled]))
                for k in ["log10_rmse", "rmse_ug_l", "r2", "median_factor_error"]
            }
        pivot = folds.pivot_table(index=["repeat", "fold"], columns="model", values="macro_waterbody_log10_rmse")
        paired = {}
        for a, b in [("mlp", "random_forest"), ("cnn", "random_forest"), ("mlp", "cnn"), ("random_forest", "ridge")]:
            diff = pivot[a] - pivot[b]
            paired[f"{a}_minus_{b}"] = {
                "mean": float(diff.mean()),
                "std": float(diff.std(ddof=1)),
                "a_better_folds": int((diff < 0).sum()),
                "total_folds": int(len(diff)),
            }
        selection = folds[folds.selected_on_inner_val].model.value_counts().to_dict()
        write_json(
            out / "cv_summary.json",
            {
                "dataset": args.dataset,
                "protocol": (
                    f"{args.repeats}x repeated GroupKFold(k={args.folds}, shuffled) on split_group; "
                    "inner 15% group holdout for ridge/PLS params and NN early stopping"
                ),
                "config": vars(args),
                "samples": len(frame),
                "split_groups": int(frame.split_group.nunique()),
                "data_hashes": quality["hashes"],
                "metrics_mean_std_over_folds": summary,
                "paired_fold_differences_macro_log10": paired,
                "inner_val_selection_counts": selection,
                "seconds": round(time.monotonic() - start, 1),
                "caveat": "汇总了原冻结测试集，结果不可用于在冻结测试上重新选模",
            },
        )
    print(pd.DataFrame({m: {k: v["mean"] for k, v in s.items() if k != "pooled_oof"} for m, s in summary.items()}).T)
    print(paired)


if __name__ == "__main__":
    main()
