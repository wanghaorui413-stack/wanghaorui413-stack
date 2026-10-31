import csv
import platform
import time
import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from torch.utils.data import DataLoader, TensorDataset
from .common import (
    ROOT,
    FEATURES,
    TARGET,
    SCHEMA_VERSION,
    named_path,
    read_json,
    write_json,
    sha256,
    atomic_directory,
    source_identity,
)
from .data import check_spectra
from .metrics import scores, invert_target
from .models import create_model


def grouped_scores(y, p, groups):
    metric = scores(y, p)
    groups = np.asarray(groups)
    metric["macro_waterbody_log10_rmse"] = float(
        np.mean([scores(y[groups == g], p[groups == g])["log10_rmse"] for g in np.unique(groups)])
    )
    return metric


def fit_network(name, x, y, xv, yv_raw, prep, args, out, groups):
    torch.manual_seed(args.seed)
    model = create_model(name)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.001)
    loader = DataLoader(
        TensorDataset(torch.tensor(x, dtype=torch.float32).unsqueeze(1), torch.tensor(y, dtype=torch.float32)),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(args.seed),
    )
    xt = torch.tensor(x, dtype=torch.float32).unsqueeze(1)
    xvt = torch.tensor(xv, dtype=torch.float32).unsqueeze(1)
    best, epoch_best, stale = float("inf"), 0, 0
    history = []
    best_metrics = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        for xb, yb in loader:
            optimizer.zero_grad()
            loss = torch.nn.functional.mse_loss(model(xb), yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            train_mse = torch.nn.functional.mse_loss(model(xt), torch.tensor(y, dtype=torch.float32)).item()
            raw_prediction = invert_target(model(xvt).numpy(), prep["ym"], prep["ys"])
        metric = grouped_scores(yv_raw, raw_prediction, groups)
        value = metric["macro_waterbody_log10_rmse"]
        if not np.isfinite(train_mse + value):
            raise ValueError("训练出现非有限损失，请检查数据与学习率")
        history.append(
            {
                "model": name,
                "epoch": epoch,
                "train_standardized_log_mse": train_mse,
                "val_log10_rmse": metric["log10_rmse"],
                "val_macro_waterbody_log10_rmse": value,
                "val_rmse_ug_l": metric["rmse_ug_l"],
            }
        )
        if value < best:
            best, epoch_best, stale, best_metrics = value, epoch, 0, metric
            torch.save(model.state_dict(), out / f"{name}.pt")
        else:
            stale += 1
        if epoch == 1 or epoch % 10 == 0:
            print(f"{name} epoch={epoch:03d} train_log_mse={train_mse:.4f} val_log10_RMSE={value:.4f}", flush=True)
        if stale >= args.patience:
            break
    return best_metrics, epoch_best, history


def train(args):
    if min(args.epochs, args.patience, args.batch_size) < 1 or not np.isfinite(args.lr) or args.lr <= 0:
        raise ValueError("epochs、patience、batch-size、lr 必须为有限正数")
    data_dir = named_path("data/prepared", args.dataset)
    quality = read_json(data_dir / "quality.json")
    if quality["schema_version"] != SCHEMA_VERSION or quality["features"] != FEATURES:
        raise ValueError("数据模式与当前代码不兼容")
    # 训练从不读取 test.csv，测试评估由另一个命令完成。
    for split in ["train", "val"]:
        if sha256(data_dir / f"{split}.csv") != quality["hashes"][split]:
            raise ValueError("准备后的数据被修改，请建立新的数据集版本")
    training = pd.read_csv(data_dir / "train.csv", dtype={"sample_id": str})
    validation = pd.read_csv(data_dir / "val.csv", dtype={"sample_id": str})
    group_column = "split_group" if "split_group" in training else "waterbody"
    if set(training[group_column]) & set(validation[group_column]):
        raise ValueError("训练/验证水体交叉")
    x, xv = check_spectra(training), check_spectra(validation)
    raw_y = training[TARGET].to_numpy(dtype=float)
    val_y = validation[TARGET].to_numpy(dtype=float)
    ym, ys = float(np.log10(raw_y).mean()), float(max(np.log10(raw_y).std(), 1e-8))
    xm, xs = x.mean(0), x.std(0).clip(1e-8)
    prep = {
        "xm": xm,
        "xs": xs,
        "ym": np.array(ym),
        "ys": np.array(ys),
        "train_min": x.min(0),
        "train_max": x.max(0),
        "target_min": np.array(raw_y.min()),
        "target_max": np.array(raw_y.max()),
    }
    x, xv = (x - xm) / xs, (xv - xm) / xs
    y = (np.log10(raw_y) - ym) / ys
    out = named_path("runs", args.run)
    final_out = out
    with atomic_directory(final_out) as out:
        np.savez(out / "preprocess.npz", **prep)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        start = time.monotonic()
        estimates = {}
        all_scores = {"mean_log": grouped_scores(val_y, np.full(len(val_y), 10**ym), validation.waterbody.to_numpy())}
        trials = {}
        params = {}
        for name, choices in [
            ("ridge", [(str(a), Ridge(alpha=a)) for a in [1.0, 10.0, 100.0, 1000.0]]),
            ("pls", [(str(n), PLSRegression(n_components=n, scale=False)) for n in [2, 5, 10]]),
        ]:
            best = float("inf")
            trials[name] = {}
            for label, estimate in choices:
                estimate.fit(x, y)
                metric = grouped_scores(
                    val_y, invert_target(estimate.predict(xv).reshape(-1), ym, ys), validation.waterbody.to_numpy()
                )
                trials[name][label] = metric
                if metric["macro_waterbody_log10_rmse"] < best:
                    best = metric["macro_waterbody_log10_rmse"]
                    estimates[name], all_scores[name], params[name] = estimate, metric, label
        forest = RandomForestRegressor(
            n_estimators=200, min_samples_leaf=2, max_features=0.7, n_jobs=2, random_state=args.seed
        )
        forest.fit(x, y)
        estimates["random_forest"] = forest
        all_scores["random_forest"] = grouped_scores(
            val_y, invert_target(forest.predict(xv), ym, ys), validation.waterbody.to_numpy()
        )
        joblib.dump(estimates, out / "baselines.joblib")
        history = []
        best_epochs = {}
        for name in ["mlp", "cnn"]:
            metric, epoch, rows = fit_network(name, x, y, xv, val_y, prep, args, out, validation.waterbody.to_numpy())
            all_scores[name], best_epochs[name] = metric, epoch
            history.extend(rows)
        selected = min(all_scores, key=lambda key: all_scores[key]["macro_waterbody_log10_rmse"])
        model_card = {
            "schema_version": SCHEMA_VERSION,
            "dataset": args.dataset,
            "features": FEATURES,
            "target": TARGET,
            "target_unit": "ug/L",
            "reflectance_unit": "sr^-1",
            "target_transform": "log10 then train-set z-score; inverse uses numerical clip [-6,6] in log10",
            "source": quality["source"],
            "config": vars(args),
            "best_epochs": best_epochs,
            "selected_model": selected,
            "selection_metric": "validation macro_waterbody_log10_rmse",
            "identity": source_identity(),
            "validation_metrics": all_scores,
            "baseline_validation_trials": trials,
            "selected_baseline_params": params,
            "dataset_quality": quality,
            "split_hashes": quality["hashes"],
            "versions": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit-learn": sklearn.__version__,
            },
            "training_seconds": round(time.monotonic() - start, 2),
            "artifact_hashes": {
                name: sha256(out / name) for name in ["preprocess.npz", "baselines.joblib", "mlp.pt", "cnn.pt"]
            },
        }
        write_json(out / "model_card.json", model_card)
        with (out / "history.csv").open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(history[0]))
            writer.writeheader()
            writer.writerows(history)
        # 用于理解波段作用，不能当作因果解释。
        pd.DataFrame(
            {
                "wavelength_nm": [int(f.split("_")[1]) for f in FEATURES],
                "random_forest_importance": forest.feature_importances_,
            }
        ).to_csv(out / "feature_importance.csv", index=False)
        print(f"Validation-selected model: {selected}; test set untouched.")
        print({name: round(metric["log10_rmse"], 4) for name, metric in all_scores.items()})
        print(f"Saved: {final_out}")
