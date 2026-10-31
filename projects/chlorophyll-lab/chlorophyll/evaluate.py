import numpy as np
import pandas as pd
from .common import FEATURES, TARGET, named_path, read_json, write_json, sha256, atomic_directory
from .inference import Predictor
from .metrics import scores
from .report import write_reports


def evaluate(run_name):
    run = named_path("runs", run_name)
    if (run / "test_metrics.json").exists():
        raise FileExistsError("该实验已完成独立测试，请直接查看已有报告")
    predictor = Predictor(run_name)
    card = predictor.card
    data_dir = named_path("data/prepared", card["dataset"])
    path = data_dir / "test.csv"
    if sha256(path) != card["split_hashes"]["test"]:
        raise ValueError("测试数据与训练时冻结的数据版本不一致")
    data = pd.read_csv(path, dtype={"sample_id": str})
    quality = card["dataset_quality"]
    test_groups = set(data["split_group"] if "split_group" in data else data.waterbody)
    if test_groups & (set(quality["split_groups"]["train"]) | set(quality["split_groups"]["val"])):
        raise ValueError("独立测试水体与训练/验证组有交叉")
    y = data[TARGET].to_numpy(dtype=float)
    predictions = predictor.all_predictions(data[FEATURES].to_numpy(dtype=float))
    table = data[["sample_id", "waterbody"]].copy()
    table["observed_ug_l"] = y
    for name, value in predictions.items():
        table[name] = value
    final = run / "evaluations" / "final"
    with atomic_directory(final) as evaluation:
        table.to_csv(evaluation / "test_predictions.csv", index=False)
        result = {
            "samples": len(data),
            "waterbodies": int(data.waterbody.nunique()),
            "split_groups": len(test_groups),
            "selected_model": card["selected_model"],
            "test_hash": sha256(path),
            "metrics": {name: scores(y, p) for name, p in predictions.items()},
        }
        groups = []
        for group, part in table.groupby("waterbody"):
            for name in predictions:
                groups.append(
                    {
                        "waterbody": group,
                        "model": name,
                        "samples": len(part),
                        **scores(part.observed_ug_l.to_numpy(), part[name].to_numpy()),
                    }
                )
        by_group = pd.DataFrame(groups)
        by_group.to_csv(evaluation / "metrics_by_waterbody.csv", index=False)
        result["macro_waterbody_log10_rmse"] = by_group.groupby("model").log10_rmse.mean().to_dict()
        bins = [0, 1, 10, 100, np.inf]
        concentrations = []
        for low, high in zip(bins[:-1], bins[1:]):
            part = table[(table.observed_ug_l >= low) & (table.observed_ug_l < high)]
            if len(part):
                for name in predictions:
                    concentrations.append(
                        {
                            "range_ug_l": f"[{low},{high})",
                            "model": name,
                            "samples": len(part),
                            **scores(part.observed_ug_l.to_numpy(), part[name].to_numpy()),
                        }
                    )
        pd.DataFrame(concentrations).to_csv(evaluation / "metrics_by_concentration.csv", index=False)
        # 完成全部计算后写入最终评估标记，避免后续意外覆盖。
        write_json(evaluation / "test_metrics.json", result)
        write_json(evaluation / "prediction_manifest.json", {"sha256": sha256(evaluation / "test_predictions.csv")})
    rebuild_report(run_name)
    print(pd.DataFrame(result["metrics"]).T.to_string())
    print(f"Frozen serving model: {card['selected_model']}; report: {run / 'report.html'}")
    return result


def rebuild_report(run_name):
    """只使用冻结预测重建展示文件，不重新接触测试标签或推理。"""
    import shutil

    run = named_path("runs", run_name)
    frozen = run / "evaluations" / "final"
    card = read_json(run / "model_card.json")
    result = read_json(frozen / "test_metrics.json")
    manifest = read_json(frozen / "prediction_manifest.json")
    if sha256(frozen / "test_predictions.csv") != manifest["sha256"]:
        raise ValueError("冻结预测被修改")
    table = pd.read_csv(frozen / "test_predictions.csv", dtype={"sample_id": str})
    write_reports(run, card, result, table)
    for name in [
        "test_predictions.csv",
        "test_metrics.json",
        "metrics_by_waterbody.csv",
        "metrics_by_concentration.csv",
    ]:
        shutil.copy2(frozen / name, run / name)
    return run / "report.html"
