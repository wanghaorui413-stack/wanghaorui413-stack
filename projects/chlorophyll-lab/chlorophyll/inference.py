from pathlib import Path
import hashlib
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
import torch
from .common import FEATURES, SCHEMA_VERSION, ARCHITECTURE_VERSION, PREPROCESS_VERSION, named_path, read_json, sha256
from .data import read_prediction_csv
from .metrics import invert_target
from .models import create_model

# 分组交叉验证中，神经网络/线性模型对超出训练光谱范围的输入会外推到10^6 μg/L。
# 越界样本只输出空值与原因，不输出数值；宽容带按训练范围的比例给出。
# 光谱在范围内时神经网络仍可能失控，故再与随机森林（输出受训练标签约束）对照。
BAND_RANGE_MARGIN = 0.05
TARGET_RANGE_FACTOR = 2.0
REFERENCE_MAX_LOG10_GAP = 1.0
NETWORKS = ["mlp", "cnn"]


def domain_check(x, predictions, prep, reference=None):
    """返回每条样本的拦截原因（空字符串表示通过）。reference 为随机森林预测，可选。"""
    span = prep["train_max"] - prep["train_min"]
    low, high = prep["train_min"] - BAND_RANGE_MARGIN * span, prep["train_max"] + BAND_RANGE_MARGIN * span
    bands = ((x < low) | (x > high)).any(axis=1)
    target = (predictions < prep["target_min"] / TARGET_RANGE_FACTOR) | (
        predictions > prep["target_max"] * TARGET_RANGE_FACTOR
    )
    disagree = np.zeros(len(x), dtype=bool)
    if reference is not None:
        disagree = np.abs(np.log10(predictions) - np.log10(reference)) > REFERENCE_MAX_LOG10_GAP
    reasons = np.full(len(x), "", dtype=object)
    reasons[disagree] = "disagrees_with_random_forest"
    reasons[target] = "prediction_outside_training_range"
    reasons[bands] = "spectrum_outside_training_range"
    return reasons


class Predictor:
    def __init__(self, run="gloria_v1", package_path=None):
        self.run = Path(package_path).resolve() if package_path else named_path("runs", run)
        self.card = read_json(self.run / "model_card.json")
        if self.card["schema_version"] != SCHEMA_VERSION or self.card["features"] != FEATURES:
            raise ValueError("模型输入模式与当前代码不兼容")
        for key, expected in [
            ("architecture_version", ARCHITECTURE_VERSION),
            ("preprocess_version", PREPROCESS_VERSION),
        ]:
            if self.card.get("identity", {}).get(key, self.card.get(key, expected)) != expected:
                raise ValueError(f"模型{key}不兼容")
        for name, digest in self.card["artifact_hashes"].items():
            if sha256(self.run / name) != digest:
                raise ValueError(f"模型产物 {name} 被修改，请使用完整实验目录")
        with np.load(self.run / "preprocess.npz") as archive:
            self.prep = {key: archive[key].copy() for key in archive.files}
        # 仅加载本程序在本地生成、经完整性校验的训练产物。
        self.baselines = None
        self.networks = {}
        self.instance_id = sha256(self.run / "model_card.json")

    def _network(self, name):
        if name not in self.networks:
            model = create_model(name)
            model.load_state_dict(torch.load(self.run / f"{name}.pt", map_location="cpu", weights_only=True))
            model.eval()
            self.networks[name] = model
        return self.networks[name]

    def all_predictions(self, x, models=None):
        x = np.asarray(x, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(FEATURES) or not np.isfinite(x).all():
            raise ValueError("输入必须为有限数值组成的[N,71]矩阵")
        z = (x - self.prep["xm"]) / self.prep["xs"]
        if len(x) == 0:
            raise ValueError("输入样本为空")
        names = list(models or ["mean_log", "ridge", "pls", "random_forest", "mlp", "cnn"])
        if set(names) - {"mean_log", "ridge", "pls", "random_forest", "mlp", "cnn"}:
            raise ValueError("未知模型")
        standardized = {}
        if "mean_log" in names:
            standardized["mean_log"] = np.zeros(len(z))
        for name in names:
            if name in ["ridge", "pls", "random_forest"]:
                if self.baselines is None:
                    self.baselines = joblib.load(self.run / "baselines.joblib")
                standardized[name] = np.asarray(self.baselines[name].predict(z)).reshape(-1)
        for name in [n for n in names if n in ["mlp", "cnn"]]:
            model = self._network(name)
            chunks = []
            with torch.no_grad():
                for start in range(0, len(z), 512):
                    chunks.append(model(torch.tensor(z[start : start + 512], dtype=torch.float32).unsqueeze(1)).numpy())
            standardized[name] = np.concatenate(chunks)
        return {name: invert_target(value, self.prep["ym"], self.prep["ys"]) for name, value in standardized.items()}

    def from_csv(self, text, model="selected", compare=False):
        frame, x = read_prediction_csv(text)
        chosen = self.card["selected_model"] if model == "selected" else model
        predictions = self.all_predictions(x, None if compare else [chosen])
        if chosen not in predictions:
            raise ValueError(f"未知模型 {chosen}")
        result = frame[["sample_id"]].copy()
        if "waterbody" in frame:
            result["waterbody"] = frame.waterbody
        result["selected_model"] = chosen
        result["chlorophyll_ug_l"] = predictions[chosen]
        for name, value in predictions.items():
            result[f"{name}_ug_l"] = value
        z = (x - self.prep["xm"]) / self.prep["xs"]
        result["max_abs_z"] = np.abs(z).max(axis=1)
        result["bands_outside_training_fraction"] = ((x < self.prep["train_min"]) | (x > self.prep["train_max"])).mean(
            axis=1
        )
        result["outside_target_training_range"] = (predictions[chosen] < self.prep["target_min"]) | (
            predictions[chosen] > self.prep["target_max"]
        )
        reference = None
        if chosen in NETWORKS:
            reference = predictions.get("random_forest")
            if reference is None:
                reference = self.all_predictions(x, ["random_forest"])["random_forest"]
        reasons = domain_check(x, predictions[chosen], self.prep, reference)
        rejected = reasons != ""
        result["status"] = np.where(rejected, "rejected", "ok")
        result["rejection_reason"] = reasons
        value_columns = ["chlorophyll_ug_l", *[f"{name}_ug_l" for name in predictions]]
        result.loc[rejected, value_columns] = np.nan
        result["model_instance_sha256"] = self.instance_id
        result["input_sha256"] = hashlib.sha256(text.encode("utf-8")).hexdigest()
        result["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
        result["schema_version"] = SCHEMA_VERSION
        return result


def predict_file(run, input_path, output_path, model="selected", package_path=None):
    output_path = Path(output_path)
    if output_path.exists():
        raise FileExistsError("预测输出已存在，请使用新文件名")
    predictor = Predictor(run, package_path=package_path)
    result = predictor.from_csv(Path(input_path).read_text(encoding="utf-8-sig"), model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False)
    print(result[["sample_id", "selected_model", "chlorophyll_ug_l", "max_abs_z"]].to_string(index=False))
    rejected = int((result.status == "rejected").sum())
    if rejected:
        print(f"已拦截 {rejected} 条越出训练范围的样本，浓度留空，原因见 rejection_reason 列")
    print(f"Saved {len(result)} predictions: {output_path.resolve()}")
