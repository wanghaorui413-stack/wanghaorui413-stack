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
    print(f"Saved {len(result)} predictions: {output_path.resolve()}")
