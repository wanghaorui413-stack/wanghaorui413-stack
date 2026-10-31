"""传感器专用模型：GLORIA光谱按实际RSR积分，不制造71波段卫星输入。"""

import re
import zipfile
from pathlib import Path
import urllib.request
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from .common import ROOT, TARGET, atomic_directory, write_json, read_json, sha256, source_identity
from .metrics import scores

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A"]


def read_rsr(text):
    curves = {}
    band = None
    for line in text.splitlines():
        match = re.match(r";;\s*BAND\s+(\S+)", line)
        if match:
            band = "B" + match.group(1)
            curves[band] = []
        elif line.strip() and not line.startswith(";") and band:
            wavelength, response = map(float, line.split()[:2])
            curves[band].append((wavelength, response))
    result = {}
    for band in BANDS:
        values = np.asarray(curves[band])
        wave, response = values[:, 0], values[:, 1]
        if len(wave) < 2 or (np.diff(wave) <= 0).any() or (response < 0).any():
            raise ValueError("无效RSR")
        dx = np.diff(wave)
        trapezoid = np.r_[dx[0] / 2, (dx[:-1] + dx[1:]) / 2, dx[-1] / 2]
        weights = response * trapezoid
        result[band] = (wave, weights / weights.sum())
    return result


def train_satellite_model(sensor="S2A_MSI", dataset="gloria_v2", output=None):
    if sensor not in ["S2A_MSI", "S2B_MSI"]:
        raise ValueError("当前模型仅支持S2A_MSI/S2B_MSI")
    rsr = ROOT / "data" / "satellite" / "rsr" / f"{sensor}.txt"
    rsr.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://raw.githubusercontent.com/acolite/acolite/main/data/RSR/{sensor}.txt"
    if not rsr.exists():
        with urllib.request.urlopen(url, timeout=60) as response:
            rsr.write_bytes(response.read())
    curves = read_rsr(rsr.read_text())
    directory = ROOT / "data" / "prepared" / dataset
    parts = {
        name: pd.read_csv(directory / f"{name}.csv", dtype={"sample_id": str}) for name in ["train", "val", "test"]
    }
    with zipfile.ZipFile(ROOT / "data" / "gloria" / "raw" / "GLORIA-2022.zip") as archive:
        spectra = pd.read_csv(archive.open("GLORIA_2022/GLORIA_Rrs.csv")).set_index("GLORIA_ID")
    arrays = {}
    for split, frame in parts.items():
        source = spectra.loc[frame.sample_id]
        convolved = []
        for band in BANDS:
            wavelengths, weights = curves[band]
            if not np.allclose(wavelengths, np.round(wavelengths)):
                raise ValueError("RSR需要显式光谱重采样适配")
            columns = [f"Rrs_{int(w)}" for w in wavelengths]
            convolved.append(source[columns].to_numpy() @ weights)
        x = np.column_stack(convolved)
        valid = np.isfinite(x).all(axis=1)
        arrays[split] = (frame.loc[valid].copy(), x[valid])
    training, x = arrays["train"]
    validation, xv = arrays["val"]
    if min(len(training), len(validation)) < 20:
        raise ValueError("可用卫星响应卷积样本不足")
    model = RandomForestRegressor(n_estimators=200, min_samples_leaf=2, max_features=1.0, n_jobs=2, random_state=42)
    model.fit(x, np.log10(training[TARGET]))
    val_prediction = 10 ** model.predict(xv)
    metric = scores(validation[TARGET].to_numpy(), val_prediction)
    group_errors = []
    for name, indices in validation.groupby("waterbody").indices.items():
        group_errors.append(scores(validation[TARGET].to_numpy()[indices], val_prediction[indices])["log10_rmse"])
    metric["macro_waterbody_log10_rmse"] = float(np.mean(group_errors))
    output = Path(output) if output else ROOT / "releases" / f"{sensor}_gloria_v2"
    with atomic_directory(output) as stage:
        joblib.dump(model, stage / "model.joblib")
        write_json(
            stage / "model.json",
            {
                "sensor": sensor,
                "bands": BANDS,
                "input_unit": "sr^-1",
                "target_unit": "ug/L",
                "validation_status": "spectral convolution only; satellite matchup unvalidated",
                "validation_metrics": metric,
                "sample_counts": {k: len(v[0]) for k, v in arrays.items()},
                "dataset": dataset,
                "quality_sha256": sha256(directory / "quality.json"),
                "rsr_url": url,
                "rsr_sha256": sha256(rsr),
                "rsr_definition": "ACOLITE bundled response; exact file hash frozen",
                "identity": source_identity(),
                "artifact_sha256": sha256(stage / "model.joblib"),
                "training_min": x.min(0).tolist(),
                "training_max": x.max(0).tolist(),
            },
        )
    return output


class SatellitePredictor:
    def __init__(self, model_dir):
        self.directory = Path(model_dir)
        self.card = read_json(self.directory / "model.json")
        if self.card["bands"] != BANDS or self.card["input_unit"] != "sr^-1":
            raise ValueError("卫星模型契约不匹配")
        if sha256(self.directory / "model.joblib") != self.card["artifact_sha256"]:
            raise ValueError("卫星模型校验失败")
        self.model = joblib.load(self.directory / "model.joblib")

    def predict(self, rrs):
        values = np.asarray(rrs, dtype=float)
        if values.ndim != 2 or values.shape[1] != 7 or not np.isfinite(values).all():
            raise ValueError("需要有限的[N,7] Rrs")
        return np.power(10.0, np.clip(self.model.predict(values), -6, 6))
