import csv
import io
import unicodedata
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from .common import (
    ROOT,
    FEATURES,
    TARGET,
    WAVELENGTHS,
    SCHEMA_VERSION,
    named_path,
    sha256,
    write_json,
    atomic_directory,
)
from .download import download, CITATION, DOI
from .grouping import assign_split_groups


def normalize_group(value):
    return " ".join(unicodedata.normalize("NFKC", str(value)).casefold().split())


def read_csv_checked(text):
    header = next(csv.reader(io.StringIO(text)), [])
    if len(header) != len(set(header)):
        raise ValueError("CSV 表头含重复列，必须先修复")
    if not header:
        raise ValueError("CSV 为空")
    return pd.read_csv(io.StringIO(text), dtype={"sample_id": str, "waterbody": str})


def check_spectra(frame):
    absent = sorted(set(FEATURES) - set(frame.columns))
    if absent:
        raise ValueError(f"缺少必要波段（示例）：{absent[:5]}。需要400–750nm、间隔5nm的Rrs。")
    x = frame[FEATURES].to_numpy(dtype=np.float64)
    if len(x) == 0 or not np.isfinite(x).all():
        raise ValueError("光谱必须非空且无缺失/无穷值")
    # 范围是宽松输入校验，不是精度保证；不强制把轻微负 Rrs 截断为零。
    if ((x < -0.05) | (x > 1)).any():
        raise ValueError("Rrs 超出宽松范围 [-0.05,1] sr^-1，请检查单位和缺测码")
    return x


def read_prediction_csv(text):
    frame = read_csv_checked(text)
    if not {"sample_id"}.issubset(frame.columns):
        raise ValueError("预测 CSV 需要 sample_id 列及所有 rrs_波长 列")
    if len(frame) > 10000:
        raise ValueError("单次预测最多10000条样本")
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("sample_id 必须非空且唯一")
    x = check_spectra(frame)
    return frame, x


def split_groups(frame, seed=42):
    groups = (
        frame["split_group"].to_numpy() if "split_group" in frame else frame.waterbody.map(normalize_group).to_numpy()
    )
    if len(set(groups)) < 12:
        raise ValueError("至少需要12个独立水体组；不能用逐样本随机划分替代")
    first = GroupShuffleSplit(n_splits=1, test_size=0.15, random_state=seed)
    trainval, test = next(first.split(frame, groups=groups))
    second = GroupShuffleSplit(n_splits=1, test_size=0.15 / 0.85, random_state=seed + 1)
    ti, vi = next(second.split(trainval, groups=groups[trainval]))
    output = {"train": trainval[ti], "val": trainval[vi], "test": test}
    for name, indices in output.items():
        if len(indices) < 20:
            raise ValueError(f"{name}只有{len(indices)}条样本，需检查水体组大小及划分方案")
    return output


def gloria_frame():
    archive_path = download()
    with zipfile.ZipFile(archive_path) as archive:

        def read(name):
            matches = [n for n in archive.namelist() if Path(n).name == name]
            if len(matches) != 1:
                raise ValueError(f"GLORIA archive contains ambiguous {name}")
            return pd.read_csv(archive.open(matches[0]))

        meta, spectra, qc = read("GLORIA_meta_and_lab.csv"), read("GLORIA_Rrs.csv"), read("GLORIA_qc_flags.csv")
    # 官方 QC 表尾部有一个全空行；只剔除全空行，不掩盖有内容但缺 ID 的记录。
    empty_qc_rows = int(qc.isna().all(axis=1).sum())
    qc = qc.dropna(how="all")
    joined = meta.merge(spectra, on="GLORIA_ID", validate="one_to_one").merge(qc, on="GLORIA_ID", validate="one_to_one")
    if len(joined) != len(meta) or len(joined) != len(spectra) or len(joined) != len(qc):
        raise ValueError("GLORIA_ID 无法完整一一匹配")
    # 只使用 Chla，绝不把 Chla_plus_phaeo 混入叶绿素a标签。
    qc_ok = joined["Flagged"].eq(0) & joined["Suspect"].eq(0)
    kept = joined.loc[qc_ok].copy()
    frame = pd.DataFrame(
        {
            "sample_id": kept.GLORIA_ID,
            "waterbody": kept.Site_name,
            "country": kept.Country,
            "measured_at": kept.Date_Time_UTC,
            "source_id": kept.Dataset_ID,
            TARGET: kept.Chla,
        }
    )
    for wavelength, column in zip(WAVELENGTHS, FEATURES):
        frame[column] = kept[f"Rrs_{wavelength}"]
    source = {
        "name": "GLORIA-2022",
        "doi": DOI,
        "citation": CITATION,
        "license": "CC-BY-4.0",
        "archive_sha256": sha256(archive_path),
        "source_rows": len(joined),
        "removed_qc": int((~qc_ok).sum()),
        "empty_qc_rows": empty_qc_rows,
        "target_field": "Chla; Chla_plus_phaeo is never substituted",
        "target_conversion": "mg/m^3 equals ug/L numerically",
        "reflectance": "Rrs in sr^-1; exact source wavelengths selected, no interpolation",
    }
    return frame, source


def prepare(dataset="gloria", custom_csv=None, source_name=None, input_contract=None):
    if custom_csv is None:
        frame, source = gloria_frame()
    else:
        if not source_name:
            raise ValueError("自有数据必须提供 --source-name，记录来源和单位已核实的信息")
        custom_csv = Path(custom_csv)
        frame = read_csv_checked(custom_csv.read_text(encoding="utf-8-sig"))
        source = {
            "name": source_name,
            "input_file": str(custom_csv.resolve()),
            "sha256": sha256(custom_csv),
            "source_rows": len(frame),
            "target_unit": "ug/L",
            "reflectance_unit": "sr^-1",
            "removed_qc": 0,
        }
        contract = input_contract or {
            "rrs_unit": "sr^-1",
            "target_unit": "ug/L",
            "declaration": "trusted library caller",
        }
        if contract["rrs_unit"] not in ["sr^-1", "water_leaving_reflectance"] or contract["target_unit"] not in [
            "ug/L",
            "mg/m3",
            "mg/L",
        ]:
            raise ValueError("输入单位声明不受支持")
        source["input_contract"] = contract
        if contract["target_unit"] == "mg/L":
            frame[TARGET] = pd.to_numeric(frame[TARGET], errors="coerce") * 1000
        if contract["rrs_unit"] == "water_leaving_reflectance":
            frame[FEATURES] = frame[FEATURES].apply(pd.to_numeric, errors="coerce") / np.pi
    required = {"sample_id", "waterbody", TARGET, *FEATURES}
    if not required.issubset(frame.columns):
        raise ValueError(f"训练数据缺少列：{sorted(required - set(frame.columns))[:8]}")
    if frame.sample_id.isna().any() or frame.sample_id.duplicated().any():
        raise ValueError("样本ID为空或重复，不能可靠建立实验记录")
    frame = frame.copy()
    reasons = {}

    def remove(mask, reason):
        nonlocal frame
        reasons[reason] = int(mask.sum())
        frame = frame.loc[~mask].copy()

    y = pd.to_numeric(frame[TARGET], errors="coerce")
    remove(~np.isfinite(y) | (y <= 0), "missing_or_nonpositive_chlorophyll")
    frame[TARGET] = pd.to_numeric(frame[TARGET])
    remove(
        frame.waterbody.isna() | frame.waterbody.map(normalize_group).isin(["", "nan", "unknown", "none"]),
        "missing_waterbody",
    )
    frame["waterbody"] = frame.waterbody.map(normalize_group)
    numeric = frame[FEATURES].apply(pd.to_numeric, errors="coerce")
    remove(~np.isfinite(numeric.to_numpy()).all(axis=1), "incomplete_spectra")
    frame[FEATURES] = numeric.loc[frame.index]
    bad = ((frame[FEATURES] < -0.05) | (frame[FEATURES] > 1)).any(axis=1)
    remove(bad, "outside_broad_rrs_range")
    # 全局去掉完全一致光谱，避免换个ID后复制光谱进入不同分区。
    remove(frame.duplicated(subset=FEATURES, keep="first"), "exact_duplicate_spectra")
    frame = frame.reset_index(drop=True)
    check_spectra(frame)
    frame, grouping_audit = assign_split_groups(frame)
    splits = split_groups(frame)
    destination = named_path("data/prepared", dataset)
    final_destination = destination
    with atomic_directory(final_destination) as destination:
        frame.to_csv(destination / "clean.csv", index=False)
        membership = frame[["sample_id", "waterbody"]].copy()
        membership["split"] = ""
        stats = {}
        for split, indices in splits.items():
            part = frame.iloc[indices]
            part.to_csv(destination / f"{split}.csv", index=False)
            membership.loc[indices, "split"] = split
            stats[split] = {
                "samples": len(part),
                "waterbodies": part.waterbody.nunique(),
                "chlorophyll_min_ug_l": float(part[TARGET].min()),
                "chlorophyll_max_ug_l": float(part[TARGET].max()),
            }
        membership.to_csv(destination / "split_membership.csv", index=False)
        frame.loc[splits["test"][:5], ["sample_id", "waterbody", *FEATURES]].to_csv(
            destination / "prediction_example.csv", index=False
        )
        quality = {
            "schema_version": SCHEMA_VERSION,
            "dataset": dataset,
            "features": FEATURES,
            "wavelengths_nm": WAVELENGTHS,
            "target": TARGET,
            "target_unit": "ug/L",
            "rrs_unit": "sr^-1",
            "source": source,
            "removed_sequentially": reasons,
            "retained_samples": len(frame),
            "split_method": "connected components of waterbody names, compound names and source_id; seeds 42/43",
            "grouping_audit": grouping_audit,
            "split_groups": {key: sorted(frame.iloc[idx].split_group.unique().tolist()) for key, idx in splits.items()},
            "splits": stats,
            "hashes": {key: sha256(destination / f"{key}.csv") for key in splits},
            "limitations": [
                "水体名称已标准化，但未完成全球别名和空间边界消歧",
                "剔除缺标签和质量标记样本后，仅代表保留子集",
                "原位Rrs模型不能直接接受未完成大气校正的卫星数据",
            ],
        }
        write_json(destination / "quality.json", quality)
    print(f"Prepared {dataset}: {len(frame)} paired samples")
    print(stats)
    return final_destination
