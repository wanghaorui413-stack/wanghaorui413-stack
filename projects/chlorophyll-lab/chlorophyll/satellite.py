"""ACOLITE水离反射率适配。卫星匹配精度未经验证；不接受L2A地表反射率冒充Rrs。"""

import math
import subprocess
import tempfile
import shutil
from pathlib import Path
import numpy as np
from .common import atomic_directory, sha256, write_json

BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8A"]
NOMINAL = [492, 560, 665, 704, 740, 783, 865]


def identify_sensor(attrs):
    sensor = str(attrs.get("sensor", ""))
    if sensor not in {"S2A_MSI", "S2B_MSI"}:
        raise ValueError("仅支持明确标记S2A_MSI或S2B_MSI的ACOLITE产品；S2C不受支持")
    return sensor


def match_bands(variables):
    """按水离物理量前缀与波长匹配，不将rhos/rhot当成水离Rrs。"""
    output = []
    for target in NOMINAL:
        candidates = []
        for name in variables:
            prefix = name.split("_")[0]
            if prefix not in {"Rrs", "rhow"}:
                continue
            try:
                wave = float(name.split("_")[1])
            except (IndexError, ValueError):
                continue
            if abs(wave - target) <= 10:
                candidates.append((0 if prefix == "Rrs" else 1, abs(wave - target), name))
        if not candidates:
            raise ValueError(f"缺少{target}nm附近的Rrs/rhow；rhos和rhot不可替代")
        candidates.sort()
        if len(candidates) > 1 and candidates[0][:2] == candidates[1][:2]:
            raise ValueError("波段匹配有歧义")
        output.append(candidates[0][2])
    if len(set(output)) != 7:
        raise ValueError("七个传感器波段不能映射到同一变量")
    return output


def as_rrs(values, variable):
    values = np.asarray(values, dtype=np.float64)
    if variable.startswith("Rrs_"):
        return values
    if variable.startswith("rhow_"):
        return values / math.pi
    raise ValueError("只接受Rrs或rhow；必须先完成水体大气/反光校正")


def validate_grid(x, y):
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 1 or y.ndim != 1 or min(len(x), len(y)) < 2:
        raise ValueError("需要至少2×2像元的一维投影坐标网格")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("网格坐标非有限值")
    dx, dy = np.diff(x), np.diff(y)
    if not np.allclose(dx, 20, atol=0.01) or not np.allclose(dy, -20, atol=0.01):
        raise ValueError("需要从西向东、北向南排列的20米网格；请在ACOLITE设置s2_target_res=20")
    return (float(x[0] - 10), float(y[0] + 10))


def inspect_product(path):
    with open_netcdf(path) as dataset:
        sensor = identify_sensor(dataset.attrs)
        variables = match_bands(dataset.variables)
        validate_grid(dataset["x"].values, dataset["y"].values)
        return {
            "sensor": sensor,
            "bands": BANDS,
            "variables": variables,
            "has_quality_mask": "l2_flags" in dataset,
            "resolution_m": 20,
            "input_sha256": sha256(path),
            "input": str(Path(path).resolve()),
        }


def write_acolite_settings(input_path, output, limit):
    if (
        len(limit) != 4
        or not all(np.isfinite(limit))
        or not (-90 <= limit[0] < limit[2] <= 90 and -180 <= limit[1] < limit[3] <= 180)
    ):
        raise ValueError("limit必须为合法south west north east地理范围")
    path = Path(output) / "settings.txt"
    text = (
        "\n".join(
            [
                f"inputfile={Path(input_path).resolve()}",
                f"output={Path(output).resolve()}",
                "limit=" + ",".join(map(str, limit)),
                "s2_target_res=20",
                "atmospheric_correction=True",
                "dsf_residual_glint_correction=True",
                # 官方acolite_l2w会从rhos派生大写Rrs_*并无条件写l2_flags。
                # 小写rrs_*是次表层量，不能与这里的水面以上Rrs混用。
                "rsr_version=None",
                "l2w_parameters=Rrs_*",
                "output_xy=True",
                "netcdf_projection=True",
                "l2w_mask=True",
                "l2r_export_geotiff=False",
                "l2w_export_geotiff=False",
            ]
        )
        + "\n"
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def correct(processor, input_path, output, limit, timeout=3600):
    processor, input_path, output = Path(processor), Path(input_path), Path(output)
    if not processor.is_file() or not input_path.exists():
        raise FileNotFoundError("需要已下载的官方ACOLITE处理器和真实Sentinel-2 L1C产品")
    if "MSIL2A" in input_path.name:
        raise ValueError("ACOLITE入口需要L1C产品，不接受L2A输入")
    with atomic_directory(output) as stage, tempfile.TemporaryDirectory(prefix="chlorophyll-native-") as native_name:
        native = Path(native_name)
        if not str(native).isascii():
            raise ValueError("Windows原生NetCDF需要ASCII临时目录，请设置TEMP")
        actual_processor = processor.resolve()
        if not str(actual_processor).isascii():
            runtime = actual_processor.parents[2]
            if not (runtime / "config" / "defaults.txt").exists():
                raise ValueError("请把完整官方ACOLITE运行目录放在ASCII路径")
            shutil.copytree(runtime, native / "runtime")
            actual_processor = native / "runtime" / actual_processor.relative_to(runtime)
        actual_input = input_path.resolve()
        if not str(actual_input).isascii():
            actual_input = native / input_path.name
            if input_path.is_dir():
                shutil.copytree(input_path, actual_input, ignore=shutil.ignore_patterns("*_$folder$"))
            else:
                shutil.copy2(input_path, actual_input)
        processing = native / "output"
        settings = write_acolite_settings(actual_input, processing, limit)
        command = [str(actual_processor), "--cli", "--settings", str(settings)]
        with (stage / "processor.log").open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            finally:
                log.flush()
                shutil.copy2(stage / "processor.log", output.with_name(output.name + ".last-processor.log"))
        if result.returncode != 0:
            # 复制失败日志到最终目录外，以保留可诊断证据，原子目录仍回滚。
            failed = output.with_name(output.name + ".failed.log")
            failed.write_bytes((stage / "processor.log").read_bytes())
            raise RuntimeError(f"ACOLITE失败，返回码{result.returncode}，日志{failed}")
        for item in processing.iterdir():
            if item.is_dir():
                shutil.copytree(item, stage / item.name)
            else:
                shutil.copy2(item, stage / item.name)
        products = list(stage.rglob("*L2W*.nc"))
        if not products:
            raise RuntimeError("处理器返回0但没有L2W水离Rrs产品；只有L2R/rhos不能认定链路成功")
        contracts = []
        for product in products:
            contract = inspect_product(product)
            if not contract["has_quality_mask"]:
                raise RuntimeError("生成的L2W缺少l2_flags，不能作为可预测产品提交")
            contracts.append(contract)
        input_files = (
            {str(p.relative_to(input_path)): sha256(p) for p in sorted(input_path.rglob("*")) if p.is_file()}
            if input_path.is_dir()
            else {input_path.name: sha256(input_path)}
        )
        write_json(stage / "input_manifest.json", input_files)
        write_json(
            stage / "correction_provenance.json",
            {
                "processor": str(processor.resolve()),
                "processor_sha256": sha256(processor),
                "input": str(input_path.resolve()),
                "input_sha256": sha256(input_path) if input_path.is_file() else None,
                "input_manifest_sha256": sha256(stage / "input_manifest.json"),
                "limit": list(limit),
                "resolution_m": 20,
                "rsr_version": "base/unversioned",
                "command": command,
                "products": [str(p.relative_to(stage)) for p in products],
                "validated_product_contracts": contracts,
            },
        )
    return output


def open_netcdf(path):
    """Windows原生NetCDF不支持中文文件名时，ROI使用内存映像读取。"""
    import xarray as xr

    path = Path(path)
    if str(path.resolve()).isascii():
        return xr.open_dataset(path, decode_coords="all")
    import netCDF4

    dataset = netCDF4.Dataset("inmemory", memory=path.read_bytes())
    return xr.open_dataset(xr.backends.NetCDF4DataStore(dataset), decode_coords="all")


def predict_raster(input_path, model_dir, output_dir, quality_path=None):
    import xarray as xr
    import rasterio
    from rasterio.transform import from_origin
    from pyproj import CRS
    from .satellite_model import SatellitePredictor

    input_path = Path(input_path)
    model = SatellitePredictor(model_dir)
    with open_netcdf(input_path) as data:
        info = inspect_product(input_path)
        if model.card["sensor"] != info["sensor"]:
            raise ValueError("卫星产品与专用模型传感器不一致")
        west, north = validate_grid(data.x.values, data.y.values)
        stack = np.stack([as_rrs(data[name].transpose("y", "x").values, name) for name in info["variables"]], axis=-1)
        attrs = data.attrs
        crs = None
        for name in data.variables:
            candidate = data[name].attrs
            for key in ["spatial_ref", "crs_wkt", "proj4_string"]:
                if candidate.get(key):
                    crs = CRS.from_user_input(candidate[key])
                    break
            if crs:
                break
            if "grid_mapping_name" in candidate:
                crs = CRS.from_cf(candidate)
                break
        if crs is None and attrs.get("proj4_string"):
            crs = CRS.from_user_input(attrs["proj4_string"])
        if crs is None or not crs.is_projected or any(a.unit_name != "metre" for a in crs.axis_info):
            raise ValueError("必须有明确的米制投影坐标系")
        quality_source = Path(quality_path) if quality_path else input_path
        with open_netcdf(quality_source) as quality:
            if "l2_flags" not in quality:
                raise ValueError("缺少ACOLITE l2_flags；请提供对应L2W质量产品，不能假设所有像元有效")
            if not np.array_equal(quality.x.values, data.x.values) or not np.array_equal(
                quality.y.values, data.y.values
            ):
                raise ValueError("质量掩膜与反射率网格不一致")
            flags = quality.l2_flags.transpose("y", "x").values
        valid = np.isfinite(stack).all(axis=-1) & np.isfinite(flags) & (flags == 0) & (stack >= 0).all(axis=-1)
        if not valid.any():
            raise ValueError("质量筛选后没有有效水体像元；不得把全NoData栅格当作成功反演")
        values = np.full(stack.shape[:2], -9999, dtype=np.float32)
        if valid.any():
            values[valid] = model.predict(stack[valid]).astype(np.float32)
        with atomic_directory(output_dir) as stage:
            profile = dict(
                driver="GTiff",
                height=stack.shape[0],
                width=stack.shape[1],
                count=1,
                crs=crs.to_wkt(),
                transform=from_origin(west, north, 20, 20),
                compress="deflate",
            )
            with rasterio.open(stage / "chlorophyll_ug_l.tif", "w", dtype="float32", nodata=-9999, **profile) as dst:
                dst.write(values, 1)
                dst.set_band_description(1, "chlorophyll_a_ug_L")
                dst.update_tags(
                    units="ug/L",
                    sensor=info["sensor"],
                    model_sha256=sha256(Path(model_dir) / "model.json"),
                    validation_status=model.card["validation_status"],
                )
            with rasterio.open(stage / "valid_water_mask.tif", "w", dtype="uint8", nodata=255, **profile) as dst:
                dst.write(valid.astype("uint8"), 1)
            write_json(
                stage / "provenance.json",
                {
                    **info,
                    "quality_sha256": sha256(quality_source),
                    "model_sha256": sha256(Path(model_dir) / "model.json"),
                    "valid_pixels": int(valid.sum()),
                    "total_pixels": int(valid.size),
                    "unit": "ug/L",
                    "validation_status": model.card["validation_status"],
                    "mask": "finite nonnegative seven-band Rrs and l2_flags==0",
                    "physical_conversion": "Rrs unchanged; rhow divided by pi; rhos/rhot rejected",
                },
            )
            from .satellite_report import write_satellite_report

            context = input_path.with_name(input_path.name.replace("_L2W.nc", "_L2R_rgb_rhos.png"))
            write_satellite_report(stage, context)
    return Path(output_dir)
