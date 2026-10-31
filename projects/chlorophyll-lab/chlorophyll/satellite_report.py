"""直接从冻结GeoTIFF重建可检查报告，不重新预测。"""

import base64
import html
from pathlib import Path
import numpy as np
import pandas as pd
from .common import read_json, write_json, sha256


def write_satellite_report(output_dir, context_image=None):
    import rasterio
    from pyproj import Transformer

    directory = Path(output_dir)
    provenance = read_json(directory / "provenance.json")
    with rasterio.open(directory / "chlorophyll_ug_l.tif") as source:
        data = source.read(1, masked=True)
        valid = ~np.ma.getmaskarray(data)
        values = data.compressed()
        if not len(values):
            raise ValueError("全NoData产物不可生成成功报告")
        rows, cols = np.where(valid)
        x, y = rasterio.transform.xy(source.transform, rows, cols)
        lon, lat = Transformer.from_crs(source.crs, 4326, always_xy=True).transform(x, y)
        pd.DataFrame(
            {
                "row": rows,
                "col": cols,
                "easting_m": x,
                "northing_m": y,
                "longitude": lon,
                "latitude": lat,
                "chlorophyll_ug_l": np.asarray(data[valid]),
            }
        ).to_csv(directory / "valid_pixel_predictions.csv", index=False)
        low, high = float(values.min()), float(values.max())
        rectangles = []
        for row, col, value in zip(rows, cols, np.asarray(data[valid])):
            fraction = (float(value) - low) / max(high - low, 1e-8)
            color = f"rgb({int(30 + 210 * fraction)},{int(160 - 80 * fraction)},{int(170 - 130 * fraction)})"
            rectangles.append(f'<rect x="{col}" y="{row}" width="1" height="1" fill="{color}"/>')
        image = f'<svg viewBox="0 0 {source.width} {source.height}" role="img" aria-label="有效像元浓度估计"><rect width="100%" height="100%" fill="#dce2e4"/>{"".join(rectangles)}</svg>'
        stats = {
            "valid_pixels": int(valid.sum()),
            "total_pixels": int(valid.size),
            "coverage_fraction": float(valid.mean()),
            "min_ug_l": low,
            "median_ug_l": float(np.median(values)),
            "max_ug_l": high,
            "crs": source.crs.to_string(),
            "resolution_m": list(source.res),
            "concentration_geotiff_sha256": sha256(directory / "chlorophyll_ug_l.tif"),
            "mask_sha256": sha256(directory / "valid_water_mask.tif"),
        }
    write_json(directory / "product_summary.json", stats)
    context = ""
    if context_image and Path(context_image).exists():
        encoded = base64.b64encode(Path(context_image).read_bytes()).decode()
        context = f'<section><h2>同区域ACOLITE校正反射率RGB</h2><img alt="影像上下文" src="data:image/png;base64,{encoded}"></section>'
    page = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Sentinel-2真实影像工程验收</title>
<style>body{{font:16px system-ui,"Microsoft YaHei";margin:35px auto;max-width:1100px;padding:20px;background:#f2f7f5;color:#163e37}}section{{padding:24px;background:white;border-radius:14px;margin:20px 0}}p{{line-height:1.9}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:20px}}img,svg{{width:100%;max-height:600px}}.notice{{background:#fff2d7;padding:20px}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}</style>
<h1>Sentinel-2B · 真实L1C到浓度栅格</h1><p>汉江ROI · 20米 · L1C → ACOLITE 20260421.0 → 七波段专用模型 → GeoTIFF</p>
<section class="notice"><b>处理链路已实际运行；卫星精度尚未经独立实测匹配验证。</b><p>严格质量筛选保留 {stats["valid_pixels"]} / {stats["total_pixels"]} 像元（{stats["coverage_fraction"]:.2%}）。其余像元为NoData，不能插值填充后当作完整水质图。</p></section>
<div class="grid">{context}<section><h2>仅有效像元的浓度估计</h2>{image}<p>灰色：不输出估计。色标由蓝绿（{low:.3f}）到橙红（{high:.3f} μg/L）；中位数 {stats["median_ug_l"]:.3f} μg/L。</p></section></div>
<section><h2>核验与复现</h2><p>投影：{html.escape(stats["crs"])}，像元20×20米。采用默认ACOLITE质量标记为0、七波段有限且非负的像元。未为增加覆盖率放宽质量标准。</p><p>模型来自GLORIA实测光谱按S2B实际响应函数积分后的独立随机森林；原位71波段CNN未用于填充卫星输入。此报告不提供卫星浓度实测误差或置信区间。</p><p>逐像元值见 valid_pixel_predictions.csv；完整来源、质量和模型哈希见 provenance.json；GeoTIFF校验值见 product_summary.json。</p><p>输入：{html.escape(Path(provenance["input"]).name)}</p></section></html>"""
    (directory / "report.html").write_text(page, encoding="utf-8")
    return stats
