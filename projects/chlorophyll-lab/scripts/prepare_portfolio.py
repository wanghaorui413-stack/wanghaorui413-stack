"""Build a portable, read-only project showcase from verified local evidence."""

from pathlib import Path
import hashlib
import json
import shutil
import zipfile
from html.parser import HTMLParser


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "portfolio/chlorophyll-lab"


def main():
    manifest = json.loads((ROOT / "dist/RELEASE-MANIFEST.json").read_text(encoding="utf-8"))
    for name, evidence in manifest["artifacts"].items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == evidence["sha256"], name
    OUTPUT.mkdir(parents=True, exist_ok=True)
    satellite = ROOT / "predictions/hanriver_s2b_20240531"
    for name in ["report.html", "chlorophyll_ug_l.tif", "valid_water_mask.tif", "product_summary.json"]:
        shutil.copy2(satellite / name, OUTPUT / name)
    shutil.copy2(ROOT / "dist/chlorophyll_lab-0.2.0-py3-none-any.whl", OUTPUT)
    page = """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="叶绿素a浓度研究工程：真实GLORIA光谱、神经网络与Sentinel-2影像处理。">
<title>Chlorophyll Lab · 叶绿素浓度研究工程</title><style>
:root{font-family:system-ui,"Microsoft YaHei",sans-serif;color:#153b36;background:#f0f6f3}*{box-sizing:border-box}
body{margin:0}main{max-width:1080px;margin:auto;padding:56px 24px}header{border-bottom:1px solid #bbcfc7;padding-bottom:36px}
.eyebrow{letter-spacing:.16em;color:#47766a;font-size:13px}h1{font-size:clamp(32px,6vw,60px);line-height:1.15;margin:20px 0}
p,li{line-height:1.9}h2{font-size:23px}section{margin:28px 0;padding:28px;background:white;border-radius:14px}
.grid{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}.stat{background:#153b36;color:white;padding:24px;border-radius:12px}
.stat strong{display:block;font-size:32px}.notice{background:#fff0cf;border-left:5px solid #ba7f22}
a{color:#00665b;text-underline-offset:4px}nav{display:flex;flex-wrap:wrap;gap:12px;margin:24px 0}
nav a{padding:12px 20px;border:1px solid #00665b;border-radius:7px}table{border-collapse:collapse;width:100%}td,th{text-align:left;padding:12px;border-bottom:1px solid #d8e4df}
code{overflow-wrap:anywhere}footer{font-size:14px;color:#4d6e64}@media(max-width:680px){.grid{grid-template-columns:1fr}section{padding:20px}main{padding:32px 16px}}
</style></head><body><main><header><p class="eyebrow">REMOTE SENSING / MACHINE LEARNING / REPRODUCIBLE RESEARCH</p>
<h1>Chlorophyll Lab<br>从实测光谱到真实卫星影像</h1>
<p>面向个人科研与求职展示的叶绿素 a 浓度研究工程。将公开真实数据、神经网络实验、模型发布与卫星栅格处理连接成可复现的本地流程。</p>
<nav aria-label="项目资料"><a href="report.html">查看真实卫星处理报告</a><a href="chlorophyll_lab-0.2.0-py3-none-any.whl" download>下载 Python 安装包</a></nav></header>
<section class="notice"><h2>结果的适用范围</h2><p>真实 L1C 处理链路已经执行；卫星浓度尚未经过独立实测匹配验证。严格质控只保留 124 / 35,055 个像元（0.35%），其余为 NoData。此示例不能代表完整区域水质。</p></section>
<div class="grid"><div class="stat"><strong>3,385</strong>GLORIA 真实配对样本</div><div class="stat"><strong>32 / 32</strong>本机工程测试通过</div><div class="stat"><strong>20 m</strong>真实卫星输出网格</div></div>
<section><h2>两条建模与处理路线</h2><p><b>原位光谱：</b>GLORIA Rrs → 来源与水体分组隔离 → 训练集预处理 → MLP / 1D CNN 与传统基线 → 验证水体宏平均对数误差选模 → 冻结评估 → 独立模型包。</p>
<p><b>卫星影像：</b>Sentinel-2 L1C → 官方 ACOLITE 水体大气校正与残余反光校正 → 七波段水离 Rrs → 专用随机森林 → 浓度 GeoTIFF、有效掩膜与来源报告。卫星模型以真实响应函数卷积的 GLORIA 光谱训练。</p>
<p>卫星宽波段没有被补成 71 列冒充原位神经网络输入；L2A 地表反射率没有被当作 Rrs。</p></section>
<section><h2>工程实现与验证</h2><ul><li>Python 可安装包与统一 CLI；数据、模型和预测记录版本及哈希。</li><li>暂存与原子提交支持失败重试；冻结预测可重新生成报告。</li><li>独立模型包、本地 CSV 批量预测、页面状态失效保护与结构化错误。</li><li>Windows / Python 3.12 / CPU 下 32 项测试通过；安装包在源码目录外验证，第三方运行库复用本机环境。</li></ul>
<p>原位主模型按验证集规则选中 MLP。260 条开发保留样本上的总体 log10 RMSE 为 0.2387、RMSE 为 51.69 μg/L。这是历史开发评估，不是新的外部盲测或卫星精度，也不表示所有指标均优于随机森林。</p></section>
<section><h2>真实示例与可下载结果</h2><p>韩国汉江 ROI · Sentinel-2B · 2024-05-31 · EPSG:32652 · ACOLITE 20260421.0。影像公开源对象通过 MD5 核查，校正与推理均在本机实际执行。</p>
<nav><a href="chlorophyll_ug_l.tif" download>浓度 GeoTIFF</a><a href="valid_water_mask.tif" download>有效像元掩膜</a><a href="product_summary.json">产物统计与校验值</a></nav>
<p>数据来源：<a href="https://doi.org/10.1594/PANGAEA.948492">Lehmann 等（2022），GLORIA 数据集</a>（CC BY 4.0）；卫星处理器：<a href="https://github.com/acolite/acolite">ACOLITE 官方项目</a>。</p></section>
<footer>Chlorophyll Lab 0.2 · 2026-10-31 复核 · 本页为静态项目展示，预测需在本地 Python 环境执行。</footer></main></body></html>"""
    (OUTPUT / "index.html").write_text(page, encoding="utf-8")

    class Links(HTMLParser):
        def handle_starttag(self, tag, attrs):
            if tag == "a":
                href = dict(attrs).get("href", "")
                if href and not href.startswith(("https:", "http:", "#")):
                    assert (OUTPUT / href).is_file(), href

    Links().feed(page)
    assert "C:/Users" not in page and "C:\\Users" not in page
    archive = ROOT / "portfolio/chlorophyll-lab-showcase.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as target:
        for path in sorted(OUTPUT.iterdir()):
            target.write(path, path.relative_to(OUTPUT))
    print(f"Release hashes and showcase links verified: {OUTPUT}")
    print(f"Upload bundle: {archive}")


if __name__ == "__main__":
    main()
