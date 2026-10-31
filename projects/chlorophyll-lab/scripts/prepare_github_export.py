"""Prepare an explicit source allowlist for the user's profile repository."""

from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
files = []
for directory, suffixes in [("chlorophyll", {".py", ".html"}), ("src", {".py"}), ("tests", {".py"}), ("scripts", {".py"}), ("configs", {".json"}), ("web", {".html"})]:
    files.extend(p for p in (root / directory).rglob("*") if p.suffix in suffixes and "__pycache__" not in p.parts)
files.extend(root / name for name in ["pyproject.toml", ".gitignore", "GLOSSARY.md", "requirements-engineering-lock.txt", "docs/engineering/OPERATIONS.md", "docs/engineering/RELEASE-ACCEPTANCE.md", "predictions/hanriver_s2b_20240531/product_summary.json", "predictions/hanriver_s2b_20240531/report.html"])
entries = []
for path in files:
    relative = path.relative_to(root).as_posix()
    if relative.startswith("predictions/"):
        relative = "example/" + path.name
    entries.append({"path": "projects/chlorophyll-lab/" + relative, "mode": "100644", "type": "blob", "content": path.read_text(encoding="utf-8-sig")})
readme = '''# Chlorophyll Lab 0.2

叶绿素 a 浓度研究工程：真实 GLORIA 光谱、MLP / 1D CNN、可复现实验与 Sentinel-2 卫星栅格处理。

## 已实现

- 原位 Rrs（400–750 nm，5 nm间隔）→ 数据质控、来源/水体分组隔离 → 基线与神经网络 → 宏平均对数误差选模 → 冻结评估与独立模型发布。
- 真实 Sentinel-2 L1C → 官方 ACOLITE 水体大气校正 → 七波段专用随机森林 → 20米浓度 GeoTIFF、有效掩膜与报告。
- 原子提交、失败恢复、数据/模型哈希、预测溯源、统一命令、本地页面与结构化错误。

## 本地复现

```powershell
git clone https://github.com/wanghaorui413-stack/wanghaorui413-stack.git
cd wanghaorui413-stack/projects/chlorophyll-lab
py -3.12 -m venv .venv
.venv/Scripts/python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python.exe -m pip install ".[dev,satellite]"
.venv/Scripts/python.exe -m chlorophyll --version
.venv/Scripts/python.exe -m chlorophyll download
.venv/Scripts/python.exe -m chlorophyll prepare --dataset gloria_v2
.venv/Scripts/python.exe -m chlorophyll train --config configs/train-gloria-v2.json --run research_v1
.venv/Scripts/python.exe -m chlorophyll evaluate --run research_v1
.venv/Scripts/python.exe -m chlorophyll serve --run research_v1 --port 8767 --open
```

Python 3.12 / Windows / CPU 已实测。原始数据与处理器通过脚本获取，不随源码重复上传；首次下载和训练需要时间。详见[运行手册](docs/engineering/OPERATIONS.md)。手册中父目录`.venv`及历史模型路径对应原开发工作区；本仓库可使用上述项目内`.venv`和新运行名。部分辅助脚本面向原开发环境，主线使用`python -m chlorophyll`命令。

## 验证与科学边界

2026-10-31 本机32项测试通过、无跳过；代码及格式检查、原发布文件哈希检查通过。运行测试：`python -m unittest discover -s tests -v`。未下载数据、未训练历史模型或未安装Node/遥感依赖时，相关测试会跳过；本机32项结果不等于全新环境32项离线必通过。无既有模型的工程测试使用独立临时夹具完成准备、训练、评估与恢复。

原位数据保留3,385条：训练2,811、验证314、开发保留260；验证选中MLP。保留集总体log10 RMSE=0.2387、RMSE=51.69 μg/L，属于历史开发评估，不是新的外部盲测，也不表示所有指标优于随机森林。

真实卫星示例：2024-05-31韩国汉江ROI，S2B，EPSG:32652，20米。严格质控仅124/35,055像元有效（0.35%）；卫星浓度尚未经独立实测匹配验证，不能当作完整区域水质图。卫星宽波段不补成71列套用原位CNN，L2A地表反射率不当作Rrs。

[真实影像报告HTML（下载后打开）](example/report.html) · [产物统计](example/product_summary.json) · [验收说明](docs/engineering/RELEASE-ACCEPTANCE.md)

原验收说明中的完整SAFE、处理器、历史运行、模型发布包和GeoTIFF保留在开发工作区；本次公开发布包含源码、测试、配置和离线HTML证据，不宣称这些大型文件也已上传。GitHub文件页面不会直接执行HTML；本地打开报告即可查看内嵌影像。

## 数据与第三方软件

GLORIA: Lehmann, Moritz K et al. (2022), *GLORIA – A global dataset of remote sensing reflectance and water quality from inland and coastal waters*, PANGAEA, [doi:10.1594/PANGAEA.948492](https://doi.org/10.1594/PANGAEA.948492), CC BY 4.0。代码包含质控、单位转换、分组及传感器响应积分处理。

[ACOLITE官方项目](https://github.com/acolite/acolite)与依赖遵循各自许可；处理器不在此分发。当前为个人科研展示项目，不提供业务SLA或公网预测服务。
'''
entries.append({"path": "projects/chlorophyll-lab/README.md", "mode": "100644", "type": "blob", "content": readme})
out = root / "tmp/github-export.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps(entries, ensure_ascii=False), encoding="utf-8")
print(f"Prepared {len(entries)} public source and evidence files")
