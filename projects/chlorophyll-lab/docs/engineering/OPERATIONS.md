# 工程版0.2运行手册

验证目标：Windows、Python3.12、CPU，个人科研与求职展示。项目同时保留原模拟教学路线和历史gloria_v1，工程主线使用gloria_v2及gloria_engineering_v2。

## 本机直接运行

在VS Code打开`C:\Users\DELL\Desktop\科研项目\nn_from_zero`，解释器选父目录`.venv/Scripts/python.exe`。

```powershell
..\.venv\Scripts\python.exe -m chlorophyll --version
..\.venv\Scripts\python.exe -m chlorophyll serve --run gloria_engineering_v2 --port 8767 --open
```

若8767已有本任务启动的服务，直接访问`http://127.0.0.1:8767`，不要重复启动。此页面用于原位CSV推理；较大的卫星SAFE产品通过CLI处理。

## 安装与独立工作目录

换电脑用Python3.12创建虚拟环境后，安装CPU版PyTorch，再安装项目或wheel：

```powershell
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install ".[dev,satellite]"
# 或安装dist目录的chlorophyll_lab-0.2.0-py3-none-any.whl及相应依赖
chlorophyll --version
chlorophyll --home 'D:\chlorophyll-work' download
chlorophyll --home 'D:\chlorophyll-work' prepare --dataset gloria_v2
```

`CHLOROPHYLL_HOME`环境变量也可指定工作目录。源码不必与数据、运行和发布包放在同一目录。`requirements-engineering-lock.txt`记录本机依赖版本；包含CPU版PyTorch，重建时需要其官方CPU索引。

wheel已离开源码目录安装到全新临时项目目录，验证了入口、包内页面资源和独立工作目录；这项测试复用了本机第三方运行库，不等同于新操作系统上的完整依赖安装。

## 原位研究流程

```powershell
# 已准备的gloria_v2拒绝覆盖；不要为重跑删除来源与历史结果
chlorophyll train --config configs/train-gloria-v2.json --run my_new_run
chlorophyll evaluate --run my_new_run
# 报告生成失败或需复核展示时，只从冻结预测重建，不重新选模
chlorophyll rebuild-report --run my_new_run
chlorophyll export --run my_new_run --output releases/my_model
chlorophyll predict --package releases/my_model --input data/prepared/gloria_v2/prediction_example.csv --output predictions/my_results.csv
```

显式CLI训练参数优先于JSON配置。数据准备、训练和最终评估暂存后原子提交；失败不占用最终名称。完成目录不覆盖。预测CSV包含模型实例哈希、输入哈希、UTC生成时间和模式版本。

发布包只保留选定模型与预处理；不依赖原始训练数据或全部候选模型。仅信任自行生成的本地模型包；同目录哈希用于完整性核查，不是第三方安全签名。

自有CSV从CLI导入时必须显式声明单位：

```powershell
chlorophyll prepare --dataset own_v1 --csv paired.csv --source-name '实测来源及质控说明' --rrs-unit 'sr^-1' --target-unit 'ug/L'
```

支持声明mg/m3（与ug/L数值相等）、mg/L（乘1000）和水离反射率`water_leaving_reflectance`（除π转为Rrs）。L2A地表反射率不属于该水离反射率选项。

## 卫星真实示例

已下载并实际处理的成功场景：

`S2B_MSIL1C_20240531T021529_N0510_R003_T52SCG_20240531T034003.SAFE`

来源是GCP公开Sentinel-2镜像；`scripts/fetch_gcp_l1c.py`按目录API返回的对象清单下载、断点重试并核查源MD5。每景保存独立来源清单；`scripts/select_demo_scene.py`记录候选云量，所选景产品云量约0.44%。元数据云量只是预筛选，实际像元仍严格使用ACOLITE质量标记。

官方ACOLITE 20260421.0处理器由`fetch_satellite_demo.py processor`获取，下载哈希及URL保留。Windows原生NetCDF不能直接处理当前中文绝对路径，因此`correct`在ASCII临时执行区运行官方完整处理器和输入副本，成功后提交回项目；TEMP也含非ASCII时会明确拒绝，需要先配置ASCII临时目录。对象存储的`_$folder$`不是SAFE数据，执行副本排除它们，原始来源记录保留。

```powershell
chlorophyll satellite correct --processor data/satellite/runtime/acolite_py_win/dist/acolite/acolite.exe --input data/satellite/gcp_l1c/S2B_MSIL1C_20240531T021529_N0510_R003_T52SCG_20240531T034003.SAFE --output data/satellite/my_correction --limit 37.505 126.98 37.535 127.025
chlorophyll satellite inspect --input data/satellite/hanriver_20240531_acolite/S2B_MSI_2024_05_31_02_27_13_T52SCG_L2W.nc
chlorophyll satellite predict --input data/satellite/hanriver_20240531_acolite/S2B_MSI_2024_05_31_02_27_13_T52SCG_L2W.nc --model releases/S2B_MSI_gloria_v2 --output predictions/my_satellite_product
```

输入为已生成的L2W大写Rrs产品；不把小写次表层rrs、L2A、rhos或rhot静默替代。可选接受明确定义的rhow/π。只支持已训练的S2A/S2B，拒绝S2C及未知传感器。

七波段为B2/B3/B4/B5/B6/B7/B8A；专用随机森林以GLORIA实测光谱和ACOLITE实际RSR积分后训练，记录响应文件URL与哈希。不是把卫星宽波段插成71列再送入原位CNN。

### 已生成的卫星产物

目录`predictions/hanriver_s2b_20240531/`：

- `chlorophyll_ug_l.tif`：EPSG:32652、20m、NoData=-9999，带单位及模型哈希标签。
- `valid_water_mask.tif`：有效像元1、其他0，NoData定义255。
- `valid_pixel_predictions.csv`：像元行列、投影坐标、经纬度和估计值。
- `provenance.json`、`product_summary.json`：输入、质量和模型哈希及覆盖统计。
- `report.html`：离线打开的影像上下文、浓度估计与限制说明。

默认严格规则仅保留124/35,055像元，约0.35%；没有放宽掩膜。2024-03-22的首次尝试全部被排除，保留为失败诊断；新代码会拒绝全NoData“成功产品”。

**这证明真实处理链路完成，不证明卫星浓度精度。** 卫星—实测匹配仍是独立里程碑；稀疏输出不能解释成完整区域水质图。

## 服务与检查

本地服务仅绑定127.0.0.1，限制请求体、读取时间和并发。响应包含请求ID，预期输入错误与内部500有稳定JSON协议；日志记录状态和耗时，不记录原始CSV正文。输入/模型变化会使旧结果失效，失败后不能继续导出旧结果。

```powershell
python -m unittest discover -s tests -v
python -m ruff check chlorophyll tests
python -m build --wheel --no-isolation
```

`.github/workflows/ci.yml`提供Windows检查配置；本次在本机执行了对应检查，未发布GitHub仓库或声称远程CI已运行。未承诺Linux实测、公网认证、多租户或商业SLA。
