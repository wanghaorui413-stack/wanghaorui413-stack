from pathlib import Path

source = Path("web/chlorophyll.html")
text = source.read_text(encoding="utf-8")
text = text.replace("按水体名称分组验证", "按水体与采集来源隔离验证")
text = text.replace("03 / 独立水体测试结果", "03 / 冻结分组评估")
text = text.replace("选模使用验证集 log10 RMSE", "选模指标与分组规则见实验报告")
text = text.replace("独立测试 ${q.splits.test.samples}", "保留评估 ${q.splits.test.samples}")
if 'href="/satellite-report"' not in text:
    text = text.replace(
        "查看学习曲线与散点报告</a>",
        '查看学习曲线与散点报告</a> · <a href="/satellite-report" target="_blank">真实卫星示例报告</a>',
    )
source.write_text(text, encoding="utf-8")
Path("chlorophyll/assets/chlorophyll.html").write_text(text, encoding="utf-8")
