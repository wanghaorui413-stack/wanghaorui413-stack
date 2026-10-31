import html
import math
import numpy as np
import pandas as pd


LABELS = {
    "mean_log": "训练几何均值",
    "ridge": "岭回归",
    "pls": "PLS",
    "random_forest": "随机森林",
    "mlp": "PyTorch MLP",
    "cnn": "多尺度残差 1D-CNN",
}


def learning_svg(history):
    rows = pd.read_csv(history)
    key = "val_macro_waterbody_log10_rmse" if "val_macro_waterbody_log10_rmse" in rows else "val_log10_rmse"
    hi = max(rows[key].max(), 0.01) * 1.08
    max_epoch = rows.epoch.max()
    colors = {"cnn": "#0c857c", "mlp": "#d47932"}
    polylines = []
    for name, part in rows.groupby("model"):
        pts = " ".join(
            f"{50 + (e - 1) / max(1, max_epoch - 1) * 650:.1f},{250 - v / hi * 205:.1f}"
            for e, v in zip(part.epoch, part[key])
        )
        polylines.append(f'<polyline points="{pts}" stroke="{colors[name]}" fill="none" stroke-width="2"/>')
    ticks = "".join(f'<text x="5" y="{250 - f * 205:.0f}" font-size="12">{f * hi:.2f}</text>' for f in [0, 0.5, 1])
    return f'<svg viewBox="0 0 740 290" role="img" aria-label="验证集对数RMSE学习曲线"><path d="M50 30 V250 H700" stroke="#9aacb6" fill="none"/>{ticks}{"".join(polylines)}<text x="50" y="280">1</text><text x="660" y="280">{max_epoch} epoch</text></svg>'


def scatter_svg(y, p):
    a, b = np.log10(y), np.log10(p)
    lo = math.floor(min(a.min(), b.min()))
    hi = math.ceil(max(a.max(), b.max()))
    span = max(hi - lo, 1)

    def xy(v):
        return 50 + (v - lo) / span * 410

    circles = "".join(
        f'<circle cx="{xy(v):.1f}" cy="{510 - xy(w):.1f}" r="2.3" fill="#087f78" opacity=".4"/>' for v, w in zip(a, b)
    )
    xticks = "".join(
        f'<text x="{xy(v):.1f}" y="483" text-anchor="middle" font-size="12">{v}</text><text x="38" y="{510 - xy(v):.1f}" text-anchor="end" font-size="12">{v}</text>'
        for v in range(lo, hi + 1)
    )
    return f'<svg viewBox="0 0 520 520" role="img" aria-label="独立测试集实测与预测浓度的log10散点图"><path d="M50 50 V460 H460" stroke="#9aacb6" fill="none"/><path d="M50 460 L460 50" stroke="#999" stroke-dasharray="5 5"/>{circles}{xticks}<text x="145" y="510">实测 log10(Chl-a / μg/L)</text><text x="8" y="18">预测 log10(Chl-a / μg/L)</text></svg>'


def write_reports(run, card, result, table):
    selected = card["selected_model"]
    evaluation_note = (
        "GLORIA历史开发评估：此前已查看GLORIA测试结果，本次不是全新外部独立盲测。"
        if card["source"].get("name") == "GLORIA-2022"
        else "当前评估仅针对声明的数据划分，不代表外部独立验证。"
    )
    lines = [
        "# 叶绿素 a 真实光谱反演：独立水体测试",
        "",
        f"数据：{card['source']['name']}；验证集预先选定模型：**{LABELS[selected]}**。",
        f"测试：{result['samples']} 个样本，{result['waterbodies']} 个水体名称组；选模指标：{card.get('selection_metric', 'validation log10_rmse')}。",
        "",
        "| 模型 | RMSE / μg/L | MAE / μg/L | R² | log10 RMSE | 中位误差倍数 |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    rows = []
    for name, m in result["metrics"].items():
        r2 = "N/A" if m["r2"] is None else f"{m['r2']:.4f}"
        values = [
            LABELS[name],
            f"{m['rmse_ug_l']:.3f}",
            f"{m['mae_ug_l']:.3f}",
            r2,
            f"{m['log10_rmse']:.4f}",
            f"{m['median_factor_error']:.3f}",
        ]
        lines.append("| " + " | ".join(values) + " |")
        rows.append("<tr>" + "".join(f"<td>{html.escape(v)}</td>" for v in values) + "</tr>")
    lines += [
        "",
        evaluation_note,
        "",
        "所有模型使用同一数据划分。几何均值基线和全部标准化仅使用训练集。测试结果未用于重新选择模型。",
        "",
        "log10 RMSE 衡量跨数量级误差；原始尺度 RMSE 会更受高浓度及过高预测影响。中位误差倍数越接近1越好。",
        "",
        "水体按标准化名称分组，尚未完成所有名称别名及空间边界消歧；这不是跨传感器或卫星影像验证。",
        "输入是原位遥感反射率 Rrs（sr^-1），不得直接用未完成大气校正的卫星DN值、辐亮度或无单位反射率替代。",
        "",
        f"来源：{card['source'].get('citation', card['source']['name'])}",
        "来源许可证：" + card["source"].get("license", "自有数据，使用者确认权限"),
    ]
    lines += [
        "",
        "水体宏平均 log10 RMSE（各水体等权）：",
        "",
        *[f"- {LABELS[name]}: {value:.4f}" for name, value in result.get("macro_waterbody_log10_rmse", {}).items()],
    ]
    (run / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    title = html.escape(LABELS[selected])
    source = html.escape(card["source"].get("citation", card["source"]["name"]))
    history = learning_svg(run / "history.csv")
    scatter = scatter_svg(table["observed_ug_l"].to_numpy(), table[selected].to_numpy())
    (run / "report.html").write_text(
        f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>叶绿素反演 · 独立测试报告</title><style>body{{font:15px system-ui,"Microsoft YaHei";max-width:1100px;margin:35px auto;padding:20px;color:#183c48;background:#f4f8f8}}section{{background:white;padding:26px;border-radius:14px;margin:20px 0;border:1px solid #deeaeb}}p{{line-height:1.8}}table{{width:100%;border-collapse:collapse}}th,td{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}svg{{width:100%;max-height:500px}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:25px}}@media(max-width:700px){{.grid{{grid-template-columns:1fr}}}}.table{{overflow:auto}}</style>
<h1>叶绿素 a 浓度反演 · 真实数据验证</h1><p>{html.escape(evaluation_note)}</p><p>验证集选定：<b>{title}</b>；独立测试 {result["samples"]} 个样本 / {result["waterbodies"]} 个水体名称组。</p>
<section class="table"><h2>同一独立测试集上的模型比较</h2><table><tr><th>模型</th><th>RMSE μg/L</th><th>MAE μg/L</th><th>R²</th><th>log10 RMSE</th><th>中位误差倍数</th></tr>{"".join(rows)}</table></section>
<div class="grid"><section><h2>验证学习曲线</h2><p>绿：CNN；橙：MLP。纵轴为模型卡声明的验证选模指标（新版本：水体宏平均 log10 RMSE），横轴 epoch。</p>{history}</section><section><h2>{title}：独立水体散点</h2><p>双轴为 log10 浓度；虚线为预测等于实测。</p>{scatter}</section></div>
<section><h2>如何理解结果</h2><p>浓度跨多个数量级，选模指标由模型卡冻结，新版本使用水体宏平均 log10 RMSE，同时报告样本加权及原始尺度误差。测试集只用于最终评价，不据此改选模型。水体名称分组尚未完成全球别名消歧；指标不代表卫星反演或任意新水体均能达到此精度。</p><p>{source}</p></section></html>""",
        encoding="utf-8",
    )
