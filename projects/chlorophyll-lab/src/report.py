"""仅使用标准库生成可离线查看的学习曲线。"""
import html
import numpy as np


def metrics(y, prediction):
    return {'rmse': float(np.sqrt(np.mean((y - prediction) ** 2))),
            'mae': float(np.mean(np.abs(y - prediction))),
            'r2': float(1 - np.sum((y - prediction) ** 2)
                        / np.sum((y - y.mean()) ** 2))}


def write_report(path, history, result):
    width, height, margin = 760, 300, 45
    max_y = max(max(row['train_mse'], row['val_mse']) for row in history) * 1.05
    def points(key):
        return ' '.join(f'{margin + i / max(1, len(history)-1) * (width-2*margin):.1f},'
                        f'{height-margin-row[key]/max_y*(height-2*margin):.1f}'
                        for i, row in enumerate(history))
    lines = ''.join(f'<polyline fill="none" stroke="{color}" stroke-width="2" '
                    f'points="{points(key)}"/>' for key, color in
                    [('train_mse', '#2563eb'), ('val_mse', '#ef4444')])
    ticks = ''.join(f'<text x="2" y="{height-margin-f*(height-2*margin):.0f}" '
                    f'font-size="12">{f*max_y:.2f}</text>' for f in [0, .25, .5, .75, 1])
    rows = ''.join('<tr><td>' + html.escape(name) + '</td>' +
                   ''.join(f'<td>{value[key]:.4f}</td>' for key in ['rmse','mae','r2']) + '</tr>'
                   for name, value in result['test_metrics'].items())
    path.write_text(f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8">
<title>神经网络训练记录</title><style>body{{font:16px system-ui;max-width:900px;margin:40px auto;padding:20px;color:#172033}}td,th{{padding:12px;border-bottom:1px solid #ddd;text-align:left}}svg{{width:100%;height:auto}}p{{line-height:1.8}}</style>
<h1>光谱回归 · {html.escape(result['backend'])}</h1>
<p>全部结果来自人为生成的模拟数据，仅用于学习流程，不代表真实叶绿素反演能力。</p>
<h2>学习曲线</h2><p>蓝线：训练 MSE；红线：验证 MSE。纵轴使用标准化后的目标值，横轴为 epoch（1–{len(history)}）。验证曲线用于选模型。</p>
<svg viewBox="0 0 {width} {height}" role="img" aria-label="训练和验证损失随 epoch 的变化">
<path d="M45 45 V255 H715" fill="none" stroke="#777"/>{ticks}{lines}
<text x="45" y="285">1</text><text x="690" y="285">{len(history)}</text></svg>
<p>选用第 {result['best_epoch']} 轮的模型；测试集只在选定模型后评估。RMSE 和 MAE 使用还原后的模拟目标尺度（设定单位 μg/L）。</p>
<table><tr><th>模型</th><th>RMSE ↓</th><th>MAE ↓</th><th>R² ↑</th></tr>{rows}</table>
<p>思考：神经网络是否优于岭回归？如果没有，能否解释数据规模、生成规律和模型复杂度的关系？不要根据测试结果反复选超参数。</p></html>''', encoding='utf-8')
