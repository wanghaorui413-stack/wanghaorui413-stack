"""运行 python -m src.train --backend numpy 或 --backend torch。"""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from .data import ROOT, prepare
from .models import MLP
from .report import metrics, write_report


def train_numpy(x, y, xv, yv, args, out):
    model = MLP(hidden=args.hidden, seed=args.seed)
    rng = np.random.default_rng(args.seed)
    best_loss, best_epoch, best, history = float('inf'), 0, None, []
    for epoch in range(1, args.epochs + 1):
        order = rng.permutation(len(x))
        for start in range(0, len(x), args.batch_size):
            batch = order[start:start + args.batch_size]
            _, gradients = model.loss_and_gradients(x[batch], y[batch])
            model.step(gradients, args.lr)
        train_loss = float(np.mean((model.forward(x) - y) ** 2))
        val_loss = float(np.mean((model.forward(xv) - yv) ** 2))
        if not np.isfinite(train_loss + val_loss):
            raise ValueError('损失非有限值，请检查数据或降低学习率。')
        history.append(dict(epoch=epoch, train_mse=train_loss, val_mse=val_loss))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            best = {key: value.copy() for key, value in model.params.items()}
        if epoch == 1 or epoch % 20 == 0:
            print(f'epoch={epoch:3d} train_mse={train_loss:.5f} val_mse={val_loss:.5f}')
    model.params = best
    np.savez(out / 'model.npz', **best)
    return model.forward, history, best_epoch


def train_torch(x, y, xv, yv, args, out):
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .torch_model import SpectralCNN
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    def tensor(a):
        return torch.tensor(a, dtype=torch.float32)
    xt, yt = tensor(x).unsqueeze(1), tensor(y)
    xvt, yvt = tensor(xv).unsqueeze(1), tensor(yv)
    loader = DataLoader(TensorDataset(xt, yt), batch_size=args.batch_size,
                        shuffle=True, generator=torch.Generator().manual_seed(args.seed))
    model = SpectralCNN()  # CPU 默认可复现，小数据无需 GPU
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = torch.nn.MSELoss()
    best_loss, best_epoch, history = float('inf'), 0, []
    for epoch in range(1, args.epochs + 1):
        model.train()  # 开启 dropout
        for xb, yb in loader:
            optimizer.zero_grad()      # 清掉上批次梯度
            loss = criterion(model(xb), yb)
            loss.backward()            # 自动微分
            optimizer.step()           # 更新参数
        model.eval()                   # 关闭 dropout
        with torch.no_grad():
            train_loss = criterion(model(xt), yt).item()
            val_loss = criterion(model(xvt), yvt).item()
        if not np.isfinite(train_loss + val_loss):
            raise ValueError('损失非有限值，请降低学习率或检查数据。')
        history.append(dict(epoch=epoch, train_mse=train_loss, val_mse=val_loss))
        if val_loss < best_loss:
            best_loss, best_epoch = val_loss, epoch
            torch.save(model.state_dict(), out / 'model.pt')
        if epoch == 1 or epoch % 20 == 0:
            print(f'epoch={epoch:3d} train_mse={train_loss:.5f} val_mse={val_loss:.5f}')
    model.load_state_dict(torch.load(out / 'model.pt', weights_only=True))
    model.eval()
    def predict(a):
        with torch.no_grad():
            return model(tensor(a).unsqueeze(1)).numpy()
    return predict, history, best_epoch


def main():
    parser = argparse.ArgumentParser(description='模拟高光谱回归教学项目')
    parser.add_argument('--backend', choices=['numpy', 'torch'], default='numpy')
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=None)
    parser.add_argument('--hidden', type=int, default=32, help='仅 NumPy MLP 使用')
    parser.add_argument('--seed', type=int, default=42, help='模型与批次随机种子；数据划分固定')
    parser.add_argument('--run', default=None, help='runs 下的新目录名，已有目录会拒绝覆盖')
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.hidden) < 1:
        parser.error('epochs、batch-size、hidden 必须为正整数')
    args.lr = args.lr if args.lr is not None else (0.01 if args.backend == 'numpy' else 0.001)
    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error('lr 必须是有限正数')
    name = args.run or args.backend
    if name in {'.', '..'} or Path(name).name != name or '/' in name or '\\' in name:
        parser.error('run 必须是简单目录名')
    # 先检查可选依赖，避免留下空的训练目录。
    if args.backend == 'torch':
        import torch
    out = ROOT / 'runs' / name
    out.mkdir(parents=True, exist_ok=False)
    data = prepare()
    np.savez(out / 'preprocess.npz', **{k: data[k] for k in ['xm','xs','ym','ys','wavelength']})
    np.savez(out / 'split.npz', **{k: data[k] for k in ['train','val','test']})
    x = (data['x'] - data['xm']) / data['xs']
    y = (data['y'] - data['ym']) / data['ys']
    ti, vi, si = data['train'], data['val'], data['test']
    fn = train_numpy if args.backend == 'numpy' else train_torch
    predict, history, best_epoch = fn(x[ti], y[ti], x[vi], y[vi], args, out)
    prediction = predict(x[si]) * data['ys'] + data['ym']
    # 简单基线：训练均值 + 固定 alpha 的岭回归；均不拟合测试数据。
    ridge = np.linalg.solve(x[ti].T @ x[ti] + 1.0 * np.eye(x.shape[1]), x[ti].T @ y[ti])
    result = dict(backend=args.backend, data='synthetic only', best_epoch=best_epoch,
                  sizes={k: len(data[k]) for k in ['train','val','test']},
                  config=vars(args), versions={'numpy': np.__version__},
                  test_metrics={
                      'mean_baseline': metrics(data['y'][si], np.full_like(prediction, data['ym'][0])),
                      'ridge_alpha_1': metrics(data['y'][si], (x[si] @ ridge) * data['ys'] + data['ym']),
                      args.backend: metrics(data['y'][si], prediction)})
    if args.backend == 'torch':
        result['versions']['torch'] = torch.__version__
    (out / 'metrics.json').write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding='utf-8')
    with (out / 'history.csv').open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch','train_mse','val_mse'])
        writer.writeheader()
        writer.writerows(history)
    np.savetxt(out / 'test_predictions.csv', np.column_stack([si, data['y'][si], prediction]),
               delimiter=',', header='sample_id,true,prediction', comments='')
    write_report(out / 'report.html', history, result)
    print(json.dumps(result['test_metrics'], indent=2))
    print(f'Saved: {out}')


if __name__ == '__main__':
    main()
