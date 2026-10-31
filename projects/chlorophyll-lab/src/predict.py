"""加载训练时的标准化参数和最佳权重，对 CSV 中每条光谱预测。"""
import argparse
from pathlib import Path
import numpy as np
from .models import MLP


def load_predictor(run):
    with np.load(run / 'preprocess.npz') as f:
        prep = {key: f[key].copy() for key in f.files}
    if (run / 'model.npz').exists():
        with np.load(run / 'model.npz') as f:
            model = MLP(hidden=f['w1'].shape[1])
            model.params = {key: f[key].copy() for key in f.files}
        forward = model.forward
    else:
        import torch
        from .torch_model import SpectralCNN
        model = SpectralCNN()
        model.load_state_dict(torch.load(run / 'model.pt', map_location='cpu', weights_only=True))
        model.eval()
        def forward(x):
            with torch.no_grad():
                return model(torch.tensor(x, dtype=torch.float32).unsqueeze(1)).numpy()
    def predict(x):
        x = np.atleast_2d(np.asarray(x, dtype=float))
        if x.shape[1] != 64 or not np.isfinite(x).all():
            raise ValueError('输入必须是有限数值组成的 [N,64] 光谱矩阵')
        return forward((x - prep['xm']) / prep['xs']) * prep['ys'] + prep['ym']
    return predict


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--input', type=Path, required=True, help='带一行表头的64列CSV；波段顺序须与训练一致')
    args = parser.parse_args()
    x = np.loadtxt(args.input, delimiter=',', skiprows=1, ndmin=2)
    values = load_predictor(args.run)(x)
    print('sample,predicted_chlorophyll_ug_L')
    for index, value in enumerate(values[:, 0]):
        print(f'{index},{value:.4f}')


if __name__ == '__main__':
    main()
