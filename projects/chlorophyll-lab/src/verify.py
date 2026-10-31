"""验证关键数学和流程性质，非对实现的简单复写。"""
import argparse
import json
from pathlib import Path
import numpy as np
from .data import prepare
from .models import MLP
from .predict import load_predictor
from .report import metrics


def check_gradients():
    rng = np.random.default_rng(7)
    x, y = rng.normal(size=(5, 64)), rng.normal(size=(5, 1))
    model = MLP(hidden=4)
    _, gradients = model.loss_and_gradients(x, y)
    worst = 0.0
    for key, param in model.params.items():
        for _ in range(8):
            index = tuple(rng.integers(s) for s in param.shape)
            old = param[index]
            param[index] = old + 1e-5
            plus = np.mean((model.forward(x) - y) ** 2)
            param[index] = old - 1e-5
            minus = np.mean((model.forward(x) - y) ** 2)
            param[index] = old
            numerical = (plus - minus) / 2e-5
            worst = max(worst, abs(numerical - gradients[key][index]))
    assert worst < 1e-6, worst
    print(f'PASS finite-difference gradient check, max absolute error={worst:.2e}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run', type=Path)
    args = parser.parse_args()
    check_gradients()
    data = prepare()
    sets = [set(data[k]) for k in ['train','val','test']]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(set.union(*sets)) == len(data['x'])
    assert np.allclose(data['xm'], data['x'][data['train']].mean(0))
    model = MLP(hidden=16)
    x = (data['x'][data['train'][:16]] - data['xm']) / data['xs']
    y = (data['y'][data['train'][:16]] - data['ym']) / data['ys']
    initial, _ = model.loss_and_gradients(x, y)
    for _ in range(600):
        _, grad = model.loss_and_gradients(x, y)
        model.step(grad, 0.01)
    final, _ = model.loss_and_gradients(x, y)
    assert final < initial * 0.05, (initial, final)
    print(f'PASS disjoint splits and tiny-batch overfit: {initial:.5f} -> {final:.5f}')
    if args.run:
        saved = np.loadtxt(args.run / 'test_predictions.csv', delimiter=',', skiprows=1)
        prediction = load_predictor(args.run)(data['x'][saved[:, 0].astype(int)])
        assert np.allclose(prediction[:, 0], saved[:, 2], atol=1e-4)
        report = json.loads((args.run / 'metrics.json').read_text(encoding='utf-8'))
        score = metrics(saved[:, 1:2], prediction)
        for key in score:
            assert abs(score[key] - report['test_metrics'][report['backend']][key]) < 1e-4
        assert score['rmse'] < report['test_metrics']['mean_baseline']['rmse']
        print('PASS checkpoint reload, preprocessing, metric recomputation, mean-baseline comparison')


if __name__ == '__main__':
    main()
