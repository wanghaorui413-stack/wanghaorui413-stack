"""可逐行推导的两层 MLP。批次维 B、波段维 D=64、隐藏维 H。"""
import numpy as np


class MLP:
    def __init__(self, inputs=64, hidden=32, seed=42):
        rng = np.random.default_rng(seed)
        self.params = {
            'w1': rng.normal(0, np.sqrt(1 / inputs), (inputs, hidden)),
            'b1': np.zeros((1, hidden)),
            'w2': rng.normal(0, np.sqrt(1 / hidden), (hidden, 1)),
            'b2': np.zeros((1, 1)),
        }

    def forward(self, x):
        p = self.params
        h = np.tanh(x @ p['w1'] + p['b1'])  # [B,64] @ [64,H] -> [B,H]
        return h @ p['w2'] + p['b2']       # [B,H] @ [H,1] -> [B,1]

    def loss_and_gradients(self, x, y):
        p = self.params
        h = np.tanh(x @ p['w1'] + p['b1'])
        prediction = h @ p['w2'] + p['b2']
        error = prediction - y
        loss = np.mean(error ** 2)
        dout = 2 * error / len(x)          # MSE 对输出的导数
        dh = (dout @ p['w2'].T) * (1 - h ** 2)  # tanh 的导数和链式法则
        return float(loss), {
            'w1': x.T @ dh, 'b1': dh.sum(0, keepdims=True),
            'w2': h.T @ dout, 'b2': dout.sum(0, keepdims=True),
        }

    def step(self, gradients, lr):
        for name in self.params:
            self.params[name] -= lr * gradients[name]
