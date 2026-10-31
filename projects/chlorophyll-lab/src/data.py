"""模拟光谱数据。这里的规律是人为指定的，不能作为真实水质反演证据。"""
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def generate(n=1800, seed=42):
    rng = np.random.default_rng(seed)
    wavelength = np.linspace(400, 800, 64)
    chlorophyll = rng.uniform(1, 40, n)
    sediment = rng.uniform(0, 1, n)
    illumination = rng.uniform(0.8, 1.2, n)
    # 仅为教学构造：浓度影响吸收和反射峰，另加干扰变量与噪声。
    absorption = np.exp(-((wavelength - 675) / 24) ** 2)
    green_peak = np.exp(-((wavelength - 550) / 45) ** 2)
    x = (0.12 + 0.08 * sediment[:, None]
         + 0.05 * np.log1p(chlorophyll[:, None]) * green_peak
         - 0.07 * (chlorophyll[:, None] / 40) * absorption)
    x = x * illumination[:, None] + rng.normal(0, 0.003, (n, 64))
    y = chlorophyll + rng.normal(0, 0.5, n)
    return x, y[:, None], wavelength


def prepare():
    x, y, wavelength = generate()
    order = np.random.default_rng(42).permutation(len(x))
    # 只在独立同分布的模拟数据中随机划分；真实数据必须审视站点/时间关联。
    train, val, test = order[:1260], order[1260:1530], order[1530:]
    xm, xs = x[train].mean(0), x[train].std(0).clip(1e-8)
    ym, ys = y[train].mean(0), y[train].std(0).clip(1e-8)
    directory = ROOT / 'data'
    directory.mkdir(exist_ok=True)
    np.savetxt(directory / 'sample_spectrum.csv', x[test[:1]], delimiter=',',
               header=','.join(f'r_{w:.1f}' for w in wavelength), comments='')
    np.savez(directory / 'synthetic.npz', x=x, y=y, wavelength=wavelength,
             train=train, val=val, test=test)
    return dict(x=x, y=y, train=train, val=val, test=test,
                xm=xm, xs=xs, ym=ym, ys=ys, wavelength=wavelength)
