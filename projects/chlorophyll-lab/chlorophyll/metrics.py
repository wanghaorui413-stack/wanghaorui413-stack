import numpy as np


def invert_target(z, mean, std):
    log_value = np.asarray(z, dtype=np.float64) * std + mean
    return np.power(10.0, np.clip(log_value, -6, 6))


def scores(y, p):
    y, p = np.asarray(y, dtype=np.float64), np.asarray(p, dtype=np.float64)
    if y.shape != p.shape or not np.isfinite(y).all() or not np.isfinite(p).all() or (y <= 0).any() or (p <= 0).any():
        raise ValueError("指标要求同形状、有限且为正的真实值和预测值")
    denominator = np.sum((y - y.mean()) ** 2)
    log_error = np.log10(p) - np.log10(y)
    return {
        "rmse_ug_l": float(np.sqrt(np.mean((y - p) ** 2))),
        "mae_ug_l": float(np.mean(np.abs(y - p))),
        "r2": float(1 - np.sum((y - p) ** 2) / denominator) if denominator > 0 else None,
        "log10_rmse": float(np.sqrt(np.mean(log_error**2))),
        "median_factor_error": float(np.median(np.maximum(p / y, y / p))),
        "bias_ug_l": float(np.mean(p - y)),
    }
