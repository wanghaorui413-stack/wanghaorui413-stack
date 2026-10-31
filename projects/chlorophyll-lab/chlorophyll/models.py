from torch import nn
from .common import FEATURES


class SpectralMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Flatten(),
            nn.Linear(len(FEATURES), 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


class MultiScaleResidual(nn.Module):
    """同一光谱上的窄/宽卷积核，加上直接传递的残差。"""

    def __init__(self, channels=16):
        super().__init__()
        self.narrow = nn.Conv1d(channels, channels, 3, padding=1)
        self.wide = nn.Conv1d(channels, channels, 7, padding=3)
        self.fuse = nn.Conv1d(channels * 2, channels, 1)
        self.relu = nn.ReLU()

    def forward(self, x):
        import torch

        features = torch.cat([self.relu(self.narrow(x)), self.relu(self.wide(x))], dim=1)
        return self.relu(x + self.fuse(features))


class SpectralCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 16, 5, padding=2),
            nn.ReLU(),
            MultiScaleResidual(16),
            nn.MaxPool1d(2),
            nn.Conv1d(16, 32, 5, padding=2),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(8),
            nn.Flatten(),
            nn.Linear(32 * 8, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        return self.network(x).squeeze(-1)


def create_model(name):
    if name == "cnn":
        return SpectralCNN()
    if name == "mlp":
        return SpectralMLP()
    raise ValueError(f"Unknown neural model {name}")
