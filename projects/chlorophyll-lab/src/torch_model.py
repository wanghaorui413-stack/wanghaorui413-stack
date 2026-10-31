"""框架版：Conv1d 输入是 [batch, channel, wavelength]。"""
from torch import nn


class SpectralCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),  # [B,8,32]
            nn.Conv1d(8, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool1d(2),  # [B,16,16]
            nn.Flatten(), nn.Linear(16 * 16, 32), nn.ReLU(),
            nn.Dropout(0.1), nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.network(x)
