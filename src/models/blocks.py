import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, channels=64, res_scale=0.1):
        super().__init__()

        self.body = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        )

        self.res_scale = res_scale

    def forward(self, x):
        residual = self.body(x)
        residual = residual * self.res_scale
        return x + residual