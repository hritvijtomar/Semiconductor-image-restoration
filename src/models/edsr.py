import torch
import torch.nn as nn

from blocks import ResidualBlock
from upsampling import Upsample


class EDSR(nn.Module):
    def __init__(self, scale=2, channels=64, num_blocks=16, res_scale=0.1):
        super().__init__()

        # Head: grayscale input
        self.head = nn.Conv2d(1, channels, kernel_size=3, padding=1)

        # Body: residual blocks
        body = []
        for _ in range(num_blocks):
            body.append(ResidualBlock(channels, res_scale))
        self.body = nn.Sequential(*body)

        # Learned 2× upsampling
        self.upsample = Upsample(scale, channels)

        # Tail: grayscale output
        self.tail = nn.Conv2d(channels, 1, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.head(x)

        residual = x

        x = self.body(x)

        x = x + residual

        x = self.upsample(x)

        x = self.tail(x)

        return x