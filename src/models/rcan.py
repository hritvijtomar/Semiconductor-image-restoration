import torch
import torch.nn as nn

from upsampling import Upsample


# -------------------------
# Channel attention
# -------------------------
class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        scale = self.attention(x)
        return x * scale


# -------------------------
# Residual channel attention block
# -------------------------
class RCAB(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1),
            ChannelAttention(channels, reduction),
        )

    def forward(self, x):
        return x + self.block(x)


# -------------------------
# Residual group
# -------------------------
class ResidualGroup(nn.Module):
    def __init__(self, channels, num_blocks, reduction=16):
        super().__init__()

        blocks = [RCAB(channels, reduction) for _ in range(num_blocks)]
        blocks.append(nn.Conv2d(channels, channels, 3, padding=1))

        self.group = nn.Sequential(*blocks)

    def forward(self, x):
        return x + self.group(x)


# -------------------------
# Lightweight RCAN
# -------------------------
class RCAN(nn.Module):
    def __init__(
        self,
        channels=64,
        num_groups=5,
        num_blocks=5,
        reduction=16,
        scale=2,
    ):
        super().__init__()

        self.head = nn.Conv2d(1, channels, 3, padding=1)

        groups = [
            ResidualGroup(channels, num_blocks, reduction)
            for _ in range(num_groups)
        ]
        groups.append(nn.Conv2d(channels, channels, 3, padding=1))

        self.body = nn.Sequential(*groups)

        self.upsample = Upsample(scale, channels)

        self.tail = nn.Conv2d(channels, 1, 3, padding=1)

    def forward(self, x):
        x = self.head(x)

        residual = x
        x = self.body(x)
        x = x + residual

        x = self.upsample(x)
        x = self.tail(x)

        return x