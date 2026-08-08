import torch.nn as nn

class Upsample(nn.Sequential):
    def __init__(self, scale, channels):
        layers = [
            nn.Conv2d(channels, channels * (scale ** 2), kernel_size=3, padding=1),
            nn.PixelShuffle(scale)
        ]
        super().__init__(*layers)