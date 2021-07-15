import torch
from torch import nn

from dpipe import layers


class UNet2D(nn.Module):
    def __init__(self, structure, n_chans_in, n_chans_out):
        super().__init__()
        first_channels = structure[0][0][0]
        last_channels = structure[0][-1][-1]

        self.unet = nn.Sequential(
            nn.Conv2d(n_chans_in, first_channels, kernel_size=3, padding=1, bias=False),

            layers.FPN(
                layers.ResBlock2d, nn.MaxPool2d(kernel_size=2, ceil_mode=True), nn.Identity(),
                lambda left, down: torch.add(*layers.interpolate_to_left(left, down, 1)),
                structure, kernel_size=3, padding=1
            )
        )

        self.out = nn.Sequential(layers.ResBlock2d(last_channels, last_channels, kernel_size=3, padding=1),
                                 layers.PreActivation2d(last_channels, n_chans_out, kernel_size=1))

    def forward(self, x):
        x = self.unet(x)
        return self.out(x)
