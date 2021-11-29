import torch
import torch.nn as nn


class CBR2d(nn.Module):
    def __init__(self, in_channel, upscale_factor, kernel_size=4, stride=2, padding=1, bias=True, norm='bnorm', relu=0.0):
        super().__init__()

        self.conv = nn.Conv2d(in_channels=in_channel, out_channels=in_channel * upscale_factor, kernel_size=kernel_size,
                              stride=stride, padding=padding, bias=bias)
        if not norm is None:
            self.norm = nn.BatchNorm2d(in_channel * upscale_factor) if norm == 'bnorm' else self.norm = nn.InstanceNorm2d(in_channel * upscale_factor)
        self.relu = nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)

    def forward(self, x):
        CBR2d1 = self.conv(x)
        CBR2d2 = self.norm(CBR2d1)
        CBR2d3 = self.relu(CBR2d2)
        return CBR2d3





