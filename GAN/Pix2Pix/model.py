import torch
import torch.nn as nn


class CBR2d(nn.Module):
    def __init__(self, in_channel, upscale_factor, kernel_size=4, stride=2, padding=1, bias=True, norm='bnorm', relu=0.0):
        super(CBR2d, self).__init__()
        layer = []
        layer += [nn.Conv2d(in_channels=in_channel, out_channels=in_channel * upscale_factor, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=bias)]
        if not norm is None:
            layer += [nn.BatchNorm2d(in_channel * upscale_factor) if norm == 'bnorm' else nn.InstanceNorm2d(in_channel * upscale_factor)]
        layer += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(layer)

    def forward(self, x):
        return self.cbr(x)


class DECBDR2d(nn.Module):
    def __init__(self, in_channel, downscale_factor, kernel_size=4, stride=2, padding=1, bias=True, norm='bnorm', relu=0.0, drop=0.5):
        super(DECBDR2d, self).__init__()
        layer = []
        layer += [nn.Conv2d(in_channels=in_channel, out_channels=in_channel / downscale_factor, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias)]
        if not norm is None:
            layer += [nn.BatchNorm2d(in_channel * downscale_factor) if norm == 'bnorm' else nn.InstanceNorm2d(in_channel * downscale_factor)]
        if not drop is None:
            layer += [nn.Dropout2d(drop)]
        layer += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.decbdr = nn.Sequential(layer)

    def forward(self, x):
        return self.decbdr(x)




