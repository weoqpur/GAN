import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, channel=64, norm='inorm'):
        super(Generator, self).__init__()
        # encoder
        self.enc1 = CBR2d(in_channels, channel, kernel_size=7, padding=(7-1)//2, norm=norm, relu=0.0)
        self.enc2 = CBR2d(channel, channel * 2, kernel_size=3, stride=2, norm=norm, relu=0.0)
        self.enc3 = CBR2d(channel * 2, channel * 4, kernel_size=3, stride=2, norm=norm, relu=0.0)

    def forward(self, x):


class Discriminator(nn.Module):
    def __init__(self, factor):
        super(Discriminator, self).__init__()

    def forward(self, x):


class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0):
        super(CBR2d, self).__init__()
        layer = [nn.ReflectionPad2d(padding=padding)]
        layer += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=0, bias=bias)]
        if not norm is None:
            layer += [nn.BatchNorm2d(out_channels) if norm == 'bnorm' else nn.InstanceNorm2d(out_channels)]
        layer += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layer)

    def forward(self, x):
        x2 = torch.tensor(x.clone(), dtype=torch.float32)
        return self.cbr(x2)


class DECBDR2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0):
        super(DECBDR2d, self).__init__()
        layer = []

        layer += [nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride,
                                     padding=padding, bias=bias, output_padding=padding)]

        if not norm is None:
            layer += [nn.BatchNorm2d(out_channel) if norm == 'bnorm' else nn.InstanceNorm2d(out_channel)]
        if not relu is None:
            layer += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.decbdr = nn.Sequential(*layer)

    def forward(self, x):
        x2 = torch.tensor(x.clone(), dtype=torch.float32)
        return self.decbdr(x2)






