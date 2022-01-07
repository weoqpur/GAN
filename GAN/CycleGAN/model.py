import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, channel=64, norm='inorm'):
        super(Generator, self).__init__()
        # encoder
        self.enc1 = CBR2d(in_channels, channel, kernel_size=7, padding=4, norm=norm, relu=0.0)
        self.enc2 = CBR2d(channel, channel * 2, kernel_size=3, stride=2, norm=norm, relu=0.0)
        self.enc3 = CBR2d(channel * 2, channel * 4, kernel_size=3, stride=2, norm=norm, relu=0.0)

        res = list()
        for i in range(9):
            res += ResBlock(channel * 4, channel * 4, kernel_size=3, stride=1, padding=1, norm=norm)
        self.res = nn.Sequential(*res)

        self.dec1 = DECBDR2d(4 * channel, 2 * channel, kernel_size=3, padding=1, norm=norm, relu=0.0, stride=2)
        self.dec2 = DECBDR2d(2 * channel, 1 * channel, kernel_size=3, padding=1, norm=norm, relu=0.0, stride=2)
        self.dec3 = CBR2d(1 * channel, out_channels, kernel_size=7, padding=3, norm=None, relu=None, stride=1)

    def forward(self, x):
        x = self.enc1(x)
        x = self.enc2(x)
        x = self.enc3(x)

        x = self.res(x)

        x = self.dec1(x)
        x = self.dec2(x)
        x = self.dec3(x)

        x = torch.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels, out_channels, channels=64, norm='inorm'):
        super(Discriminator, self).__init__()
        self.cbr1 = CBR2d(in_channels=in_channels, out_channels=channels, kernel_size=4, stride=2, padding=1, norm=None, relu=0.2, bias=False)
        self.cbr2 = CBR2d(in_channels=channels, out_channels=channels * 2, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.cbr3 = CBR2d(in_channels=channels * 2, out_channels=channels * 4, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.cbr4 = CBR2d(in_channels=channels * 4, out_channels=channels * 8, kernel_size=4, stride=2, padding=1, norm=norm, relu=0.2, bias=False)
        self.cbr5 = CBR2d(in_channels=channels * 8, out_channels=out_channels, kernel_size=4, stride=2, padding=1, norm=None, relu=None, bias=False)

    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        x = self.cbr4(x)
        x = self.cbr5(x)

        x = torch.sigmoid(x)

        return x

class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True, norm='bnorm', relu=0.0):
        super(ResBlock, self).__init__()
        layers = list()

        layers += [CBR2d(in_channels, out_channels, kernel_size, stride, padding, bias, norm, relu)]
        layers += [CBR2d(in_channels, out_channels, kernel_size, stride, padding, bias, norm, relu=None)]

        self.resblock = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.resblock(x)

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






