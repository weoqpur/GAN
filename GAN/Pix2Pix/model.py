import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        channel = 64
        # encoder
        self.enc1 = CBR2d(3, channel, norm=None, relu=0.2)
        self.enc2 = CBR2d(channel, channel * 2, relu=0.2)
        self.enc3 = CBR2d(channel * 2, channel * 4, relu=0.2)
        self.enc4 = CBR2d(channel * 4, channel * 8, relu=0.2)
        self.enc5 = CBR2d(channel * 8, channel * 8, relu=0.2)
        self.enc6 = CBR2d(channel * 8, channel * 8, relu=0.2)
        self.enc7 = CBR2d(channel * 8, channel * 8, relu=0.2)
        self.enc8 = CBR2d(channel * 8, channel * 8, relu=0.2)

        # decoder
        self.dec1 = DECBDR2d(channel * 8, channel * 8)
        self.dec2 = DECBDR2d(channel * 8 * 2, channel * 8)
        self.dec3 = DECBDR2d(channel * 8 * 2, channel * 8)
        self.dec4 = DECBDR2d(channel * 8 * 2, channel * 8, drop=None)
        self.dec5 = DECBDR2d(channel * 8 * 2, channel * 4, drop=None)
        self.dec6 = DECBDR2d(channel * 8, channel * 2, drop=None)
        self.dec7 = DECBDR2d(channel * 4, channel, drop=None)
        self.dec8 = DECBDR2d(channel * 2, 3, norm=None, relu=None, drop=None)

        self.tanh = nn.Tanh()

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)
        enc6 = self.enc6(enc5)
        enc7 = self.enc7(enc6)
        enc8 = self.enc8(enc7)

        dec1 = self.dec1(enc8)

        cat1 = torch.cat((dec1, enc7), dim=1)
        dec2 = self.dec2(cat1)

        cat2 = torch.cat((dec2, enc6), dim=1)
        dec3 = self.dec3(cat2)

        cat3 = torch.cat((dec3, enc5), dim=1)
        dec4 = self.dec4(cat3)

        cat4 = torch.cat((dec4, enc4), dim=1)
        dec5 = self.dec5(cat4)

        cat5 = torch.cat((dec5, enc3), dim=1)
        dec6 = self.dec6(cat5)

        cat6 = torch.cat((dec6, enc2), dim=1)
        dec7 = self.dec7(cat6)

        cat7 = torch.cat((dec7, enc1), dim=1)
        dec8 = self.dec8(cat7)

        return self.tanh(dec8)

class Discriminator(nn.Module):
    def __init__(self, factor):
        super(Discriminator, self).__init__()
        channel = 64
        self.cbr1 = CBR2d(6, channel, norm=None, relu=0.2)
        self.cbr2 = CBR2d(channel, channel * factor, relu=0.2)
        self.cbr3 = CBR2d(channel * factor, channel * factor * 2, relu=0.2)
        self.cbr4 = CBR2d(channel * factor * 2, channel * factor * 4, relu=0.2)
        self.cbr5 = CBR2d(channel * factor * 4, 1, relu=0.2)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        x = self.cbr1(x)
        x = self.cbr2(x)
        x = self.cbr3(x)
        x = self.cbr4(x)
        x = self.cbr5(x)

        return self.sig(x)

class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=True, norm='bnorm', relu=0.0):
        super(CBR2d, self).__init__()
        layer = []
        layer += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size,
                            stride=stride, padding=padding, bias=bias)]
        if not norm is None:
            layer += [nn.BatchNorm2d(out_channels) if norm == 'bnorm' else nn.InstanceNorm2d(out_channels)]
        layer += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.cbr = nn.Sequential(*layer)

    def forward(self, x):
        x2 = torch.tensor(x.clone(), dtype=torch.float32)
        return self.cbr(x2)


class DECBDR2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=4, stride=2, padding=1, bias=True, norm='bnorm', relu=0.0, drop=0.5):
        super(DECBDR2d, self).__init__()
        layer = []

        layer += [nn.ConvTranspose2d(in_channels=in_channel, out_channels=out_channel, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias, output_padding=0)]

        if not norm is None:
            layer += [nn.BatchNorm2d(out_channel) if norm == 'bnorm' else nn.InstanceNorm2d(out_channel)]
        if not drop is None:
            layer += [nn.Dropout2d(drop)]
        if not relu is None:
            layer += [nn.ReLU() if relu == 0 else nn.LeakyReLU(relu)]

        self.decbdr = nn.Sequential(*layer)

    def forward(self, x):
        x2 = torch.tensor(x.clone(), dtype=torch.float32)
        return self.decbdr(x2)






