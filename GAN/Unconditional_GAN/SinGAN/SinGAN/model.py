import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CBR2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(CBR2d, self).__init__()
        layers = list()

        layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                             stride=stride)]
        layers += [nn.BatchNorm2d(out_channels)]
        layers += [nn.LeakyReLU(0.2, inplace=True)]

        self.cbr = nn.Sequential(*layers)

    def forward(self, x):
        return self.cbr(x)

class WDiscriminator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(WDiscriminator, self).__init__()
        channels = 32
        self.head = CBR2d(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size, padding=padding,
                          stride=stride)
        self.body1 = CBR2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding,
                           stride=stride)
        self.body2 = CBR2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding,
                           stride=stride)
        self.body3 = CBR2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding,
                           stride=stride)
        self.tail = nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding,
                              stride=stride)

    def forward(self, x):
        x = self.head(x)
        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)
        x = self.tail(x)
        return x

class Generator(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride):
        super(Generator, self).__init__()
        channels = 32
        self.head = CBR2d(in_channels=in_channels, out_channels=channels, kernel_size=kernel_size, padding=padding,
                          stride=stride)
        self.body1 = CBR2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding,
                           stride=stride)
        self.body2 = CBR2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding,
                           stride=stride)
        self.body3 = CBR2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size, padding=padding,
                           stride=stride)
        self.tail = nn.Conv2d(in_channels=channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                              padding=padding)

    def forward(self, x, y):
        x = self.head(x)
        x = self.body1(x)
        x = self.body2(x)
        x = self.body3(x)
        x = self.tail(x)
        x = torch.tanh(x)
        ind = int((y.shape[2]-x.shape[2])/2)
        y = y[:,:,ind:(y.shape[2]-ind), ind:(y.shape[3]-ind)]
        return x+y
