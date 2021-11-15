# pipeline

# generator
# input -> Conv(k9n64s1)-> PReLU-> (Conv(k3n64s1)-> BN-> PReLU-> Conv(k3n64s1)-> BN) * 5->
# Conv(k3n64s1)-> PReLU-> (Conv-> PixelShuffle-> PReLU)-> Conv(k3n64s1) -> output

# discriminator
# input -> Conv(k3n64s1)-> Leaky ReLU-> (Conv(k3n64s2)-> BN-> Leaky ReLU)->
# (Conv(k3n128s1)-> BN-> Leaky ReLU)-> (Conv(k3n128s2)-> BN-> Leaky ReLU)->
# (Conv(k3n256s1)-> BN-> Leaky ReLU)-> (Conv(k3n256s2)-> BN-> Leaky ReLU)->
# (Conv(k3n512s1)-> BN-> Leaky ReLU)-> (Conv(k3n512s2)-> BN-> Leaky ReLU)->
# Conv (1024)-> Leaky ReLU-> Conv (1)-> Sigmoid -> 1 or 0

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.Conv1 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.prelu = nn.PReLU()
        self.Conv2 = nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = self.conv1(x)
        residual = self.bn1(residual)
        residual = self.prelu(residual)
        residual = self.conv2(residual)
        residual = self.bn2(residual)

        return x + residual




# Generator
class Generator(nn.Module):
    def __init__(self, scale_factor):
        upsample_block_num = int(math.log(scale_factor, 2))

        super(Generator, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=9, padding=4),
            nn.PReLU()
        )
        self.block2 = ResidualBlock(64)
        self.block3 = ResidualBlock(64)
        self.block4 = ResidualBlock(64)
        self.block5 = ResidualBlock(64)
        self.block6 = ResidualBlock(64)
        self.block7 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64)
        )
