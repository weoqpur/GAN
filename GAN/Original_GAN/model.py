import numpy as np
import torch
import torch.nn as nn

# 생성자 만들기 (Generator)
class Generator(nn.Module):
    # 초기값 설정 - 인풋 채널, 아웃풋 채널, normalization
    def __init__(self, img_shape):
        super(Generator, self).__init__()

        self.block1 = Block(100, 64 * 2, norm=False)
        self.block2 = Block(64 * 2, 64 * 4)
        self.block3 = Block(64 * 4, 64 * 8)
        self.block4 = Block(64 * 8, 64 * 16)
        self.linear = nn.Linear(1024, 256),
        self.tanh = nn.Tanh()

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.linear(x)
        x = self.tanh(x)

        return x

# DCGAN 판별자(discriminator) 만들기
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.dis = nn.Sequential(
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.dis(x)
        return x


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, norm=True):
        super(Block, self).__init__()
        layer = []
        layer += [nn.Linear(in_features=in_channel, out_features=out_channel)]
        if norm:
            layer += [nn.BatchNorm2d(out_channel, 0.8)]
        layer += (nn.LeakyReLU(0.2, inplace=True))

        self.block = nn.Sequential(*layer)

    def forward(self, x):
        return self.block(x.cloen())