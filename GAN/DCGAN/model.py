## DCGAN_faces_tutorial_clone_coding
from __future__ import print_function

import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

## 랜덤 시드 설정
manualSeed = 999
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)


dataroot = "./celeba" # 데이터 셋의 경로
workers = 2 # 데이터로더의 worker 수
batch_size = 128 # 배치 사이즈 설정
image_size = 64 # 이미지의 크기 설정
nc = 3 # 훈련 이미지의 채널 수 컬러 이미지의 경우 3개
nz = 100 # generator의 input 크기
ngf = 64 # generator의 feature map 크기
ndf = 64 # discriminator의 feature map 크기
num_epochs = 5 # training epochs의 수
lr = 0.0002 # Learning rate
beta1 = 0.5 # Adam optimizers를 위한 hyperparameter
ngpu = 0 # 사용 가능한 GPU의 수 CPU 모드에는 0을 사용


## 데이터셋 생성
dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
    transforms.Resize(image_size), # 이미지 사이즈 재설정
    transforms.CenterCrop(image_size), # 중심을 위주로 자르기
    transforms.ToTensor(), # 탠서로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    # 평균과 표준편차를 사용하여 텐서 이미지를 정규화
    # (평균, 표준) 각 채널에 대한 평균 시퀸스, 각 채널에 대한 표준 편차의 시퀸스
]))

## 데이터 로더 생성
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

## GPU 사용 여부 결정
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

## 일부 훈련 이미지 보기
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

### Generator 와 Discriminator에서 호출될 가중치 초기화
def weights_init(m):
    classname = m.__class__.__name__ # m의 클래스 이름 참조
    if classname.find('Conv') != -1: # 클래스 이름에 Conv가 있을경우 True
        nn.init.normal_(m.weight.data, 0.0, 0.02) # 입력 텐서를 정규 분포에서 가져온 값으로 채운다.
    elif classname.find('BatchNorm') != -1: # 클래스 이름에 BatchNorm이 있을경우 True
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0) # 주어진 텐서서 자리에 스칼라 값을 넣는다


## 생성자
class Generator(nn.Module):
    def __init__(self, ngpu):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input에 Z를 넣고 convolution 진행
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False), # 512개의 피처맵을 생성
            nn.BatchNorm2d(ngf * 8), # batchnormalization진행
            nn.ReLU(True),# 활성화 함수 사용
            # state size. (ngf * 8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf * 4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf * 2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False), # RGB값을 3개의 채널로 추출
            nn.Tanh() # Tanh 함수로 연산
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self, ngpu):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),# 2d convolution 연산을 진행
            nn.LeakyReLU(0.2, inplace=True), # LeakyReLU 사용
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf * 4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False), # 마지막은 채널을 하나만 출력
            nn.Sigmoid() # sigmoid함수로 연산
        )

    def forward(self, input):
        return self.main(input)


## generator를 생성
netG = Generator(ngpu).to(device)

## 사용할 수 있다면 GPU사용
if (device.type == 'cuda') and (ngpu > 1):
    netG = nn.DataParallel(netG, list(range(ngpu)))

## weights_init 함수를 적용하여 모든 가중치를 무작위로 초기화 합니다.
## to mean=0, stdev=0.2
netG.apply(weights_init)

## Discriminator
netD = Discriminator(ngpu).to(device)

## 멀티 GPU를 사용할지 말지 걱정
if (device.type == 'cuda') and (ngpu > 1):
    netD = nn.DataParallel(netD, list(range(ngpu)))

## weights_init 함수를 적용하여 모든 가중치를 무작위로 초기화 합니다.
## to mean=0, stdev=0.2
netD.apply(weights_init)

## 모델 출력
print(netG)
print(netD)

## BCELoss function 초기값 설정
criterion = nn.BCELoss()

## 시각화에 사용할 잠재 벡터 배치 생성
## generator 진행
fixed_noise = torch.randn(64, nz, 1, 1, device=device)

## 훈련 중 진짜 레이블과 가짜 레이블에 대한 규칙 설정
real_label = 1.
fake_label = 0.

## Adam optimizer 사용 G D 둘 다
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

## 훈련 루프

## 진행 상황 저장용 리스트
img_list = []
D_losses = []
G_losses = []
iters = 0

print("훈련 시작")

for epoch in range(num_epochs):
    # batch 단위로 반복
    for i, data in enumerate(dataloader, 0):

        ########################################################################
        # (1) Discriminator network 업데이트: maximize log(D(x)) + log(1 - D(G(z)))
        ########################################################################
        ## 진짜 batch 훈련 part
        netD.zero_grad() # 기울기를 0으로 설정
        # 배치를 구성
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
        # Discriminator에게 배치를 전방 전달
        output = netD(real_cpu).view(-1)
        # 모든 배치에서 손실 계산
        errD_real = criterion(output, label)
        # 후방 전달에서 Discriminator에 대한 기울기 계산
        errD_real.backword()
        D_x = output.mean().item()

        ## 가짜 batch 훈련 part
        # 평균은 0 분산은 1인 노이즈 생성
        noise = torch.randn(b_size, nz, 1, 1, device=device)
        # Generator로 가짜 이미지 배치 생성
        fake = netG(noise)
        label.fill_(fake_label)
        # 모든 가짜 배치를 D로 분류
        output = netD(fake.detach()).view(-1)
        # 모든 가짜 배치에서 D의 오차 계산
        errD_fake = criterion(output, label)
        # 이전 gradient로 누적합된 이 배치에 대한 gradient를 계산
        errD_fake.backward()
        D_G_z1 = output.mean().item()
        # 가짜 배치와 진짜 배치에 대한 합으로 Discriminator의 오차를 계산
        errD = errD_real + errD_fake
        # update D
        optimizerD.step()

        #####################################################
        # (2) Generator network 업데이트: maximize log(D(G(z)))
        #####################################################

