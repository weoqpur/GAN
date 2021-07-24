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


dataroot = "./celeba" ## 데이터 셋의 경로
workers = 2 ## 데이터로더의 worker 수
batch_size = 128 ## 배치 사이즈 설정
image_size = 64 ## 이미지의 크기 설정
nc = 3 ## 훈련 이미지의 채널 수 컬러 이미지의 경우 3개
nz = 100 ## generator의 input 크기
ngf = 64 ## generator의 feature map 크기
ndf = 64 ## discriminator의 feature map 크기
num_epochs = 5 ## training epochs의 수
lr = 0.0002 ## Learning rate
beta1 = 0.5 ## Adam optimizers를 위한 hyperparameter
ngpu = 0 ## 사용 가능한 GPU의 수 CPU 모드에는 0을 사용


## 데이터셋 생성
dataset = dset.ImageFolder(root=dataroot, transform=transforms.Compose([
    transforms.Resize(image_size), ## 이미지 사이즈 재설정
    transforms.CenterCrop(image_size), ## 중심을 위주로 자르기
    transforms.ToTensor(), ## 탠서로 변환
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ## 평균과 표준편차를 사용하여 텐서 이미지를 정규화
    ## (평균, 표준) 각 채널에 대한 평균 시퀸스, 각 채널에 대한 표준 편차의 시퀸스
]))

## 데이터 로더 생성
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

## GPU 사용 여부 결정
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# 일부 훈련 이미지 보기
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()

## Generator 와 Discriminator에서 호출될 가중치 초기화
def weights_init(m):
    classname = m.__class__.__name__ ## m의 클래스 이름 참조
    if classname.find('Conv') != -1: ## 클래스 이름에 Conv가 있을경우 True
        nn.init.normal_(m.weight.data, 0.0, 0.02) ## 입력 텐서를 정규 분포에서 가져온 값으로 채운다.
    elif classname.find('BatchNorm') != -1: ## 클래스 이름에 BatchNorm이 있을경우 True
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0) ## 주어진 텐서서 자리에 스칼라 값을 넣는다

