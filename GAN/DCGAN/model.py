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


dataroot = "./img_align_celeba" ## 데이터 셋의 경로
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