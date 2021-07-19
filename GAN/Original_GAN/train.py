## 학습을 시키기
import os
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from torchvision import transforms

from model import *

## 기본적인 변수 설정
lr = 2e-4
batch_size = 128
num_epoch = 200

data_dir = './datasets'
ckpt_dir = './checkpoint'
result_dir = './result'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## 추가적인 세팅
result_dir_train = os.path.join(result_dir, 'train')
result_dir_test = os.path.join(result_dir, 'test')

if not os.path.exists(result_dir_train):
    os.makedirs(os.path.join(result_dir_train, 'png'))
if not os.path.exists(result_dir_test):
    os.makedirs(os.path.join(result_dir_test, 'png'))
    os.makedirs(os.path.join(result_dir_test, 'numpy'))