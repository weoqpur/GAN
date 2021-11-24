import argparse
import os
from math import log10

import pandas as pd
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm

import ssim
from data_loader import TrainDataset, ValDataset, display_transform
from loss import GeneratorLoss
from model import Generator, Discriminator

parser = argparse.ArgumentParser(description='Train SRGAN')
parser.add_argument('--crop_size', default=88, type=int, help='training images crop size')
parser.add_argument('--upscale_factor', default=4, type=int, choices=[2, 4, 8], help='SR upscale factor')
parser.add_argument('--num_epochs', default=100, type=int, help='train epoch number')

opt = parser.parse_args()

# hyper parameters
crop_size = opt.crop_size
upscale_factor = opt.upscale_factor
num_epochs = opt.num_epochs

# data loading
train_set = TrainDataset('./images/train', crop_size=crop_size, upscale_factor=upscale_factor)
val_set = ValDataset('./images/val', upscale_factor=upscale_factor)
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=64, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=1, shuffle=False)

netG = Generator(upscale_factor)
print('# G parameters', sum(param.numel() for param in netG.parameters()))
netD = Discriminator()
print('# D parameters', sum(param.numel() for param in netD.parameters()))

generator_criterion = GeneratorLoss()

if torch.cuda.is_available():
    netG.cuda()
    netD.cuda()
    generator_criterion.cuda()

optimizerG = optim.Adam(netG.parameters())
optimizerD = optim.Adam(netD.parameters())

results = {'d_loss': [], 'g_loss': [], 'd_score': [], 'g_score': [], 'psnr': [], 'ssim': []}

for epoch in range(1, num_epochs + 1):
    train_bar = tqdm(train_loader)
    running_results = {'batch_sizes': 0, 'd_loss': 0, 'g_loss': 0, 'd_score': 0, 'g_score': 0}

    # network를 훈련모드로 전환
    netG.train()
    netD.train()
    with torch.autograd.set_detect_anomaly(True):
        for data, target in train_bar:
            ### g_update_first = True
            batch_size = data.size(0)
            # batch size 계산
            running_results['batch_sizes'] += batch_size
            # (1) Update D network: maximize D(x)-1-D(G(z))
            real_img = Variable(target)
            if torch.cuda.is_available():
                real_img = real_img.cuda()
            z = Variable(data)
            if torch.cuda.is_available():
                z = z.cuda()
            fake_img = netG(z)

            netD.zero_grad()
            real_out = netD(real_img).mean()
            fake_out = netD(fake_img).mean()
            d_loss = 1 - real_out + fake_out
            d_loss.backward(retain_graph=True)
            optimizerD.step()

            # (2) Update G network: minimize 1-D(G(z)) + Perception Loss + Image Loss + TV Loss
            netG.zero_grad()
            print("fake image 1" + str(fake_img.shape))
            ## 코랩 런타임 에러 방지 코드
            # fake_img = netG(z)
            # fake_out = netD(fake_img).mean()
            ##
            g_loss = generator_criterion(fake_out.clone(), fake_img.clone(), real_img.clone())
            print(g_loss)
            g_loss.backward()

            fake_img = netG(z)
            print("fake image 2" + str(fake_img.shape))
            fake_out = netD(fake_img).mean()

            optimizerG.step()

            # 최적화 전 loss
            running_results['g_loss'] += g_loss.item() * batch_size
            running_results['d_loss'] += d_loss.item() * batch_size
            running_results['d_score'] += real_out.item() * batch_size
            running_results['g_score'] += fake_out.item() * batch_size

            train_bar.set_description(desc='[%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f' % (
                epoch, num_epochs, running_results['d_loss'] / running_results['batch_sizes'],
                running_results['g_loss'] / running_results['batch_sizes'],
                running_results['d_score'] / running_results['batch_sizes'],
                running_results['g_score'] / running_results['batch_sizes']
            ))


print("성공")