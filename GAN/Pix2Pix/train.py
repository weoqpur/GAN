import os
import numpy as np
import pandas as pd
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
import torchvision.utils as utils
from utils import init_weights
from tqdm import tqdm
from torch.utils.data import DataLoader

from torchvision import transforms

from model import Generator, Discriminator
from data_loader import Dataset

parser = argparse.ArgumentParser(description='Train Pix2Pix')
parser.add_argument('--crop_size', default=256, type=int, help='training images crop size')
parser.add_argument('--num_epochs', default=200, type=int, help='training epoch number')
parser.add_argument('--scale_factor', default=2, type=int, help='Pix2Pix scale factor')
parser.add_argument('--opts', nargs='+', default=['direction'])
parser.add_argument('--batch_size', default=16, type=int, help='map data batch size')

opt = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# hyper parameters
crop_size = opt.crop_size
num_epochs = opt.num_epochs

transform = transforms.Compose([transforms.Resize((286, 286)),
                                transforms.RandomCrop(crop_size, crop_size),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                                transforms.ToTensor()])

# data loading
train_set = Dataset('./maps/train', opts=opt.opts)
val_set = Dataset('./maps/val', opts=opt.opts)
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)
val_loader = DataLoader(dataset=val_set, num_workers=0, batch_size=opt.batch_size, shuffle=False)

num_batch_train = int((train_set.__len__() / opt.batch_size) + ((train_set.__len__() / opt.batch_size) != 0))
num_batch_val = int((val_set.__len__() / opt.batch_size) + ((val_set.__len__() % opt.batch_size) != 0))

netG = Generator(opt.scale_factor).to(device)
netD = Discriminator(opt.scale_factor).to(device)

init_weights(netG)
init_weights(netD)

fn_loss = nn.BCELoss().to(device) # binary cross entropy
l1_loss = nn.L1Loss().to(device)

optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

for epoch in range(1, num_epochs + 1):
    netG.train()
    netD.train()

    loss_G_L1_train = []
    loss_G_gan_train = []
    loss_D_fake_train = []
    loss_D_real_train = []

    for batch, data in enumerate(train_loader, 1):
        label = data['label'].to(device)
        input = data['label'].to(device)

        print(input.shape)
        output = netG(input.permute(0, 3, 1, 2))

        optimizerD.zero_grad()

        fake = torch.cat((input, output), 1)
        pred_fake = netD(fake.detach())
        loss_D_fake = fn_loss(pred_fake, torch.zeros_like(pred_fake))

        real = torch.cat((input, label), 1)
        pred_real = netD(real)
        loss_D_real = fn_loss(pred_real, torch.ones_like(pred_real))

        loss_D = (loss_D_fake + loss_D_real) * 0.5

        loss_D.backward()

        optimizerD.step()


        optimizerG.zero_grad()

        fake = torch.cat((input, output), 1)
        pred_fake = netD(fake)
        loss_g_gan = fn_loss(pred_fake, torch.ones_like(pred_fake))
        loss_g_l1 = l1_loss(output, label)
        loss_g = loss_g_gan + 100 * loss_g_l1

        loss_g.backward()

        optimizerG.step()

        loss_G_L1_train += [loss_g_l1.item()]
        loss_G_gan_train += [loss_g_gan.item()]
        loss_D_real_train += [loss_D_real.item()]
        loss_D_fake_train += [loss_D_fake.item()]

        print("train: epoch %04d / %04d | batch: %04d / %04d | GAN gan %.4f | GAN L1 %.4f | DISC REAL: %.4f | DISC REAL: %.4f"
              % (num_epochs, epoch, num_batch_train, batch, np.mean(loss_G_gan_train), np.mean(loss_G_L1_train), np.mean(loss_D_real_train), np.mean(loss_D_fake_train)))







