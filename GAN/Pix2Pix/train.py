import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from utils import init_weights
from torch.utils.data import DataLoader
from torchvision import transforms

from model import Generator, Discriminator
from data_loader import Dataset, RandomCrop, Resize, Normalization

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

transform_train = transforms.Compose([
    Resize(shape=(286, 286, 3)),
    RandomCrop((256, 256)),
    Normalization(mean=0.5, std=0.5)
])

# data loading
train_set = Dataset('./maps/train', transform=transform_train)
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=opt.batch_size, shuffle=True)

num_batch_train = int((train_set.__len__() / opt.batch_size) + ((train_set.__len__() / opt.batch_size) != 0))

out_path = 'training_results/SRF_2/'
if not os.path.exists(out_path):
    os.makedirs(out_path)

netG = Generator().to(device)
netD = Discriminator(opt.scale_factor).to(device)

init_weights(netG)
init_weights(netD)

fn_loss = nn.BCELoss().to(device) # binary cross entropy
l1_loss = nn.L1Loss().to(device)

optimizerG = optim.Adam(netG.parameters(), lr=0.0002, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=0.0002, betas=(0.5, 0.999))

fn_tonumpy = lambda x: x.to('cpu').datach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean

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

        output = netG(input)

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

        if epoch % 20 == 0:
            input = fn_tonumpy(fn_denorm(input, mean=0.5, std=0.5)).squeeze()
            label = fn_tonumpy(fn_denorm(label, mean=0.5, std=0.5)).squeeze()
            output = fn_tonumpy(fn_denorm(output, mean=0.5, std=0.5)).squeeze()

            input = np.clip(input, a_min=0, a_max=1)
            label = np.clip(label, a_min=0, a_max=1)
            output = np.clip(output, a_min=0, a_max=1)

            plt.imsave(os.path.join(out_path, 'png', '%04d_input.png' % epoch/20), input[0], cmap=None)
            plt.imsave(os.path.join(out_path, 'png', '%04d_label.png' % epoch/20), label[0], cmap=None)
            plt.imsave(os.path.join(out_path, 'png', '%04d_output.png' % epoch/20), output[0], cmap=None)

    torch.save(netG.state_dict(), 'epochs/netG_epoch_%d_%d.pth' % (2, epoch))
    torch.save(netD.state_dict(), 'epochs/netD_epoch_%d_%d.pth' % (2, epoch))
    netG.eval()









