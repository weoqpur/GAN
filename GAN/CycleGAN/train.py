import os
import matplotlib.pyplot as plt
import numpy as np
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision import transforms
import itertools

from model import Generator, Discriminator
from data_loader import Dataset, RandomCrop, Resize, Normalization
from utils import load, save, set_requires_grad, init_weights

parser = argparse.ArgumentParser(description='Train the CycleGAN', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--mode', default='train', choices=['train', 'test'], type=str, dest='mode')
parser.add_argument('--train_continue', default='off', choices=['on', 'off'], type=str, dest='train_continue')

parser.add_argument('--lr', default=2e-4, type=float, dest='lr')
parser.add_argument('--batch_size', default=16, type=int)
parser.add_argument('--num_epochs', default=100, type=int)

parser.add_argument('--data_dir', default='./monet2photo', type=str, dest='data_dir')
parser.add_argument('--ckpt_dir', default='./epochs', type=str, dest='ckpt_dir')
parser.add_argument('--log_dir', default='./log', type=str, dest='log_dir')
parser.add_argument('--result_dir', default='./training_results/', type=str, dest='result_dir')

parser.add_argument('--task', default='denoising', choices=['denoising', 'inpainting', 'super_resolution'], type=str, dest='task')
parser.add_argument('--opts', nargs='+', default=['random', 30.0], dest='opts')

parser.add_argument('--ny', default=256, type=int, dest='ny')
parser.add_argument('--nx', default=256, type=int, dest='nx')
parser.add_argument('--nch', default=3, type=int, dest='nch')
parser.add_argument('--nker', default=64, type=int, dest='nker')

parser.add_argument('--wgt_cycle', default=1e1, type=float, dest='wgt_cycle')
parser.add_argument('--wgt_ident', default=5e-1, type=float, dest='wgt_ident')
parser.add_argument('--norm', default='inorm', type=str, dest='norm')

parser.add_argument('--learning_type', default='plain', choices=['plain', 'residual'], type=str, dest='learning_type')

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# training parameters
mode = args.mode
train_continue = args.train_continue

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch

data_dir = args.data_dir
ckpt_dir = args.ckpt_dir
log_dir = args.log_dir
result_dir = args.result_dir

task = args.task
opts = [args.opts[0], np.asarray(args.opts[1:]).astype(np.float)]

ny = args.ny
nx = args.nx
nch = args.nch
nker = args.nker

wgt_cycle = args.wgt_cycle
wgt_ident = args.wgt_ident
norm = args.norm

learning_type = args.learning_type

transform_train = transforms.Compose([
    Resize(shape=(286, 286, 3)),
    RandomCrop((256, 256)),
    Normalization(mean=0.5, std=0.5)
])

# data loading
train_set = Dataset(data_dir=data_dir, transform=transform_train, data_type='both')
train_loader = DataLoader(dataset=train_set, num_workers=0, batch_size=batch_size, shuffle=True)

num_batch_train = int((train_set.__len__() / batch_size) + ((train_set.__len__() / batch_size) != 0))

if not os.path.exists(result_dir):
    os.makedirs(os.path.join(result_dir, 'train'))
result_dir_train = os.path.join(result_dir, 'train')

netG_x2y = Generator(in_channels=nch, out_channels=nch, norm=norm).to(device)
netG_y2x = Generator(in_channels=nch, out_channels=nch, norm=norm).to(device)

netD_x = Discriminator(in_channels=nch, out_channels=1, norm=norm).to(device)
netD_y = Discriminator(in_channels=nch, out_channels=1, norm=norm).to(device)

# Gaussian distribution N(0, 0.02)
init_weights(netG_x2y)
init_weights(netG_y2x)
init_weights(netD_x)
init_weights(netD_y)

# loss func
fn_cycle = nn.L1Loss().to(device)
fn_gan = nn.BCELoss().to(device)
fn_ident = nn.L1Loss().to(device)

optimizerG = optim.Adam(itertools.chain(netG_x2y.parameters(), netG_y2x.parameters()), lr=lr, betas=(0.5, 0.999))
optimizerD = optim.Adam(itertools.chain(netD_x.parameters(), netD_y.parameters()), lr=lr, betas=(0.5, 0.999))

fn_tonumpy = lambda x: x.to('cpu').datach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean

st_epoch = 1

if mode == 'train':
    if train_continue == 'on':
        netG_x2y, netG_y2x, netD_x, netD_y, optimizerG, optimizerD, st_epoch = load(
            ckpt_dir, netG_x2y, netG_y2x, netD_x, netD_y, optimizerG, optimizerD
        )
    for epoch in range(st_epoch, num_epoch + 1):
        netG_x2y.train()
        netG_y2x.train()
        netD_x.train()
        netD_y.train()

        loss_G_x2y_train = []
        loss_G_y2x_train = []
        loss_D_x_train = []
        loss_D_y_train = []
        loss_cycle_x_train = []
        loss_cycle_y_train = []
        loss_ident_x_train = []
        loss_ident_y_train = []

        for batch, data in enumerate(train_loader, 1):
            input_x = data['data_a'].to(device)
            input_y = data['data_b'].to(device)

            # forward
            output_y = netG_x2y(input_x)
            recon_x = netG_y2x(output_y)

            output_x = netG_y2x(input_y)
            recon_y = netG_x2y(output_x)

            # backward D
            set_requires_grad([netD_x, netD_y], requires_grad=True)
            optimizerD.zero_grad()

            # netD_x
            pred_real_x = netD_x(input_x)
            pred_fake_x = netD_x(output_x.detach())

            loss_D_x_real = fn_gan(pred_real_x, torch.ones_like(pred_real_x))
            loss_D_x_fake = fn_gan(pred_fake_x, torch.zeros_like(pred_fake_x))
            loss_D_x = 0.5 * (loss_D_x_real + loss_D_x_fake)

            # netD_y
            pred_real_y = netD_y(input_y)
            pred_fake_y = netD_y(output_x.detach())

            loss_D_y_real = fn_gan(pred_real_y, torch.ones_like(pred_fake_y))
            loss_D_y_fake = fn_gan(pred_real_y, torch.zeros_like(pred_fake_y))
            loss_D_y = 0.5 * (loss_D_y_real + loss_D_y_fake)

            loss_D = loss_D_x + loss_D_y

            loss_D.backward()
            optimizerD.step()

            # backward G
            optimizerG.zero_grad()

            pred_fake_x = netD_x(output_x)
            pred_fake_y = netD_y(output_y)

            ident_x = netG_y2x(input_x)
            ident_y = netG_x2y(input_y)

            loss_G_x2y = fn_gan(pred_fake_x, torch.ones_like(pred_fake_x))
            loss_G_y2x = fn_gan(pred_fake_y, torch.ones_like(pred_fake_y))
            loss_cycle_x = fn_cycle(recon_x, input_x)
            loss_cycle_y = fn_cycle(recon_y, input_y)
            loss_ident_x = fn_ident(input_x, ident_x)
            loss_ident_y = fn_ident(input_y, ident_y)

            loss_G = (loss_G_x2y + loss_G_y2x) + wgt_cycle * (loss_cycle_x + loss_cycle_y) + wgt_cycle * \
                    wgt_ident * (loss_ident_x + loss_ident_y)

            loss_G.backward()
            optimizerG.step()

            # 손실함수 계산
            loss_G_x2y_train += [loss_G_x2y.item()]
            loss_G_y2x_train += [loss_G_y2x.item()]

            loss_D_x_train += [loss_D_x.item()]
            loss_D_y_train += [loss_D_y.item()]

            loss_cycle_x_train += [loss_cycle_x.item()]
            loss_cycle_y_train += [loss_cycle_y.item()]

            loss_ident_x_train += [loss_ident_x.item()]
            loss_ident_y_train += [loss_ident_y.item()]

            print("train: epoch %04d / %04d | batch: %04d / %04d | GAN x2y %.4f y2x %.4f | DISC x %.4f y %.4f |"
                  "CYCLE x %.4f y %.4f | IDENT x %.4f y %.4f |"
                  % (num_epoch, epoch, num_batch_train, batch, np.mean(loss_G_x2y_train), np.mean(loss_G_y2x_train),
                     np.mean(loss_D_x_train), np.mean(loss_D_y_train), np.mean(loss_cycle_x_train), np.mean(loss_cycle_y_train),
                     np.mean(loss_ident_x_train), np.mean(loss_ident_y_train)))

            if epoch % 20 == 0:
                input_x = fn_tonumpy(fn_denorm(input_x, mean=0.5, std=0.5)).squeeze()
                input_y = fn_tonumpy(fn_denorm(input_y, mean=0.5, std=0.5)).squeeze()
                output_x = fn_tonumpy(fn_denorm(output_x, mean=0.5, std=0.5)).squeeze()
                output_y = fn_tonumpy(fn_denorm(output_y, mean=0.5, std=0.5)).squeeze()

                input_x = np.clip(input_x, a_min=0, a_max=1)
                input_y = np.clip(input_y, a_min=0, a_max=1)
                output_x = np.clip(output_x, a_min=0, a_max=1)
                output_y = np.clip(output_y, a_min=0, a_max=1)

                plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input_x.png' % epoch/20), input_x[0], cmap=None)
                plt.imsave(os.path.join(result_dir_train, 'png', '%04d_input_y_.png' % epoch/20), input_y[0], cmap=None)
                plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output_x.png' % epoch/20), output_x[0], cmap=None)
                plt.imsave(os.path.join(result_dir_train, 'png', '%04d_output_x.png' % epoch/20), output_x[0], cmap=None)

        torch.save(netG_x2y.state_dict(), 'epochs/netG_x2y_epoch_%d_%d.pth' % (2, epoch))
        torch.save(netG_y2x.state_dict(), 'epochs/netG_y2x_epoch_%d_%d.pth' % (2, epoch))
        netG_x2y.eval()
        netG_y2x.eval()









