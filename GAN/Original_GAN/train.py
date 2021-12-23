## 학습을 시키기
import os

import matplotlib.pyplot as plt
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

from torchvision import datasets
from torchvision import transforms
from model import *
from utils import init_weights
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

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

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(0.5, 0.5)
])

train_set = datasets.MNIST(root='dataset/', transform=transform, download=True)
train_loader = torch.utils.data.DataLoader(train_set, batch_size, shuffle=True, num_workers=0)

num_batch_train = int((train_set.__len__() / batch_size) + ((train_set.__len__() / batch_size) != 0))

out_path = "training_results/"
if not os.path.exists(out_path):
    os.makedirs(out_path)

netG = Generator(100).to(device)
netD = Discriminator().to(device)

init_weights(netG)
init_weights(netD)

loss = nn.BCELoss().to(device)

optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(0.5, 0.999))
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(0.5, 0.999))

writer_fake = SummaryWriter(log_dir=os.path.join('./log', 'fake'))
writer_real = SummaryWriter(log_dir=os.path.join('./log', 'real'))

fn_tonumpy = lambda x: x.to('cpu').datach().numpy().transpose(0, 2, 3, 1)
fn_denorm = lambda x, mean, std: (x * std) + mean



for epoch in range(1, num_epoch + 1):
    netG.train()
    netD.train()

    loss_train = []
    loss_D_fake_train = []
    loss_D_real_train = []

    for batch, (img, label) in enumerate(train_loader, 1):

        valid = Variable(torch.FloatTensor(img.size(0), 1).fill_(1.0), requires_grad=False)
        fake = Variable(torch.FloatTensor(img.size(0), 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(img.type(torch.FloatTensor))

        optimizerG.zero_grad()

        # noise as generator input
        z = Variable(torch.FloatTensor(np.random.normal(0, 1, (img.shape[0], 100))))

        gen_imgs = netG(z)

        g_loss = loss(netD(gen_imgs), valid)

        g_loss.backward()
        optimizerG.step()

        optimizerD.zero_grad()

        real_loss = loss(netD(real_imgs), valid)
        fake_loss = loss(netD(gen_imgs.detach()), fake)
        d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizerD.step()

        loss_train += [g_loss.item()]
        loss_D_fake_train += [fake_loss.item()]
        loss_D_real_train += [real_loss.item()]

        print(
            "train: epoch %04d / %04d | batch: %04d / %04d | GAN %.4f | DISC REAL: %.4f | DISC REAL: %.4f"
            % (num_epoch, epoch, num_batch_train, batch, np.mean(loss_train), np.mean(loss_D_real_train),
               np.mean(loss_D_fake_train)))
        if epoch % 10 == 0:
            gen_imgs = fn_tonumpy(fn_denorm(gen_imgs, mean=0.5, std=0.5)).squeeze()

            gen_imgs = np.clip(gen_imgs, a_min=0, a_max=1)

            plt.imsave(os.path.join(gen_imgs, 'png', '%04d_input.png' % epoch/10), gen_imgs[0], cmap=None)

    if epoch % 10 == 0:
        torch.save(netG.state_dict(), 'epochs/netG_epoch_%d.pth' % epoch)
        torch.save(netD.state_dict(), 'epochs/netD_epoch_%d.pth' % epoch)