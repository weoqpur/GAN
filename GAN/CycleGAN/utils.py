import os
import torch
import torch.nn as nn
import numpy as np

from scipy.stats import poisson
from skimage.transform import rescale, resize

# network grad 설정
def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad


# 가중치 초기화
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def save(ckpt_dir, netG_x2y, netG_y2x, netD_x, netD_y, optimG, optimD, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netG_x2y': netG_x2y.state_dict(), 'netG_y2x': netG_y2x.state_dict(),
                'netD_x': netD_x.state_dict, 'netD_y': netD_y.state_dict(),
                'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
               '%s/model_epoch%d.pth' % (ckpt_dir, epoch))

def load(ckpt_dir, netG_x2y, netG_y2x, netD_x, netD_y, optimG, optimD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG_x2y, netG_y2x, netD_x, netD_y, optimG, optimD, epoch

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst = [f for f in ckpt_lst if f.endswith('pth')]
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit(), f))))

    dict_model = torch.load('%s/%s' % (ckpt_dir, ckpt_lst[-1]), map_location=device)

    netG_x2y.load_state_dict(dict_model['netG_x2y'])
    netG_y2x.load_state_dict(dict_model['netG_y2x'])
    netD_x.load_state_dict(dict_model['netD_x'])
    netD_y.load_state_dict(dict_model['netD_y'])
    optimG.load_state_dict(dict_model['optimG'])
    optimD.load_state_dict(dict_model['optimD'])
    epoch = int(ckpt_lst[-1].split('epoch')[1].split('.pth')[0])

    return netG_x2y, netG_y2x, netD_x, netD_y, optimG, optimD, epoch


def add_sampling(img, type='random', opts=None):
    sz = img.shape

    if type == 'uniform':
        ds_y = opts[0].astype(np.int)
        ds_x = opts[1].astype(np.int)

        msk = np.zeros(sz)
        msk[::ds_y, ::ds_x, :] = 1

        dst = img * msk
    elif type == 'random':
        prob = opts[0]
        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd > prob).astype(np.float)

        dst = img * msk

    elif type == 'gaussian':
        x0 = opts[0]
        y0 = opts[1]
        sgmx = opts[2]
        sgmy = opts[3]

        a = opts[4]

        ly = np.linspace(-1, 1, sz[0])
        lx = np.linspace(-1, 1, sz[1])

        x, y = np.meshgrid(lx, ly)

        gaus = a * np.exp(-((x - x0)**2/(2*sgmx**2) + (y - y0)**2/(2*sgmy**2)))
        gaus = np.tile(gaus[:, :, np.newaxis], (1, 1, sz[2]))
        rnd = np.random.rand(sz[0], sz[1], sz[2])
        msk = (rnd < gaus).astype(np.float)

        dst = img * msk

    return dst

## noise 추가
def add_noise(img, type='random', opts=None):
    sz = img.shape

    if type == 'random':
        sgm = opts[0]
        noise = sgm/255 * np.random.randn(sz[0], sz[1], sz[2])
        dst = img + noise

    elif type == 'poisson':
        dst = poisson.rvs(img)
        noise = dst - img

    return dst

## low-resolution 추가
def add_blur(img, type='bilinear', opts=None):
    # rescale option들
    if type == 'nearest': order = 0
    elif type == 'bilinear': order = 1
    elif type == 'biquadratic': order = 2
    elif type == 'bicubic': order = 3
    elif type == 'biquartic': order = 4
    elif type == 'biquintic': order = 5

    sz = img.shape

    dw = opts[0]

    if len(opts) == 1:
        keepdim = True
    else:
        keepdim = opts[1]

    dst = resize(img, output_shape=(sz[0] // dw, sz[1] // dw, sz[2]), order=order)

    if keepdim:
        dst = resize(dst, output_shape=(sz[0], sz[1], sz[2]), order=order)

    return dst

