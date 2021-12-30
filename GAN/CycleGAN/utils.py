import torch
import numpy as np

from scipy.stats import poisson
from skimage.transform import rescale, resize

# 가중치 초기화
def init_weights(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

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
    if type == 'nearest':
        order = 0
    elif type == 'bilinear':
        order = 1
    elif type == 'biquadratic':
        order = 2
    elif type == 'bicubic':
        order = 3
    elif type == 'biquartic':
        order = 4
    elif type == 'biquintic':
        order = 5

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

