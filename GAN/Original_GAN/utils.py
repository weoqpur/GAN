import os
import numpy as np

import torch
import torch.nn as nn

# Network weights 초기화
def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__

        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)


            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)

    net.apply(init_func)

## 네트워크의 가중치를 저장
def save(ckpt_dir, netG, netD, optimG, optimD, epoch):
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    torch.save({'netG': netG.state_dict(), 'netD': netD.state_dict(),
                'optimG': optimG.state_dict(), 'optimD': optimD.state_dict()},
               "%s/model_epoch%d.pth" % (ckpt_dir, epoch))


## 네트워크 로드
def load(ckpt_dir, netG, netD, optimG, optimD):
    if not os.path.exists(ckpt_dir):
        epoch = 0
        return netG, netD, optimG, optimD, epoch

    ckpt_lst = os.listdir(ckpt_dir)
    ckpt_lst.sort(key=lambda f: int(''.join(filter(str.isdigit, f))))

    dict_model = torch.load('%s%s' % (ckpt_dir, ckpt_lst[-1]))

    netG.load_state_dict(dict_model['netG'])
    netD.load_state_dict(dict_model['netD'])
    optimG.load_state_dict(dict_model['optimG'])
    optimD.load_state_dict(dict_model['optimD'])

    epoch = int(ckpt_lst[-1].split('epoch')[1].split('pth')[0])

    return netG, netD, optimG, optimD, epoch