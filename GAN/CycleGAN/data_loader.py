import os.path
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

import torch.nn as nn
import torch
import numpy as np
from torch.utils.data.dataset import Dataset
from skimage.transform import resize
from utils import add_noise, add_sampling, add_blur

# 확장자 판별 return: bool
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class Dataset(Dataset):
    def __init__(self, data_dir, transform=None, data_type='both'):
        super(Dataset, self).__init__()
        self.data_dir_a = data_dir + 'A'
        self.data_dir_b = data_dir + 'B'

        if os.path.exists(self.data_dir_a):
            lst_data_a = [x for x in os.listdir(self.data_dir_a)]
            lst_data_a.sort()
        else:
            lst_data_a = list()

        if os.path.exists(self.data_dir_b):
            lst_data_b = [x for x in os.listdir(self.data_dir_b)]
            lst_data_b.sort()
        else:
            lst_data_b = list()

        self.lst_data_a = lst_data_a
        self.lst_data_b = lst_data_b

        self.transform = transform
        self.totensor = ToTensor()
        self.data_type = data_type

    def __len__(self):
        if self.data_type == 'both':
            if len(self.lst_data_a) < len(self.lst_data_b):
                return len(self.lst_data_a)
            else:
                return len(self.lst_data_b)
        elif self.data_type == 'a':
            return len(self.lst_data_a)
        elif self.data_type == 'b':
            return len(self.lst_data_b)

    def __getitem__(self, index):

        data = {}

        if self.data_type == 'a' or self.data_type == 'both':
            data_a = plt.imread(os.path.join(self.data_dir_a, self.lst_data_a[index]))

            if data_a.dtype == np.uint8:
                data_a = data_a / 255.0
            if data_a.ndim == 2:
                data_a = data_a[:, :, np.newaxis]

            data['data_a'] = data_a

        if self.data_type == 'b' or self.data_type == 'both':
            data_b = plt.imread(os.path.join(self.data_dir_b, self.lst_data_b[index]))

            if data_b.dtype == np.uint8:
                data_b = data_b / 255.0
            if data_b.ndim == 2:
                data_b = data_b[:, :, np.newaxis]

            data['data_b'] = data_b

        if self.transform:
            data = self.transform(data)

        data = self.totensor(data)

        return data


class ToTensor(object):
    def __call__(self, data):
        for key, value in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data


class Normalization(object):
    def __init__(self, mean=0.5, std=0.5):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        for key, value in data.items():
            data[key] = (value - self.mean) / self.std

        return data


class RandomCrop(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):

        key = list(data.keys())[0]

        h, w = data[key].shape[:2]
        new_h, new_w = self.shape

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        id_y = np.arange(top, top + new_h, 1)[:, np.newaxis]
        id_x = np.arange(left, left + new_w, 1)

        for key, value in data.items():
            data[key] = value[id_y, id_x]

        return data


class Resize(object):
    def __init__(self, shape):
        self.shape = shape

    def __call__(self, data):
        for key, value in data.items():
            data[key] = resize(value, output_shape=(self.shape[0], self.shape[1], self.shape[2]))

        return data