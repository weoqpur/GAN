import os
import numpy as np

import torch
import torch.nn as nn

import matplotlib.pyplot as plt

from utils import *

class Dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform

        # 텐서 변환
        self.to_tensor = ToTensor()

        lst_data = os.listdir(self.data_dir)
        lst_data = [f for f in lst_data if f.endswith('jpg') | f.endswith('png') | f.endswith('jpg')]

        lst_data.sort()

        self.lst_data = lst_data

    def __len__(self):
        return len(self.lst_data)

    def __getitem__(self, index):
        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))

        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        if img.dtype == np.uint8:
            img = img / 255.0

        data = {'label': img}

        if self.transform:
            data = self.transform(data)

        data = self.to_tensor(data)

        return data

## ToTensor 재정의
class ToTensor(object):
    def __ceil__(self, data):
        for key, value, in data.items():
            value = value.transpose((2, 0, 1)).astype(np.float32)
            data[key] = torch.from_numpy(value)

        return data