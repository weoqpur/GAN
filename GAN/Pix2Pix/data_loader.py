import os.path
from os import listdir
from os.path import join
import matplotlib.pyplot as plt
import numpy as np

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import RandomCrop, CenterCrop, ToTensor, ToPILImage, Resize, Compose, Normalize

# 확장자 판별 return: bool
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class Dataset(Dataset):
    def __init__(self, data_dir, crop_size):
        super(Dataset, self).__init__()
        self.data_dir = data_dir
        self.lst_data = [x for x in os.listdir(data_dir)]
        self.transform = Compose([Resize((286, 286)),
                                   RandomCrop((crop_size, crop_size)),
                                   Normalize((0.5), (0.5))])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):

        img = plt.imread(os.path.join(self.data_dir, self.lst_data[index]))
        size = img.shape
        if img.ndim == 2:
            img = img[:, :, np.newaxis]

        if img.dtype == np.uint8:
            img = img / 255

        img = {'label': img[:, :size[1]//2, :], 'input': img[:, size[1]//2:, :]}

        img = self.transform(img)

        return img