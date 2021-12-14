from os import listdir
from os.path import join

import numpy as np
from PIL import Image
from torch.utils.data.dataset import Dataset
from torchvision.transforms import RandomCrop, CenterCrop, ToTensor, ToPILImage, Resize, Compose, Normalize

# 확장자 판별 return: bool
def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])

class Dataset(Dataset):
    def __init__(self, data_dir):
        super(Dataset, self).__init__()
        self.image_filenames = [join(data_dir, x) for x in listdir(data_dir) if is_image_file(x)]
        if self.image_filenames.dtype == np.uint8:
            self.image_filenames = self.image_filenames/255
        self.transform_train = Compose([Resize(shape=(286, 286, 3)),
                                   RandomCrop((256, 256)),
                                   Normalize((0.5), (0.5))])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, index):

        img = self.transform_train(Image.open(self.image_filenames[index]))
        sz = img.shape

        img = {'label': img[:, :sz[1]//2, :], 'input': img[:, sz[1]//2:, :]}

        return img