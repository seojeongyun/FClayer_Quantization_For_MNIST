from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import os
import struct
import numpy as np

from torch.utils.data import Dataset
import numpy as np
from scipy import io

import torch
import torchvision.transforms as transforms
from PIL import Image
import glob


class data_loader(Dataset):
    def __init__(self, path, height=28, width=28, augmentation=False, task='train'):
        super().__init__()
        assert task == 'train' or task == 'val' or task == 'test', f'Invalid task...'
        #
        self.mnist = self.get_mnist(task, path)
        #
        self.dataset_path = path
        self.task = task
        #
        self.img_size = (height, width)
        self.augmentation = augmentation
        #
        self.fn_transform = self.get_transformer()

    def get_mnist(self):
        if self.task is "training":
            fname_img = os.path.join(self.dataset_path, 'train-images.idx3-ubyte')
            fname_lbl = os.path.join(self.dataset_path, 'train-labels.idx1-ubyte')
        elif self.task is "testing":
            fname_img = os.path.join(self.dataset_path, 't10k-images.idx3-ubyte')
            fname_lbl = os.path.join(self.dataset_path, 't10k-labels.idx1-ubyte')
        else:
            raise ValueError("dataset must be 'testing' or 'training'")

        # Load everything in some numpy arrays
        with open(fname_lbl, 'rb') as flbl:
            magic, num = struct.unpack(">II", flbl.read(8))
            lbl = np.fromfile(flbl, dtype=np.int8)

        with open(fname_img, 'rb') as fimg:
            magic, num, rows, cols = struct.unpack(">IIII", fimg.read(16))
            img = np.fromfile(fimg, dtype=np.uint8).reshape(len(lbl), rows, cols)

        get_img = lambda idx: (lbl[idx], img[idx])

        mnist_data = []
        for i in range(len(lbl)):
            mnist_data.append(get_img(i))

        return mnist_data

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        img = torch.tensor(self.mnist[index][1]) / 255.
        label = self.mnist[index][0]

        return img, label

    @staticmethod
    def make_one_hot_vec(label):
        labels = []
        for idx, value in enumerate(label):
            one_hot_vec = np.zeros(10)
            one_hot_vec[value] = 1
            labels.append(one_hot_vec)
        return torch.tensor(np.array(labels))

    @staticmethod
    def collate_fn(batch):
        """Merges a list of samples to form a mini-batch of Tensor(s)"""
        inputs, labels = zip(*batch)
        #
        #
        inputs = torch.cat(inputs, dim=0)
        labels = torch.cat(labels, dim=0)
        labels = data_loader.make_one_hot_vec(labels)
        #
        return inputs, labels


if __name__ == '__main__':
    object = data_loader(path='/storage/hrlee/WDM/wdmmix_new/sample_train/',
                 height=52,
                 width=52,
                 augmentation=True,
                 task='train'
                 )
    img, label = object.__getitem__(0)
    #
    loader = torch.utils.data.DataLoader(
        object,
        batch_size=5,
        collate_fn=object.collate_fn
    )
    #
    for batch_id, data in enumerate(loader):
        image, label = data[0], data[1]
