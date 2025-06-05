from torch.utils import data
import numpy as np
import torch

class GTSRB(data.Dataset):
    def __init__(self, train = True, transform = None):
        prefix = "/home/test/backdoor/test/"
        np_data = None
        if train:
            np_data = np.load(prefix + "data/GTSRB_train.npz")
        else:
            np_data = np.load(prefix + "data/GTSRB_test.npz")
        self.x = np_data["arr_0"]
        self.y = np_data["arr_1"]

        self._transform = transform

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        img = self.x[index]  # (32, 32, 3)  0-1
        label = int(self.y[index])
        if self._transform != None:
            img = self._transform(img)
        img = img.to(dtype = torch.float32)
        return img, label
