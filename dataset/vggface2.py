import os

import numpy as np
import PIL.Image
import torch
from torch.utils import data
from torchvision import transforms
import random
import shutil

class VGG_Faces2(data.Dataset):

    def __init__(self, image_list_file, train = True, transform = None, horizontal_flip=False):
        assert os.path.exists(image_list_file), "image_list_file: {} not found.".format(image_list_file)
        self.image_list_file = image_list_file
        self._transform = transform
        self.horizontal_flip = horizontal_flip
        self.train = train

        self.img_info = []
        with open(image_list_file, "r") as file:
            for line in file:
                strs = line.split('-')
                self.img_info.append(
                    {
                        'img': strs[0],
                        'label': int(strs[1]),
                    }
                )

    def __len__(self):
        return len(self.img_info)

    def __getitem__(self, index):
        info = self.img_info[index]
        img_file = info['img']
        img = PIL.Image.open(img_file)
        img = transforms.Resize(120)(img)
        if self.train:
            img = transforms.RandomCrop(112)(img)
            # img = transforms.RandomGrayscale(p=0.2)(img)
        else:
            img = transforms.CenterCrop(112)(img)
        if self.horizontal_flip:
            img = transforms.functional.hflip(img)

        img = np.array(img, dtype=np.uint8)
        assert len(img.shape) == 3  # assumes color images and no alpha channel

        label = info['label']
        # class_id = info['cid']
        if self._transform != None:
            return self.transform(img, self._transform), label #, img_file #, class_id
        else:
            return img, label #, img_file #, class_id

    def transform(self, img, transform):
        img1 = transform(img)
        return img1

if __name__ == '__main__':
    select_tr = []
    select_val = []
    index = 0

    data_dir = '/home/test/backdoor/test/vggface2/train'
    train_dir = '/home/test/backdoor/test/vggface2/random_select_400_100/train'
    val_dir = '/home/test/backdoor/test/vggface2/random_select_400_100/val'

    for subfold in os.listdir(data_dir):
        full_path = os.path.join(data_dir, subfold)
        if os.path.isdir(full_path) and index < 20:
            for _, _, files in os.walk(full_path):
                count = len(files)
                if count > 500:
                    all_indices = list(range(count))
                    train_indices = random.sample(all_indices, 400)
                    rest_indices = [item for i, item in enumerate(all_indices) if i not in train_indices]
                    val_indices = random.sample(rest_indices, 100)

                    # has_intersection = any(item in train_indices for item in val_indices)
                    # print(has_intersection)
                    os.makedirs(train_dir + '/' + str(index), exist_ok=True)
                    os.makedirs(val_dir + '/' + str(index), exist_ok=True)
                    for i, file in enumerate(files):
                        if i in train_indices:
                            shutil.copy(os.path.join(full_path, file), os.path.join(train_dir, str(index), file))
                            select_tr.append((os.path.join(train_dir, str(index), file), index))
                        if i in val_indices:
                            shutil.copy(os.path.join(full_path, file), os.path.join(val_dir, str(index), file))
                            select_val.append((os.path.join(val_dir, str(index), file), index))

                    index += 1

    with open('/home/test/backdoor/test/vggface2/random_select_400_100/train.txt', 'w') as file:
        for item in select_tr:
            file.write(item[0] + '-' + str(item[1]) + '\n')
    with open('/home/test/backdoor/test/vggface2/random_select_400_100/val.txt', 'w') as file:
        for item in select_val:
            file.write(item[0] + '-' + str(item[1]) + '\n')
    print('end')
