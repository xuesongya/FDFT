import numpy as np
import torch
import os
import random
import cv2
import enc_img

import torch.nn.functional as F
from pytorch_msssim import ssim

from PIL import Image
from torchvision import transforms

dataset_m = 'cifar10'

def set_dataset_mode(value):
    global dataset_m
    dataset_m = value

def generate_poisoned_trainset(method, dataset, target, random_indices, trigger_path, **kwargs):
    if method == 'fdft':
        return generate_poisoned_trainset_fdft(dataset, target, random_indices, trigger_path, **kwargs)
    elif method == 'badnets':
        return generate_poisoned_trainset_badnets(dataset, target, random_indices, trigger_path)
    elif method == 'blended':
        return generate_poisoned_trainset_blended(dataset, target, random_indices, trigger_path)
    elif method == 'issba':
        return generate_poisoned_trainset_issba(dataset, target, random_indices, trigger_path)

def generate_poisoned_testset(method, dataset, target, trigger_path, **kwargs):
    if method == 'fdft':
        return generate_poisoned_testset_fdft(dataset, target, trigger_path, **kwargs)
    elif method == 'badnets':
        return generate_poisoned_testset_badnets(dataset, target, trigger_path)
    elif method == 'blended':
        return generate_poisoned_testset_blended(dataset, target, trigger_path)
    elif method == 'issba':
        return generate_poisoned_testset_issba(dataset, target, trigger_path)

def generate_poisoned_trainset_fdft(dataset, target, random_indices, trigger_path, m = 1.0):
    print(f'm:{m}')
    tri = RGB2YUV(read_trigger(trigger_path, 5))
    if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
        tri = DCT(tri, 32)
    else:
        tri = DCT(tri, 5)  #cifar用32， vggface用5

    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]        # tensor, cifar数据集 shape[3, 32, 32]  value[0, 1]

        if i in random_indices:
            img = img.permute(1, 2, 0).unsqueeze(0)
            img *= 255
            img = img.numpy().astype(np.uint8)  # tensor转numpy
            img = RGB2YUV(img)
            if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
                img = DCT(img, 32)
            else:
                img = DCT(img, 112)  #cifar用32， vggface用112

            for i in range(img.shape[0]):
                for ch in [1, 2]:
                    img[i][ch][img.shape[2] - tri.shape[2] : img.shape[2], img.shape[3] - tri.shape[3] : img.shape[3]] += m * tri[0][ch]
            
            if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
                img = IDCT(img, 32)
            else:
                img = IDCT(img, 112)  #cifar用32， vggface用112
            img = YUV2RGB(img)
            img /= 255

            img = torch.from_numpy(img).squeeze(0).permute(2, 0, 1)  # numpy转tensor
            img = torch.clamp(img, 0, 1)
            dataset_.append((img, target))
        else:
            dataset_.append((img, data[1]))
    return dataset_

def generate_poisoned_testset_fdft(dataset, target, trigger_path, m = 1.0):
    tri = RGB2YUV(read_trigger(trigger_path, 5))
    if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
        tri = DCT(tri, 32)
    else:
        tri = DCT(tri, 5)  #cifar用32， vggface用5

    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if label == target:
            continue
        
        img = img.permute(1, 2, 0).unsqueeze(0)
        img *= 255
        img = img.numpy().astype(np.uint8)  # tensor转numpy
        img = RGB2YUV(img)
        if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
            img = DCT(img, 32)
        else:
            img = DCT(img, 112) #cifar用32， vggface用112

        for i in range(img.shape[0]):
            for ch in [1, 2]:
                img[i][ch][img.shape[2] - tri.shape[2] : img.shape[2], img.shape[3] - tri.shape[3] : img.shape[3]] += m * tri[0][ch]
        
        if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
            img = IDCT(img, 32)
        else:
            img = IDCT(img, 112)  #cifar用32， vggface用112
        img = YUV2RGB(img)
        img /= 255
        img = torch.from_numpy(img).squeeze(0).permute(2, 0, 1)  # numpy转tensor

        img = torch.clamp(img, 0, 1)          
        dataset_.append((img, target))
    return dataset_

def generate_poisoned_trainset_badnets(dataset, target, random_indices, trigger_path):
    if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
        trigger_img = Image.open(trigger_path).convert('RGB').resize((5,5))
    else:
        trigger_img = Image.open(trigger_path).convert('RGB').resize((10,10))

    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]   # tensor, cifar数据集 shape[3, H, W]  value[0, 1]
        if i in random_indices:

            to_pil = transforms.ToPILImage()
            img = to_pil(img)
            if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
                img.paste(trigger_img, (16 - 3, 16 - 3, 16 + 2, 16 + 2))
            else:
                img.paste(trigger_img, (56 - 5, 56 - 5, 56 + 5, 56 + 5))   #cifar用32， vggface用112

            # 定义一个转换操作，将PIL图像转换为Tensor
            transform = transforms.Compose([
                transforms.ToTensor(),  # 将PIL图像转换为Tensor
            ])
            # 应用转换
            img = transform(img)

            img = torch.clamp(img, 0, 1)
            dataset_.append((img, target))
        else:
            dataset_.append((img, data[1]))         
    return dataset_

def generate_poisoned_testset_badnets(dataset, target, trigger_path):
    if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
        trigger_img = Image.open(trigger_path).convert('RGB').resize((5,5))
    else:
        trigger_img = Image.open(trigger_path).convert('RGB').resize((10,10))

    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if label == target:
            continue
        
        to_pil = transforms.ToPILImage()
        img = to_pil(img)
        if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
            img.paste(trigger_img, (16 - 3, 16 - 3, 16 + 2, 16 + 2))
        else:
            img.paste(trigger_img, (56 - 5, 56 - 5, 56 + 5, 56 + 5)) # cifar用32， vggface用112

        # 定义一个转换操作，将PIL图像转换为Tensor
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将PIL图像转换为Tensor
        ])
        # 应用转换
        img = transform(img)

        img = torch.clamp(img, 0, 1)          
        dataset_.append((img, target))
    return dataset_

def generate_poisoned_trainset_blended(dataset, target, random_indices, trigger_path):
    if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
        trigger_img = Image.open(trigger_path).convert('RGB').resize((32,32))
    else:
        trigger_img = Image.open(trigger_path).convert('RGB').resize((112,112))  # cifar用32， vggface用112
    trigger = np.array(trigger_img).astype(np.float32) / 255.0
    trigger = torch.tensor(trigger).permute(2, 0, 1)  # Convert to CHW format
    alpha = 0.2

    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]   # tensor, cifar数据集 shape[3, 32, 32]  value[0, 1]
        if i in random_indices:
            img = (1 - alpha) * img + alpha * trigger

            img = torch.clamp(img, 0, 1)
            dataset_.append((img, target))
        else:
            dataset_.append((img, data[1]))         
    return dataset_

def generate_poisoned_testset_blended(dataset, target, trigger_path):
    if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
        trigger_img = Image.open(trigger_path).convert('RGB').resize((32,32))
    else:
        trigger_img = Image.open(trigger_path).convert('RGB').resize((112,112))  # cifar用32， vggface用112
    trigger = np.array(trigger_img).astype(np.float32) / 255.0
    trigger = torch.tensor(trigger).permute(2, 0, 1)  # Convert to CHW format
    alpha = 0.2

    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if label == target:
            continue
        
        img = (1 - alpha) * img + alpha * trigger

        img = torch.clamp(img, 0, 1)          
        dataset_.append((img, target))
    return dataset_


def generate_poisoned_trainset_issba(dataset, target, random_indices, trigger_path):

    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]   # tensor, cifar数据集 shape[3, 32, 32]  value[0, 1]
        if i in random_indices:
            if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
                transform = transforms.Compose([
                    transforms.Resize((112, 112))
                ])

                img = transform(img)
            img = img.permute(1, 2, 0)
            # img, _ = enc_img.encode_image(img.numpy())

            img = torch.from_numpy(img).to(torch.float32)
            img = img.permute(2, 0, 1)

            if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
                transform = transforms.Compose([
                    transforms.Resize((32, 32)),
                ])
                # 应用转换
                img = transform(img)
            # print(img.dtype)
            img /= 255

            img = torch.clamp(img, 0, 1)
            dataset_.append((img, target))
        else:
            dataset_.append((img, data[1]))         
    return dataset_

def generate_poisoned_testset_issba(dataset, target, trigger_path):

    dataset_ = list()
    for i in range(len(dataset)):
        data = dataset[i]
        img = data[0]
        label = data[1]
        if label == target:
            continue
        
        if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
            transform = transforms.Compose([
                transforms.Resize((112, 112))
            ])
            img = transform(img)
        

        img = img.permute(1, 2, 0)
        img, _ = enc_img.encode_image(img.numpy())

        img = torch.from_numpy(img).to(torch.float32)
        img = img.permute(2, 0, 1)

        if dataset_m == 'cifar10' or dataset_m == 'GTSRB':
            transform = transforms.Compose([
                transforms.Resize((32, 32)),
            ])
            # 应用转换
            img = transform(img)
        img /= 255

        img = torch.clamp(img, 0, 1)          
        dataset_.append((img, target))
    return dataset_


def read_trigger(trigger_path, trigger_size):
    trigger_img = Image.open(trigger_path).convert('RGB')
    trigger_img = trigger_img.resize((trigger_size, trigger_size))
    trigger_img = np.array(trigger_img)
    trigger_img = trigger_img.reshape(1, trigger_size, trigger_size, 3)
    return trigger_img

def RGB2YUV(x_rgb):
    x_yuv = np.zeros(x_rgb.shape, dtype=np.float32)
    for i in range(x_rgb.shape[0]):
        img = cv2.cvtColor(x_rgb[i].astype(np.uint8), cv2.COLOR_RGB2YCrCb)
        x_yuv[i] = img
    return x_yuv

def YUV2RGB(x_yuv):
    x_rgb = np.zeros(x_yuv.shape, dtype=np.float32)
    for i in range(x_yuv.shape[0]):
        img = cv2.cvtColor(x_yuv[i].astype(np.uint8), cv2.COLOR_YCrCb2RGB)
        x_rgb[i] = img
    return x_rgb


def DCT(x_train, window_size):
    x_dct = np.zeros((x_train.shape[0], x_train.shape[3], x_train.shape[1], x_train.shape[2]), dtype=np.float32)
    x_train = np.transpose(x_train, (0, 3, 1, 2))

    for i in range(x_train.shape[0]):
        for ch in range(x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    sub_dct = cv2.dct(x_train[i][ch][w:w+window_size, h:h+window_size].astype(np.float32))
                    x_dct[i][ch][w:w+window_size, h:h+window_size] = sub_dct
    return x_dct


def IDCT(x_train, window_size):
    x_idct = np.zeros(x_train.shape, dtype=np.float32)

    for i in range(x_train.shape[0]):
        for ch in range(0, x_train.shape[1]):
            for w in range(0, x_train.shape[2], window_size):
                for h in range(0, x_train.shape[3], window_size):
                    sub_idct = cv2.idct(x_train[i][ch][w:w+window_size, h:h+window_size].astype(np.float32))
                    x_idct[i][ch][w:w+window_size, h:h+window_size] = sub_idct
    x_idct = np.transpose(x_idct, (0, 2, 3, 1))
    return x_idct


def compute_psnr(image1, image2):
    """Compute PSNR between two images."""
    mse = F.mse_loss(image1, image2).item()
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(1.0 / torch.sqrt(torch.tensor(mse)))
    return psnr.item()

def compute_l0_norm(residual):
    # print(f'residual:{residual.shape}')
    """Compute the L0 norm of the residual."""
    return torch.sum(residual != 0).item()

def compute_linf_norm(residual):
    # print(f'residual:{residual.shape}')
    """Compute the L-infinity norm of the residual."""
    return torch.max(torch.abs(residual)).item()

def compute_ssim(image1, image2):
    """Compute SSIM between two images."""
    # Apply the transform to images
    img1_tensor = image1.unsqueeze(0)  # Add batch dimension
    img2_tensor = image2.unsqueeze(0)  # Add batch dimension

    # Ensure both images have the same dimensions
    if img1_tensor.size() != img2_tensor.size():
        raise ValueError("Input images must have the same dimensions")

    # Compute SSIM
    ssim_value = ssim(img1_tensor, img2_tensor, data_range=1.0, size_average=True)
    return ssim_value.item()


def set_random_seed(seed = 10):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

class MyDataset(torch.utils.data.Dataset):
   
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample, label = self.data[idx][0], self.data[idx][1]
        if self.transform:
            sample = self.transform(sample)
        return (sample, label)
    

def train_step(model, criterion, optimizer, data_loader, device):
    model.train()
    total_correct = 0
    total_loss = 0.0
    for i, (images, labels) in enumerate(data_loader):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        output = model(images)
        loss = criterion(output, labels)
        pred = output.data.max(1)[1]
        total_correct += pred.eq(labels.view_as(pred)).sum()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc


def test_step(model, criterion, data_loader, device):
    model.eval()
    total_correct = 0
    total_loss = 0.0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_loader):
            images, labels = images.to(device), labels.to(device)
            output = model(images)
            total_loss += criterion(output, labels).item()
            pred = output.data.max(1)[1]
            total_correct += pred.eq(labels.data.view_as(pred)).sum()
    loss = total_loss / len(data_loader)
    acc = float(total_correct) / len(data_loader.dataset)
    return loss, acc