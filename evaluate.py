from torchvision import datasets, transforms
from util import *
import argparse
from dataset import vggface2
from torchmetrics.multimodal import CLIPImageQualityAssessment
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser(description='Train Backdoored Model')
parser.add_argument('--dataset', type=str, default='cifar10')
parser.add_argument('--y_target', type=int, default=0)
parser.add_argument('--method', type=str, default='issba')
args = parser.parse_args()

trigger = {
    'fdft' : '/home/test/backdoor/test/data/trigger_white.png',
    'badnets' : '/home/test/backdoor/test/data/trigger_white.png',
    'blended' : '/home/test/backdoor/test/data/hk.png',
    'issba' : '',
}

class pair_iter(torch.utils.data.dataset.Dataset):
    def __init__(self,x):
        self.data = x
    def __getitem__(self, item):
        img = self.data[item]
        return img
    def __len__(self):
        return len(self.data)

if __name__ == '__main__':

    test_dataset = None

    if args.dataset == 'cifar10':
        test_dataset = datasets.CIFAR10(root='dataset', 
                                        train=False,
                                        transform=transforms.ToTensor(), 
                                        download=True)
    elif args.dataset == 'vggface2':
        test_dataset = vggface2.VGG_Faces2('/home/test/backdoor/test/vggface2/random_select_400_100/val.txt', 
                                           train = False, 
                                           transform=transforms.ToTensor())
        set_dataset_mode('vggface2')
    
    trigger_path = trigger[args.method]
    poison_test_set = generate_poisoned_testset(args.method, test_dataset, args.y_target, trigger_path)

    length = len(test_dataset)
    length1 = len(poison_test_set)

    print(f'{length} and {length1}')

    filter_dataset = list()
    for i in range(len(test_dataset)):
        data = test_dataset[i]
        img = data[0]
        label = data[1]
        if label == args.y_target:
            continue
        filter_dataset.append((img, args.y_target))

    psnr = 0
    ssim_ = 0
    l0 = 0
    l_inf = 0

    for i in range(length1):
        data1 = filter_dataset[i]
        img1 = data1[0]

        data2 = poison_test_set[i]
        img2 = data2[0]
        residual = img2 - img1
        # Image.fromarray((residual*255).permute(1,2,0).numpy().astype(np.uint8)).save('/home/test/attack/SIBA/test/r.png')

        v1 = compute_psnr(img1, img2)
        v2 = compute_ssim(img1, img2)
        v3 = compute_l0_norm(img2 - img1)
        v4 = compute_linf_norm(img2 - img1)

        psnr += v1
        ssim_ += v2
        l0 += v3
        l_inf += v4
    print(f'{args.method} psnr : {psnr/length},  ssim : {ssim_/length},  l0 : {l0/length},  l_inf : {l_inf/length}')

    clip_1 = 0
    clip_c_1 = 0
    clip_2 = 0
    clip_c_2 = 0
    clip_3 = 0
    clip_c_3 = 0
    clip_4 = 0
    clip_c_4 = 0
    clip_5 = 0
    clip_c_5 = 0
    clip_6 = 0
    clip_c_6 = 0

    image_pair = []
    for i in range(length1):
        data1 = filter_dataset[i]
        img1 = data1[0]

        data2 = poison_test_set[i]
        img2 = data2[0]
        image_pair.append((img1, img2))
    
    pair_dataset = pair_iter(image_pair)
    pair_loader = DataLoader(pair_dataset, batch_size=100, shuffle=False, num_workers=0, pin_memory=False)

    for index, pair in enumerate(pair_loader):
        img11 = pair[0].to('cuda:2')
        img22 = pair[1].to('cuda:2')
        with torch.cuda.device('cuda:2'):
            metric = CLIPImageQualityAssessment(prompts=('quality','noisiness','contrast', 'natural', 'brightness', 'sharpness'))
            v6 = metric(img11)
            v5 = metric(img22)
            # print(f'posion:{v5}--clean:{v6}')


        clip_1 += v5['quality'].sum().cpu().numpy()
        clip_c_1 += v6['quality'].sum().cpu().numpy()
        clip_2 += v5['noisiness'].sum().cpu().numpy()
        clip_c_2 += v6['noisiness'].sum().cpu().numpy()
        clip_3 += v5['contrast'].sum().cpu().numpy()
        clip_c_3 += v6['contrast'].sum().cpu().numpy()
        clip_4 += v5['natural'].sum().cpu().numpy()
        clip_c_4 += v6['natural'].sum().cpu().numpy()
        clip_5 += v5['brightness'].sum().cpu().numpy()
        clip_c_5 += v6['brightness'].sum().cpu().numpy()
        clip_6 += v5['sharpness'].sum().cpu().numpy()
        clip_c_6 += v6['sharpness'].sum().cpu().numpy()

        if (i+1) % 100 == 0:
            print(f'{i+1} is done!')
    
    
    print(f'Clip_quality:{clip_1/length}, clip_clean_quality:{clip_c_1/length}, Clip_noisiness:{clip_2/length}, clip_clean_noisiness:{clip_c_2/length}, \
          Clip_contrast:{clip_3/length}, clip_clean_contrast:{clip_c_3/length}, Clip_natural:{clip_4/length}, clip_clean_natural:{clip_c_4/length}, \
          Clip_brightness:{clip_5/length}, clip_clean_brightness:{clip_c_5/length},Clip_sharpness:{clip_6/length}, clip_clean_sharpness:{clip_c_6/length}')


