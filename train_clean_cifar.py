from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import logging
import time
import models
from util import *
from dataset.GTSRB import GTSRB

parser = argparse.ArgumentParser(description='Train clean model')
parser.add_argument('--model', type=str, default='resnet18', 
                    choices=['resnet18', 'resnet34', 'vgg16_bn', 'vgg19_bn'])
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=2048)
parser.add_argument('--save_surrogate', type=str, default='save_surrogate')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dataset', type=str, default='GTSRB', choices=['cifar10', 'vggface2', 'GTSRB'])
args = parser.parse_args()


def main(args):
    
    set_random_seed(args.seed)
    device = f'cuda:{args.device}'
    set_dataset_mode(args.dataset)

    train_dataset = None
    test_dataset = None
    class_num = 10


    train_transform = transforms.Compose([
                transforms.Pad(4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32),
                transforms.ToTensor()])
    
    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='dataset', 
                                        train=True, 
                                        transform=train_transform, 
                                        download=True)
        test_dataset = datasets.CIFAR10(root='dataset', 
                                        train=False, 
                                        transform=transforms.ToTensor(), 
                                        download=True)
        class_num = 10
    elif args.dataset == 'GTSRB':
        train_dataset = GTSRB(train=True, transform=transforms.ToTensor())
        test_dataset = GTSRB(train=False, transform=transforms.ToTensor())
        class_num = 43

    train_loader = DataLoader(train_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=True, 
                            num_workers=4) 
    test_loader = DataLoader(test_dataset, 
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=4)

    save_dir = args.save_surrogate + '_' + args.dataset
    os.makedirs(save_dir + '/' + args.model, exist_ok=True)
    
    model = getattr(models, args.model)(num_classes=class_num).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)
    model_optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=0.9,
            nesterov=True,
            weight_decay=5e-4)
    scheduler = MultiStepLR(
            model_optimizer, 
            milestones=[60, 90], 
            gamma=0.1)

    logger = logging.getLogger()
    logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.DEBUG,
            handlers=[
                logging.FileHandler(os.path.join(save_dir, args.model, 'output_clean.log')),
                logging.StreamHandler()
            ])
    logger.info(args)
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t CleanLoss \t CleanACC')
    for epoch in range(args.epochs):
        start = time.time()
        lr = model_optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_step(model, criterion, model_optimizer, train_loader, device)
        cl_test_loss, cl_test_acc = test_step(model, criterion, test_loader, device)
        scheduler.step()
        end = time.time()
        logger.info(
                '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f',
                epoch, lr, end - start, train_loss, train_acc,
                cl_test_loss, cl_test_acc)

    torch.save(model.state_dict(), os.path.join(save_dir, args.model, "benign_model.pth"))

if __name__ == '__main__':
    main(args)