from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import models
import logging
import time
from util import *
from dataset.GTSRB import GTSRB

parser = argparse.ArgumentParser(description='Train Backdoored Model')
parser.add_argument('--model', default='resnet18')
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.1)
parser.add_argument('--seed', type=int, default=2048)
parser.add_argument('--y_target', type=int, default=0)
parser.add_argument('--poison_rate', type=float, default=0.01)
parser.add_argument('--save_dir', type=str, default='save_backdoor')
parser.add_argument('--method', type=str, default='fdft')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'vggface2', 'GTSRB'])
parser.add_argument('--m', type=float, default=1.0)
args = parser.parse_args()

trigger = {
    'fdft' : '/home/test/backdoor/test/data/trigger_white.png',
    'badnets' : '/home/test/backdoor/test/data/trigger_white.png',
    'blended' : '/home/test/backdoor/test/data/hk.png',
    'issba' : '',
}

def main(args):

    set_random_seed(args.seed)
    device = f'cuda:{args.device}'
    set_dataset_mode(args.dataset)

    train_dataset = None
    test_dataset = None
    class_num = 10

    if args.dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root='dataset', 
                                        train=True, 
                                        transform=transforms.ToTensor(), 
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

    shuffle = np.random.permutation(len(train_dataset))
    total_poison = int(len(train_dataset)*args.poison_rate)
    k = 0
    random_indices = []
    for i in shuffle:
        if train_dataset[i][1] != args.y_target and k < total_poison:
            random_indices.append(i)
            k += 1

    trigger_path = trigger[args.method]

    poison_train_set = generate_poisoned_trainset(args.method, train_dataset, args.y_target, random_indices, trigger_path, m = args.m)
    poison_test_set = generate_poisoned_testset(args.method, test_dataset, args.y_target, trigger_path, m = args.m)

    train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Pad(4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32),
                transforms.ToTensor()])
    poison_train_set = MyDataset(poison_train_set, train_transform)
    poison_train_loader = DataLoader(poison_train_set, 
                                    batch_size=args.batch_size, 
                                    shuffle=True, 
                                    num_workers=4)
    clean_test_loader = DataLoader(test_dataset, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=4)
    trigger_loader = DataLoader(poison_test_set, 
                                batch_size=args.batch_size, 
                                shuffle=False, 
                                num_workers=4)

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

    base_dir = args.save_dir + '_' + args.dataset
    os.makedirs(base_dir + '/' + args.method + '/' + args.model, exist_ok=True)
    logger = logging.getLogger()
    logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.DEBUG,
            handlers=[
                logging.FileHandler(os.path.join(base_dir, args.method, args.model, 'output.log')),
                logging.StreamHandler()
            ])
    logger.info(args)
    logger.info('PosionRate\t Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    for epoch in range(args.epochs):
        start = time.time()
        lr = model_optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_step(model, criterion, model_optimizer, poison_train_loader, device)
        cl_test_loss, cl_test_acc = test_step(model, criterion, clean_test_loader, device)
        po_test_loss, po_test_acc = test_step(model, criterion, trigger_loader, device)
        scheduler.step()
        end = time.time()
        logger.info(
                '%.4f \t %d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                args.poison_rate, epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
                cl_test_loss, cl_test_acc)

    torch.save(model.state_dict(), os.path.join(base_dir, args.method, args.model, "backdoor_model.pth"))


if __name__ == '__main__':
    main(args)