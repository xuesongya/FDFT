from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
import argparse
import models
import logging
import time
from util import *
from dataset import vggface2

parser = argparse.ArgumentParser(description='Train Backdoored Model')
parser.add_argument('--model', default='iresnet18')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--seed', type=int, default=2048)
parser.add_argument('--y_target', type=int, default=0)
parser.add_argument('--poison_rate', type=float, default=0.01)
parser.add_argument('--save_dir', type=str, default='save_backdoor_vggface2')
parser.add_argument('--method', type=str, default='fdft')
parser.add_argument('--device', type=int, default=0)
parser.add_argument('--dataset', type=str, default='GTSRB', choices=['cifar10', 'vggface2', 'GTSRB'])
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

    train_dataset = vggface2.VGG_Faces2('/home/test/backdoor/test/vggface2/random_select_400_100/train.txt', 
                             train = True, 
                             transform=transforms.ToTensor())
    test_dataset = vggface2.VGG_Faces2('/home/test/backdoor/test/vggface2/random_select_400_100/val.txt', 
                            train = False, 
                            transform=transforms.ToTensor())

    shuffle = np.random.permutation(len(train_dataset))
    total_poison = int(len(train_dataset)*args.poison_rate)
    k = 0
    random_indices = []
    for i in shuffle:
        if train_dataset[i][1] != args.y_target and k < total_poison:
            random_indices.append(i)
            k += 1

    trigger_path = trigger[args.method]

    poison_train_set = generate_poisoned_trainset(args.method, train_dataset, args.y_target, random_indices, trigger_path)
    poison_test_set = generate_poisoned_testset(args.method, test_dataset, args.y_target, trigger_path)

    train_transform = transforms.Compose([
                transforms.ToPILImage(),
                transforms.RandomHorizontalFlip(),
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

    model = getattr(models, args.model)(num_classes=20).to(device)

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

    os.makedirs(args.save_dir + '/' + args.method + '/' + args.model, exist_ok=True)
    logger = logging.getLogger()
    logging.basicConfig(
            format='[%(asctime)s] - %(message)s',
            datefmt='%Y/%m/%d %H:%M:%S',
            level=logging.DEBUG,
            handlers=[
                logging.FileHandler(os.path.join(args.save_dir, args.method, args.model, 'output.log')),
                logging.StreamHandler()
            ])
    logger.info(args)
    logger.info('Epoch \t lr \t Time \t TrainLoss \t TrainACC \t PoisonLoss \t PoisonACC \t CleanLoss \t CleanACC')
    for epoch in range(args.epochs):
        start = time.time()
        lr = model_optimizer.param_groups[0]['lr']
        train_loss, train_acc = train_step(model, criterion, model_optimizer, poison_train_loader, device)
        cl_test_loss, cl_test_acc = test_step(model, criterion, clean_test_loader, device)
        po_test_loss, po_test_acc = test_step(model, criterion, trigger_loader, device)
        scheduler.step()
        end = time.time()
        logger.info(
                '%d \t %.3f \t %.1f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f \t %.4f',
                epoch, lr, end - start, train_loss, train_acc, po_test_loss, po_test_acc,
                cl_test_loss, cl_test_acc)

    torch.save(model.state_dict(), os.path.join(args.save_dir, args.method, args.model, "backdoor_model.pth"))


if __name__ == '__main__':
    main(args)