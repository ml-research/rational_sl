'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import pickle
import os
import argparse
from rtpt import RTPT
from models import *
# from utils import progress_bar

from rational.torch import Rational, RecurrentRational

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--lr', default=0.01, type=float, help='learning rate')
parser.add_argument('--nn_type', '-nt', action='store',
                    help='activation function to be used, either lrelu, rn, rrn or a combination in the form of rXrX where X is either a number or nothing')
parser.add_argument('--seed', '-s', action='store', type=int,
                    help='activation function to be used')
parser.add_argument('--epochs', '-e', action='store', type=int, default=60,
                    help='Number of epochs')
parser.add_argument('--arch', type=str, required=True)
parser.add_argument('--dataset', type=str, required=True,
                    choices=["cifar100", "cifar10"])
parser.add_argument('--init', "-i", type=str, default="xavier",
                    choices=["xavier", "he", "no"])
parser.add_argument('--batch_norm', '-bn',  action='store_true')
args = parser.parse_args()

args.arch = args.arch.lower()

print(args.init)

np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Set all environment deterministic to seed {args.seed}")


device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

if args.dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    nb_outputs = 10

elif args.dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR100(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR100(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
    nb_outputs = 100


classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

if args.nn_type == "lrelu":
    activation_type = nn.LeakyReLU
elif args.nn_type == "rn":
    activation_type = Rational
elif args.nn_type == "rrn":
    activation_type = RecurrentRational()
else:
    activation_type = []
    for num in args.nn_type.split('r')[1:]:
        if num == "":
            activation_type.append(Rational)
        else:
            assert num in [str(i + 1) for i in range(20)]
            rec_rat = RecurrentRational()
            for _ in range(int(num)):
                activation_type.append(rec_rat)
# Model
print('==> Building model..')
if args.arch == "vgg8":
    net = VGG8(activation_type, args.batch_norm, nb_outputs)
elif "vgg" in args.arch:
    net = VGG(args.arch.upper(), activation_type, args.batch_norm, nb_outputs)
elif args.arch == "lenet":
    if args.batch_norm:
        net = LeNet_bn(activation_type, nb_outputs)
    else:
        net = LeNet(activation_type, nb_outputs)
elif args.arch == "resnet18":
    net = ResNet18(activation_type)
# net = PreActResNet18(activation_type)
# net = GoogLeNet(activation_type)
# net = DenseNet121()
# net = ResNeXt29_2x64d()
# net = MobileNet()
# net = MobileNetV2()
# net = DPN92()
# net = ShuffleNetG2()
# net = SENet18()
# net = ShuffleNetV2(1)
# net = EfficientNetB0()
# net = RegNetX_200MF()
# net = SimpleDLA()
else:
    print("Arch not covered")
    exit(1)
net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

print(net)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=args.lr,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40, 50])
trains_losses = []
trains_acc = []
tests_losses = []
tests_acc = []

def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
        m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        m.bias.data.fill_(0.01)

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        m.bias.data.fill_(0.01)

if args.init == "xavier":
    net.apply(weights_init_xavier)
    print("xavier weight init")
elif args.init == "he":
    net.apply(weights_init_kaiming)
    print("he/kaiming weight init")

# print(net)

# Training
def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = net(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        # progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #              % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
    trains_losses.append(train_loss/(batch_idx+1))
    trains_acc.append(100.*correct/total)

def test(epoch):
    global best_acc
    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
            #              % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
    tests_losses.append(test_loss/(batch_idx+1))
    tests_acc.append(100.*correct/total)

rtpt = RTPT(name_initials='QD', experiment_name=f'RationCifar10_{args.nn_type}', max_iterations=args.epochs)

# Start the RTPT tracking
rtpt.start()
for epoch in range(args.epochs):
    train(epoch)
    test(epoch)
    scheduler.step()
    rtpt.step()

# args.dataset = "cifar10"
dirs = [f"scores_sl/{args.arch}_scores_{args.dataset}/",
        f"trained_networks/{args.arch}_models_{args.dataset}/"]

for fold in dirs:
    if not os.path.exists(fold):
        os.makedirs(fold)

suffix = f"{args.arch}_{args.nn_type}_{args.seed}_{args.init}"
if args.batch_norm:
    suffix += "_bn"

with open(f"{dirs[0]}/scores_{suffix}.pkl", "wb") as scoref:
    pickle.dump({"test/loss": tests_losses, "test/accuracy@1": tests_acc,
                 "train/loss": trains_losses, "train/accuracy@1": trains_acc},
                scoref)

with open(f"{dirs[1]}/models_{suffix}.pkl", "wb") as modelf:
    pickle.dump(net, modelf)
