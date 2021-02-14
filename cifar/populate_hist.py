'''Train CIFAR10 with PyTorch.'''
import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision
import torchvision.transforms as transforms
import pickle
import os
import argparse
from models import *
from utils import progress_bar
from rational.torch import Rational


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
        # loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        # trains_losses.append(train_loss/(batch_idx+1))
        # trains_acc.append(100.*correct/total)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--saved_model', '-sm', action='store', required=True,
                    help='path to the model to be loaded (pickle format)')

args = parser.parse_args()


assert 'rn' in args.saved_model
seed = int(args.saved_model.split("rn_")[-1][0])
dataset = args.saved_model.split("cifar")[1]
if dataset[:3] == "100":
    dataset = "cifar100"
elif dataset[:2] == "10":
    dataset = "cifar10"
else:
    print("dataset not recognised")
    exit(1)
print(dataset)

np.random.seed(seed)
torch.manual_seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
print(f"Set all environment deterministic to seed {seed}")


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

if dataset == "cifar10":
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
elif dataset == "cifar100":
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=2)
else:
    print("No dataset found")
    exit(1)

classes = ('plane', 'car', 'bird', 'cat', 'deer',
           'dog', 'frog', 'horse', 'ship', 'truck')

# load_model
net = pickle.load(open(args.saved_model, "rb"))
print(net)
rationals = []
for el in net.modules():
    if type(el) is Rational:
        el.input_retrieve_mode(auto_stop=False)
        rationals.append(el)


net = net.to(device)
if device == 'cuda':
    net = torch.nn.DataParallel(net)
    cudnn.benchmark = True

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1,
                      momentum=0.9, weight_decay=5e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

with torch.no_grad():
    train(0)
    # test(0)

# for rat in rationals:
#     rat.training_mode()
#     rat.best_fitted_function = None
#     rat.show()

filename = args.saved_model.split("/")[-1]
subdir = args.saved_model.split("/")[-2]
save_dir = f"trained_networks/populated/{subdir}"
print(save_dir)

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


with open(f"{save_dir}/{filename}", "wb") as modelf:
    pickle.dump(net, modelf)
