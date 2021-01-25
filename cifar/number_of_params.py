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
from utils import progress_bar

from rational.torch import Rational, RecurrentRational

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--nn_type', '-nt', action='store',
                    choices=["lrelu", "rn", "rrn"],
                    help='activation function to be used')
parser.add_argument('--arch', type=str, required=True)
parser.add_argument('--batch_norm', '-bn',  action='store_true')
args = parser.parse_args()



device = 'cpu'
nb_outputs = 100

if args.nn_type == "lrelu":
    activation_type = nn.LeakyReLU
elif args.nn_type == "rn":
    activation_type = Rational
elif args.nn_type == "rrn":
    activation_type = RecurrentRational()
# Model
print('==> Building model..')
if args.arch == "vgg8":
    net = VGG8(activation_type, False, nb_outputs)
elif "vgg" in args.arch:
    net = VGG(args.arch.upper(), activation_type, False, nb_outputs)
elif args.arch == "lenet":
    net = LeNet(activation_type, nb_outputs)
elif args.arch == "resnet18":
    net = ResNet18(activation_type)

nb_params = count_parameters(net)
print(f"{args.arch} with {args.nn_type} has {nb_params} parameters")
