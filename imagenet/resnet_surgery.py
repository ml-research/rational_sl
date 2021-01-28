import os
import pickle
import sys
import torch
import torch.nn as nn
import torchvision
import numpy as np
import random
# from rational.torch import Rational
import argparse
from rtpt import RTPT
from utils import choose_layer, compute_number_of_exps, make_deter, accuracy, \
    perform_surgery, identity_rational, one_train, one_validate, make_loaders, \
    augmented_print, revalidate, retrain


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-s', '--seed', default=0, type=int,
                    help='random seed to be fixed')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-i', '--iter', default=100, type=int,
                    help='iter')
parser.add_argument('-j', '--num_workers', default=1, type=int,
                    help='iter')
parser.add_argument('--eval_original', default=False, action="store_true",
                    help='if true, evaluation of the original model')
parser.add_argument('--use_id', default=False, action="store_true",
                    help='if true, replace a module with an Identity Layer')
parser.add_argument('--use_rat', default=False, action="store_true",
                    help='if true, replace a module with a Rational')
parser.add_argument('-e', '--epochs', default=15, type=int,
                    help='number of epochs for retraining')
sp_help = 'For --surgered_part:\n' + \
          'either random or $l.$k with $l in [1:4] ' + \
          '(corresponding to the layer) and $k in [1:23]' + \
          '(For example: 2.3 for layer2 block3)'
parser.add_argument('--surgered_part', default="random", action="store",
                    help=sp_help)
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

args = parser.parse_args()
args.print_freq = 1
args.gpus = 0
args.clip_grad_value = 5

train_loader, val_loader = make_loaders(args)

criterion = nn.CrossEntropyLoss().cuda()

resnet = torchvision.models.resnet101(pretrained=True).cuda()


model_save_folder = "surgery/models/"
score_save_folder = "surgery/scores/"
for fold in [model_save_folder, score_save_folder]:
    if not os.path.exists(fold):
        os.makedirs(fold)


total_exps = compute_number_of_exps(args)
rtpt = RTPT(name_initials='QD', experiment_name=f'ResNetSurgery_s{args.seed}',
            max_iterations=args.epochs)
rtpt.start()

if args.eval_original:
    train_loss, train_acc1, train_acc5 = one_train(resnet, rtpt, train_loader, criterion)
    test_loss, test_acc1, test_acc5 = one_validate(resnet, rtpt, val_loader, criterion)
    print("Original Resnet:")
    print(f"Train : Loss: {train_loss} | acc@1: {train_acc1} | acc5: {train_acc5}")
    print(f"Test  : Loss: { test_loss} | acc@1: { test_acc1} | acc5: { test_acc5}")
    filename = "original_resnet"
    score_dict = {"train_loss": train_loss, "test_loss": test_loss,
                  "train_acc1": train_acc1, "test_acc1": test_acc1,
                  "train_acc5": train_acc5, "test_acc5": test_acc5}
    pickle.dump(score_dict, open(os.path.join(score_save_folder, filename) + ".pkl", 'wb'))

layer, block_n = choose_layer(args, sp_help)

if args.use_id:
    make_deter(args.seed)
    id_module = nn.Identity()
    perform_surgery(resnet, layer, block_n, id_module)
    print(f"Identity Surgery performed on {layer} block n {block_n}")

    train_loss, train_acc1, train_acc5 = one_train(resnet, rtpt, train_loader, criterion)
    test_loss, test_acc1, test_acc5 = one_validate(resnet, rtpt, val_loader, criterion)

    print("Modified Resnet: (Identity)")
    print(f"Train : Loss: {train_loss} | acc@1: {train_acc1} | acc5: {train_acc5}")
    print(f"Test  : Loss: { test_loss} | acc@1: { test_acc1} | acc5: { test_acc5}")
    filename = f"surgered_resnet_id_{layer}_block{block_n}"
    score_dict = {"layer": layer, "block_n": block_n,
                  "train_loss": train_loss, "test_loss": test_loss,
                  "train_acc1": train_acc1, "test_acc1": test_acc1,
                  "train_acc5": train_acc5, "test_acc5": test_acc5}
    pickle.dump(score_dict, open(os.path.join(score_save_folder, filename) + ".pkl", 'wb'))
    pickle.dump(resnet, open(os.path.join(model_save_folder, filename) + ".pkl", 'wb'))


if args.use_rat:
    make_deter(args.seed)
    rat_module = identity_rational()
    # rat_module.input_retrieve_mode(auto_stop=False, bin_width=0.01)
    # new_module = Rational()
    for param in resnet.parameters():
        param.requires_grad_(False)
    params_to_update = perform_surgery(resnet, layer, block_n, rat_module)
    blue, normal = '\033[94m', '\033[0m'
    print(f"{blue} Rational Surgery performed on {layer} block n {block_n} {normal}")
    optimizers = [torch.optim.SGD(params_to_update, lr=args.lr, momentum=0.9)]
    train_losses, train_accs1, train_accs5 = list(), list(), list()
    test_losses, test_accs1, test_accs5 = list(), list(), list()
    augmented_print(resnet)
    # exit()
    for epoch in range(args.epochs):
        train_loss, train_acc1, train_acc5 = retrain(train_loader, resnet, criterion, optimizers, None, epoch, args)
        test_loss, test_acc1, test_acc5 = revalidate(val_loader, resnet, criterion, epoch, args)
        train_losses.append(train_loss); train_accs1.append(train_acc1); train_accs5.append(train_acc5)
        test_losses.append(test_loss); test_accs1.append(test_acc1); test_accs5.append(test_acc5)
        print(f"Epoch {epoch} Modified Resnet: (Rational)")
        print(f"Train : Loss: {train_loss} | acc@1: {train_acc1} | acc5: {train_acc5}")
        print(f"Test  : Loss: { test_loss} | acc@1: { test_acc1} | acc5: { test_acc5}")
        rtpt.step()
    filename = f"surgered_trained_rat_{layer}_block{block_n}"
    score_dict = {"layer": layer, "block_n": block_n,
                  "train_loss": train_losses, "test_loss": test_losses,
                  "train_acc1": train_accs1, "test_acc1": test_accs1,
                  "train_acc5": train_accs5, "test_acc5": test_accs5}
    pickle.dump(score_dict, open(os.path.join(score_save_folder, filename) + ".pkl", 'wb'))
    pickle.dump(resnet, open(os.path.join(model_save_folder, filename) + ".pkl", 'wb'))
    print(f"Saved score in {os.path.join(score_save_folder, filename)}")
    print(f"Saved model in {os.path.join(model_save_folder, filename)}")
