import numpy as np
import torch

torch.manual_seed(17)
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = False
np.random.seed(17)

import argparse
import os
import random
import shutil
import time
import warnings
import sys
import pickle
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import models as models
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

from rtpt import RTPT
from rational.torch import RecurrentRational, Rational

model_names = sorted(name for name in models.__dict__
                     if name.islower() and not name.startswith("__")
                     and callable(models.__dict__[name]))

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet18',
                    choices=model_names,
                    help='model architecture: ' +
                         ' | '.join(model_names) +
                         ' (default: resnet18)')
parser.add_argument('-j', '--workers', default=3, type=int, metavar='N',
                    help='number of data loading workers (default: 4), (rule of thump: (2 to 4) * numgpus)')
parser.add_argument('--epochs', default=350, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr_pade', '--learning-rate-pade', default=0.1, type=float,
                    metavar='LR', help='initial learning rate for activation functions', dest='lr_pade')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--momentum_pade', default=0.9, type=float, metavar='M',
                    help='momentum')
# parser.add_argument('--dropout', default=0., type=float, metavar='M',
#                    help='dropout')
parser.add_argument('--wd', '--weight-decay', default=1e-5, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('--pade_optimization', action='store_true')
parser.add_argument('--resume_with_lr', default=-1, type=float,
                    help='lr to resume with, if -1 then load lr from saved model')
parser.add_argument('--resume_with_lr_pade', default=-1, type=float,
                    help='lr to resume with, if -1 then load lr from saved model')
parser.add_argument('--clip_grad_value', default=5, type=int,
                    help='')

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=17, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpus', default=None, type=str,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--selected_activation', default='relu6', type=str,
                    help='one of the standard activation functions or a pade activation function')
parser.add_argument('--save_path', type=str, required=True)
# parser.add_argument('--lr_scheduler', default='', type=str)
# parser.add_argument('--lr_scheduler_pade', default='', type=str)
parser.add_argument('--lr_min', default=0., type=float)
parser.add_argument('--lr_min_pade', default=0., type=float)
parser.add_argument('--init', "-i", type=str, default="xavier",
                    choices=["xavier", "he", "no"])
best_acc1 = 0
report_dict = {'train/loss': [], 'train/accuracy@1': [], 'train/accuracy@5': [], \
              'validate/loss': [], 'validate/accuracy@1': [], 'validate/accuracy@5': []}

def weights_init_xavier(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data, gain=nn.init.calculate_gain('leaky_relu'))
        if m.bias is not None:
            m.bias.data.fill_(0.01)
    elif isinstance(m, nn.Linear):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def weights_init_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight.data)
        if m.bias is not None:
            m.bias.data.fill_(0.01)

def totimestring(seconds):
    day = seconds // (24 * 3600)
    time = seconds % (24 * 3600)
    hour = time // 3600
    time %= 3600
    minutes = time // 60
    return "{}d:{}h:{}m".format(int(day), int(hour), int(minutes))


# CUDA_VISIBLE_DEVICES=8,9 python model_imagenet/main.py -a resnet101 --seed 17 --lr 0.1 --lr_pade 0.1  --save_path=experiments/paper_imagenet/resnet101 --pade_optimization --selected_activation elu --gpu 0,1 /datasets/imagenet --epochs=100 -b 256 -j 5
def main():
    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpus is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    # Simply call main_worker function
    main_worker(list(map(int, args.gpus.strip().split(','))), args)


def update_srelu(args, model, train_loader):
    print("Updating srelu")
    model.eval()

    # TODO only work for 1 shared actv function per layer
    tr = dict()
    hook_handles = list()

    def srelu_hook(self, input, output):
        x = input[0] #.detach().cpu().view([-1]).numpy()
        k = (0.9 * torch.abs(x)).max()
        tr_ = torch.max(x.max(), other=k)
        tr[id(self)] = torch.max(tr[id(self)], other=tr_)

    def register_srelu_hook(m):
        classname = m.__class__.__name__
        if 'srelu' in classname.lower():
            tr[id(m)] = torch.tensor(0, device='cuda', dtype=torch.float)
            # print("Register", classname, id(m))
            hook_handle = m.register_forward_hook(srelu_hook)
            hook_handles.append(hook_handle)

    model.apply(register_srelu_hook)

    print(len(train_loader))
    with torch.no_grad():
        for batch_idx, (input, _) in tqdm(enumerate(train_loader)):
            if args.gpus is not None:
                input = input.cuda()
            output = model(input)

    for hook_handle in hook_handles:
        hook_handle.remove()

    def update_srelu_(m):
        classname = m.__class__.__name__
        if 'srelu' in classname.lower():
            m.tr.data = torch.tensor(tr[id(m)], device='cuda')

    model.apply(update_srelu_)

    for key in tr.keys():
        tr[key] = tr[key].cpu()

    del tr
    print("Finished updating srelu")


def main_worker(gpu, args):
    global best_acc1

    args.gpus = gpu
    torch.set_num_threads(len(gpu) * 6)
    num_workers = len(gpu) * args.workers
    if args.gpus is not None:
        print("Use GPU: {} for training".format(args.gpus))

    selected_actf = args.selected_activation
    if selected_actf == "lrelu":
        actv_function = nn.LeakyReLU
    elif selected_actf == "rn":
        actv_function = Rational
    elif selected_actf == "rrn":
        actv_function = RecurrentRational()

    args.save_path = os.path.join(args.save_path, selected_actf)

    if args.arch != 'mobilenet_v2' and 'efficientnet' not in args.arch and \
        'resnet' not in args.arch and 'vgg' not in args.arch:
        raise ValueError('Pade not implemented for selected network architecture')

    # create model
    if args.pretrained:
        print("=> using pre-trained model '{}'".format(args.arch))
        model = models.__dict__[args.arch](pretrained=True, activation_func=actv_function)  # , drop_rate=args.dropout
    else:
        print("=> creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch](activation_func=actv_function)  # , drop_rate=args.dropout

    print(args.gpus)
    # DataParallel will divide and allocate batch_size to all available GPUs
    if args.arch.startswith('alexnet') or args.arch.startswith('vgg'):
        model.features = torch.nn.DataParallel(model.features, device_ids=args.gpus)
        model.cuda()
    else:
        model = torch.nn.DataParallel(model, device_ids=args.gpus).cuda()

    if args.init == "xavier":
        model.apply(weights_init_xavier)
        print("xavier weight init")
    elif args.init == "he":
        model.apply(weights_init_kaiming)
        print("he/kaiming weight init")
    # define loss function (criterion) and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    params = list()
    params_activation = list()
    params_activation_srelu = list()
    params_activation_apl = list()

    for p in model.named_parameters():
        if 'weight_center' in p[0] or 'numerator' in p[0] or '_denominator' in p[0]:
            if p[1].requires_grad:
                params_activation.append(p[1])
        elif p[0].endswith(".tr") or p[0].endswith(".tl") or p[0].endswith(".ar") or p[0].endswith(".al"):
            if p[1].requires_grad:
                params_activation_srelu.append(p[1])
        elif p[0].endswith(".apl_a") or p[0].endswith(".apl_b"):
            if p[1].requires_grad:
                params_activation_apl.append(p[1])
        else:
            params.append(p[1])

    # setup of MobileNetV2 https://arxiv.org/pdf/1801.04381.pdf
    #scheduler = None
    scheduler_activation = None
    scheduler_srelu = None
    scheduler_apl = None

    scheduler_plateau = None
    scheduler_activation_plateau = None
    scheduler_srelu_plateau = None
    scheduler_apl_plateau = None

    optimizer_activation = None
    optimizer_apl = None
    optimizer_srelu = None

    if 'efficientnet' in args.arch:
        # base_lr 0.016 for batch_size 256
        # optimizer = torch.optim.RMSprop(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                    nesterov=False)

        scheduler = {
            """"scheduler": torch.optim.lr_scheduler.StepLR(optimizer,
                                                         step_size=3,
                                                         gamma=0.97),"""
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                              milestones=[30, 60, 90],
                                                              gamma=0.1),
            "optimizer": optimizer, "lr_min": args.lr_min
        }
        """scheduler_plateau = {
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                                    factor=0.2,
                                                                    verbose=True,
                                                                    patience=5,
                                                                    min_lr=1e-4),
            "optimizer": optimizer, "lr_min": args.lr_min
        }"""
        if "rn" in selected_actf:
            print("Optimizing with Pade activation")
            # optimizer_activation = torch.optim.RMSprop(params_activation, lr=args.lr_pade, momentum=args.momentum_pade,
            #                                       weight_decay=0)
            optimizer_activation = torch.optim.SGD(params_activation,
                                                   lr=args.lr_pade,
                                                   momentum=args.momentum_pade,
                                                   weight_decay=0)
            scheduler_activation = {
                """"scheduler": torch.optim.lr_scheduler.StepLR(optimizer_activation,
                                                             step_size=3,
                                                             gamma=0.97),"""
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer_activation,
                                                                  milestones=[30, 60, 90],
                                                                  gamma=0.1),
                "optimizer": optimizer_activation, "lr_min": args.lr_min_pade
            }

            """scheduler_activation_plateau = {
                "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer_activation,
                                                                        factor=0.2,
                                                                        verbose=True,
                                                                        patience=5, min_lr=1e-4),
                "optimizer": optimizer_activation, "lr_min": args.lr_min_pade
            }"""
        elif "srelu" == selected_actf:
            optimizer_srelu = torch.optim.SGD(params_activation_srelu,
                                              lr=args.lr,
                                              momentum=args.momentum,
                                              weight_decay=0)
            scheduler_srelu = {
                """"scheduler": torch.optim.lr_scheduler.StepLR(optimizer_activation,
                                                             step_size=3,
                                                             gamma=0.97),"""
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer_srelu,
                                                                  milestones=[30, 60, 90],
                                                                  gamma=0.1),
                "optimizer": optimizer_srelu, "lr_min": args.lr_min
            }
        elif "apl" == selected_actf:
            optimizer_apl = torch.optim.SGD(params_activation_apl,
                                            lr=args.lr,
                                            momentum=args.momentum,
                                            weight_decay=0)
            scheduler_apl = {
                """"scheduler": torch.optim.lr_scheduler.StepLR(optimizer_activation,
                                                             step_size=3,
                                                             gamma=0.97),"""
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer_apl,
                                                                  milestones=[30, 60, 90],
                                                                  gamma=0.1),
                "optimizer": optimizer_apl, "lr_min": args.lr_min
            }
        else:
            print("Opimizing with some standard activation")
            optimizer_activation = None
            optimizer_apl = None
            optimizer_srelu = None
            scheduler_activation = None
            scheduler_srelu = None
            scheduler_apl = None

    elif args.arch == 'mobilenet_v2' or 'resnet' in args.arch or 'densenet' in args.arch or 'vgg' in args.arch:
        ## mobilenet Paper hyperparameters
        optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay,
                                    nesterov=False)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60, 90, 120], gamma=0.1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.97)  # 0.93
        scheduler = {
            # "scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98),
            "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                              milestones=[30, 60, 90],
                                                              gamma=0.1),
            "optimizer": optimizer, "lr_min": args.lr_min
        }
        if "rn" in selected_actf:
            print("Optimizing with Pade activation")
            optimizer_activation = torch.optim.SGD(params_activation, lr=args.lr_pade, momentum=args.momentum_pade,
                                                   weight_decay=0)
            scheduler_activation = {
                # "scheduler": torch.optim.lr_scheduler.StepLR(optimizer_activation, step_size=1, gamma=0.98),
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer_activation,
                                                                  milestones=[30, 60, 90],
                                                                  gamma=0.1),
                "optimizer": optimizer_activation, "lr_min": args.lr_min_pade}
            optimizer_srelu = None
            scheduler_srelu = None

            optimizer_apl = None
            scheduler_apl = None
        elif "srelu" == selected_actf:
            print("Optimizing with SReLU activation")
            optimizer_srelu = torch.optim.SGD(params_activation_srelu, lr=args.lr,
                                              momentum=args.momentum,
                                              weight_decay=0)
            scheduler_srelu = {
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer_srelu,
                                                                  milestones=[30, 60, 90],
                                                                  gamma=0.1),
                "optimizer": optimizer_srelu, "lr_min": args.lr_min}
            optimizer_activation = None
            scheduler_activation = None

            optimizer_apl = None
            scheduler_apl = None
        elif "apl" == selected_actf:
            print("Optimizing with APL activation")
            optimizer_apl = torch.optim.SGD(params_activation_apl, lr=args.lr,
                                            momentum=args.momentum,
                                            weight_decay=0.001)
            scheduler_apl = {
                # "scheduler": torch.optim.lr_scheduler.StepLR(optimizer_activation, step_size=1, gamma=0.98),
                "scheduler": torch.optim.lr_scheduler.MultiStepLR(optimizer_apl,
                                                                  milestones=[30, 60, 90],
                                                                  gamma=0.1),
                "optimizer": optimizer_apl, "lr_min": args.lr_min}
            optimizer_activation = None
            scheduler_activation = None

            optimizer_srelu = None
            scheduler_srelu = None
        else:
            print("Opimizing with some standard activation")
            optimizer_activation = None
            scheduler_activation = None

            optimizer_srelu = None
            scheduler_srelu = None

            optimizer_apl = None
            scheduler_apl = None
    else:
        raise ValueError("network not implemented")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            best_acc1 = checkpoint['best_acc1']
            if args.gpus is not None:
                # best_acc1 may be from a checkpoint from a different GPU
                best_acc1 = best_acc1.cuda()
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            if args.resume_with_lr >= 0:
                optimizer.param_groups[0]["lr"] = args.resume_with_lr
            """try:
                scheduler.load_state_dict(checkpoint['scheduler'])
            except:
                pass"""

            if optimizer_activation is not None:
                optimizer_activation.load_state_dict(checkpoint['optimizer_acitivation_func'])
                if args.resume_with_lr_pade >= 0:
                    optimizer_activation.param_groups[0]["lr"] = args.resume_with_lr_pade
                """try:
                    scheduler_activation.load_state_dict(checkpoint['scheduler_activation'])
                except:
                    pass"""
            if optimizer_srelu is not None:
                optimizer_srelu.load_state_dict(checkpoint['optimizer_srelu'])

            if optimizer_apl is not None:
                optimizer_apl.load_state_dict(checkpoint['optimizer_apl'])

            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    optimizers = list()
    optimizers.append(optimizer)
    if optimizer_activation is not None:
        optimizers.append(optimizer_activation)
    #if optimizer_srelu is not None: # call after freeze epoch is reached
    #    optimizers.append(optimizer_srelu)
    if optimizer_apl is not None:
        optimizers.append(optimizer_apl)

    schedulers = list()
    if scheduler is not None:
        schedulers.append(scheduler)
    if scheduler_activation is not None:
        schedulers.append(scheduler_activation)
    if scheduler_srelu is not None:
        schedulers.append(scheduler_srelu)
    if scheduler_apl is not None:
        schedulers.append(scheduler_apl)

    schedulers_plateau = list()
    if scheduler_plateau is not None:
        schedulers_plateau.append(scheduler_plateau)
    if scheduler_activation_plateau is not None:
        schedulers_plateau.append(scheduler_activation_plateau)
    if scheduler_srelu_plateau is not None:
        schedulers_plateau.append(scheduler_srelu_plateau)
    if scheduler_apl_plateau is not None:
        schedulers_plateau.append(scheduler_apl_plateau)

    # cudnn.benchmark = True

    # Data loading code
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(
        traindir,
        transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    srelu_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1024, shuffle=True,
        num_workers=num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True)

    if args.evaluate:
        validate(val_loader, model, criterion, 0, args)
        return

    rtpt = RTPT(name_initials='QD', experiment_name='ratio_imnet', max_iterations=args.epochs)
    rtpt.start()
    for epoch in range(args.start_epoch, args.epochs):
        if "srelu" == selected_actf and epoch == srelu_update_epoch:
            update_srelu(args, model.module, srelu_loader)
            optimizers = optimizers + [optimizer_srelu]
            save_checkpoint({
                'epoch': epoch,
                'arch': args.arch,
                'activation_function': selected_actf,
                'state_dict': model.state_dict(),
                'best_acc1': best_acc1,
                'optimizer': optimizer.state_dict(),
                'optimizer_acitivation_func': None if optimizer_activation is None else optimizer_activation.state_dict(),
                'optimizer_srelu': None if optimizer_srelu is None else optimizer_srelu.state_dict(),
                'optimizer_apl': None if optimizer_apl is None else optimizer_apl.state_dict(),
                'scheduler': scheduler["scheduler"].state_dict(),
                'scheduler_activation': None if optimizer_activation is None else scheduler_activation[
                    "scheduler"].state_dict()
            }, is_best=False, filename='checkpoint_sreluUpdated.pth.tar', args=args)
        # train for one epoch
        train(train_loader, model, criterion, optimizers, schedulers, epoch, args)

        # evaluate on validation set
        acc1, val_loss = validate(val_loader, model, criterion, epoch, args)

        for scheduler_epoch in schedulers_plateau:
            scheduler_epoch["scheduler"].step(val_loss)

        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)

        rtpt.step()

        save_checkpoint({
            'epoch': epoch + 1,
            'arch': args.arch,
            'activation_function': selected_actf,
            'state_dict': model.state_dict(),
            'best_acc1': best_acc1,
            'optimizer': optimizer.state_dict(),
            'optimizer_acitivation_func': None if optimizer_activation is None else optimizer_activation.state_dict(),
            'optimizer_srelu': None if optimizer_srelu is None else optimizer_srelu.state_dict(),
            'optimizer_apl': None if optimizer_apl is None else optimizer_apl.state_dict(),
            'scheduler': scheduler["scheduler"].state_dict(),
            'scheduler_activation': None if optimizer_activation is None else scheduler_activation[
                "scheduler"].state_dict()
        }, is_best, args=args)

    ##
        rtpt.step()

    if "resnet" in args.arch:
        scores_folder = "scores_resnet"
        models_folder = "models_resnet"
    elif "mobilenet" in args.arch:
        scores_folder = "scores_mobilenet"
        models_folder = "models_mobilenet"

    for folder in [scores_folder, models_folder]:
        if not os.path.exists(folder):
            os.makedirs(folder)
    filepart = f"imagenet_{args.arch}_{args.selected_activation}_{args.seed}"
    with open(f"{scores_folder}/scores_{filepart}.pkl", "wb") as score_file:
        pickle.dump(report_dict, score_file)
    torch.save(model, f"{models_folder}/model_{filepart}.pth")
    print("Saved scores in :" + f"{scores_folder}/scores_{filepart}.pkl")
    print("Saved model in :" + f"{models_folder}/model_{filepart}.pth")



def train(train_loader, model, criterion, optimizers, schedulers, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1,
                             top5, prefix="Epoch: [{}]".format(epoch))

    n_batches = len(train_loader)
    print("Num Batches per epoch: ", n_batches)
    # switch to train mode
    model.train()

    end = time.time()
    global report_dict

    for batch_idx, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpus is not None:
            input = input.cuda()
            target = target.cuda()

        # compute output
        output = model(input)
        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1.item(), input.size(0))
        top5.update(acc5.item(), input.size(0))

        # compute gradient and do SGD step
        for optimizer in optimizers:
            optimizer.zero_grad()

        loss.backward()

        clip_grad_norm_(model.parameters(), args.clip_grad_value, norm_type=2)

        # update step
        for optimizer in optimizers:
            optimizer.step()
        if schedulers is not None:
            for idx_scheduler, scheduler in enumerate(schedulers):
                if scheduler["scheduler"].__class__.__name__ == 'CyclicLR':
                    scheduler["scheduler"].step()
                else:
                    if batch_idx == n_batches - 1:
                        if scheduler["optimizer"].param_groups[0]["lr"] > scheduler["lr_min"]:
                            scheduler["scheduler"].step(epoch=epoch)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()


        if batch_idx % args.print_freq == 0:
            progress.print(batch_idx)

    report_dict['train/loss'].append(losses.avg)
    report_dict['train/accuracy@1'].append(top1.avg)
    report_dict['train/accuracy@5'].append(top5.avg)

    return top1.avg, top5.avg, losses.avg


def validate(val_loader, model, criterion, epoch, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(val_loader), batch_time, losses, top1, top5,
                             prefix='Test: ')

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            if args.gpus is not None:
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.print(i)

        # TODO: this should also be done with the ProgressMeter
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    report_dict['validate/loss'].append(losses.avg)
    report_dict['validate/accuracy@1'].append(top1.avg)
    report_dict['validate/accuracy@5'].append(top5.avg)

    return top1.avg, top5.avg, losses.avg


def save_checkpoint(state, is_best, args, filename='checkpoint.pth.tar'):
    save_path_checkpoint = os.path.join(args.save_path, filename)
    save_path_checkpoint_best = os.path.join(args.save_path, 'model_best.pth.tar')
    torch.save(state, save_path_checkpoint)
    if is_best:
        shutil.copyfile(save_path_checkpoint, save_path_checkpoint_best)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum()
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == '__main__':
    main()
