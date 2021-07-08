import random
import numpy as np
import torch
import os
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from rational.torch import Rational
from torch.nn.modules.module import _addindent
import time
from torch.nn.utils import clip_grad_norm_


def make_loaders(args):
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
        train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True,
        num_workers=args.num_workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])),
        batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True)
    return train_loader, val_loader

def choose_parts(args, sp_help):
    bottleneck_dict = {"layer1": 3, "layer2": 4, "layer3": 23, "layer4": 3}
    surg_list =  args.surgered_part.replace(' ', '').split(',')
    parts = []
    for oper in surg_list:
        if oper != "random":
            if not (args.use_id or args.use_rat or args.eval_id or args.reuse_rat):
                print('No use of giving bottleneck to perform surgery on if ' +
                      '"--use_id" and "--use_rat are not given"')
                exit(1)
            try:
                lay, blo = oper.split('.')
                if int(lay) not in range(1, 5):
                    print(sp_help)
                    print("Available layers: 1, 2, 3, 4")
                    exit(1)
                n_blocks = bottleneck_dict["layer" + lay]
                if int(blo) not in range(n_blocks):
                    print("For --surgered_part:")
                    print(f"Available blocks for layer{lay}: " + \
                          str(list(range(n_blocks))))
                    exit(1)
                layer = "layer" + lay
                block_n = blo
            except Exception as e:
                if type(e) is not SystemExit:
                    print(sp_help)
                exit(1)
        else:
            layer = np.random.choice(list(bottleneck_dict.keys()))
            block_n = str(np.random.randint(bottleneck_dict[layer]))
        parts.append((layer, block_n))
    return parts


def compute_number_of_exps(args):
    total_exps = 0
    if args.eval_original:
        total_exps += 1
    if args.eval_id:
        total_exps += 1
    if args.use_id:
        total_exps += args.epochs
    if args.use_rat:
        total_exps += args.epochs
    if args.reuse_rat:
        total_exps += args.epochs
    return total_exps


def make_deter(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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


def perform_surgery(network, layer, block, new_module, seperate_rational=False):
    update_params = list()
    block_int = int(block)
    if type(new_module) is Rational:
        if not seperate_rational:
            update_params.extend([new_module.numerator, new_module.denominator])
    if layer == 'layer1':
        network.layer1.__setattr__(block, new_module)
        if block_int > 0:
            previous_bn = network.layer1[block_int - 1]
        else:
            previous_bn = None
            print("No previous block found, as surgery performed on first block of first layer")
        if block_int + 1 < len(network.layer1):
            next_bn = network.layer1[block_int+1]
        else:
            next_bn = network.layer2[0]
    elif layer == 'layer2':
        network.layer2.__setattr__(block, new_module)
        if block_int > 0:
            previous_bn = network.layer2[block_int - 1]
        else:
            previous_bn = network.layer1[-1]
        if block_int + 1 < len(network.layer2):
            next_bn = network.layer2[block_int+1]
        else:
            next_bn = network.layer3[0]
    elif layer == 'layer3':
        network.layer3.__setattr__(block, new_module)
        if block_int > 0:
            previous_bn = network.layer3[block_int - 1]
        else:
            previous_bn = network.layer2[-1]
        if block_int + 1 < len(network.layer3):
            next_bn = network.layer3[block_int+1]
        else:
            next_bn = network.layer4[0]
    elif layer == 'layer4':
        network.layer4.__setattr__(block, new_module)
        if block_int > 0:
            previous_bn = network.layer4[block_int - 1]
        else:
            previous_bn = network.layer3[-1]
        if block_int + 1 < len(network.layer4):
            next_bn = network.layer4[block_int+1]
        else:
            next_bn = None
            print("No next block found, as surgery performed on last block of last layer")
    else:
        print("Could not perform surgery")
        exit(1)
    if previous_bn is not None:
        update_params.extend(list(previous_bn.parameters()))
        for param in previous_bn.parameters():
            param.requires_grad = True
    if next_bn is not None:
        update_params.extend(list(next_bn.parameters()))
        for param in next_bn.parameters():
            param.requires_grad = True
    if not seperate_rational:
        return update_params
    else:
        return update_params, [new_module.numerator, new_module.denominator]


def identity_rational():
    from rational.torch import Rational
    rat = Rational()
    rat.numerator = torch.nn.Parameter(torch.tensor([0., 1., 0., 0., 0., 0.],
                                                    device=rat.device,
                                                    requires_grad=True))
    rat.denominator = torch.nn.Parameter(torch.tensor([0., 0., 0., 0.],
                                                      device=rat.device,
                                                      requires_grad=True))
    rat.init_approximation = "identity"
    return rat

def __new__repr__(self):
    requires_grad = False
    # We treat the extra repr like the sub-module, one item per line
    if all([elem.requires_grad for elem in list(self.parameters())]):
        requires_grad = True
    extra_lines = []
    extra_repr = self.extra_repr()
    # empty string will be split into list ['']
    if extra_repr:
        extra_lines = extra_repr.split('\n')
    child_lines = []
    for key, module in self._modules.items():
        mod_str = repr(module)
        mod_str = _addindent(mod_str, 2)
        child_lines.append('(' + key + '): ' + mod_str)
    lines = extra_lines + child_lines

    main_str = self._get_name() + '('
    if lines:
        # simple one-liner info, which most builtin Modules will use
        if len(extra_lines) == 1 and not child_lines:
            main_str += extra_lines[0]
        else:
            main_str += '\n  ' + '\n  '.join(lines) + '\n'

    main_str += ')'
    if requires_grad:
        return '\033[92m' + main_str + '\033[0m'
    else:
        return main_str

def replace_repr(module):
    if type(module) == torchvision.models.resnet.Bottleneck:
        module.__class__.__repr__ = __new__repr__
    if type(module) == Rational:
        module.__class__.__repr__ = __new__repr__

def augmented_print(network):
    network.apply(replace_repr)
    print(network)
    text = "\n\t* The green blocks are the one for whom requires_grad == True\n"
    print('\033[92m' + text + '\033[0m')


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


def retrain(train_loader, model, criterion, optimizers, schedulers, epoch, args, learning=True):
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


        if learning:
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


    return losses.avg, top1.avg, top5.avg


def revalidate(val_loader, model, criterion, epoch, args):
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

    return losses.avg, top1.avg, top5.avg
