import os
import pickle
import torch
import torch.nn as nn
import torchvision
import argparse
from rtpt import RTPT
from utils import choose_parts, compute_number_of_exps, make_deter, perform_surgery,\
    identity_rational, make_loaders, augmented_print, revalidate, retrain
from time import sleep

blue, normal, yellow = '\033[94m', '\033[0m', '\033[93m'


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('-s', '--seed', default=0, type=int,
                    help='random seed to be fixed')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N',
                    help='mini-batch size (default: 128), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('-i', '--iter', default=100, type=int,
                    help='iter')
parser.add_argument('-j', '--num_workers', default=1, type=int,
                    help='iter')
parser.add_argument('--eval_original', default=False, action="store_true",
                    help='if true, evaluation of the original model')
parser.add_argument('--eval_id', default=False, action="store_true",
                    help='if true, replace a module with an Identity Layer and eval')
parser.add_argument('--use_id', default=False, action="store_true",
                    help='if true, replace a module with an Identity Layer and retrain')
parser.add_argument('--use_rat', default=False, action="store_true",
                    help='if true, replace a module with a Rational')
parser.add_argument('--rat_lr', type=float, default=None,
                    help='if true, use diferent lr for the rational than the layers')
parser.add_argument('--reuse_rat', action="store", default="",
                    help='path to saved agent')
parser.add_argument('-e', '--epochs', default=15, type=int,
                    help='number of epochs for retraining')
sp_help = 'For --surgered_part:\n' + \
          'either random or $l.$k with $l in [1:4] ' + \
          '(corresponding to the layer) and $k in [1:23]' + \
          '(For example: 2.3 for layer2 block3)'
parser.add_argument('--surgered_part', default=["random"], action="store",
                    type=str, help=sp_help)
parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')

args = parser.parse_args()
args.print_freq = 1
args.gpus = 0
args.clip_grad_value = 5
make_deter(args.seed)


train_loader, val_loader = make_loaders(args)

criterion = nn.CrossEntropyLoss().cuda()

if args.reuse_rat:
    print(f"{yellow}Reused pretrained surgered Net from:")
    print(f"{args.reuse_rat}{normal}")
    resnet = pickle.load(open(args.reuse_rat, "rb"))
else:
    print(f"{yellow}Loaded original trained resnet from Pytorch{normal}")
    resnet = torchvision.models.resnet101(pretrained=True).cuda()


model_save_folder = "surgery/models/"
score_save_folder = "surgery/scores/"
for fold in [model_save_folder, score_save_folder]:
    if not os.path.exists(fold):
        os.makedirs(fold)


total_exps = compute_number_of_exps(args)
rtpt = RTPT(name_initials='QD', experiment_name=f'ResNetSurgery_s{args.seed}',
            max_iterations=total_exps)
rtpt.start()

if args.eval_original:
    epoch = 0
    with torch.no_grad():
        train_loss, train_acc1, train_acc5 = retrain(train_loader, resnet, criterion, None, None, epoch, args, False)
        test_loss, test_acc1, test_acc5 = revalidate(val_loader, resnet, criterion, epoch, args)
    print("Original Resnet:")
    print(f"Train : Loss: {train_loss} | acc@1: {train_acc1} | acc5: {train_acc5}")
    print(f"Test  : Loss: { test_loss} | acc@1: { test_acc1} | acc5: { test_acc5}")
    filename = "original_resnet"
    score_dict = {"train_loss": train_loss, "test_loss": test_loss,
                  "train_acc1": train_acc1, "test_acc1": test_acc1,
                  "train_acc5": train_acc5, "test_acc5": test_acc5}
    pickle.dump(score_dict, open(os.path.join(score_save_folder, filename) + ".pkl", 'wb'))
    rtpt.step()

parts = choose_parts(args, sp_help)

if args.eval_id:
    id_module = nn.Identity()
    for layer, block_n in parts:
        perform_surgery(resnet, layer, block_n, id_module)
        print(f"Identity Surgery performed on {layer} block n {block_n}")

    epoch = 0
    with torch.no_grad():
        train_loss, train_acc1, train_acc5 = retrain(train_loader, resnet, criterion, None, None, epoch, args, False)
        test_loss, test_acc1, test_acc5 = revalidate(val_loader, resnet, criterion, epoch, args)

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
    rtpt.step()


if args.use_id:
    id_module = nn.Identity()
    for param in resnet.parameters():
        param.requires_grad_(False)
    params_to_update = []
    filename = f"surgered_trained_id"
    for layer, block_n in parts:
        params_to_update.extend(perform_surgery(resnet, layer, block_n, id_module))
        print(f"{blue} Rational Surgery performed on {layer} block n {block_n} {normal}")
        filename += f'_{layer}_block{block_n}'
    sleep(3)
    optimizers = [torch.optim.SGD(params_to_update, lr=args.lr, momentum=0.9)]
    train_losses, train_accs1, train_accs5 = list(), list(), list()
    test_losses, test_accs1, test_accs5 = list(), list(), list()
    augmented_print(resnet)
    for epoch in range(args.epochs):
        train_loss, train_acc1, train_acc5 = retrain(train_loader, resnet, criterion, optimizers, None, epoch, args)
        test_loss, test_acc1, test_acc5 = revalidate(val_loader, resnet, criterion, epoch, args)
        train_losses.append(train_loss); train_accs1.append(train_acc1); train_accs5.append(train_acc5)
        test_losses.append(test_loss); test_accs1.append(test_acc1); test_accs5.append(test_acc5)
        print(f"Epoch {epoch} Modified Resnet: (Rational)")
        print(f"Train : Loss: {train_loss} | acc@1: {train_acc1} | acc5: {train_acc5}")
        print(f"Test  : Loss: { test_loss} | acc@1: { test_acc1} | acc5: { test_acc5}")
        rtpt.step()

    score_dict = {"layer": layer, "block_n": block_n,
                  "train_loss": train_losses, "test_loss": test_losses,
                  "train_acc1": train_accs1, "test_acc1": test_accs1,
                  "train_acc5": train_accs5, "test_acc5": test_accs5}
    pickle.dump(score_dict, open(os.path.join(score_save_folder, filename) + ".pkl", 'wb'))
    pickle.dump(resnet, open(os.path.join(model_save_folder, filename) + ".pkl", 'wb'))
    print(f"Saved score in {os.path.join(score_save_folder, filename)}")
    print(f"Saved model in {os.path.join(model_save_folder, filename)}")


if args.use_rat:
    rat_module = identity_rational()
    for param in resnet.parameters():
        param.requires_grad_(False)
    filename = f"surgered_trained_rat"
    F_layers, rationals = [], []
    params_to_update = []
    for layer, block_n in parts:
        if args.rat_lr is not None:
            print(f"{blue} Rational Surgery performed on {layer} block n {block_n}{normal}")
            F_layers_around, rational = perform_surgery(resnet, layer, block_n, rat_module, True)
            F_layers.extend(F_layers_around)
            rationals.extend(rational)
        else:
            print(f"{blue} Rational Surgery performed on {layer} block n {block_n} {normal}")
            params_to_update.extend(perform_surgery(resnet, layer, block_n, rat_module))
        filename += f'_{layer}_block{block_n}'
    if args.rat_lr is not None:
        print(f'{blue}\tDifferent lrs for rat and blocks{normal}')
        optimizers = [torch.optim.SGD(F_layers, lr=args.lr, momentum=0.9),
                      torch.optim.SGD(rationals, lr=args.rat_lr, momentum=0.9)]
        filename += f'_ratlr{args.rat_lr}'
    else:
        optimizers = [torch.optim.SGD(params_to_update, lr=args.lr, momentum=0.9)]
    sleep(3)
    train_losses, train_accs1, train_accs5 = list(), list(), list()
    test_losses, test_accs1, test_accs5 = list(), list(), list()
    augmented_print(resnet)
    for epoch in range(args.epochs):
        train_loss, train_acc1, train_acc5 = retrain(train_loader, resnet, criterion, optimizers, None, epoch, args)
        test_loss, test_acc1, test_acc5 = revalidate(val_loader, resnet, criterion, epoch, args)
        train_losses.append(train_loss); train_accs1.append(train_acc1); train_accs5.append(train_acc5)
        test_losses.append(test_loss); test_accs1.append(test_acc1); test_accs5.append(test_acc5)
        print(f"Epoch {epoch} Modified Resnet: (Rational)")
        print(f"Train : Loss: {train_loss} | acc@1: {train_acc1} | acc5: {train_acc5}")
        print(f"Test  : Loss: { test_loss} | acc@1: { test_acc1} | acc5: { test_acc5}")
        rtpt.step()

    score_dict = {"layer": layer, "block_n": block_n,
                  "train_loss": train_losses, "test_loss": test_losses,
                  "train_acc1": train_accs1, "test_acc1": test_accs1,
                  "train_acc5": train_accs5, "test_acc5": test_accs5}
    pickle.dump(score_dict, open(os.path.join(score_save_folder, filename) + ".pkl", 'wb'))
    pickle.dump(resnet, open(os.path.join(model_save_folder, filename) + ".pkl", 'wb'))
    print(f"Saved score in {os.path.join(score_save_folder, filename)}")
    print(f"Saved model in {os.path.join(model_save_folder, filename)}")

if args.reuse_rat:
    if ',' in args.surgered_part:
        print(f"{yellow}Multi surgery not implemented for successive retraining{normal}")
        exit(1)
    rat_module = identity_rational()
    # rat_module.input_retrieve_mode(auto_stop=False, bin_width=0.01)
    # new_module = Rational()
    for param in resnet.parameters():
        param.requires_grad_(False)
    params_to_update = perform_surgery(resnet, layer, block_n, rat_module)
    blue, normal = '\033[94m', '\033[0m'
    print(f"{blue} Rational Surgery performed on {layer} block n {block_n} {normal}")
    optimizers = [torch.optim.SGD(params_to_update, lr=0.001, momentum=0.9)]
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
    original_filaname = args.reuse_rat.split("/")[-1].replace('.pkl', '')
    filename = f"{original_filaname}_then_{layer}_block{block_n}"
    score_dict = {"layer": layer, "block_n": block_n,
                  "train_loss": train_losses, "test_loss": test_losses,
                  "train_acc1": train_accs1, "test_acc1": test_accs1,
                  "train_acc5": train_accs5, "test_acc5": test_accs5}
    pickle.dump(score_dict, open(os.path.join(score_save_folder, filename) + ".pkl", 'wb'))
    pickle.dump(resnet, open(os.path.join(model_save_folder, filename) + ".pkl", 'wb'))
    print(f"Saved score in {os.path.join(score_save_folder, filename)}")
    print(f"Saved model in {os.path.join(model_save_folder, filename)}")
