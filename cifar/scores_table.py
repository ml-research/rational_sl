import pickle
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Plotting')
parser.add_argument('-bn', '--batch_norm', action='store_true')
parser.add_argument('--save', action='store_true')
parser.add_argument('-m','--mixed', action='store',
                    choices=['lenet', 'vgg8', None], default=None)
args = parser.parse_args()


def format_text(m, sd):
    text = ""
    if str(m)[0] == '0':
        text += str(np.round(m, 2))[1:]
    else:
        text += str(np.round(m, 1))
    text += "±"
    if str(sd)[0] == '0':
        text += str(np.round(sd, 1))[1:]
    else:
        text += str(np.round(sd, 0))
    return text

eval_meths = ['train/accuracy@1', 'test/accuracy@1']
datasets = ["cifar10", "cifar100"]

if not args.batch_norm and args.mixed is None:
    architectures = ["lenet", "vgg8"]
    net_types = ["lrelu", "rn", "rrn"]
    scores = dict(keys=net_types)
    rows = []
    for dataset in datasets:
        for eval_meth in eval_meths:
            row = []
            for arch in architectures:
                base_folder = f"scores_sl/{arch}_scores_{dataset}"
                for af in net_types:
                    all_scores = []
                    for seed in range(5):
                        filename = f"scores_{arch}_{af}_{seed}_xavier.pkl"
                        try:
                            all_scores.append(pickle.load(open(f"{base_folder}/{filename}", "rb"))[eval_meth][:60])
                        except FileNotFoundError:
                            continue
                    f_mean = np.mean(all_scores, 0)[-1]
                    f_std = np.std(all_scores, 0)[-1]
                    text = format_text(f_mean, f_std)
                    row.append(text)
            rows.append(row)

    col_ind = pd.MultiIndex.from_product([architectures, net_types],
                                       names=["architecture", "net_type"])
    row_ind = pd.MultiIndex.from_product([datasets, eval_meths],
                                       names=["dataset", "eval"])
    df = pd.DataFrame(rows, index=row_ind, columns=col_ind)
    if args.save:
        df.to_csv('cifar_all_nets.csv')
    print(df)

if args.batch_norm:
    architectures = ["lenet", "vgg8", "vgg11"]
    net_types = ["lrelu", "rn", "rrn"]
    scores = dict(keys=net_types)
    rows = []
    indexes = []
    eval_meth = "train/accuracy@1"
    for dataset in ["cifar10", "cifar100"]:
        for suffix in ['', '_bn']:
            row = []
            indexes.append(dataset + suffix)
            for arch in architectures:
                base_folder = f"scores_sl/{arch}_scores_{dataset}"
                for af in net_types:
                    all_scores = []
                    for seed in range(5):
                        filename = f"scores_{arch}_{af}_{seed}_xavier{suffix}.pkl"
                        try:
                            all_scores.append(pickle.load(open(f"{base_folder}/{filename}", "rb"))[eval_meth])
                        except FileNotFoundError:
                            continue
                    f_mean = np.mean(all_scores, 0)[-1]
                    f_std = np.std(all_scores, 0)[-1]
                    text = format_text(f_mean, f_std)
                    row.append(text)
            rows.append(row)

    index = pd.MultiIndex.from_product([architectures, net_types],
                                       names=["architecture", "net_type"])
    df = pd.DataFrame(rows, index=indexes, columns=index)

    print(df)
    exit()
# import ipdb; ipdb.set_trace()

if args.mixed is not None:
    rows = []
    indexes = []
    architectures = [args.mixed]
    if args.mixed == "lenet":
        net_types = ["rn", "rrn", "r2rr", "rr2r", "rrr2"]
    elif args.mixed == "vgg8":
        net_types = ["rn", "rrn", "r2rrr", "rr2rr", "rrr2r", "rrrr2",
                     "r2r2r", "rr2r2", "r2rr2",
                     "r3rr", "rr3r", "rrr3", "r3r2", "r2r3", "r4r", "rr4"]
    scores = dict(keys=net_types)
    for dataset in datasets:
        for eval_meth in eval_meths:
            row = []
            for arch in architectures:
                base_folder = f"scores_sl/{arch}_scores_{dataset}"
                for af in net_types:
                    all_scores = []
                    for seed in range(5):
                        filename = f"scores_{arch}_{af}_{seed}_xavier.pkl"
                        try:
                            all_scores.append(pickle.load(open(f"{base_folder}/{filename}", "rb"))[eval_meth])
                        except FileNotFoundError:
                            continue
                    f_mean = np.mean(all_scores, 0)[-1]
                    f_std = np.std(all_scores, 0)[-1]
                    text = format_text(f_mean, f_std)
                    row.append(text)
            rows.append(row)

    index = pd.MultiIndex.from_product([architectures, net_types],
                                       names=["architecture", "net_type"])
    row_ind = pd.MultiIndex.from_product([datasets, eval_meths],
                                       names=["dataset", "eval"])
    df = pd.DataFrame(rows, index=row_ind, columns=index)
    if args.save:
        df.to_csv(f'cifar_{args.mixed}_selected_r.csv')
        print(f"Saved in cifar_{args.mixed}_selected_r.csv")
    print(df)
