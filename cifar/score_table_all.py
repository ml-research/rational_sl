import pickle
import argparse
import numpy as np
import pandas as pd

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Plotting')
parser.add_argument('-bn', action='store_true')
parser.add_argument('-m','--mixed', action='store', choices=['lenet', 'vgg8', None], default=None)
args = parser.parse_args()

def format_text(m, sd):
    text = ""
    if str(m)[0] == '0':
        text += str(np.round(m, 2))[1:]
    else:
        text += str(np.round(m, 1))
    text += "±"
    if str(sd)[0] == '0':
        text += str(np.round(sd, 2))[1:]
    else:
        text += str(np.round(sd, 1))
    return text

eval_meth = 'test/accuracy@1'

if not args.bn and args.mixed is None:
    architectures = ["lenet", "vgg8", "vgg11", "vgg19"]
    net_types = ["lrelu", "rn", "rrn"]
    scores = dict(keys=net_types)
    rows = []
    for dataset in ["cifar10", "cifar100"]:
        row = []
        for arch in architectures:
            base_folder = f"scores_sl/{arch}_scores_{dataset}"
            for af in net_types:
                all_scores = []
                for seed in range(3):
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

    df = pd.DataFrame(rows, index=["cifar10", "cifar100"], columns=index)
    print(df)

if args.bn:
    architectures = ["lenet", "vgg8", "vgg11"]
    net_types = ["lrelu", "rn", "rrn"]
    scores = dict(keys=net_types)
    rows = []
    indexes = []
    for dataset in ["cifar10", "cifar100"]:
        for suffix in ['', '_bn']:
            row = []
            indexes.append(dataset + suffix)
            for arch in architectures:
                base_folder = f"scores_sl/{arch}_scores_{dataset}"
                for af in net_types:
                    all_scores = []
                    for seed in range(3):
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
# import ipdb; ipdb.set_trace()

if args.mixed is not None:
    architectures = [args.mixed]
    if args.mixed == "lenet":
        net_types = ["rn", "rrn", "r2rr", "rr2r", "rrr2"]
    elif args.mixed == "vgg8":
        net_types = ["rn", "rrn", "r2rrr", "rr2rr", "rrr2r", "rrrr2"]
    scores = dict(keys=net_types)
    rows = []
    indexes = []
    for dataset in ["cifar10", "cifar100"]:
        row = []
        indexes.append(dataset)
        for arch in architectures:
            base_folder = f"scores_sl/{arch}_scores_{dataset}"
            for af in net_types:
                all_scores = []
                for seed in range(3):
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
    df = pd.DataFrame(rows, index=indexes, columns=index)

    print(df)
