import pickle
import matplotlib.pyplot as plt
import argparse
import numpy as np


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Plotting')
parser.add_argument('--arch', type=str, required=True)
args = parser.parse_args()

base_folder = f"scores_sl/{args.arch}_scores_cifar100"
net_types = ["lrelu", "rn", "rrn"]
eval_meth = 'train/loss'
scores = dict(keys=net_types)
for af in net_types:
    all_scores = []
    for seed in range(3):
        filename = f"scores_{args.arch}_{af}_{seed}_xavier.pkl"
        try:
            all_scores.append(pickle.load(open(f"{base_folder}/{filename}", "rb"))[eval_meth][10:])
        except FileNotFoundError:
            continue
    means = np.mean(all_scores, 0)
    stds = np.std(all_scores, 0)
    plt.plot(range(len(means)), means, label=af)
    plt.fill_between(range(len(means)), (means + stds), (means - stds), alpha=0.4)

plt.title(f"{args.arch}_{eval_meth}")
plt.legend()
# plt.savefig(f"{args.arch}_{eval_meth}.png".replace("/", ":"))
plt.show()
