import pickle
import argparse
from rational.torch import Rational, RecurrentRational
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--saved_model', '-sm', action='store', required=True,
                    help='path to the populated model to be loaded (pickle format)')
parser.add_argument('--score', '-sc', action='store_true',
                    help='if provided, the score of this seed is added to the title')
parser.add_argument('--subselect', action='store_true',
                    help='if provided, select a subpart')
parser.add_argument('--print_dist', action='store_true',
                    help='print the distance between the functions')
args = parser.parse_args()


def decompose_model_path(model_path):
    assert 'rn' in model_path
    assert 'populated' in model_path
    filename =  model_path.split("/")[-1]
    subdir = model_path.split("/")[-2]
    dataset = subdir.split('_')[-1]
    if "_bn" in filename:
        bn = "(bn)"
        filename = filename.replace('_bn', '')
    else:
        bn = ""
    if len(filename.split('_')) == 5:
        _, arch, nt, seed, _ = filename.split('_')
    elif len(filename.split('_')) == 4:
        _, arch, nt, seed = filename.split('_')
    else:
        print("unknown filename type")
        exit(1)
    return arch, nt, dataset, seed, bn


arch, nt, dataset, seed, bn = decompose_model_path(args.saved_model)

net = pickle.load(open(args.saved_model, "rb"))
rationals = []
for el in net.modules():
    if type(el) is RecurrentRational or (type(el) is Rational and 'rrn' in args.saved_model):
        rationals.append(el)
        break
    elif type(el) is Rational:
        rationals.append(el)

if args.print_dist:
    def neural_dist(func_a, func_b, x_range):
        """
        In the rational module, the fit function gives the distance between the two functions
        """
        func_a = func_a.numpy()
        func_b = func_b.numpy()
        return func_a.fit(func_b, x_range=x_range)[1]
    assert len(rationals) == 4
    import numpy as np
    import torch
    inp = np.arange(-2, 2, 0.1)
    first, second, third, fourth = rationals
    print("neural_dist")
    print(f"First - Second: {neural_dist(first, second, inp):.2f}")
    print(f"Second - Third: {neural_dist(third, second, inp):.2f}")
    print(f"Second - Fourth: {neural_dist(fourth, second, inp):.2f}")
    print(f"First - Third: {neural_dist(first, third, inp):.2f}")
    print(f"First - Fourth: {neural_dist(fourth, first, inp):.2f}")
    print(f"Third - Fourth: {neural_dist(third, fourth, inp):.2f}")

# print(net)
if len(bn) > 0:
    bn_suffix = "_bn"
else:
    bn_suffix = ""

if len(rationals) == 1:
    fig = plt.gcf()
    axes = [plt.gca()]
elif len(rationals) <= 5:
    fig, axes = plt.subplots(1, len(rationals), figsize=(2.5 * len(rationals), 2))
else:
    print("Not implemented for networks with more than 5 rats, please implement")
    exit(1)
# fig, axes = plt.subplots(1, len(rationals), figsize=(8, 1.5))

grey_color = (0.5, 0.5, 0.5, 0.25)
subrange = (-2.5, 2.5)

for rat, ax in zip(rationals, axes):
    plot_dict = rat.show(display=False)
    hist = plot_dict["hist"]
    line = plot_dict["line"]
    selection = (line['x'] > subrange[0]) & (line['x'] < subrange[1])
    if args.subselect:
        line['x'] = line['x'][selection]
        line['y'] = line['y'][selection]
        hist['freq'] = hist['freq'][selection]
        hist['bins'] = hist['bins'][selection]
    ax.plot(line['x'], line['y'])
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(axis='y', labelsize=8.5)
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))

    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.bar(hist["bins"], hist["freq"], width=hist["width"],
            color=grey_color, edgecolor=grey_color)


# plt.suptitle(f"{arch} on {dataset} {bn}")
plt.savefig(f"images/{arch}_{nt}_{dataset}{bn_suffix}.svg", format="svg")
# plt.savefig(f"images/{arch}_{nt}_{dataset}", format="svg")
plt.show()
