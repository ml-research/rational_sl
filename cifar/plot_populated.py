import pickle
import argparse
from rational.torch import Rational, RecurrentRational
import matplotlib.pyplot as plt
import seaborn as sns

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--saved_model', '-sm', action='store', required=True,
                    help='path to the populated model to be loaded (pickle format)')
parser.add_argument('--score', '-sc', action='store_true',
                    help='if provided, the score of this seed is added to the title')
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
    if type(el) is Rational:
        rationals.append(el)

# print(net)
if len(bn) > 0:
    bn_suffix = "_bn"
else:
    bn_suffix = ""

if args.score:
    import pickle
    scores_path = f"scores_sl/{arch}_scores_{dataset}/scores_{arch}_{nt}_{seed}_xavier{bn_suffix}.pkl"
    print(scores_path)
    scores = pickle.load(open(scores_path, "rb"))

# if len(rationals) <= 5:
fig, axes = plt.subplots(1, len(rationals), figsize=(2.5 * len(rationals), 2.8))

grey_color = (0.5, 0.5, 0.5, 0.25)
for rat, ax in zip(rationals, axes):
    plot_dict = rat.show(display=False)
    hist = plot_dict["hist"]
    line = plot_dict["line"]
    ax.plot(line['x'], line['y'])
    # ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    # ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.bar(hist["bins"], hist["freq"], width=hist["width"],
            color=grey_color, edgecolor=grey_color)
    if args.score:
        plt.plot([], [], ' ', label=f"tr/l:{scores['train/loss'][-1]}")
        plt.plot([], [], ' ', label=f"te/l:{scores['test/loss'][-1]}")
plt.suptitle(f"{arch} on {dataset} {bn}")
plt.savefig(f"images/{arch}_{nt}_{dataset}{bn_suffix}.svg", format="svg")
plt.legend()
# plt.savefig(f"images/{arch}_{nt}_{dataset}", format="svg")
plt.show()
