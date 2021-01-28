import pickle
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import sys

fs = 12
grey_color = (0.5, 0.5, 0.5, 0.25)
plot_dict_list = pickle.load(open('mnist_vgg_pau_(acc99.04%).fig.pkl', 'rb'))
plot_dict_list.append(pickle.load(open('mnist_vgg_recurrent_pau_(acc99.53%).fig.pkl', 'rb'))[0])
fig, axes = plt.subplots(2, len(plot_dict_list), figsize=(12, 3.2))
plt.subplots_adjust(wspace=0.27, hspace=0.26, bottom=0.2)
axes[0][0].set_ylabel("mnist", fontsize=fs)
i = 1
for hist_dict, ax in zip(plot_dict_list, axes[0]):
    hist, line = hist_dict["hist"], hist_dict["line"]
    ax.plot(line['x'], line['y'])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(integer=True, nbins=3))
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.bar(hist["bins"], hist["freq"][:-1], width=hist["width"],
            color=grey_color, edgecolor=grey_color)
    i += 1

plot_dict_list = pickle.load(open('fmnist_vgg_pau_(acc92.34%).fig.pkl', 'rb'))
plot_dict_list.append(pickle.load(open('fmnist_vgg_recurrent_pau_(acc92.57%).fig.pkl', 'rb'))[0])
axes[1][0].set_ylabel("fmnist", position=(-10, 0.5), fontsize=fs)
i = 1
for hist_dict, ax in zip(plot_dict_list, axes[1]):
    hist, line = hist_dict["hist"], hist_dict["line"]
    if i != len(plot_dict_list):
        ax.set_xlabel(f"Layer {i}", fontsize=fs)
    else:
        ax.set_xlabel("Reccurent Version", fontsize=fs)
    ax.plot(line['x'], line['y'])
    ax2 = ax.twinx()
    ax2.set_yticks([])
    ax2.bar(hist["bins"], hist["freq"][:-1], width=hist["width"],
            color=grey_color, edgecolor=grey_color)
    i += 1

if "--save" in sys.argv:
    fig.savefig("vgg_on_f+mnist_new.svg", format="svg")
else:
    plt.show()
