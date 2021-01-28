import pickle
from rational.torch import Rational
import matplotlib.pyplot as plt
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("-f", help="Pickle file filename of the Rational NN",
                    dest="filename")
parser.add_argument("--save", help="Save in an SVG_file",
                    action="store_true", dest="save", default=False)

args = parser.parse_args()
if "_rn_" not in args.filename:
    print("Please provide the file with \"_rn_\" inside, will try to " +
          "automatically both the rn and rrn files")


mod = pickle.load(open(args.filename, "rb"))
rats = []


for mo in mod.modules():
    if type(mo) is Rational:
        rats.append(mo)


rrn_filename = args.filename.split('_rn_')
rrn_filename = rrn_filename[0] + '_rrn_' + rrn_filename[1]
if rrn_filename in os.listdir():
    mod = pickle.load(open(rrn_filename, "rb"))
    for mo in mod.modules():
        if type(mo) is Rational:
            rats.append(mo)
            print("Found and Loaded Reccurent Rational Network")
            break

# import ipdb; ipdb.set_trace()


if "mobilenet" in args.filename:
    fig, axes = plt.subplots(6, 6, figsize=(20, 12))
elif "resnet18" in args.filename:
    fig, axes = plt.subplots(3, 6, figsize=(20, 6))

i = 1
for rat, ax in zip(rats, axes.flatten()):
    if not 'best_fitted_function' in dir(rat):
        rat.best_fitted_function = None
    import physt
    # import ipdb; ipdb.set_trace()
    from rational.utils.histograms_cupy import Histogram
    # if type(rat.distribution) is physt.histogram1d.Histogram1D:
    #     rat.distribution = Histogram()._from_physt(rat.distribution)
    hist_dict = rat.show(display=False)
    hist, line = hist_dict["hist"], hist_dict["line"]
    if i != len(rats):
        lab = f"Layer {i}"
    else:
        lab = f"Reccurent version"
    ax.plot(line['x'], line['y'], label=lab)
    ax2 = ax.twinx()
    ax2.set_yticks([])
    grey_color = (0.5, 0.5, 0.5, 0.25)
    ax2.bar(hist["bins"], hist["freq"], width=hist["width"],
            color=grey_color, edgecolor=grey_color)
    i += 1
    ax.legend()


if args.save:
    fig_filename = args.filename.split("populated")[0] + "all_functions.svg"
    plt.savefig(fig_filename, format="svg")
    print(f"Saved in {fig_filename}")
else:
    plt.show()
