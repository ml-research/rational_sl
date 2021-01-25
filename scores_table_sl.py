import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import sys

print("\n" + "-"* 60)
print(" " * 12 + "FMNIST" + " " * 12)
rows = []
for dataset in ["mnist", "fmnist"]:
    header = ["Network"]
    row = [dataset]
    # for nn in ["vgg11", "vgg19", "lenet"]: # vgg11
    for nn in ["lenet", "vgg8"]: # vgg11
        save_folder = f"scores_sl/{nn}_scores_{dataset}"
        for act in ["lrelu", "rn", "rrn"]:
            all_scores = []
            for seed in range(5):
                filename = f"scores_{nn}_{act}_{seed}.pkl"
                sc_seed = pickle.load(open(f"{save_folder}/{filename}", "rb"))
                all_scores.append(sc_seed[1])
            header.append(f'{nn}_{act}')
            all_scores = np.array(all_scores)
            mean = np.mean(all_scores, 0)[-1]
            std = np.std(all_scores, 0)[-1]
            # import ipdb; ipdb.set_trace()
            prec = 2
            row.append(f"{np.round(mean, prec)} ± {np.round(std, prec)}")
    rows.append(row)

score_df = pd.DataFrame(rows, columns=header)
print(score_df)
print("\n" + "-"* 60)
if "--store" in sys.argv:
    store_path = f"../scores_tables/scores_table_fmnist.csv"
    score_df.to_csv(store_path, index=False)
    print(f"stored in {store_path}")


print("\n" + "-"* 60)
print(" " * 25 + "CIFAR10" )
rows = []
# for dataset in ["mnist", "fmnist", "cifar10"]:
nets = ["vgg11", "vgg19", "lenet"]
acts = ["lrelu", "rn", "rrn"]
header = ["Network"] + acts
for dataset in ["cifar10"]:
    for nn in nets: # vgg11
        row = [nn]
    # for nn in ["vgg11", "lenet"]: # vgg11
        save_folder = f"scores_sl/{nn}_scores_{dataset}"
        for act in acts:
            all_scores = []
            for seed in range(5):
                filename = f"scores_{nn}_{act}_{seed}.pkl"
                sc_seed = pickle.load(open(f"{save_folder}/{filename}", "rb"))
                all_scores.append(sc_seed[1])
            all_scores = np.array(all_scores)
            mean = np.mean(all_scores, 0)[-1]
            std = np.std(all_scores, 0)[-1]
            # import ipdb; ipdb.set_trace()
            prec = 2
            row.append(f"{np.round(mean, prec)} ± {np.round(std, prec)}")
        rows.append(row)

score_df = pd.DataFrame(rows, columns=header)
print(score_df)


print("\n" + "-"* 60)
print(" " * 12 + "IMAGENET")

rows = []
dataset = "imagenet"
for eval_met in ["train/accuracy@1", "validate/accuracy@1"]:
    header = ["Network"]
    row = [eval_met]
    # for nn in ["vgg11", "vgg19", "lenet"]: # vgg11
    # for nn in ["mobilenet_v2"]: # vgg11
    # for nn in ["mobilenet_v2", "resnet18"]: # vgg11
    for nn in ["mobilenet_v2"]: # vgg11
        save_folder = f"scores_sl/{nn}_scores_{dataset}"
        # for act in ["lrelu", "rn", "rrn"]:
        for act in ["rrn"]:
            all_scores = []
            for seed in range(5):
                filename = f"scores_{dataset}_{nn}_{act}_{seed}.pkl"
                try:
                    sc_seed = pickle.load(open(f"{save_folder}/{filename}", "rb"))
                except FileNotFoundError:
                    print(filename)
                    continue
                all_scores.append(sc_seed[eval_met])
            header.append(f'{nn}_{act}')
            all_scores = np.array(all_scores)
            # import ipdb; ipdb.set_trace()
            mean = np.mean(all_scores, 0)[-1]
            std = np.std(all_scores, 0)[-1]
            prec = 2
            row.append(f"{np.round(mean, prec)} ± {np.round(std, prec)}")
    rows.append(row)

score_df = pd.DataFrame(rows, columns=header)
print(score_df)
if "--store" in sys.argv:
    store_path = f"../scores_tables/scores_table_imagenet.csv"
    score_df.to_csv(store_path, index=False)
    print(f"stored in {store_path}")
