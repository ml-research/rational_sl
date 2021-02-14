import pickle
import numpy as np
import pandas as pd
import sys

rows = []
datasets = ["mnist", "fmnist"]
architectures = ["lenet", "vgg8"]
net_types = ["lrelu", "rn", "rrn"]
for dataset in datasets:
    row = []
    for nn in architectures: # vgg11
        save_folder = f"scores_sl/{nn}_scores_{dataset}"
        for act in net_types:
            all_scores = []
            for seed in range(5):
                filename = f"scores_{nn}_{act}_{seed}.pkl"
                sc_seed = pickle.load(open(f"{save_folder}/{filename}", "rb"))
                all_scores.append(sc_seed[1])
            all_scores = np.array(all_scores)
            mean = np.mean(all_scores, 0)[-1]
            std = np.std(all_scores, 0)[-1]
            prec = 2
            row.append(f"{np.round(mean, prec)} ± {np.round(std, prec)}")
    rows.append(row)
header = pd.MultiIndex.from_product([architectures, net_types],
                                    names=["architecture", "net_type"])
score_df = pd.DataFrame(rows, index=datasets, columns=header)
print(score_df)
if "--store" in sys.argv:
    store_path = f"scores_tables/scores_table_fmnist.csv"
    score_df.to_csv(store_path)
    print(f"stored in {store_path}")
