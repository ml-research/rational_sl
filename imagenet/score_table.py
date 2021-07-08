import pickle
import numpy as np
import pandas as pd
import sys


rows = []
dataset = "imagenet"
for eval_met in ["train/accuracy@1", "validate/accuracy@1"]:
    header = ["Network"]
    row = [eval_met]
    for nn in ["mobilenet_v2", "resnet18"]: # vgg11
    # for nn in ["mobilenet_v2"]: # vgg11
        save_folder = f"scores_sl/{nn}_scores_{dataset}"
        for act in ["lrelu", "rn", "rrn"]:
        # for act in ["rrn"]:
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
