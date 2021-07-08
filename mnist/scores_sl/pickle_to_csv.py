import pickle
from termcolor import colored
import pandas as pd
import os
import numpy as np

def swap_columns(df):
    df.rename(columns={'test/loss':'train/loss',
                              'train/loss':'test/loss',
                              'test/accuracy@1':'train/accuracy@1',
                              'train/accuracy@1':'test/accuracy@1'}, inplace=True)

for (root, dirs, files) in os.walk('.', topdown=True):
    for file in files:
        filepath = os.path.join(root, file)
        if '.pkl' in filepath:
            import ipdb; ipdb.set_trace()
            if '.pkl' in filepath:
                with open(filepath, 'rb') as file:
                    scores = pickle.load(file)
                assert type(scores) is list
                scores_df = pd.DataFrame(scores)
                if np.mean(scores["test/loss"]) < np.mean(scores["train/loss"]):
                    if np.mean(scores["test/accuracy@1"]) < np.mean(scores["train/accuracy@1"]):
                        import ipdb; ipdb.set_trace() # should not swap
                    print(colored('Swapping test and train for consistency', 'yellow'))
                    swap_columns(scores_df)
                # else:
                    # import ipdb; ipdb.set_trace() # should not swap
                csv_path = filepath.replace('.pkl', '.csv')
                print(colored(f"{filepath} -> {csv_path}", 'green'), end="")
                scores_df.to_csv(csv_path)
                with open(filepath, 'wb') as file2:
                    pickle.dump(scores, file2)
                    print(colored("Corrected", "green"))
