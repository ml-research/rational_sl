import os
import pickle


def reorder_keys(scores_dict):
    green, red = "\033[1;31;40m", "\033[1;32;40m"
    if scores_dict["train_loss"][-1] > 10:
        print(f"{green} Weird score, changing the keys")
        scores_dict["train_loss"], scores_dict["train_acc1"], scores_dict["train_acc5"] = scores_dict["train_acc5"], scores_dict["train_loss"], scores_dict["train_acc1"]
        scores_dict["test_loss"], scores_dict["test_acc1"], scores_dict["test_acc5"] = scores_dict["test_acc5"], scores_dict["test_loss"], scores_dict["test_acc1"]
    else:
        print(f"{red} Scores seem to be normal, no change performed")

for filename in os.listdir():
    if filename[-4:] == ".pkl" and 'rat' in filename:
        scores_d = pickle.load(open(filename, 'rb'))
        reorder_keys(scores_d)
        pickle.dump(scores_d, open(filename, 'wb'))
