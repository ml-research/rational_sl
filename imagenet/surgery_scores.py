import pickle
import pandas as pd
import numpy as np
import sys

save_folder = 'surgery/scores'
original_scores = pickle.load(open(f'{save_folder}/original_resnet.pkl', 'rb'))


def metrics_from_filename(filename):
    if 'resnet_id' in filename: # no epochs -> only surgered
        scores = pickle.load(open(f'{save_folder}/{filename}', 'rb'))
        all_metrics = scores['train_acc1'], scores['test_acc1']
        return np.array(all_metrics)
    else:
        scores = pickle.load(open(f'{save_folder}/{filename}', 'rb'))
        all_metrics = max(scores['train_acc1']), max(scores['test_acc1'])
        return np.array(all_metrics)

def percent_recover(evaluated, original, surgered):
    return (evaluated - surgered)/(original - surgered) * 100


blocks = ["2.3", "3.13", "3.19", "4.2"]
indexes = pd.MultiIndex.from_product([['train', 'test'], ["Std. (Veit et al)", "rationals"]], names=["Eval", "Lesion"])
all_df = pd.DataFrame([], index=indexes)
for block in blocks:
    layer, block_n = block.split('.')
    ori_all_eval = original_scores['train_acc1'], original_scores['test_acc1']
    surg_all_eval = metrics_from_filename(f"surgered_resnet_id_layer{layer}_block{block_n}.pkl")
    rat_all_eval = metrics_from_filename(f"surgered_trained_rat_layer{layer}_block{block_n}.pkl")
    trid_all_eval = metrics_from_filename(f"surgered_trained_id_layer{layer}_block{block_n}.pkl")
    percent_tr_id = percent_recover(trid_all_eval, ori_all_eval, surg_all_eval)
    percent_tr_rat = percent_recover(rat_all_eval, ori_all_eval, surg_all_eval)
    data = np.array([percent_tr_id, percent_tr_rat]).T.ravel()
    df = pd.DataFrame(data, index=indexes, columns=[f"L{layer}.B{block_n}"])
    all_df = all_df.join(df)


# all_df = all_df.drop("train_loss")
print(all_df)
if '--save' in sys.argv:
    all_df.to_csv('surgery_compare_scores.csv', float_format='%.1f')

# indexes = pd.MultiIndex.from_product([['train_loss', 'train_acc1', 'test_acc1'], ["original", "successive_rational"]], names=["Eval", "Network"])
# all_df = pd.DataFrame([], index=indexes)
# successive_trained = ['3.14-3.17', '3.14-3.19', '3.20-3.12', '3.20-3.15', '3.20-3.16']
# for block in successive_trained:
#     firstop, secondop = block.split('-')
#     layer1, block_n1 = firstop.split('.')
#     layer2, block_n2 = secondop.split('.')
#     filename_rat = f"surgered_trained_rat_layer{layer1}_block{block_n1}_then_layer{layer2}_block{block_n2}.pkl"
#     rat_scores = pickle.load(open(f'{save_folder}/{filename_rat}', 'rb'))
#     ori_all_eval = original_scores['train_loss'], original_scores['train_acc1'], original_scores['test_acc1']
#     rat_all_eval = min(rat_scores['train_loss']), max(rat_scores['train_acc1']), max(rat_scores['test_acc1'])
#     data = np.array([ori_all_eval, rat_all_eval]).T.ravel()
#     df = pd.DataFrame(data, index=indexes, columns=[f"l{layer1}b{block_n1}->l{layer2}b{block_n2}"])
#     all_df = all_df.join(df)
#
# print(all_df)

# lrs = ['original', "identity", '0.001', '0.002', '0.005', '0.01']
# indexes = pd.MultiIndex.from_product([['train_loss', 'train_acc1', 'test_acc1'], lrs], names=["Eval", "lr"])
# all_df = pd.DataFrame([], index=indexes)
# rat_lr = ['0.001', '0.002', '0.005', '0.01']
# for block in ["2.3", "3.13", "3.20"]:
#     layer1, block_n1 = block.split('.')
#     ori_all_eval = original_scores['train_loss'], original_scores['train_acc1'], original_scores['test_acc1']
#     try:
#         filename_trid = f"surgered_trained_id_layer{layer1}_block{block_n1}.pkl"
#         id_metrics = metrics_from_filename(filename_trid)
#     except FileNotFoundError:
#         id_metrics = None, None, None
#     filename_rat = f"surgered_trained_rat_layer{layer1}_block{block_n1}.pkl"
#     rat_metrics = metrics_from_filename(filename_rat)
#     filename_rat002 = f"surgered_trained_rat_layer{layer1}_block{block_n1}_ratlr0.002.pkl"
#     rat002_metrics = metrics_from_filename(filename_rat002)
#     filename_rat005 = f"surgered_trained_rat_layer{layer1}_block{block_n1}_ratlr0.005.pkl"
#     rat005_metrics = metrics_from_filename(filename_rat005)
#     filename_rat01 = f"surgered_trained_rat_layer{layer1}_block{block_n1}_ratlr0.01.pkl"
#     rat01_metrics = metrics_from_filename(filename_rat01)
#     data = np.array([ori_all_eval, id_metrics, rat_all_eval, rat002_metrics, rat005_metrics, rat01_metrics]).T.ravel()
#     df = pd.DataFrame(data, index=indexes, columns=[f"l{layer1}b{block_n1}"])
#     all_df = all_df.join(df)
#
# print(all_df)
