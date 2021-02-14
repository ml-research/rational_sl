# Rational Networks on ImageNet

## Normal training
This part describe normal training on imagenet, you need to get [the imagenet dataset](http://www.image-net.org/) on your own.
For the following examples, `/datasets/imagenet` is the path to imagenet dataset
```
# To train on architecture mobilenet_v2 with a rational network (rn) (the path to imagenet dataset need to be given) using 5 workers
python3 train_imagenet.py -a mobilenet_v2 --lr 0.1 --lr_rat 0.1  --save_path=experiments/paper_imagenet/mobilenet --selected_activation rn --gpu 0 /datasets/imagenet --epochs=100 -b 64 -j 5 --seed 0

# To populate the histogram after training, use the resume argument
python3 imagenet_populate_hist.py -a mobilenet_v2 --lr 0.1 --lr_rat 0.1 --save_path=experiments/paper_imagenet/mobilenet --selected_activation rn --gpu 0 /datasets/imagenet --epochs=100 -b 256 -j 1 --seed 0 --resume experiments/paper_imagenet/mobilenet/rn/checkpoint.pth.tar

# To plot the rational learned functions' profile provide the populated graph
python3 imagenet_graphs.py -f populated_nets/mobilenet_v2_rn_populated.pkl
```


## Lesioning (or surgery)
```
# To evaluate the original ResNet101 network
python3 resnet_surgery.py /datasets/imagenet/ --eval_original -b 32

# To perform lesioning on block 2.3 and evaluate the resulting network (without retraining)
python3 resnet_surgery.py /datasets/imagenet/ --eval_id --surgered_part '2.3' -b 32

# To perform lesioning on block 3.13 and retrain for 20 epochs (default 15)
python3 resnet_surgery.py /datasets/imagenet/ --use_id --surgered_part '3.13' -b 32 -e 20

# To perform lesioning on block 3.13 and replace with a `identity rational` function
python3 resnet_surgery.py /datasets/imagenet/ --use_rat --surgered_part '3.13' -b 32 -e 20

# To perform lesioning on blocks 3.13 and 3.19 and replace with a rational function
python3 resnet_surgery.py /datasets/imagenet/ --use_rat --surgered_part '3.13,3.19' -b 32 -e 20
```


## Score Tables
```
For the table of train and test accuracies of different trained architecures with different network types
python3 score_table.py

# For the table showing train and test accuracies of different lesioned layers
python3 surgery_scores.py

```

## Additionally: Number of parameters in specific block
```
To get the number of params (and total proportion) of the block no 2 of layer 4
python3 nb_params_in_block.py --surgered_part 4.2
```
