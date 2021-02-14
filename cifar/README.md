# Rational Networks on CIFAR10 and CIFAR100

This repository evaluate rational networks on CIFAR(10/100).
It's based on [kangliu pytorch-cifar repository](https://github.com/kuangliu/pytorch-cifar)

## Training
```
# To train a LeNet rational network (rn) on with seed 1 on cifar 10
python3 train.py --arch lenet -s 1 -nt rn --dataset cifar10

# To train a VGG8 recurrent rational network (rrn) with seed 3 on cifar 100 for 20 epochs
python3 train.py --arch vgg8 -s 3 -nt rrn --dataset cifar100 -e 20
```

## Populate histogram
```
# To populate the input distribution's histogram for a saved model (-sm):
python3 populate_hist.py -sm trained_networks/lenet_models_cifar100/models_lenet_rrn_0_xavier.pkl
```

## Plots
```
# For the recurrent rational function's profile with the distance between the function
python3 plot_populated.py -sm trained_networks/populated/lenet_models_cifar10/models_lenet_rrn_0_xavier.pkl

# For the successive rational functions' profiles with the distance between the function
python3 plot_populated.py -sm trained_networks/populated/lenet_models_cifar10/models_lenet_rn_0_xavier.pkl --print_dist

# For plot of the different network types on vgg8
python3 score_evolution.py --arch vgg8 --eval_meth test/accuracy@1
```

## Score Tables
```
# For the table showing train and test accuracies of different net types with different architectures
python3 scores_table.py

# For the table with train and test accuracies of vgg8 with mixture of rationals and recurrent rationals
python3 scores_table.py -m vgg8
```

## Additionally: Number of parameters
```
To get the number of params of vgg19 equipped with LeakyReLU (lrelu) network
python3 number_of_params.py --arch vgg19 -nt lrelu
```
