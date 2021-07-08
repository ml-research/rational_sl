# Rational Networks on MNIST and FMNIST


## Normal training
This repository evaluate rational networks on MNIST/FMNIST.

## Training
```
# To train a LeNet rational network (rn) on with seed 1 on mnist
python3 train.py --batch-size 128 --dataset mnist --arch lenet --optimizer sgd --seed 0

# To train a VGG8 recurrent rational network (rrn) with seed 3 on fmnist for 7 epochs, and save the trained model
python3 train.py --batch-size 128 --dataset fmnist --arch vgg8 --epochs 7 --optimizer sgd --seed 3 --save
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
python3 score_table.py -m vgg8
```
