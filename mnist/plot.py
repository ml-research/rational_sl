import torch
import numpy as np

import pickle
import argparse
import torch.nn.functional as F
import matplotlib
import os
from rational.torch import Rational, RecurrentRationalModule
from torchvision import datasets, transforms
from mnist import VGG, LeNet5


torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_anomaly_enabled(True)

font = {'family': 'normal',
        'weight': 'bold',
        'size': 22}

matplotlib.rc('font', **font)


def test(args, model, device, test_loader, epoch):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    acc = 100. * correct / len(test_loader.dataset)

    print('\nTest set: Epoch: {}, Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(epoch, test_loss,
                                                                                            correct,
                                                                                            len(test_loader.dataset),
                                                                                            acc))
    return acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=17, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--dataset', type=str, default='mnist',
                        help='dataset to use')
    parser.add_argument('--arch', type=str, required=True)
    parser.add_argument('--init', type=str, default="", choices=["", "xavier", "he"])
    args = parser.parse_args()

    networks = dict({
        "vgg": VGG,
        "lenet": LeNet5,
    })

    network = networks[args.arch]
    # activation_function_keys = [x for x in list(actfvs.keys()) if 'pau' in x]
    # activation_function_keys = ['pau']
    # activation_function_keys = ['recurrent_pau']
    activation_function_keys = ['pau', 'recurrent_pau']
    optimizer = 'sgd'
    epochs = ['final']
    for activation_function_key in activation_function_keys:
        for epoch in epochs:
            print("---" * 42)
            print("Starting with dataset: {}, activation function: {}".format(args.dataset, activation_function_key))
            print("---" * 42)
            load_path = 'examples/runs/mnist/paper_{}_{}_{}{}_seed{}/'.format(args.dataset, args.arch, optimizer,
                                                                              "_init_{}".format(args.init) if args.init != "" else "",
                                                                 args.seed) + activation_function_key
            use_cuda = not args.no_cuda and torch.cuda.is_available()
            torch.manual_seed(args.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(args.seed)

            device = torch.device("cuda" if use_cuda else "cpu")

            kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
            if args.dataset == 'mnist':
                test_loader = torch.utils.data.DataLoader(
                    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                    batch_size=args.batch_size, shuffle=True, **kwargs)
                lr_scheduler_milestones = [30, 60, 90]  # Simple CNN with 3 Conv
                # lr_scheduler_milestones = [40, 80]  # VGG
            elif args.dataset == 'fmnist':
                test_loader = torch.utils.data.DataLoader(
                    datasets.FashionMNIST('../data', train=False, transform=transforms.Compose([
                        transforms.Resize((32, 32)),
                        transforms.ToTensor(),
                        transforms.Normalize((0.1307,), (0.3081,))
                    ])),
                    batch_size=args.batch_size, shuffle=True, **kwargs)
                lr_scheduler_milestones = [40, 80]
            else:
                raise ValueError('dataset error')

            model = network(activation_func=activation_function_key).to(device)
            model.load_state_dict(torch.load(os.path.join(load_path, 'model_{}.pt'.format(epoch))))
            paus = list()

            for name, layer in model.named_modules():
                if isinstance(layer, Rational):
                    layer.input_retrieve_mode(max_saves=10)
                    paus.append(('rational', name, layer))
                if isinstance(layer, RecurrentRationalModule):
                    layer.input_retrieve_mode(max_saves=10)
                    paus.append(('recurrent_rational', name, layer))

            if len(paus) > 0:
                os.makedirs(os.path.join(load_path, 'plots'), exist_ok=True)
                # dict(model.named_parameters())["features.3.0.bias"][0]
                # dict(model.named_parameters())["features.4.2.numerator"][0]
                print("Starting model eval")
                acc = test(args, model, device, test_loader, epoch)
                print("Finished model eval -> Plot")
                # fig = plt.figure(1, figsize=(6*len(paus),6))
                fig_dicts = []
                for i, p in enumerate(paus):
                    fig = p[2].show(display=False)
                    print(fig)
                    fig_dicts.append(fig)
                pickle.dump(fig_dicts, open(f'{args.dataset}_{args.arch}_{activation_function_key}_(acc{acc}%).fig.pkl', "wb"))
            else:
                print("No Rational Activations found. Exit without plotting")


if __name__ == '__main__':
    main()
