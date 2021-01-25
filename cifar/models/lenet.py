'''LeNet in PyTorch.'''
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    def __init__(self, act_type, nb_outputs):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, nb_outputs)
        if type(act_type) == list:
            if not len(act_type) == 4:
                from pprint import pprint
                pprint(act_type)
                print("Need 4 activations in total for Lenet")
                exit(1)
            else:
                print("Creating a mixed Rational Version")
                self.act1, self.act2, self.act3, self.act4 = [act() for act in act_type]
        else:
            self.act1 = act_type()
            self.act2 = act_type()
            self.act3 = act_type()
            self.act4 = act_type()

    def forward(self, x):
        out = self.act1(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = self.act2(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.act3(self.fc1(out))
        out = self.act4(self.fc2(out))
        out = self.fc3(out)
        return out


class LeNet_bn(nn.Module):
    def __init__(self, act_type, nb_outputs):
        super(LeNet_bn, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.bn1 = nn.BatchNorm2d(6)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)
        self.fc1   = nn.Linear(16*5*5, 120)
        self.bn3 = nn.BatchNorm1d(120)
        self.fc2   = nn.Linear(120, 84)
        self.bn4 = nn.BatchNorm1d(84)
        self.fc3   = nn.Linear(84, nb_outputs)
        self.act1 = act_type()
        self.act2 = act_type()
        self.act3 = act_type()
        self.act4 = act_type()

    def forward(self, x):
        out = self.act1(self.bn1(self.conv1(x)))
        out = F.max_pool2d(out, 2)
        out = self.act2(self.bn2(self.conv2(out)))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.act3(self.bn3(self.fc1(out)))
        out = self.act4(self.bn4(self.fc2(out)))
        out = self.fc3(out)
        return out
