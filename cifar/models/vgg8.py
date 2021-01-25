import torch
import torch.nn as nn
import torch.nn.functional


def vgg_block(num_convs, in_channels, num_channels, actv_function, batch_norm):
    layers = []
    for i in range(num_convs):
        layers += [nn.Conv2d(in_channels=in_channels, out_channels=num_channels, kernel_size=3, padding=1)]
        in_channels = num_channels
        if batch_norm:
            layers += [nn.BatchNorm2d(num_channels)]
    layers += [actv_function()]
    layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    return nn.Sequential(*layers)


class VGG8(nn.Module):
    def __init__(self, act_type, batch_norm, nb_outputs):
        super(VGG8, self).__init__()
        self.act_type = act_type
        self.conv_arch = ((1, 3, 64), (1, 64, 128), (2, 128, 256), (2, 256, 512), (2, 512, 512))
        self.batch_norm = batch_norm
        layers = []
        if type(act_type) == list:
            if not len(act_type) == 5:
                from pprint import pprint
                pprint(act_type)
                print("Need 5 activations in total for vgg8")
                exit(1)
            else:
                print("Creating a mixed Rational Version")
                for ((num_convs, in_channels, num_channels), act) in zip(self.conv_arch, self.act_type):
                    layers += [vgg_block(num_convs, in_channels, num_channels, act, batch_norm)]
        else:
            for (num_convs, in_channels, num_channels) in self.conv_arch:
                layers += [vgg_block(num_convs, in_channels, num_channels, self.act_type, batch_norm)]
        self.features = nn.Sequential(*layers)
        self.classifier = nn.Linear(512, nb_outputs)

    def forward(self, x):
        out = self.features(x)
        out = out.view(-1, 512)
        out = self.classifier(out)
        return out
