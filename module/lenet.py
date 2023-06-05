import torch.nn as nn
from collections import OrderedDict


class LeNet4(nn.Module):
    """
    Adapted from https://github.com/activatedgeek/LeNet-5
    """
    def __init__(self, num_channels = 3, num_classes = 10):
        super().__init__()

        self.layer1 = nn.Conv2d(num_channels, 6, kernel_size=(5, 5))
        self.max_pool1 =  nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        # self.skip = nn.Conv2d(6, 120, kernel_size=(14,14))
        self.skip = nn.Linear(1176, 120)
        self.layer2 = nn.Conv2d(6, 16, kernel_size=(5, 5))
        self.max_pool2 =  nn.MaxPool2d(kernel_size=(2, 2), stride=2)
        self.layer3 = nn.Conv2d(16, 120, kernel_size=(5, 5))
        self.act = nn.ReLU()
        self.fc = nn.Linear(120, num_classes)

    def forward(self, img, skip_weight = 1, feat_weight=1):
        x = self.act(self.layer1(img))
        x = self.max_pool1(x)
        skip_out = self.skip(x.view(x.size(0),-1))
        # skip_out = skip_out.view(skip_out.size(0),-1)
        x = self.act(self.layer2(x))
        x = self.max_pool2(x)
        x = self.act(self.layer3(x))
        x = x.view(x.size(0), -1)
        out = self.fc(skip_weight * skip_out + feat_weight * x)
        return out