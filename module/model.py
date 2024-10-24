import torch
import torch.nn as nn
from module.resnet import resnet20
from torchvision.models import resnet18, resnet34
from collections import OrderedDict


class MLPHiddenlayers(nn.Module):
    def __init__(self, num_layers=1):
        super(MLPHiddenlayers, self).__init__()
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.hidden_layers.append(nn.Linear(100, 100))
        self.act = nn.ReLU()

    def forward(self, x):
        for layer in self.hidden_layers:
            x = self.act(layer(x))
        return x


class CMNISTDeNetDMModel(nn.Module):
    def __init__(self, debias_hidden_layers=3, bias_hidden_layers=5, num_classes=10, stage='1'):
        super(CMNISTDeNetDMModel, self).__init__()
        if stage == '1':
            self.debias_branch = nn.Sequential(
                OrderedDict([('c1', nn.Linear(3 * 28 * 28, 100)),
                             ('r1', nn.ReLU()),
                             ('s1', MLPHiddenlayers(num_layers=debias_hidden_layers - 2))]))

        elif stage == '2':
            self.debias_branch = nn.Sequential(
                OrderedDict([('c2', nn.Linear(3 * 28 * 28, 100)),
                             ('r2', nn.ReLU()),
                             ('s2', MLPHiddenlayers(num_layers=bias_hidden_layers - 2))]))

        self.bias_branch = nn.Sequential(nn.Linear(3 * 28 * 28, 100),
                                         nn.ReLU(),
                                         MLPHiddenlayers(num_layers=bias_hidden_layers - 2)
                                         )
        self.classifier = nn.Linear(100, num_classes)

    def forward(self, x, debias_weight=1, bias_weight=1):
        x = x.view(x.size(0), -1)
        x_debias = self.debias_branch(x)
        x_bias = self.bias_branch(x)
        feat = debias_weight * x_debias + bias_weight * x_bias
        x = self.classifier(feat)
        return x


class CCIFARDeNetDMModel(nn.Module):
    def __init__(self, num_classes=10, stage='1'):
        super(CCIFARDeNetDMModel, self).__init__()
        self.bias_branch = resnet20(num_classes=num_classes)
        self.bias_branch.linear = nn.Identity()

        for params in self.bias_branch.linear.parameters():
            params.requires_grad = False
        if stage == '1':
            self.debias_branch = nn.Sequential(
                OrderedDict([('c1', nn.Conv2d(3, 32, kernel_size=(5, 5))),
                             ('b1', nn.BatchNorm2d(32)), ('r1', nn.ReLU()),
                             ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                             ('c2', nn.Conv2d(32, 64, kernel_size=(5, 5))),
                             ('b2', nn.BatchNorm2d(64)), ('r2', nn.ReLU()),
                             ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                             ('c3', nn.Conv2d(64, 64, kernel_size=(5, 5))),
                             ('b3', nn.BatchNorm2d(64)), ('r3', nn.ReLU()),
                             ('f1', nn.Flatten(start_dim=1))]))
        elif stage == '2':
            self.debias_backbone = resnet18(pretrained=True)
            self.debias_backbone.fc = nn.Identity()
            for params in self.debias_backbone.fc.parameters():
                params.requires_grad = False
            self.debias_branch = nn.Sequential(
                OrderedDict([('b1', self.debias_backbone),
                             ('l1', nn.Linear(512, 64))])
            )

        self.classifier = nn.Linear(64, num_classes)
        self.act = nn.ReLU()

    def forward(self, x, debias_weight=1, bias_weight=1):
        x_bias = self.bias_branch(x)
        x_debias = self.debias_branch(x)
        feat = debias_weight * x_debias + bias_weight * x_bias
        x = self.classifier(feat)
        return x


class BFFHQDeNetDModel(nn.Module):
    def __init__(self, num_classes=2, stage='1'):
        super(BFFHQDeNetDModel, self).__init__()
        self.bias_branch = resnet18(pretrained=False)
        self.bias_branch.fc = nn.Identity()
        for params in self.bias_branch.fc.parameters():
            params.requires_grad = False

        if stage == '1':
            self.debias_branch = nn.Sequential(
                OrderedDict([
                    ('c1', nn.Conv2d(3, 64, kernel_size=(7, 7))),
                    ('b1', nn.BatchNorm2d(64)), ('r1', nn.ReLU(inplace=True)),
                    ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ('c2', nn.Conv2d(64, 128, kernel_size=(3, 3))),
                    ('b2', nn.BatchNorm2d(128)), ('r2', nn.ReLU(inplace=True)),
                    ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ('c3', nn.Conv2d(128, 512, kernel_size=(3, 3))),
                    ('s3', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                    ('b3', nn.BatchNorm2d(512)), ('r3', nn.ReLU(inplace=True)),
                    ('c4', nn.Conv2d(512, 512, kernel_size=(3, 3))),
                    ('b4', nn.BatchNorm2d(512)),
                    ('r4', nn.ReLU(inplace=True)),
                    ('a1', nn.AdaptiveAvgPool2d((1, 1))),
                    ('f1', nn.Flatten(start_dim=1))]))

        elif stage == '2':
            self.debias_branch = resnet18(pretrained=True)
            self.debias_branch.fc = nn.Identity()
            for params in self.debias_branch.fc.parameters():
                params.requires_grad = False
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, debias_weight=1, bias_weight=1):
        x_bias = self.bias_branch(x)
        x_debias = self.debias_branch(x)
        feat = debias_weight * x_debias + bias_weight * x_bias
        x = self.classifier(feat)
        return x


class CelebADeNetDMModel(nn.Module):
    def __init__(self, num_classes=2):
        super(CelebADeNetDMModel, self).__init__()
        self.bias_branch = resnet18(pretrained=True)
        self.bias_branch.fc = nn.Identity()
        for params in self.bias_branch.fc.parameters():
            params.requires_grad = False

        # self.debias_branch = nn.Sequential(OrderedDict([('c1', nn.Conv2d(3, 64, kernel_size=(7, 7))),
        #                                                      ('b1', nn.BatchNorm2d(64)), ('r1', nn.ReLU(inplace=True)),
        #                                                      ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #                                                      ('c2', nn.Conv2d(64, 128, kernel_size=(3, 3))),
        #                                                      ('b2', nn.BatchNorm2d(128)), ('r2', nn.ReLU(inplace=True)),
        #                                                      ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #                                                      ('c3', nn.Conv2d(128, 512, kernel_size=(3, 3))),
        #                                                      ('s3', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
        #                                                      ('b3', nn.BatchNorm2d(512)), ('r3', nn.ReLU(inplace=True)),
        #                                                      ('c4', nn.Conv2d(512, 512, kernel_size=(3, 3))),
        #                                                      ('b4', nn.BatchNorm2d(512)),
        #                                                      ('r4', nn.ReLU(inplace=True))]))
        self.debias_branch = nn.Sequential(*list(resnet18(pretrained=False).children())[0:5])
        self.dim_transform = nn.Linear(64, 512)
        self.debias_branch = nn.Sequential(OrderedDict([('c1', nn.Conv2d(3, 64, kernel_size=(7, 7))),
                                                        ('b1', nn.BatchNorm2d(64)), ('r1', nn.ReLU(inplace=True)),
                                                        ('s1', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                                                        ('c2', nn.Conv2d(64, 128, kernel_size=(3, 3))),
                                                        ('b2', nn.BatchNorm2d(128)), ('r2', nn.ReLU(inplace=True)),
                                                        ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                                                        ('c3', nn.Conv2d(128, 512, kernel_size=(3, 3))),
                                                        ('s3', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
                                                        ('b3', nn.BatchNorm2d(512)), ('r3', nn.ReLU(inplace=True)),
                                                        ('c4', nn.Conv2d(512, 512, kernel_size=(3, 3))),
                                                        ('b4', nn.BatchNorm2d(512)),
                                                        ('r4', nn.ReLU(inplace=True))]))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x, debias_weight=1, bias_weight=1):
        x_bias = self.bias_branch(x)
        x_debias = self.avg_pool(self.debias_branch(x))
        x_debias = self.dim_transform(torch.flatten(x_debias, start_dim=1))
        feat = debias_weight * x_debias + bias_weight * x_bias
        x = self.classifier(feat)
        return x
