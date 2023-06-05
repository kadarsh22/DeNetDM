import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_classes = 10):
        super(MLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3 * 28*28, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )
        self.classifier = nn.Linear(100, num_classes)

    def forward(self, x, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        feat = x = self.feature(x)
        x = self.classifier(x)

        if return_feat:
            return x, feat
        else:
            return x

class MLP_Skip(nn.Module):
    def __init__(self, num_layers = 4, num_classes = 10):
        super(MLP_Skip, self).__init__()
        self.feature1 = nn.Linear(3 * 28*28, 100) 
        self.skip = nn.Linear(100, 100)
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(100, 100))
        self.classifier = nn.Linear(100, num_classes)
        self.act = nn.ReLU()


    def forward(self, x, skip_weight=1, feat_weight=1, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        feat = self.act(self.feature1(x))
        skip_out = self.act(self.skip(feat))
        for layer in self.layers:
            feat = self.act(layer(feat))
        feat = skip_weight * skip_out + feat_weight * feat
        x = self.classifier(feat)
        if return_feat:
            return x, feat
        else:
            return x