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
    def __init__(self, num_classes = 10):
        super(MLP_Skip, self).__init__()
        self.feature1 = nn.Linear(3 * 28*28, 100)
        self.skip = nn.Linear(100,100) 
        self.feature2 = nn.Linear(100, 100)
        self.feature3 = nn.Linear(100 , 100)
        self.classifier = nn.Linear(100, num_classes)
        self.act = nn.ReLU()

    def forward(self, x, skip_weight=1, feat_weight=1, return_feat=False):
        x = x.view(x.size(0), -1) / 255
        feat = self.act(self.feature1(x))
        skip_out = self.act(self.skip(feat))
        feat2 = self.act(self.feature2(feat))
        feat3 = self.act(self.feature3(feat2))
        feat = skip_weight * skip_out + feat_weight * feat3
        # feat = torch.cat((skip_out, feat3), dim=1)
        x = self.classifier(feat)
        
        if return_feat:
            return x, feat
        else:
            return x