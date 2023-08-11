import torch
import torch.nn as nn
from torchvision.models import resnet18

class Vanilla(nn.Module):
    def __init__(self, num_classes=10):
        super(Vanilla, self).__init__()
        self.resnet_feature_extractor = resnet18(pretrained=True)
        for param in self.resnet_feature_extractor.parameters():
            param.requires_grad = False
        self.resnet_feature_extractor.fc = nn.Linear(self.resnet_feature_extractor.fc.in_features, num_classes)
        

    def forward(self, x):
        x = self.resnet_feature_extractor(x)
        return x

class MLP(nn.Module):
    def __init__(self, num_classes=10, return_feat=False):
        super(MLP, self).__init__()
        self.feature = nn.Sequential(
            nn.Linear(3 * 32*32, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU()
        )
        self.classifier = nn.Linear(100, num_classes)
        self.return_feat = return_feat

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.feature(x)
        x = self.classifier(x)

        if self.return_feat:
            return x, feat
        else:
            return x


class Skip(nn.Module):
    def __init__(self, num_layers=1):
        super(Skip, self).__init__()
        self.skip_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.skip_layers.append(nn.Linear(512, 512))
        self.act = nn.ReLU()

    def forward(self, x):
        for layer in self.skip_layers:
            x = self.act(layer(x))
        return x

class Product_Of_Experts(nn.Module):
    def __init__(self, skip_layers=0, main_layers=5, num_classes=2):
        super(Product_Of_Experts, self).__init__()
        self.resnet_feature_extractor = resnet18(pretrained=True)
        self.resnet_feature_extractor.fc = nn.Identity()
        for param in self.resnet_feature_extractor.parameters():
            param.requires_grad = False
        self.feature2 = nn.Linear(512, 512)
        self.main = Skip(num_layers=main_layers - 2)
        self.classifier = nn.Linear(512, num_classes)
        self.act = nn.ReLU()
        self.skip_weight = 1
        self.feature_weight = 1

    def forward(self, x):
        resnet_features = self.resnet_feature_extractor(x)
        feat = self.act(self.feature2(resnet_features))
        x_main = self.main(feat)
        feat = self.skip_weight * resnet_features + self.feature_weight * x_main
        x = self.classifier(feat)
        return x