import torch
import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, num_layers = 4, num_classes = 10):
        super(MLP, self).__init__()
        self.feature1 = nn.Linear(3 * 32 * 32, 100) 
        self.layers = nn.ModuleList()
        for _ in range(num_layers - 2):
            self.layers.append(nn.Linear(100, 100))
        self.classifier = nn.Linear(100, num_classes)
        self.act = nn.ReLU()

    def forward(self, x, return_feat=False):
        x = x.view(x.size(0), -1)
        feat = self.act(self.feature1(x))
        for layer in self.layers:
            feat = self.act(layer(feat))
        x = self.classifier(feat)
        if return_feat:
            return x, feat
        else:
            return x

# class MLP(nn.Module):
#     def __init__(self, num_classes = 10):
#         super(MLP, self).__init__()
#         self.feature = nn.Sequential(
#             nn.Linear(3 * 32*32, 100),
#             nn.ReLU(),
#             nn.Linear(100, 100),
#             nn.ReLU(),
#             nn.Linear(100, 100),
#             nn.ReLU()
#         )
#         self.classifier = nn.Linear(100, num_classes)

#     def forward(self, x, return_feat=False):
#         x = x.view(x.size(0), -1) / 255
#         feat = x = self.feature(x)
#         x = self.classifier(x)

#         if return_feat:
#             return x, feat
#         else:
#             return x

class Branch(nn.Module):
    def __init__(self, num_layers = 1, num_classes = 10):
        super(Branch, self).__init__()
        self.skip_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.skip_layers.append(nn.Linear(100, 100))
        self.act = nn.ReLU()
        self.classifier = nn.Linear(100, num_classes)

    def forward(self, x):
        for layer in self.skip_layers:
            x = self.act(layer(x))
        out = self.classifier(x)
        return out, x
    
class Skip(nn.Module):
    def __init__(self, num_layers = 1):
        super(Skip, self).__init__()
        self.skip_layers = nn.ModuleList()
        for _ in range(num_layers):
            self.skip_layers.append(nn.Linear(100, 100))
        self.act = nn.ReLU()
    
    def forward(self, x):
        for layer in self.skip_layers:
            x = self.act(layer(x))
        return x


class Product_Of_Experts(nn.Module):
    def __init__(self, skip_layers = 3, main_layers = 5, num_classes = 10):
        super(Product_Of_Experts, self).__init__()
        self.feature1 = nn.Linear(3 * 32*32, 100) 
        self.feature2 = nn.Linear(3 * 32*32, 100) 
        self.skip = Skip(num_layers=skip_layers-2)
        self.main = Skip(num_layers=main_layers-2)
        self.classifier = nn.Linear(100, num_classes)
        self.act = nn.ReLU()
        self.skip_weight = 1
        self.feature_weight = 1
 
    def forward(self, x, return_feat=False):
        x = x.view(x.size(0), -1) 
        feat = self.act(self.feature1(x))
        x_skip = self.skip(feat)

        feat = self.act(self.feature2(x))
        x_main = self.main(feat)

        feat = self.skip_weight * x_skip + self.feature_weight * x_main
        x = self.classifier(feat)
        if return_feat:
            return x, feat
        else:
            return x
