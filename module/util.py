import torch.nn as nn
from module.resnet import resnet20
from module.mlp import Product_Of_Experts
from module.lenet import LeNet4
from torchvision.models import resnet18, resnet50


def get_model(model_tag, num_classes, num_layers=6):
    print(model_tag)
    if model_tag == "ResNet20":
        return resnet20(num_classes)
    elif model_tag == "ResNet20Skip":
        return resnet20_skip(num_classes)
    elif model_tag == "ResNet18":
        model = resnet18(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(512, num_classes)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        model.skip.requires_grad = True
        return model
    elif model_tag == "ResNet50":
        model = resnet50(pretrained=True)
        for param in model.parameters():
            param.requires_grad = False
        model.fc = nn.Linear(2048, num_classes)
        model.fc.weight.requires_grad = True
        model.fc.bias.requires_grad = True
        return model
    elif model_tag == "MLP":
        return MLP(num_classes=num_classes)
    elif model_tag == "MLP_Skip":
        return MLP_Skip(num_layers=num_layers, num_classes=num_classes)
    elif model_tag == 'Lenet':
        return LeNet4(num_classes = num_classes)
    elif model_tag == 'Product_Of_Experts':
        return Product_Of_Experts(skip_layers=8, main_layers=8, num_classes=10)
    else:
        raise NotImplementedError
