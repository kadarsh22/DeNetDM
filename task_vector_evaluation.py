import torch
import os
from util import eval_single_epoch, flatten_params, assign_weights, evaluate
from module.model import MultiCMNISTDeCAMModel
from torch.utils.data import DataLoader
from data.util import get_dataset

data_dir = os.path.join('../data/', 'cmnist')


def get_difference_model(gce_model, erm_model, net):
    w1 = flatten_params(gce_model)
    w2 = flatten_params(erm_model)
    net = assign_weights(net, w2 + 0.5* (w2 - w1)).cuda()
    return net


gce_model = MultiCMNISTDeCAMModel(debias_hidden_layers=5, num_classes=10)
print(gce_model)
erm_model = MultiCMNISTDeCAMModel(debias_hidden_layers=5, num_classes=10)
net = MultiCMNISTDeCAMModel(debias_hidden_layers=5, num_classes=10)

gce_model.load_state_dict(torch.load('results/cmnist/ColoredMNIST-Skewed0.01-Severity4/stage1/2/gce_model.th'))
erm_model.load_state_dict(torch.load('results/cmnist/ColoredMNIST-Skewed0.01-Severity4/stage1/2/erm_model.th'))
difference_model = get_difference_model(gce_model, erm_model, net)
torch.save(difference_model.state_dict(), 'pretrained_models/difference_model.th')

valid_dataset = get_dataset('ColoredMNIST-Skewed0.01-Severity4', data_dir=data_dir, dataset_split="eval",
                            transform_split="eval")
valid_loader = DataLoader(valid_dataset, batch_size=256, shuffle=True, num_workers=0, pin_memory=True,
                          drop_last=True)

gce_model.cuda()
accs, accs_aligned, accs_conflict = evaluate(gce_model, valid_loader, debias_weight=1, bias_weight=1)
print(accs)
print(accs_aligned)
print(accs_conflict)
acc = eval_single_epoch(gce_model, valid_loader)
print(acc)

erm_model.cuda()
accs, accs_aligned, accs_conflict = evaluate(erm_model, valid_loader, debias_weight=1, bias_weight=1)
print(accs)
print(accs_aligned)
print(accs_conflict)
acc = eval_single_epoch(erm_model, valid_loader)
print(acc)

accs, accs_aligned, accs_conflict = evaluate(difference_model, valid_loader, debias_weight=1, bias_weight=1)
print(accs)
print(accs_aligned)
print(accs_conflict)
acc = eval_single_epoch(difference_model, valid_loader)
print(acc)