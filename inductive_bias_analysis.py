import os
import sys

from tqdm import tqdm
import wandb
import torch
from torch.utils.data import DataLoader
from config import ex
from data.util import get_dataset, IdxDataset
import torch.nn.functional as F
from module.util import get_model
from util import MultiDimAverageMeter


@ex.capture
def train(
        main_tag,
        dataset_tag,
        model_tag,
        data_dir,
        log_dir,
        random_seed,
        device,
        target_attr_idx,
        bias_attr_idx,
        num_epochs,
        main_valid_freq,
        main_batch_size,
        main_log_freq,
        main_optimizer_tag,
        main_learning_rate,
        main_weight_decay
):
    print('Beginning Stage 1')
    device = torch.device(device)

    train_dataset = get_dataset(
        dataset_tag + 'conflict60k',
        data_dir=data_dir,
        dataset_split="train",
        transform_split="train"
    )

    valid_dataset = get_dataset(
        dataset_tag,
        data_dir=data_dir,
        dataset_split="eval",
        transform_split="eval"
    )

    train_target_attr = train_dataset.attr[:, target_attr_idx]
    train_bias_attr = train_dataset.attr[:, bias_attr_idx]
    attr_dims = [torch.max(train_target_attr).item() + 1, torch.max(train_bias_attr).item() + 1]
    num_classes = attr_dims[0]

    train_dataset = IdxDataset(train_dataset)
    valid_dataset = IdxDataset(valid_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=main_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=main_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=False
    )

    # define model and optimizer
    model = get_model(model_tag, num_classes, stage='1').to(device)
    print(model)

    if main_optimizer_tag == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
            momentum=0.9,
        )
    elif main_optimizer_tag == "Adam":
        optimizer = torch.optim.Adam(model.linear_decodable_layer.parameters(),
                                     lr=main_learning_rate,
                                     weight_decay=main_weight_decay)

    elif main_optimizer_tag == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )
    else:
        raise NotImplementedError

    label_criterion = torch.nn.CrossEntropyLoss(reduction="none")

    save_path = os.path.join(log_dir, dataset_tag, 'stage1', str(random_seed))
    os.makedirs(save_path, exist_ok=True)

    # # define evaluation function
    def evaluate_shape(model, data_loader, debias_weight=1, bias_weight=1):
        model.eval()
        attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
        for _, data, attr in data_loader:
            label = attr[:, target_attr_idx]
            data = data.to(device)
            attr = attr.to(device)
            label = label.to(device)
            with torch.no_grad():
                logit = model(data, debias_weight=debias_weight, bias_weight=bias_weight)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()

            attr = attr[:, [target_attr_idx, bias_attr_idx]]

            attrwise_acc_meter.add(correct.cpu(), attr.cpu())
        eye_tsr = torch.eye(num_classes)
        accs = attrwise_acc_meter.get_mean()
        accs_aligned = accs[eye_tsr > 0.0].mean().item()
        accs_conflict = accs[eye_tsr == 0.0].mean().item()
        accs = torch.mean(accs).item()
        return accs, accs_aligned, accs_conflict

    def evaluate_color(model, data_loader, debias_weight=1, bias_weight=1):
        model.eval()
        attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
        for _, data, attr in data_loader:
            label = attr[:, bias_attr_idx]
            data = data.to(device)
            attr = attr.to(device)
            label = label.to(device)
            with torch.no_grad():
                logit = model(data, debias_weight=debias_weight, bias_weight=bias_weight)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()

            attr = attr[:, [target_attr_idx, bias_attr_idx]]

            attrwise_acc_meter.add(correct.cpu(), attr.cpu())
        eye_tsr = torch.eye(num_classes)
        accs = attrwise_acc_meter.get_mean()
        accs_aligned = accs[eye_tsr > 0.0].mean().item()
        accs_conflict = accs[eye_tsr == 0.0].mean().item()
        accs = torch.mean(accs).item()
        return accs, accs_aligned, accs_conflict

    valid_conflict_best = 0

    # set all other train/ metrics to use this step
    wandb.define_metric("acc-debiased-branch/*", step_metric="epoch")
    wandb.define_metric("acc-biased-branch/*", step_metric="epoch")

    for i in range(10):
        linear_decodable_dict = {'debias': (0, 1), 'bias': (1, 0)}
        for name, weight in linear_decodable_dict.items():
            for idx in range(2):
                bias_weight, debias_weight = weight
                model.load_state_dict(torch.load('results/cmnist/ColoredMNIST-Skewed0.01-Severity4/stage1/5/' + str(i)
                                                 + 'debiased_model_stage1.th'), strict=False)
                optimizer = torch.optim.Adam(model.linear_decodable_layer.parameters(), lr=main_learning_rate,
                                             weight_decay=main_weight_decay)
                for epoch in range(num_epochs):
                    model.train()

                    for _, data, attr in train_loader:
                        data = data.to(device)
                        attr = attr.to(device)
                        label = attr[:, idx]

                        logit = model(data, bias_weight=bias_weight, debias_weight=debias_weight)
                        loss_per_sample = label_criterion(logit.squeeze(1), label)
                        loss = loss_per_sample.mean()

                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                if idx == 0:
                    acc_shape, _, _ = evaluate_shape(model, valid_loader, debias_weight=debias_weight,
                                                     bias_weight=bias_weight)
                    print('idx no : ' + str(i) + ' name : ' + str(name) + ' shape_acc : ' + str(acc_shape))
                else:
                    acc_color, _, _ = evaluate_color(model, valid_loader, debias_weight=debias_weight,
                                                     bias_weight=bias_weight)
                    print('idx no : ' + str(i) + ' name : ' + str(name) + ' color_acc : ' + str(acc_shape))


