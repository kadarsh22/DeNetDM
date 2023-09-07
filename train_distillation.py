import os
import pickle
from tqdm import tqdm
import torch.nn as nn
from datetime import datetime
import wandb
import random
from module.mlp import Resnet_Skip_Model, Resnet_Product_Of_Experts
from module.resnet import resnet20
import numpy as np

import torch
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, WeightedRandomSampler
from torch.utils.data.dataset import Subset
from torchvision import transforms as T
import torch.nn.functional as F
import torchvision
import matplotlib.pyplot as plt
import sys

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

from config import ex
from data.util import get_dataset, IdxDataset, ZippedDataset
from module.util import get_model
from util import MultiDimAverageMeter

def set_seed(seed: int = 172) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")



@ex.automain
def train(
        main_tag,
        dataset_tag,
        model_tag,
        data_dir,
        random_seed,
        log_dir,
        device,
        target_attr_idx,
        bias_attr_idx,
        main_num_steps,
        main_valid_freq,
        main_batch_size,
        main_log_freq,
        main_optimizer_tag,
        main_learning_rate,
        main_weight_decay,
):
    wandb.login()
    seed = random_seed
    wandb.init(project="multibias-classifier-training", entity="causality-and-robustness-of-classifiers",
               sync_tensorboard=True)
    wandb.run.name = "KD"
    wandb.run.log_code(".")
    wandb.config.update({"dataset_tag": dataset_tag, "algorithm": 'vanilla'})
    artifact = wandb.Artifact(wandb.run.name, type='model')
    set_seed(seed=seed)

    device = torch.device(device)
    start_time = datetime.now()
    writer = SummaryWriter(os.path.join(log_dir, "summary", main_tag))

    train_dataset = get_dataset(
        dataset_tag,
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
    attr_dims = []
    attr_dims.append(torch.max(train_target_attr).item() + 1)
    attr_dims.append(torch.max(train_bias_attr).item() + 1)
    num_classes = attr_dims[0]

    train_dataset = IdxDataset(train_dataset)
    valid_dataset = IdxDataset(valid_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=main_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=256,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True
    )
    
    final_model_dir = os.path.join(log_dir, "best_model" + str(seed) + ".th")

    model_dir = os.path.join(log_dir, "best_model_skip" + str(seed) + ".th")
    teacher = Resnet_Skip_Model(num_classes=num_classes).to(device)
    teacher.load_state_dict(torch.load(model_dir)['state_dict'],strict=False)
    
    student_dir = os.path.join(log_dir, "best_model_distilled" + str(seed) + ".th")
    student = resnet20(num_classes=num_classes)
    student.to(device)
    
    
    if main_optimizer_tag == "SGD":
        optimizer = torch.optim.SGD(
            student.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
            momentum=0.9,
        )
    elif main_optimizer_tag == "Adam":
        optimizer = torch.optim.Adam(
            student.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )
    elif main_optimizer_tag == "AdamW":
        optimizer = torch.optim.AdamW(
            student.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )
    else:
        raise NotImplementedError

    ce_loss = nn.CrossEntropyLoss(reduction="none")

    # define evaluation function
    def evaluate(model, data_loader):
        model.eval()
        acc = 0
        attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
        for index, data, attr in tqdm(data_loader, leave=False):
            label = attr[:, target_attr_idx]
            data = data.to(device)
            attr = attr.to(device)
            label = label.to(device)
            with torch.no_grad():
                logit = model(data)
                pred = logit.data.max(1, keepdim=True)[1].squeeze(1)
                correct = (pred == label).long()

            attr = attr[:, [target_attr_idx, bias_attr_idx]]

            attrwise_acc_meter.add(correct.cpu(), attr.cpu())

        accs = attrwise_acc_meter.get_mean()

        model.train()

        return accs

    teacher.eval()
    student.train()

    valid_attrwise_accs_list = []
    skip_conflict_acc_list = [0]
    T = 2 
    for step in tqdm(range(main_num_steps)):
        try:
            index, data, attr = next(train_iter)
        except:
            train_iter = iter(train_loader)
            _ , data, attr = next(train_iter)

        data = data.to(device)
        attr = attr.to(device)

        with torch.no_grad():
                teacher_logits = teacher(data)
        
        student_logits = student(data)
        soft_targets = nn.functional.softmax(teacher_logits / T, dim=-1)
        soft_prob = nn.functional.log_softmax(student_logits / T, dim=-1)
      
        loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (T**2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % main_log_freq == 0:
            loss = loss.detach().cpu()
            writer.add_scalar("loss_distil/train", loss, step)

            # bias_attr = attr[:, bias_attr_idx]  # oracle
            # loss_per_sample = loss_per_sample.detach()
            # if (label == bias_attr).any().item():
            #     aligned_loss = loss_per_sample[label == bias_attr].mean()
            #     writer.add_scalar("loss/train_aligned", aligned_loss, step)

            # if (label != bias_attr).any().item():
            #     skewed_loss = loss_per_sample[label != bias_attr].mean()
            #     writer.add_scalar("loss/train_skewed", skewed_loss, step)

        if step % main_valid_freq == 0:

            valid_attrwise_accs = evaluate(student, valid_loader)
            valid_attrwise_accs_list.append(valid_attrwise_accs)
            valid_accs = torch.mean(valid_attrwise_accs)
            writer.add_scalar("acc_distil/valid", valid_accs, step)
            eye_tsr = torch.eye(num_classes)
            writer.add_scalar(
                "acc_distil/valid_aligned",
                valid_attrwise_accs[eye_tsr > 0.0].mean(),
                step
            )
            writer.add_scalar(
                "acc_distil/valid_skewed",
                valid_attrwise_accs[eye_tsr == 0.0].mean(),
                step
            )

            val_acc =  valid_attrwise_accs[eye_tsr == 0.0].mean()
            best_acc = max(skip_conflict_acc_list)
            skip_conflict_acc_list.append(val_acc)
            if best_acc < val_acc:
                state_dict = {
                    'steps': step,
                    'state_dict': student.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                with open(student_dir, "wb") as f:
                    torch.save(state_dict, f)
                wandb.log({
                "best_acc_distilled_model": val_acc,
                })

    model_b = Resnet_Product_Of_Experts(num_classes=num_classes)
    model_b.to(device)
    model_b.load_state_dict(torch.load(model_dir)['state_dict'],strict=True)
    model_b.feature_weight = 1
    model_b.skip_weight = 0

    model_d = resnet20(num_classes=num_classes)
    model_d.to(device)
    model_d.load_state_dict(torch.load(student_dir)['state_dict'], strict=True)

    model_b.eval()
    model_d.train()

    main_learning_rate = 1e-4
    optimizer = torch.optim.Adam(model_d.parameters(), lr=main_learning_rate, weight_decay=main_weight_decay)

    valid_attrwise_accs_list = []
    best_acc_list = [0]
    main_num_steps = 15000
    for step in tqdm(range(main_num_steps)):
        try:
            index, data, attr = next(train_iter)
        except:
            train_iter = iter(train_loader)
            _ , data, attr = next(train_iter)

        data = data.to(device)
        attr = attr.to(device)

        label = attr[:, target_attr_idx]

        with torch.no_grad():
            outputs = F.softmax(model_b(data),dim=1)
        _, predicted = torch.max(outputs.data, 1)
        loss_weight = (predicted != label).long().detach()

        logit = model_d(data)
        loss_per_sample = ce_loss(logit.squeeze(1), label)
        loss = (loss_per_sample*loss_weight).mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % main_log_freq == 0:
            loss = loss.detach().cpu()
            writer.add_scalar("loss/train", loss, step)

            # bias_attr = attr[:, bias_attr_idx]  # oracle
            # loss_per_sample = loss_per_sample.detach()
            # if (label == bias_attr).any().item():
            #     aligned_loss = loss_per_sample[label == bias_attr].mean()
            #     writer.add_scalar("loss/train_aligned", aligned_loss, step)

            # if (label != bias_attr).any().item():
            #     skewed_loss = loss_per_sample[label != bias_attr].mean()
            #     writer.add_scalar("loss/train_skewed", skewed_loss, step)

        if step % main_valid_freq == 0:

            valid_attrwise_accs = evaluate(model_d, valid_loader)
            valid_attrwise_accs_list.append(valid_attrwise_accs)
            valid_accs = torch.mean(valid_attrwise_accs)
            writer.add_scalar("acc/valid", valid_accs, step)
            eye_tsr = torch.eye(num_classes)
            writer.add_scalar(
                "acc/valid_aligned",
                valid_attrwise_accs[eye_tsr > 0.0].mean(),
                step
            )
            writer.add_scalar(
                "acc/valid_skewed",
                valid_attrwise_accs[eye_tsr == 0.0].mean(),
                step
            )

            val_acc =  valid_attrwise_accs[eye_tsr == 0.0].mean()
            best_acc = max(best_acc_list)
            best_acc_list.append(val_acc)
            if best_acc < val_acc:
                state_dict = {
                    'steps': step,
                    'state_dict': model_d.state_dict(),
                    'optimizer': optimizer.state_dict(),
                }
                with open(final_model_dir, "wb") as f:
                    torch.save(state_dict, f)
                wandb.log({
                "best_acc_final_model": val_acc,
                })
        

