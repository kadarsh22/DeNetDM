import os
import pickle
from tqdm import tqdm
from datetime import datetime
import wandb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, BatchSampler, WeightedRandomSampler
from torch.utils.data.dataset import Subset
from torchvision import transforms as T
import torchvision
import matplotlib.pyplot as plt

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

from config import ex
from data.util import get_dataset, IdxDataset, ZippedDataset
from module.loss import GeneralizedCELoss
from module.util import get_model
from util import MultiDimAverageMeter, EMA
import random
from loss import FocalLoss


def set_seed(seed: int = 42) -> None:
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
        log_dir,
        device,
        gamma,
        target_attr_idx,
        bias_attr_idx,
        main_num_steps,
        main_valid_freq,
        main_batch_size,
        main_optimizer_tag,
        main_learning_rate,
        main_weight_decay,
):
    print(dataset_tag)
    wandb.login()

    wandb.init(project="bias_mitigation_server", entity="causality-and-robustness-of-classifiers",
               sync_tensorboard=True)
    wandb.run.name = "ours_simul_gce_focal_" + str(dataset_tag) + "_" + 'gamma' + str(gamma)
    wandb.run.log_code(".")
    wandb.config.update({"dataset_tag": dataset_tag,
                         "gamma": gamma, "algorithm": "ours_simul_gce_focal"})
    artifact = wandb.Artifact(wandb.run.name, type='model')

    set_seed()
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

    # make loader
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

      # define model and optimizer
    model_b = get_model(model_tag, attr_dims[0]).to(device)
    model_d = get_model(model_tag, attr_dims[0]).to(device)
    if main_optimizer_tag == "SGD":
        optimizer_b = torch.optim.SGD(
            model_b.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
            momentum=0.9,
        )
        optimizer_d = torch.optim.SGD(
            model_d.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
            momentum=0.9,
        )
    elif main_optimizer_tag == "Adam":
        optimizer_b = torch.optim.Adam(
            model_b.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )
        optimizer_d = torch.optim.Adam(
            model_d.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )
    elif main_optimizer_tag == "AdamW":
        optimizer_b = torch.optim.AdamW(
            model_b.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )
        optimizer_d = torch.optim.AdamW(
            model_d.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )
    else:
        raise NotImplementedError

    # define loss
    criterion = nn.CrossEntropyLoss(reduction='none')
    debias_criterion = FocalLoss(gamma=gamma)
    bias_criterion = GeneralizedCELoss()

    sample_loss_ema_b = EMA(torch.LongTensor(train_target_attr), alpha=0.7)
    sample_loss_ema_d = EMA(torch.LongTensor(train_target_attr), alpha=0.7)

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

    def plot_confusion_matrix(model, valid_loader, device, model_name):
        correct = 0
        total = 0
        ground_truths = []
        predictions = []
        for index, data, attr in valid_loader:
            images = data.to(device)
            attr = attr.to(device)
            labels = attr[:, target_attr_idx]
            ground_truths.append(labels)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            predictions.append(predicted)
            total += labels.size(0)
            correct += (predicted.cpu() == labels.cpu()).sum()
        ground_truths = torch.stack(ground_truths).cpu().view(-1).numpy()
        predictions = torch.stack(predictions).cpu().view(-1).numpy()
        class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
        wandb.log({str(model_name) + "_conf_mat": wandb.plot.confusion_matrix(probs=None,
                                                           y_true=ground_truths, preds=predictions,
                                                           class_names=class_names)})

    def visualise_model_predictions(model, valid_loader, device ,model_name):
        data = [(images, torch.max(model(images.to(device)).data, 1)[1]) for index, images, attr in valid_loader]
        img_size = data[0][0].shape[-1]
        x = torch.stack([d[0] for d in data]).view(-1, 3, img_size, img_size)
        l = torch.stack([d[1] for d in data]).view(-1)
        images = []
        for i in range(10):
            if x[l.cpu() == i][:10].shape[0] == 10:
                images.append(x[l.cpu() == i][:10])
            else:
                images.append(torch.zeros((10, 3, img_size, img_size)))
        images = torch.stack(images).view(-1, 3, img_size, img_size)
        grid_img = torchvision.utils.make_grid(images[:100], nrow=10, normalize=False)
        plt.imshow(grid_img.permute(1, 2, 0).cpu().data)
        wandb.log({model_name + "_predictions": wandb.Image(grid_img)})

     # jointly training biased/de-biased model
    valid_attrwise_accs_list = []
    
    for step in tqdm(range(main_num_steps)):

        # train main model
        try:
            index, data, attr = next(train_iter)
        except:
            train_iter = iter(train_loader)
            index, data, attr = next(train_iter)

        data = data.to(device)
        attr = attr.to(device)
        label = attr[:, target_attr_idx]

        logit_b = model_b(data)
        if np.isnan(logit_b.mean().item()):
            print(logit_b)
            raise NameError('logit_b')
        logit_d = model_d(data)
 
        loss_per_sample_b = bias_criterion(logit_b, label)
        if np.isnan(loss_per_sample_b.mean().item()):
            raise NameError('loss_b_update')
        
        loss_per_sample_d = debias_criterion(logit_d.squeeze(1), label, logit_b)
        if np.isnan(loss_per_sample_d .mean().item()):
            raise NameError('loss_d_update')
        loss =  loss_per_sample_b.mean() + loss_per_sample_d.mean()

        optimizer_b.zero_grad()
        optimizer_d.zero_grad()
        loss.backward()
        optimizer_b.step()
        optimizer_d.step()

        main_log_freq = 10
        if step % main_log_freq == 0 and step != 0:
            loss = loss.detach().cpu()
            writer.add_scalar("loss/train", loss, step)

            bias_attr = attr[:, bias_attr_idx]  # oracle
            loss_per_sample_b = loss_per_sample_b.detach()
            loss_per_sample_d = loss_per_sample_d.detach()
            if (label == bias_attr).any().item():
                aligned_loss_b = loss_per_sample_b[label == bias_attr].mean()
                aligned_loss_d = loss_per_sample_d[label == bias_attr].mean()
                writer.add_scalar("loss/b_train_aligned", aligned_loss_b, step)
                writer.add_scalar("loss/d_train_aligned", aligned_loss_d, step)

            if (label != bias_attr).any().item():
                skewed_loss_b = loss_per_sample_b[label != bias_attr].mean()
                skewed_loss_d = loss_per_sample_d[label != bias_attr].mean()
                writer.add_scalar("loss/b_train_skewed", skewed_loss_b, step)
                writer.add_scalar("loss/d_train_skewed", skewed_loss_d, step)

        if step % main_valid_freq == 0:
            valid_attrwise_accs_b = evaluate(model_b, valid_loader)
            valid_attrwise_accs_d = evaluate(model_d, valid_loader)
            valid_attrwise_accs_list.append(valid_attrwise_accs_d)
            valid_accs_b = torch.mean(valid_attrwise_accs_b)
            writer.add_scalar("acc/b_valid", valid_accs_b, step)
            valid_accs_d = torch.mean(valid_attrwise_accs_d)
            writer.add_scalar("acc/d_valid", valid_accs_d, step)

            eye_tsr = torch.eye(attr_dims[0]).long()

            writer.add_scalar(
                "acc/b_valid_aligned",
                valid_attrwise_accs_b[eye_tsr == 1].mean(),
                step,
            )
            writer.add_scalar(
                "acc/b_valid_skewed",
                valid_attrwise_accs_b[eye_tsr == 0].mean(),
                step,
            )
            writer.add_scalar(
                "acc/d_valid_aligned",
                valid_attrwise_accs_d[eye_tsr == 1].mean(),
                step,
            )
            writer.add_scalar(
                "acc/d_valid_skewed",
                valid_attrwise_accs_d[eye_tsr == 0].mean(),
                step,
            )

    os.makedirs(os.path.join(log_dir, "result", main_tag), exist_ok=True)
    result_path = os.path.join(log_dir, "result", main_tag, "result.th")
    valid_attrwise_accs_list = torch.cat(valid_attrwise_accs_list)
    with open(result_path, "wb") as f:
        torch.save({"valid/attrwise_accs": valid_attrwise_accs_list}, f) 
        
    visualise_model_predictions(model_b, valid_loader, device, 'biased_model')
    plot_confusion_matrix(model_b, valid_loader, device, 'biased_model')
    visualise_model_predictions(model_d, valid_loader, device, "debiased_model")
    plot_confusion_matrix(model_d, valid_loader, device, "debiased_model")

    model_path = os.path.join(log_dir, "result", main_tag, "model.th")
    state_dict = {
        'steps': step,
        'state_dict': model_d.state_dict(),
        'state_dict_gce': model_b.state_dict(),
        'optimizer': optimizer_d.state_dict(),
    }
    with open(model_path, "wb") as f:
        torch.save(state_dict, f)
    artifact.add_file(model_path)
    wandb.run.log_artifact(artifact)

