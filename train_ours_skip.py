import os
import pickle
from tqdm import tqdm
from datetime import datetime
import wandb
import random

import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

from module.loss import CosineSimilarityLoss
from config import ex
from data.util import get_dataset, IdxDataset, ZippedDataset
from module.util import get_model
from util import MultiDimAverageMeter
import torch.nn as nn
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
        target_attr_idx,
        bias_attr_idx,
        main_num_steps,
        main_valid_freq,
        main_batch_size,
        main_optimizer_tag,
        main_learning_rate,
        main_weight_decay,
):
    wandb.login()

    wandb.init(project="bias_mitigation_server", entity="causality-and-robustness-of-classifiers",
               sync_tensorboard=True)
    wandb.run.name = "vanilla_skip_focal_loss"
    wandb.run.log_code(".")
    wandb.config.update({"dataset_tag": dataset_tag, "algorithm": 'vanilla'})
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

    align_dataset = get_dataset(
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

    align_loader = DataLoader(
        align_dataset,
        batch_size=main_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True
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
    model = get_model(model_tag, num_classes).to(device)
    # model.load_state_dict(torch.load('pretrained_models/vanilla_skip_best/model.th')['state_dict'], strict=True)

    if main_optimizer_tag == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
            momentum=0.9,
        )
    elif main_optimizer_tag == "Adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )
    elif main_optimizer_tag == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )
    else:
        raise NotImplementedError

    label_criterion = torch.nn.CrossEntropyLoss(reduction="none")
    focal_criterion = FocalLoss(gamma=1)
    print(model)
    # define evaluation function
    def evaluate(model, data_loader, skip_weight=1, feat_weight=1):
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

    def plot_confusion_matrix(model, valid_loader, device, name):
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
        wandb.log({str(name): wandb.plot.confusion_matrix(probs=None,
                                                          y_true=ground_truths, preds=predictions,
                                                          class_names=class_names)})

    def visualise_training_data(loader, target_attr_idx):
        data = [(images, attr[:, target_attr_idx]) for index, images, attr in loader]
        img_size = data[0][0].shape[-1]
        x = torch.stack([d[0] for d in data]).view(-1, 3, img_size, img_size)
        l = torch.stack([d[1] for d in data]).view(-1)
        images = []
        for i in range(10):
            images.append(x[l == i][:10])
        images = torch.stack(images).view(-1, 3, img_size, img_size)
        grid_img = torchvision.utils.make_grid(images[:100].squeeze(1), nrow=10, normalize=True)
        wandb.log({"training_data": wandb.Image(grid_img)})

    def visualise_model_predictions(model, valid_loader, device, plot_name):
        if plot_name == "predictions-aligned_data-skip" or plot_name == "predictions-aligned_data-feature" or plot_name == "predictions-aligned_data":
            data = [(images, torch.max(model(images.to(device)).data, 1)[1]) for images, attr in valid_loader]
        else:
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
        wandb.log({plot_name: wandb.Image(grid_img)})

    # define extracting indices function
    def get_align_skew_indices(lookup_list, indices):
        '''
        lookup_list:
            A list of non-negative integer. 0 should indicate bias-align sample and otherwise(>0) indicate bias-skewed sample.
            Length of (lookup_list) should be the number of unique samples
        indices:
            True indices of sample to look up.
        '''
        pseudo_bias_label = lookup_list[indices]
        skewed_indices = (pseudo_bias_label != 0).nonzero().squeeze(1)
        aligned_indices = (pseudo_bias_label == 0).nonzero().squeeze(1)

        return aligned_indices, skewed_indices

    valid_attrwise_accs_list = []
    wandb.watch(model, log='all', log_freq=100)
    visualise_training_data(train_loader, target_attr_idx)


    for step in tqdm(range(main_num_steps)):
        try:
            index, data, attr = next(train_iter)
        except:
            train_iter = iter(train_loader)
            index, data, attr = next(train_iter)

        data = data.to(device)
        attr = attr.to(device)
        model.zero_grad()
        label = attr[:, target_attr_idx]

        ## get x_ref
        model.skip_weight = 0
        model.feature_weight = 1
        x_ref = model(data)

        model.skip_weight = 1
        model.feature_weight = 0
        logit = model(data)
        loss_per_sample = focal_criterion(logit.squeeze(1), label, x_ref.detach())
        loss = loss_per_sample.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ## get x_ref
        model.zero_grad()
        model.skip_weight = 1
        model.feature_weight = 0
        x_ref = model(data)

        model.skip_weight = 0
        model.feature_weight = 1
        logit = model(data)
        loss_per_sample = focal_criterion(logit.squeeze(1), label, x_ref.detach())
        loss = loss_per_sample.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        main_log_freq = 10
        if step % main_log_freq == 0:
            loss = loss.detach().cpu()
            writer.add_scalar("loss/train", loss, step)

            bias_attr = attr[:, bias_attr_idx]  # oracle
            loss_per_sample = loss_per_sample.detach()
            if (label == bias_attr).any().item():
                aligned_loss = loss_per_sample[label == bias_attr].mean()
                writer.add_scalar("loss/train_aligned", aligned_loss, step)

            if (label != bias_attr).any().item():
                skewed_loss = loss_per_sample[label != bias_attr].mean()
                writer.add_scalar("loss/train_skewed", skewed_loss, step)

        if step % main_valid_freq == 0:
            valid_attrwise_accs = evaluate(model, valid_loader)
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

    os.makedirs(os.path.join(log_dir, "result", main_tag), exist_ok=True)
    result_path = os.path.join(log_dir, "result", main_tag, "result.th")
    valid_attrwise_accs_list = torch.cat(valid_attrwise_accs_list)
    with open(result_path, "wb") as f:
        torch.save({"valid/attrwise_accs": valid_attrwise_accs_list}, f)

    #  model_analysis
    visualise_model_predictions(model, valid_loader, device, "predictions")
    visualise_model_predictions(model, align_loader, device, "predictions-aligned_data")
    plot_confusion_matrix(model, valid_loader, device, "overall")

    # individual component analysis
    model.skip_weight = 1
    model.feature_weight = 0
    visualise_model_predictions(model, valid_loader, device, "predictions-skip")
    visualise_model_predictions(model, align_loader, device, "predictions-aligned_data-skip")
    plot_confusion_matrix(model, valid_loader, device, "skip-only")
    valid_attrwise_accs = evaluate(model, valid_loader)
    valid_accs = torch.mean(valid_attrwise_accs)
    wandb.log({"acc/valid-skip": valid_accs.item()})
    wandb.log({"acc/valid_aligned-skip": valid_attrwise_accs[eye_tsr > 0.0].mean()})
    wandb.log({"acc/valid_skewed-skip": valid_attrwise_accs[eye_tsr == 0.0].mean()})

    # individual component analysis
    model.skip_weight = 0
    model.feature_weight = 1
    visualise_model_predictions(model, valid_loader, device, "predictions-feature")
    visualise_model_predictions(model, align_loader, device, "predictions-aligned_data-feature")
    plot_confusion_matrix(model, valid_loader, device, "feature_only")
    valid_attrwise_accs = evaluate(model, valid_loader)
    valid_accs = torch.mean(valid_attrwise_accs)
    wandb.log({"acc/valid-feature": valid_accs.item()})
    wandb.log({"acc/valid_aligned-feature": valid_attrwise_accs[eye_tsr > 0.0].mean()})
    wandb.log({"acc/valid_skewed-feature": valid_attrwise_accs[eye_tsr == 0.0].mean()})

    model_path = os.path.join(log_dir, "result", main_tag, "model.th")
    state_dict = {
        'steps': step,
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    with open(model_path, "wb") as f:
        torch.save(state_dict, f)
    artifact.add_file(model_path)
    wandb.run.log_artifact(artifact)
