import os
from tqdm import tqdm
from datetime import datetime
import wandb
import random
from data.bffhq import bFFHQDataset
from torchvision import transforms as T
from torchvision.models import resnet18
from module.mlp import MLP_Skip

import numpy as np

import torch
from torch.utils.data import DataLoader
import torchvision
import matplotlib.pyplot as plt

import warnings

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter

from config import ex
from data.util import get_dataset, IdxDataset, ZippedDataset
from module.util import get_model
from util import MultiDimAverageMeter


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

    wandb.init(project="multibias-classifier-training", entity="causality-and-robustness-of-classifiers",
               sync_tensorboard=True)
    wandb.run.name = "vanilla_bffhq_scratch"
    wandb.run.log_code(".")
    wandb.config.update({"dataset_tag": dataset_tag, "algorithm": 'vanilla'})
    artifact = wandb.Artifact(wandb.run.name, type='model')
    set_seed()

    device = torch.device(device)
    start_time = datetime.now()
    writer = SummaryWriter(os.path.join(log_dir, "summary", main_tag))

    train_transform = T.Compose([
        T.Resize((224, 224)),
        T.RandomCrop(224, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    )
    test_transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]
    )
    train_dataset = bFFHQDataset(
        '/home/prathosh/data/bffhq', 'train', transform=train_transform)

    valid_dataset = bFFHQDataset('/home/prathosh/data/bffhq/', 'valid', transform=test_transform)
    test_dataset = bFFHQDataset('/home/prathosh/data/bffhq', 'test', transform=test_transform)

    attr_dims = [2, 2]
    target_attr_idx = bFFHQDataset.target_attr_index
    bias_attr_idx = bFFHQDataset.bias_attr_index

    num_classes = 1
    attr_dims = attr_dims
    eye_tsr = torch.eye(attr_dims[0]).long()

    train_loader = DataLoader(train_dataset, batch_size=main_batch_size, shuffle=True)
    val_loader = DataLoader(valid_dataset, batch_size=main_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=main_batch_size, shuffle=False)

    # define model and optimizer
    model = resnet18(pretrained=False)
    model.fc = torch.nn.Linear(512, 1)
    
    model.to(device)
    if main_optimizer_tag == "SGD":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
            momentum=0.9,
        )
    elif main_optimizer_tag == "Adam":
        optimizer = torch.optim.Adam(
            model.fc.parameters(),
            lr=1e-4,
            weight_decay=0,
        )
    elif main_optimizer_tag == "AdamW":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=main_learning_rate,
            weight_decay=main_weight_decay,
        )
    else:
        raise NotImplementedError

    label_criterion = torch.nn.BCEWithLogitsLoss(reduction="none")

    # define evaluation function
    @torch.no_grad()
    def evaluate(model, loader, split):
        model.eval()

        total_correct = 0
        total_num = 0

        attrwise_acc_meter = MultiDimAverageMeter([2, 2])

        pbar = tqdm(loader, dynamic_ncols=True,
                    desc='evaluating ...')
        for img, all_attr_label in pbar:
            img = img.to(device, non_blocking=True)
            target_attr_label = all_attr_label[:, target_attr_idx]
            target_attr_label = target_attr_label.to(
                device, non_blocking=True)
            cls_out = model(img)
            if isinstance(cls_out, tuple):
                logits = cls_out[0]
            else:
                logits = cls_out
            prob = torch.sigmoid(logits).squeeze(-1)
            pred = prob > 0.5
            correct = (pred == target_attr_label).long()
            total_correct += correct.sum().item()
            total_num += correct.size(0)
            attrwise_acc_meter.add(correct.cpu(), all_attr_label)

        global_acc = total_correct / total_num
        log_dict = {f'{split}_global_acc': global_acc}

        multi_dim_color_acc = attrwise_acc_meter.get_mean()
        confict_align = ['conflict', 'align']
        total_acc_align_conflict = 0
        for color in range(2):
            color_mask = eye_tsr == color
            acc = multi_dim_color_acc[color_mask].mean().item()
            align_conflict_str = confict_align[color]
            log_dict[f'{split}_{align_conflict_str}_acc'] = acc
            total_acc_align_conflict += acc

        log_dict[f'{split}_unbiased_acc'] = total_acc_align_conflict / 2
        model.train()

        return log_dict

    def visualise_training_data(loader, target_attr_idx):
        train_iter = iter(train_loader)
        data, attr = next(train_iter)
        img_size = data[0][0].shape[-1]
        x = data
        l = attr[:, target_attr_idx]
        images = []
        for i in range(2):
            images.append(x[l == i][:10])
        images = torch.stack(images).view(-1, 3, img_size, img_size)
        grid_img = torchvision.utils.make_grid(images[:100].squeeze(1), nrow=10, normalize=True)
        wandb.log({"training_data": wandb.Image(grid_img)})

    def visualise_model_predictions(model, valid_loader, device, plot_name):
        if plot_name != "predictions":
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


    valid_attrwise_accs_list = []
    wandb.watch(model, log='all', log_freq=100)
    visualise_training_data(train_loader, target_attr_idx)
    main_valid_freq = 100
    
    for step in tqdm(range(main_num_steps)):
        try:
            index, data, attr = next(train_iter)
        except:
            train_iter = iter(train_loader)
            data, attr = next(train_iter)

        data = data.to(device)
        attr = attr.to(device)

        label = attr[:, target_attr_idx]

        logit = model(data)
        loss_per_sample = label_criterion(logit.squeeze(1), label.float())

        loss = loss_per_sample.mean()

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        main_log_freq = 100
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
            test_accuracy = evaluate(model, test_loader, 'eval')
            wandb.log(test_accuracy)

    # visualise_model_predictions(model, valid_loader, device, "predictions")
    # visualise_model_predictions(model, align_loader, device, "predictions-aligned_data")

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
