import os
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

    train_loader_for_eval = DataLoader(
        train_dataset,
        batch_size=main_batch_size,
        shuffle=False,
        num_workers=16,
        pin_memory=True,
        drop_last=False
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
        optimizer = torch.optim.Adam(model.parameters(),
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
    def evaluate(model, data_loader, debias_weight=1, bias_weight=1):
        model.eval()
        attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
        for _, data, attr in tqdm(data_loader, leave=False):
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

    valid_conflict_best = 0

    # set all other train/ metrics to use this step
    wandb.define_metric("acc-poe/*", step_metric="epoch")
    wandb.define_metric("acc-debiased-branch/*", step_metric="epoch")
    wandb.define_metric("acc-biased-branch/*", step_metric="epoch")
    wandb.define_metric("loss-poe/*", step_metric="epoch")
    wandb.define_metric("train-acc/*", step_metric="epoch")

    for epoch in range(num_epochs):
        model.train()

        for _, data, attr in tqdm(train_loader):
            data = data.to(device)
            attr = attr.to(device)
            label = attr[:, target_attr_idx]

            logit = model(data)
            loss_per_sample = label_criterion(logit.squeeze(1), label)
            loss = loss_per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch % main_log_freq) == 0:
            loss = loss.detach().cpu()
            wandb.log({"loss-poe/train": loss, "epoch": epoch})

            bias_attr = attr[:, bias_attr_idx]  # oracle
            loss_per_sample = loss_per_sample.detach()
            if (label == bias_attr).any().item():
                aligned_loss = loss_per_sample[label == bias_attr].mean()
                wandb.log({"loss-poe/train_aligned": aligned_loss, "epoch": epoch})

            if (label != bias_attr).any().item():
                skewed_loss = loss_per_sample[label != bias_attr].mean()
                wandb.log({"loss-poe/train_skewed": skewed_loss, "epoch": epoch})

        if (epoch % main_valid_freq) == 0:
            valid_accs, valid_aligned, valid_conflict = evaluate(model, valid_loader)
            wandb.log({"acc-poe/valid": valid_accs, "epoch": epoch})
            wandb.log({"acc-poe/valid_aligned": valid_aligned, "epoch": epoch})
            wandb.log({"acc-poe/valid_skewed": valid_conflict, "epoch": epoch})

            valid_accs, valid_aligned, valid_conflict = evaluate(model, valid_loader, debias_weight=1, bias_weight=0)
            wandb.log({"acc-debiased-branch/valid": valid_accs, "epoch": epoch})
            wandb.log({"acc-debiased-branch/valid_aligned": valid_aligned, "epoch": epoch})
            wandb.log({"acc-debiased-branch/valid_skewed": valid_conflict, "epoch": epoch})

            if valid_conflict > valid_conflict_best:
                debiased_model_path = os.path.join(save_path, 'debiased_model_stage1.th')
                torch.save(model.state_dict(), debiased_model_path)
                wandb.save(debiased_model_path)
                valid_conflict_best = valid_conflict
                wandb.log({"acc-debiased-branch/valid_best": valid_conflict_best, "epoch": epoch})

            valid_accs, valid_aligned, valid_conflict = evaluate(model, valid_loader, debias_weight=0, bias_weight=1)
            wandb.log({"acc-biased-branch/valid-branch1": valid_accs, "epoch": epoch})
            wandb.log({"acc-biased-branch/valid_aligned": valid_aligned, "epoch": epoch})
            wandb.log({"acc-biased-branch/valid_skewed": valid_conflict, "epoch": epoch})

            # ##Training accuracies

            valid_accs, valid_aligned, valid_conflict = evaluate(model, train_loader_for_eval)
            wandb.log({"train-acc/acc-poe/train": valid_accs, "epoch": epoch})
            wandb.log({"train-acc/acc-poe/train_aligned": valid_aligned, "epoch": epoch})
            wandb.log({"train-acc/acc-poe/train_skewed": valid_conflict, "epoch": epoch})

            valid_accs, valid_aligned, valid_conflict = evaluate(model, train_loader_for_eval, debias_weight=1,
                                                                 bias_weight=0)
            wandb.log({"train-acc/acc-debiased-branch/train": valid_accs, "epoch": epoch})
            wandb.log({"train-acc/acc-debiased-branch/train_aligned": valid_aligned, "epoch": epoch})
            wandb.log({"train-acc/acc-debiased-branch/train_skewed": valid_conflict, "epoch": epoch})

            valid_accs, valid_aligned, valid_conflict = evaluate(model, train_loader_for_eval, debias_weight=0,
                                                                 bias_weight=1)
            wandb.log({"train-acc/acc-biased-branch/train-branch1": valid_accs, "epoch": epoch})
            wandb.log({"train-acc/acc-biased-branch/train_aligned": valid_aligned, "epoch": epoch})
            wandb.log({"train-acc/acc-biased-branch/train_skewed": valid_conflict, "epoch": epoch})
