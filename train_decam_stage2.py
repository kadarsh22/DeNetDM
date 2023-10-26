import os
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import ex
from data.util import get_dataset, IdxDataset
from module.util import get_model
from util import MultiDimAverageMeter
import torch.nn.functional as F


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
        stage2_num_epochs,
        main_valid_freq,
        main_log_freq,
        stage2_main_batch_size,
        stage2_main_learning_rate,
        stage2_main_weight_decay,
        stage2_poe_weight,
        stage2_dist_weight,
        stage2_T,

):
    print('Beginning Stage 2')
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
        batch_size=stage2_main_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=False
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=stage2_main_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=False
    )

    model = get_model(model_tag, num_classes, stage='2').to(device)
    print(model)
    model.load_state_dict(
        torch.load(os.path.join(log_dir, dataset_tag, 'stage1', str(random_seed), 'debiased_model_stage1.th')),
        strict=False)

    teacher = get_model(model_tag, num_classes, stage='1').to(device)
    print(teacher)
    teacher.load_state_dict(
        torch.load(os.path.join(log_dir, dataset_tag, 'stage1', str(random_seed), 'debiased_model_stage1.th')),
        strict=True)

    optimizer = torch.optim.Adam(list(model.debias_branch.parameters()) + list(model.classifier.parameters()),
                                 lr=stage2_main_learning_rate,
                                 weight_decay=stage2_main_weight_decay)
    label_criterion = torch.nn.CrossEntropyLoss(reduction="none")

    save_path = os.path.join(log_dir, dataset_tag, 'stage2', str(random_seed))
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

    accs, accs_aligned, accs_conflict = evaluate(model, valid_loader, debias_weight=1, bias_weight=0)
    print('Accuracy valid Model_D: %.4f Aligned Acc : %.4f Conflict Acc %.4f ' % (accs, accs_aligned, accs_conflict))

    wandb.log({"stage2/acc-debiased-branch/valid": accs, "epoch": -1})
    wandb.log({"stage2/acc-debiased-branch/valid_aligned": accs_aligned, "epoch": -1})
    wandb.log({"stage2/acc-debiased-branch/valid_skewed": accs_conflict, "epoch": -1})

    accs, accs_aligned, accs_conflict = evaluate(model, train_loader, debias_weight=0, bias_weight=1)
    print('Accuracy Train Model_B: %.4f Aligned Acc : %.4f Conflict Acc %.4f ' % (accs, accs_aligned, accs_conflict))

    wandb.log({"stage2/acc-biased-branch/valid": accs, "epoch": -1})
    wandb.log({"stage2/acc-biased-branch/valid_aligned": accs_aligned, "epoch": -1})
    wandb.log({"stage2/acc-biased-branch/valid_skewed": accs_conflict, "epoch": -1})

    accs, accs_aligned, accs_conflict = evaluate(teacher, valid_loader, debias_weight=1, bias_weight=0)
    print('Accuracy Teacher Model: %.4f Aligned Acc : %.4f Conflict Acc %.4f ' % (accs, accs_aligned, accs_conflict))

    valid_conflict_best = 0
    teacher.eval()
    wandb.define_metric("stage2/*", step_metric="epoch")
    for epoch in range(stage2_num_epochs):
        model.train()
        for _, data, attr in tqdm(train_loader):
            data = data.to(device)
            attr = attr.to(device)
            label = attr[:, target_attr_idx]

            logit = model(data)
            loss_per_sample = label_criterion(logit.squeeze(1), label)
            poe_loss = loss_per_sample.mean()

            with torch.no_grad():
                teacher_logits = teacher(data, debias_weight=1, bias_weight=0)
            student_logits = model(data, debias_weight=1, bias_weight=0)
            soft_targets = F.softmax(teacher_logits / stage2_T, dim=-1)
            soft_prob = F.log_softmax(student_logits / stage2_T, dim=-1)
            distillation_loss = -torch.sum(soft_targets * soft_prob) / soft_prob.size()[0] * (stage2_T ** 2)

            loss = stage2_poe_weight * poe_loss + stage2_dist_weight * distillation_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch % main_log_freq) == 0:
            loss = loss.detach().cpu()
            wandb.log({"stage2/loss-poe/train": loss, "epoch": epoch})

            bias_attr = attr[:, bias_attr_idx]  # oracle
            loss_per_sample = loss_per_sample.detach()
            if (label == bias_attr).any().item():
                aligned_loss = loss_per_sample[label == bias_attr].mean()
                wandb.log({"stage2/loss-poe/train_aligned": aligned_loss, "epoch": epoch})

            if (label != bias_attr).any().item():
                skewed_loss = loss_per_sample[label != bias_attr].mean()
                wandb.log({"stage2/loss-poe/train_skewed": skewed_loss, "epoch": epoch})

        if (epoch % main_valid_freq) == 0:
            valid_accs, valid_aligned, valid_conflict = evaluate(model, valid_loader)
            wandb.log({"stage2/acc-poe/valid": valid_accs, "epoch": epoch})
            wandb.log({"stage2/acc-poe/valid_aligned": valid_aligned, "epoch": epoch})
            wandb.log({"stage2/acc-poe/valid_skewed": valid_conflict, "epoch": epoch})

            valid_accs, valid_aligned, valid_conflict = evaluate(model, valid_loader, debias_weight=1, bias_weight=0)
            wandb.log({"stage2/acc-debiased-branch/valid": valid_accs, "epoch": epoch})
            wandb.log({"stage2/acc-debiased-branch/valid_aligned": valid_aligned, "epoch": epoch})
            wandb.log({"stage2/acc-debiased-branch/valid_skewed": valid_conflict, "epoch": epoch})

            if dataset_tag != 'bFFHQ':
                if valid_accs > valid_conflict_best:
                    debiased_model_path = os.path.join(save_path, 'debiased_model_stage2.th')
                    torch.save(model.state_dict(), debiased_model_path)
                    wandb.save(debiased_model_path)
                    valid_conflict_best = valid_accs
                    wandb.log({"stage2/acc-debiased-branch/valid_best": valid_conflict_best, "epoch": epoch})

            elif dataset_tag == 'bFFHQ':
                if valid_conflict > valid_conflict_best:
                    debiased_model_path = os.path.join(save_path, 'debiased_model_stage2.th')
                    torch.save(model.state_dict(), debiased_model_path)
                    wandb.save(debiased_model_path)
                    valid_conflict_best = valid_conflict
                    wandb.log({"stage2/acc-debiased-branch/valid_best": valid_conflict_best, "epoch": epoch})

            valid_accs, valid_aligned, valid_conflict = evaluate(model, valid_loader, debias_weight=0, bias_weight=1)
            wandb.log({"stage2/acc-biased-branch/valid-branch1": valid_accs, "epoch": epoch})
            wandb.log({"stage2/acc-biased-branch/valid_aligned": valid_aligned, "epoch": epoch})
            wandb.log({"stage2/acc-biased-branch/valid_skewed": valid_conflict, "epoch": epoch})
