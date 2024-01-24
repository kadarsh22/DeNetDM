import os
from tqdm import tqdm
import wandb
import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from config import ex
from data.util import get_dataset, IdxDataset
from module.util import get_model
from data.waterbirds import get_waterbird_dataloader
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
        stage2_main_optimizer_tag,
        stage2_poe_weight,
        stage2_dist_weight,
        stage2_T,

):
    print('Beginning Stage 2')
    device = torch.device(device)

    num_classes = 2
    train_loader = get_waterbird_dataloader(stage2_main_batch_size, 0.95, split='train')
    valid_loader = get_waterbird_dataloader(stage2_main_batch_size, 0.95, split='test')

    # define model and optimizer
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

    if stage2_main_optimizer_tag == 'SGD':
        optimizer = torch.optim.SGD(list(model.debias_branch.parameters()) + list(model.classifier.parameters()),
                                 lr=stage2_main_learning_rate,
                                 weight_decay=stage2_main_weight_decay)
    if stage2_main_optimizer_tag == 'Adam':
        optimizer = torch.optim.Adam(list(model.debias_branch.parameters()) + list(model.classifier.parameters()),
                                 lr=stage2_main_learning_rate,
                                 weight_decay=stage2_main_weight_decay)
    label_criterion = torch.nn.CrossEntropyLoss(reduction="none")

    save_path = os.path.join(log_dir, dataset_tag, 'stage2', str(random_seed))
    os.makedirs(save_path, exist_ok=True)

    # # define evaluation function
    @torch.no_grad()
    def evaluate(model, loader, branch, debias_weight=1, bias_weight=0):
        model.eval()

        total_correct = 0
        total_num = 0
        total_correct_group_zero = 0
        total_num_group_zero = 0
        total_correct_group_one = 0
        total_num_group_one = 0
        total_correct_group_two = 0
        total_num_group_two = 0
        total_correct_group_three = 0
        total_num_group_three = 0

        pbar = tqdm(loader, dynamic_ncols=True, desc='evaluating ...')
        for img, all_attr_label_, env_idx in pbar:
            all_attr_label = torch.stack(all_attr_label_).T.to(device)
            img = img.to(device, non_blocking=True)
            target_attr_label = all_attr_label[:, target_attr_idx]
            target_attr_label = target_attr_label.to(
                device, non_blocking=True)
            cls_out = model(img, debias_weight, bias_weight)
            if isinstance(cls_out, tuple):
                logits = cls_out[0]
            else:
                logits = cls_out
            pred = logits.data.max(1, keepdim=True)[1].squeeze(1)

            # average group_accuracy
            correct = (pred == target_attr_label).long()
            total_correct += correct.sum().item()
            total_num += correct.size(0)

            # group_zero_accuracy
            correct = (pred == target_attr_label).long()[env_idx == 0]
            total_correct_group_zero += correct.sum().item()
            total_num_group_zero += correct.size(0)

            # group_one_accuracy
            correct = (pred == target_attr_label).long()[env_idx == 1]
            total_correct_group_one += correct.sum().item()
            total_num_group_one += correct.size(0)

            # group_two_accuracy
            correct = (pred == target_attr_label).long()[env_idx == 2]
            total_correct_group_two += correct.sum().item()
            total_num_group_two += correct.size(0)

            # group_three_accuracy
            correct = (pred == target_attr_label).long()[env_idx == 3]
            total_correct_group_three += correct.sum().item()
            total_num_group_three += correct.size(0)

        avg_group_acc = total_correct / total_num
        group_zero_acc = total_correct_group_zero / total_num_group_zero
        group_one_acc = total_correct_group_one / total_num_group_one
        group_two_acc = total_correct_group_two / total_num_group_two
        group_three_acc = total_correct_group_three / total_num_group_three
        worst_group_acc = min(group_zero_acc, group_one_acc, group_two_acc, group_three_acc)
        wandb.log({"acc/" + str(branch) + "/group0": group_zero_acc,
                   "acc/" + str(branch) + "/group1": group_one_acc,
                   "acc/" + str(branch) + "/group2": group_two_acc,
                   "acc/" + str(branch) + "/group3": group_three_acc,
                   "acc/" + str(branch) + "/worst_group_acc": worst_group_acc,
                   "epoch": epoch})
    

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

    wandb.define_metric("loss-poe/*", step_metric="epoch")
    wandb.define_metric("acc-debiased-branch/*", step_metric="epoch")
    wandb.define_metric("acc-biased-branch/*", step_metric="epoch")
    wandb.define_metric("acc/*", step_metric="epoch")
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
            wandb.log({"loss-poe/train": loss, "epoch": epoch})

            bias_attr = attr[bias_attr_idx].to(device)  # oracle
            loss_per_sample = loss_per_sample.detach()
            if (label == bias_attr).any().item():
                aligned_loss = loss_per_sample[label == bias_attr].mean()
                wandb.log({"loss-poe/train_aligned": aligned_loss, "epoch": epoch})

            if (label != bias_attr).any().item():
                skewed_loss = loss_per_sample[label != bias_attr].mean()
                wandb.log({"loss-poe/train_skewed": skewed_loss, "epoch": epoch})

        if epoch % main_log_freq == 0 and epoch > 1:
            evaluate(model, valid_loader, 'skip', debias_weight=1, bias_weight=0)
            # for bird_id in range(2):
                # visualise_model_predictions(model, test_loader, device, 'skip-group-' + str(bird_id), debias_weight=1,
                #                             bias_weight=0)

            evaluate(model, valid_loader, 'target', debias_weight=0, bias_weight=1)
            # for bird_id in range(2):
                # visualise_model_predictions(model, test_loader, device, 'feature-group' + str(bird_id), debias_weight=0,
                #                             bias_weight=1)
