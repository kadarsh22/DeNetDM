import os
from tqdm import tqdm
import wandb
import torch
from config import ex
from data.waterbirds import get_waterbird_dataloader
from module.util import get_model
from util import MultiDimAverageMeter, add_identifier_to_keys
import torchvision
import matplotlib.pyplot as plt


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

    num_classes = 2
    train_loader = get_waterbird_dataloader(main_batch_size, 0.95, split='train')
    test_loader = get_waterbird_dataloader(main_batch_size, 0.95, split='test')

    # define model and optimizer
    model = get_model(model_tag, num_classes, stage='1').to(device)
    print(model)

    # bias_optimizer = torch.optim.Adam(
    #         model.bias_branch.parameters(),
    #         lr=1e-2,
    #         weight_decay=main_weight_decay,
    #     )
    
    # debias_optimizer = torch.optim.SGD(
    #         list(model.debias_branch.parameters()) + list(model.classifier.parameters()),
    #         lr=main_learning_rate,
    #         weight_decay=1e-4,
    #         momentum=0.9,
    #     )

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

    # define evaluation function
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
        bias_aligned_acc = (total_correct_group_zero + total_correct_group_three) /  (total_num_group_zero + total_num_group_three)
        bias_conflict_acc  = (total_correct_group_one + total_correct_group_two) /  (total_num_group_one + total_num_group_two)
        worst_group_acc = min(group_zero_acc, group_one_acc, group_two_acc, group_three_acc)
        wandb.log({"acc/" + str(branch) + "/group0": group_zero_acc,
                   "acc/" + str(branch) + "/group1": group_one_acc,
                   "acc/" + str(branch) + "/group2": group_two_acc,
                   "acc/" + str(branch) + "/group3": group_three_acc,
                   "acc/" + str(branch) + "/bias_aligned" : bias_aligned_acc,
                   "acc/" + str(branch) + "/bias_conflict" : bias_conflict_acc,
                   "acc/" + str(branch) + "/worst_group_acc": worst_group_acc,
                   "epoch": epoch})

    
    def visualise_model_predictions(model, valid_loader, device, plot_name, debias_weight=1, bias_weight=0):
        if plot_name != "predictions":
            data = [(images, torch.max(model(images.to(device), debias_weight, bias_weight).data, 1)[1], attr[0]) for
                    images, attr, _ in valid_loader]
        else:
            data = [(images, torch.max(model(images.to(device), debias_weight, bias_weight).data, 1)[1]) for
                    index, images, attr in valid_loader]
        img_size = data[0][0].shape[-1]
        true_labels_birds = torch.stack([d[2] for d in data]).view(-1)
        x = torch.stack([d[0] for d in data]).view(-1, 3, img_size, img_size)
        l = torch.stack([d[1] for d in data]).view(-1)

        images = []
        for i in range(2):
            if x[l.cpu() == i][:10].shape[0] == 10:
                select = torch.clone(x[l.cpu() == i][:10])
                images.append(select)
            else:
                images.append(torch.zeros((10, 3, img_size, img_size)))
        images = torch.stack(images).view(-1, 3, img_size, img_size)
        grid_img = torchvision.utils.make_grid(images[:100], nrow=10, normalize=True)
        plt.imshow(grid_img.permute(1, 2, 0).cpu().data)
        wandb.log({plot_name: wandb.Image(grid_img)})

    wandb.define_metric("loss-poe/*", step_metric="epoch")
    wandb.define_metric("acc-debiased-branch/*", step_metric="epoch")
    wandb.define_metric("acc-biased-branch/*", step_metric="epoch")
    wandb.define_metric("acc/*", step_metric="epoch")
    target_attr_idx = 0
    bias_attr_idx = 1
    for epoch in range(num_epochs):
        model.train()
        for data, attr, _ in tqdm(train_loader):
            data = data.to(device)
            label = attr[target_attr_idx].to(device)

            logit = model(data)
            loss_per_sample = label_criterion(logit.squeeze(1), label)
            loss = loss_per_sample.mean()

            optimizer.zero_grad()
            # bias_optimizer.zero_grad()
            # debias_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # bias_optimizer.step()
            # debias_optimizer.step()

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
            evaluate(model, test_loader, 'skip', debias_weight=1, bias_weight=0)
            for bird_id in range(2):
                visualise_model_predictions(model, test_loader, device, 'skip-group-' + str(bird_id), debias_weight=1,
                                            bias_weight=0)

            evaluate(model, test_loader, 'target', debias_weight=0, bias_weight=1)
            for bird_id in range(2):
                visualise_model_predictions(model, test_loader, device, 'feature-group' + str(bird_id), debias_weight=0,
                                            bias_weight=1)
                
        debiased_model_path = os.path.join(save_path, 'debiased_model_stage1_' + str(epoch) + '_.th')
        torch.save(model.state_dict(), debiased_model_path)
        wandb.save(debiased_model_path)