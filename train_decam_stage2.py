import os
from tqdm import tqdm
import wandb
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
import warnings
import torch.nn.functional as F
from config import ex
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    from torch.utils.tensorboard import SummaryWriter


from data.util import get_dataset, IdxDataset, NewDataset, transforms
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
    
    if dataset_tag != "bFFHQ":
        train_dataset = IdxDataset(train_dataset)
        valid_dataset = IdxDataset(valid_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=stage2_main_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=stage2_main_batch_size,
        shuffle=True,
        num_workers=16,
        pin_memory=True,
        drop_last=False
    )

    model_b = get_model(model_tag, num_classes).to(device)
    model_b.load_state_dict(torch.load(os.path.join(log_dir, dataset_tag, 'stage1', str(random_seed), 'biased_model_stage1.th')))

    model_d = get_model(model_tag, num_classes).to(device)
    model_d.load_state_dict(torch.load(os.path.join(log_dir, dataset_tag, 'stage1', str(random_seed), 'debiased_model_stage1.th')))

    optimizer = torch.optim.Adam(model_d.parameters(), lr=stage2_main_learning_rate, weight_decay=stage2_main_weight_decay)
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

    def get_misclassified_data_loader(model_b_eval, loader):
        model_b_eval.eval()
        pseudo_conflict_indices = torch.tensor([]).to(device)
        for idx, data, attr in loader:
            data = data.to(device)
            attr = attr.to(device)
            label = attr[:, target_attr_idx].to(device)
            with torch.no_grad():
                outputs = F.softmax(model_b_eval(data, debias_weight=0, bias_weight=1), dim=1)
                _, predicted = torch.max(outputs.data, 1)
                misclassified = (predicted != label).long()
                pseudo_conflict_indices = torch.cat((pseudo_conflict_indices, idx[misclassified==1].to(device))).long()
        return pseudo_conflict_indices

    accs, accs_aligned, accs_conflict = evaluate(model_d, valid_loader, debias_weight=1, bias_weight=0)
    print('Accuracy Train : %.4f Aligned Acc : %.4f Conflict Acc %.4f ' % (accs, accs_aligned, accs_conflict))

    accs, accs_aligned, accs_conflict = evaluate(model_b, train_loader, debias_weight=0, bias_weight=1)
    print('Accuracy Train : %.4f Aligned Acc : %.4f Conflict Acc %.4f ' % (accs, accs_aligned, accs_conflict))

    misclassified_indices = get_misclassified_data_loader(model_b, train_loader)
    updated_train_dataset = torch.utils.data.Subset(train_dataset, misclassified_indices)
    train_loader = DataLoader(
        updated_train_dataset,
        batch_size=stage2_main_batch_size,
        shuffle=True,
        drop_last=False
    )
    valid_best = 0
    for epoch in range(stage2_num_epochs):
        model_d.train()
        for _, data, attr in tqdm(train_loader):
            data = data.to(device)
            attr = attr.to(device)
            label = attr[:, target_attr_idx]

            logit = model_d(data, debias_weight=1, bias_weight=0)
            loss_per_sample = label_criterion(logit.squeeze(1), label)
            loss = loss_per_sample.mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch % main_log_freq) == 0:
            loss = loss.item()
            wandb.log({"stage2/loss-model_d/train": loss})

        if (epoch % main_valid_freq) == 0:
            valid_accs, valid_aligned, valid_conflict = evaluate(model_d, valid_loader, debias_weight=1,
                                                                 bias_weight=0)
            wandb.log({"stage2/acc-debiased-branch/valid": valid_accs})
            wandb.log({"stage2/acc-debiased-branch/valid_aligned": valid_aligned})
            wandb.log({"stage2/acc-debiased-branch/valid_skewed": valid_conflict})

            best_model_path = os.path.join(save_path, 'debiased_model_stage2.th')
            if dataset_tag != "bFFHQ":
                if valid_accs > valid_best:
                    torch.save(model_d.state_dict(), best_model_path)
                    valid_best = valid_accs
                    wandb.log({"acc-debiased-branch/valid_best": valid_best})
                    wandb.save(best_model_path)
            else:
                if valid_conflict > valid_best:
                    torch.save(model_d.state_dict(), best_model_path)
                    valid_best = valid_conflict
                    wandb.log({"acc-debiased-branch/valid_best": valid_best})
                    wandb.save(best_model_path)


  

