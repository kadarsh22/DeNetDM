import os
import wandb
import random
import numpy as np
import torch
from config import ex
from train_decam_stage1 import train as train_stage1
from train_decam_stage2 import train as train_stage2


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
def main(random_seed, dataset_tag, train=True):
    wandb.login()
    seed = random.randint(0,10000)

    wandb.init(project="DeNetDM", sync_tensorboard=True)
    wandb.run.name = 'DeNetDM_' + dataset_tag + '_seed_' + str(seed)
    wandb.run.log_code(".")
    set_seed(seed=seed)

    if train:
        train_stage1(random_seed=seed)
        train_stage2(random_seed=seed)
    else:
        train_stage2(random_seed=seed, eval_only=True)
