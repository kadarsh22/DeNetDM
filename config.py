import os
import logging
from sacred import Experiment
from sacred.observers import MongoObserver

ex = Experiment("main")
logger = logging.getLogger("debias")
ch = logging.StreamHandler()
formatter = logging.Formatter(
    "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s", datefmt="%H:%M:%S"
)
ch.setFormatter(formatter)
logger.addHandler(ch)
logger.setLevel("INFO")
ex.logger = logger


@ex.config
def get_config():
    device = 0
    log_dir = None
    data_dir = None

    main_tag = None

    dataset_tag = None
    model_tag = None

    target_attr_idx = None
    bias_attr_idx = None

    main_num_steps = None
    main_valid_freq = None
    epochs = None

    main_batch_size = 256
    main_optimizer_tag = 'Adam'
    main_learning_rate = 1e-3
    main_weight_decay = 0.0
    gamma = None

    main_save_logits = False


# User Configuration
@ex.named_config
def server_user_lff():
    log_dir = "/home/user/workspace/debias/log_lff"
    data_dir = "/home/user/datasets/debias"


@ex.named_config
def server_user_vanilla():
    log_dir = "/home/user/workspace/debias/log_vanilla"
    data_dir = "/home/user/datasets/debias"


@ex.named_config
def server_user_gce():
    log_dir = "/home/user/workspace/debias/log_gce"
    data_dir = "/home/user/datasets/debias"

@ex.named_config
def server_user_ours():
    log_dir = "/home/user/workspace/debias/log_ours"
    data_dir = "/home/user/datasets/debias"

@ex.named_config
def server_user_ours_simul_gce_focal():
    log_dir = "/home/user/workspace/debias/log_ours_simul_gce_focal"
    data_dir = "/home/user/datasets/debias"

@ex.named_config
def remote_user_vanilla():
    log_dir = "results/log_vanilla"
    data_dir = "../data/"


@ex.named_config
def remote_user_gce():
    log_dir = "results/logs_gce"
    data_dir = "../data/"


@ex.named_config
def remote_user_ours():
    log_dir = "results/log_ours"
    data_dir = "../data/"

@ex.named_config
def remote_user_ours_simul_gce_focal():
    log_dir = "results/log_ours_simul_gce_focal"
    data_dir = "../data/"

@ex.named_config
def remote_user_lff():
    log_dir = "results/log_lff"
    data_dir = "../data/"


# Dataset Configuration

@ex.named_config
def colored_mnist(log_dir):
    dataset_tag = "ColoredMNIST"
    model_tag = "MLP"
    main_num_steps = 235 * 100
    target_attr_idx = 0
    bias_attr_idx = 1
    main_valid_freq = 235
    main_tag = "ColoredMNIST"
    main_batch_size = 256
    log_dir = os.path.join(log_dir, 'colored_mnist')


@ex.named_config
def corrupted_cifar10(log_dir):
    dataset_tag = "CorruptedCIFAR10"
    model_tag = 'ResNet20'
    target_attr_idx = 0
    bias_attr_idx = 1
    main_num_steps = 196 * 200
    main_valid_freq = 196
    main_batch_size = 256
    main_tag = "CorruptedCIFAR10"
    gamma = None
    log_dir = os.path.join(log_dir, 'corrupted_cifar')


@ex.named_config
def celeba(log_dir):
    dataset_tag = 'CelebA'
    model_tag = 'ResNet18'
    target_attr_idx = 9
    bias_attr_idx = 20
    main_num_steps = 636 * 200
    main_valid_freq = 636
    main_batch_size = 256
    main_learning_rate = 1e-4
    main_weight_decay = 1e-4
    main_tag = 'CelebA-{}-{}'.format(target_attr_idx, bias_attr_idx)
    log_dir = os.path.join(log_dir, 'celeba')


@ex.named_config
def type0(dataset_tag, main_tag):
    dataset_tag += "-Type0"
    main_tag += "-Type0"


@ex.named_config
def type1(dataset_tag, main_tag):
    dataset_tag += "-Type1"
    main_tag += "-Type1"


@ex.named_config
def skewed0(dataset_tag, main_tag):
    dataset_tag += "-Skewed0.9"
    main_tag += "-Skewed0.9"


@ex.named_config
def skewed1(dataset_tag, main_tag):
    dataset_tag += "-Skewed0.05"
    main_tag += "-Skewed0.05"


@ex.named_config
def skewed2(dataset_tag, main_tag):
    dataset_tag += "-Skewed0.02"
    main_tag += "-Skewed0.02"


@ex.named_config
def skewed3(dataset_tag, main_tag):
    dataset_tag += "-Skewed0.01"
    main_tag += "-Skewed0.01"


@ex.named_config
def skewed4(dataset_tag, main_tag):
    dataset_tag += "-Skewed0.005"
    main_tag += "-Skewed0.005"


@ex.named_config
def severity1(dataset_tag, main_tag):
    dataset_tag += "-Severity1"
    main_tag += "-Severity1"


@ex.named_config
def severity2(dataset_tag, main_tag):
    dataset_tag += "-Severity2"
    main_tag += "-Severity2"


@ex.named_config
def severity3(dataset_tag, main_tag):
    dataset_tag += "-Severity3"
    main_tag += "-Severity3"


@ex.named_config
def severity4(dataset_tag, main_tag):
    dataset_tag += "-Severity4"
    main_tag += "-Severity4"


@ex.named_config
def gce_model_path1(dataset_tag, main_tag):
    gce_model_path = '/home/user/workspace/debias/log_lff/corrupted_cifar/result/CorruptedCIFAR10-Type0-Skewed0.05-Severity1/model.th'


@ex.named_config
def gce_model_path2(dataset_tag, main_tag):
    gce_model_path = '/home/user/workspace/debias/log_lff/corrupted_cifar/result/CorruptedCIFAR10-Type0-Skewed0.05-Severity2/model.th'


@ex.named_config
def gce_model_path3(dataset_tag, main_tag):
    gce_model_path = '/home/user/workspace/debias/log_lff/corrupted_cifar/result/CorruptedCIFAR10-Type0-Skewed0.05-Severity3/model.th'


@ex.named_config
def gce_model_path4(dataset_tag, main_tag):
    gce_model_path = '/home/user/workspace/debias/log_lff/corrupted_cifar/result/CorruptedCIFAR10-Type0-Skewed0.05-Severity4/model.th'


# Method Configuration

@ex.named_config
def adam(main_tag):
    main_optimizer_tag = "Adam"
    main_learning_rate = 1e-3
    main_weight_decay = 0
    main_tag += "_Adam"


@ex.named_config
def adamw(main_tag):
    main_optimizer_tag = "AdamW"
    main_learning_rate = 1e-3
    main_weight_decay = 5e-3
    main_tag += "_AdamW"


@ex.named_config
def log_epochs(main_tag, epochs):
    main_tag += "_epochs_{}".format(epochs)
    if "ColoredMNIST" in main_tag:
        main_num_steps = 235 * epochs
    elif "CorruptedCIFAR10" in main_tag:
        main_num_steps = 196 * epochs
    elif 'CelebA' in main_tag:
        main_num_steps = 636 * epochs


@ex.named_config
def reverse(main_tag):
    main_tag += '_reverse'
    target_attr_idx = 1
    bias_attr_idx = 0

@ex.named_config
def gamma0(dataset_tag, main_tag):
    gamma = 0

@ex.named_config
def gamma1(dataset_tag, main_tag):
    gamma = 1


@ex.named_config
def gamma2(dataset_tag, main_tag):
    gamma = 2


@ex.named_config
def gamma4(dataset_tag, main_tag):
    gamma = 4


@ex.named_config
def gamma8(dataset_tag, main_tag):
    gamma = 8


@ex.named_config
def gamma16(dataset_tag, main_tag):
    gamma = 16


@ex.named_config
def gamma32(dataset_tag, main_tag):
    gamma = 32
