import os
import logging
from sacred import Experiment

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
    random_seed = 67


@ex.named_config
def colored_mnist():
    dataset_tag = "ColoredMNIST"
    data_dir = os.path.join('/vol/research/project_storage', 'ColoredMNIST')
    log_dir = os.path.join('results', 'ColoredMNIST')
    model_tag = "CMNISTDeCAMModel"
    num_epochs = 100
    target_attr_idx = 0
    bias_attr_idx = 1
    main_valid_freq = 1
    main_log_freq = 1
    main_tag = "ColoredMNIST"
    main_batch_size = 64
    main_optimizer_tag = 'Adam'
    main_learning_rate = 1e-3
    main_weight_decay = 1e-3

    stage2_num_epochs = 100
    stage2_main_batch_size = 64
    stage2_main_learning_rate = 1e-3
    stage2_main_weight_decay = 0
    stage2_poe_weight = 1
    stage2_dist_weight = 1
    stage2_T = 2


@ex.named_config
def corrupted_cifar10():
    dataset_tag = "CorruptedCIFAR10"
    data_dir = os.path.join('../data/', 'corrupted-cifar10')
    log_dir = os.path.join('results', 'corrupted-cifar10')
    model_tag = 'CCIFARDeCAMModel'
    num_epochs = 100
    target_attr_idx = 0
    bias_attr_idx = 1
    main_valid_freq = 1
    main_log_freq = 1
    main_tag = "CorruptedCIFAR10"
    main_batch_size = 256
    main_optimizer_tag = 'Adam'
    main_learning_rate = 1e-3
    main_weight_decay = 1e-3

    stage2_num_epochs = 200
    stage2_main_batch_size = 256
    stage2_main_learning_rate = 1e-4
    stage2_main_weight_decay = 0.0
    stage2_poe_weight = 1
    stage2_dist_weight = 1
    stage2_T = 2


@ex.named_config
def bffhq():
    dataset_tag = "bFFHQ"
    data_dir = os.path.join('../data/', 'bffhq')
    log_dir = os.path.join('results', 'bffhq')
    model_tag = 'bFFHQDeCAMModel'
    num_epochs = 10
    target_attr_idx = 0
    bias_attr_idx = 1
    main_valid_freq = 1
    main_log_freq = 1
    main_tag = "bffhq"
    main_batch_size = 64
    main_optimizer_tag = 'Adam'
    main_learning_rate = 1e-3
    main_weight_decay = 0.0

    stage2_num_epochs = 100
    stage2_main_batch_size = 64
    stage2_main_learning_rate = 1e-4
    stage2_main_weight_decay = 0.0
    stage2_poe_weight = 1
    stage2_dist_weight = 0
    stage2_T = 2


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
