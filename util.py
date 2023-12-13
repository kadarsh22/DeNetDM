import io
import torch
import numpy as np
from tqdm import tqdm


class MultiDimAverageMeter(object):
    def __init__(self, dims):
        self.dims = dims
        self.cum = torch.zeros(np.prod(dims))
        self.cnt = torch.zeros(np.prod(dims))
        self.idx_helper = torch.arange(np.prod(dims), dtype=torch.long).reshape(
            *dims
        )

    def add(self, vals, idxs):
        flattened_idx = torch.stack(
            [self.idx_helper[tuple(idxs[i])] for i in range(idxs.size(0))],
            dim=0,
        )
        self.cum.index_add_(0, flattened_idx, vals.view(-1).float())
        self.cnt.index_add_(
            0, flattened_idx, torch.ones_like(vals.view(-1), dtype=torch.float)
        )

    def get_mean(self):
        return (self.cum / self.cnt).reshape(*self.dims)

    def reset(self):
        self.cum.zero_()
        self.cnt.zero_()


class EMA:

    def __init__(self, label, alpha=0.9):
        self.label = label
        self.alpha = alpha
        self.parameter = torch.zeros(label.size(0))
        self.updated = torch.zeros(label.size(0))

    def update(self, data, index):
        self.parameter[index] = self.alpha * self.parameter[index] + (1 - self.alpha * self.updated[index]) * data
        self.updated[index] = 1

    def max_loss(self, label):
        label_index = np.where(self.label == label)[0]
        return self.parameter[label_index].max()


def add_identifier_to_keys(dictionary, identifier):
    modified_dict = {}
    for key, value in dictionary.items():
        modified_key = f"{identifier}_{key}"
        modified_dict[modified_key] = value
    return modified_dict


import math
import numpy as np
import os
import random
import sys
import time
import torch
import torch.nn as nn
import functools
from pathlib import Path
from torch.distributions.beta import Beta
from os import path as osp
from shutil import get_terminal_size
from torchvision.utils import make_grid
from collections import defaultdict
from torch.distributions.uniform import Uniform
import torchvision
from sklearn.metrics import confusion_matrix

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def get_between_class_variance(prototypes):
    n = prototypes.size(0)
    dist = torch.pow(prototypes, 2).sum(dim=1, keepdim=True).expand(n, n)
    dist = dist + dist.t()
    dist.addmm_(1, -2, prototypes, prototypes.t())
    dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability
    avg_dist = dist.mean()
    return avg_dist


def scandir(dir_path, suffix=None, recursive=False):
    """Scan a directory to find the interested files.

    Args:
        dir_path (str | obj:`Path`): Path of the directory.
        suffix (str | tuple(str), optional): File suffix that we are
            interested in. Default: None.
        recursive (bool, optional): If set to True, recursively scan the
            directory. Default: False.

    Returns:
        A generator for all the interested files with relative pathes.
    """
    if isinstance(dir_path, (str, Path)):
        dir_path = str(dir_path)
    else:
        raise TypeError('"dir_path" must be a string or Path object')

    if (suffix is not None) and not isinstance(suffix, (str, tuple)):
        raise TypeError('"suffix" must be a string or tuple of strings')

    root = dir_path

    def _scandir(dir_path, suffix, recursive):
        for entry in os.scandir(dir_path):
            if not entry.name.startswith('.') and entry.is_file():
                rel_path = osp.relpath(entry.path, root)
                if suffix is None:
                    yield rel_path
                elif rel_path.endswith(suffix):
                    yield rel_path
            else:
                if recursive:
                    yield from _scandir(
                        entry.path, suffix=suffix, recursive=recursive)
                else:
                    continue

    return _scandir(dir_path, suffix=suffix, recursive=recursive)


def mkdir_or_exist(dir_name, mode=0o777):
    if dir_name == '':
        return
    dir_name = osp.expanduser(dir_name)
    os.makedirs(dir_name, mode=mode, exist_ok=True)


def get_time_str():
    return time.strftime('%Y%m%d_%H%M%S', time.localtime())


def check_resume(opt, resume_iter):
    """Check resume states and pretrain_model paths.

    Args:
        opt (dict): Options.
        resume_iter (int): Resume iteration.
    """
    logger = get_root_logger()
    if opt['path']['resume_state']:
        # ignore pretrained model paths
        if opt['path'].get('pretrain_model_g') is not None or opt['path'].get(
                'pretrain_model_d') is not None:
            logger.warning(
                'pretrain_model path will be ignored during resuming.')

        # set pretrained model paths
        opt['path']['pretrain_model_g'] = osp.join(opt['path']['models'],
                                                   f'net_g_{resume_iter}.pth')
        logger.info(
            f"Set pretrain_model_g to {opt['path']['pretrain_model_g']}")

        opt['path']['pretrain_model_d'] = osp.join(opt['path']['models'],
                                                   f'net_d_{resume_iter}.pth')
        logger.info(
            f"Set pretrain_model_d to {opt['path']['pretrain_model_d']}")


def mkdir_and_rename(path):
    """mkdirs. If path exists, rename it with timestamp and create a new one.

    Args:
        path (str): Folder path.
    """
    if osp.exists(path):
        new_name = path + '_archived_' + get_time_str()
        print(f'Path already exists. Rename it to {new_name}', flush=True)
        os.rename(path, new_name)
    mkdir_or_exist(path)


def make_exp_dirs(opt):
    """Make dirs for experiments."""
    path_opt = opt['path'].copy()
    if opt['is_train']:
        mkdir_and_rename(path_opt.pop('experiments_root'))
    elif opt['is_incremental']:
        mkdir_and_rename(path_opt.pop('incremental_root'))
    else:
        mkdir_and_rename(path_opt.pop('results_root'))
    path_opt.pop('strict_load')
    for key, path in path_opt.items():
        if 'pretrain' not in key and 'resume' not in key and 'base' not in key and 'index' not in key:
            mkdir_or_exist(path)


def set_random_seed(seed):
    """Set random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_gpu(gpu):
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu
    print('using gpu:', gpu)


def crop_border(imgs, crop_border):
    """Crop borders of images.

    Args:
        imgs (list[ndarray] | ndarray): Images with shape (h, w, c).
        crop_border (int): Crop border for each end of height and weight.

    Returns:
        list[ndarray]: Cropped images.
    """
    if crop_border == 0:
        return imgs
    else:
        if isinstance(imgs, list):
            return [
                v[crop_border:-crop_border, crop_border:-crop_border, ...]
                for v in imgs
            ]
        else:
            return imgs[crop_border:-crop_border, crop_border:-crop_border,
                   ...]


def tensor2img(tensor, out_type=np.uint8, min_max=(0, 1)):
    """Convert torch Tensors into image numpy arrays.

    After clamping to [min, max], values will be normalized to [0, 1].

    Args:
        tensor (Tensor or list[Tensor]): Accept shapes:
            1) 4D mini-batch Tensor of shape (B x 3/1 x H x W);
            2) 3D Tensor of shape (3/1 x H x W);
            3) 2D Tensor of shape (H x W).
            Tensor channel should be in RGB order.
        out_type (numpy type): output types. If ``np.uint8``, transform outputs
            to uint8 type with range [0, 255]; otherwise, float type with
            range [0, 1]. Default: ``np.uint8``.
        min_max (tuple[int]): min and max values for clamp.

    Returns:
        (Tensor or list): 3D ndarray of shape (H x W x C) OR 2D ndarray of
        shape (H x W). The channel order is BGR.
    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list)
             and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError(
            f'tensor or list of tensors expected, got {type(tensor)}')

    if torch.is_tensor(tensor):
        tensor = [tensor]
    result = []
    for _tensor in tensor:
        _tensor = _tensor.squeeze(0).float().detach().cpu().clamp_(*min_max)
        _tensor = (_tensor - min_max[0]) / (min_max[1] - min_max[0])

        n_dim = _tensor.dim()
        if n_dim == 4:
            img_np = make_grid(
                _tensor, nrow=int(math.sqrt(_tensor.size(0))),
                normalize=False).numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :],
                                  (1, 2, 0))  # HWC, BGR
        elif n_dim == 3:
            img_np = _tensor.numpy()
            img_np = np.transpose(img_np[[2, 1, 0], :, :],
                                  (1, 2, 0))  # HWC, BGR
        elif n_dim == 2:
            img_np = _tensor.numpy()
        else:
            raise TypeError('Only support 4D, 3D or 2D tensor. '
                            f'But received with dimension: {n_dim}')
        if out_type == np.uint8:
            # Unlike MATLAB, numpy.unit8() WILL NOT round by default.
            img_np = (img_np * 255.0).round()
        img_np.astype(out_type)
        result.append(img_np)
    if len(result) == 1:
        result = result[0]
    return result


def dir_size(path):
    total_size = 0
    file_num = 0
    for dir in os.listdir(path):
        sub_path = osp.join(path, dir)
        if os.path.isfile(sub_path):
            file_num += 1
            total_size += os.path.getsize(sub_path)
        elif os.path.isdir(sub_path):
            sz, fn = dir_size(sub_path)
            total_size += sz
            file_num += fn
    return total_size, file_num


class ProgressBar(object):
    """A progress bar that can print the progress.

    Modified from:
    https://github.com/hellock/cvbase/blob/master/cvbase/progress.py
    """

    def __init__(self, task_num=0, bar_width=50, start=True):
        self.task_num = task_num
        max_bar_width = self._get_max_bar_width()
        self.bar_width = (
            bar_width if bar_width <= max_bar_width else max_bar_width)
        self.completed = 0
        if start:
            self.start()

    def _get_max_bar_width(self):
        terminal_width, _ = get_terminal_size()
        max_bar_width = min(int(terminal_width * 0.6), terminal_width - 50)
        if max_bar_width < 10:
            print(f'terminal width is too small ({terminal_width}), '
                  'please consider widen the terminal for better '
                  'progressbar visualization')
            max_bar_width = 10
        return max_bar_width

    def start(self):
        if self.task_num > 0:
            sys.stdout.write(f"[{' ' * self.bar_width}] 0/{self.task_num}, "
                             f'elapsed: 0s, ETA:\nStart...\n')
        else:
            sys.stdout.write('completed: 0, elapsed: 0s')
        sys.stdout.flush()
        self.start_time = time.time()

    def update(self, msg='In progress...'):
        self.completed += 1
        elapsed = time.time() - self.start_time + 1e-8
        fps = self.completed / elapsed
        if self.task_num > 0:
            percentage = self.completed / float(self.task_num)
            eta = int(elapsed * (1 - percentage) / percentage + 0.5)
            mark_width = int(self.bar_width * percentage)
            bar_chars = '>' * mark_width + '-' * (self.bar_width - mark_width)
            sys.stdout.write('\033[2F')  # cursor up 2 lines
            sys.stdout.write(
                '\033[J'
            )  # clean the output (remove extra chars since last display)
            sys.stdout.write(
                f'[{bar_chars}] {self.completed}/{self.task_num}, '
                f'{fps:.1f} task/s, elapsed: {int(elapsed + 0.5)}s, '
                f'ETA: {eta:5}s\n{msg}\n')
        else:
            sys.stdout.write(
                f'completed: {self.completed}, elapsed: {int(elapsed + 0.5)}s,'
                f' {fps:.1f} tasks/s')
        sys.stdout.flush()


def sample_data(data, labels, way, shots):
    batch = []
    if isinstance(labels, list):
        class_labels = labels[0]
    else:
        class_labels = labels

    label_list = torch.unique(class_labels)
    index_dic = defaultdict(list)

    for index, label in enumerate(class_labels):
        index_dic[label.item()].append(index)

    for key in index_dic.keys():
        index_dic[key] = torch.tensor(index_dic[key])

    classes_set = torch.randperm(len(label_list))[:way]
    for cl in classes_set:
        class_label = label_list[cl]
        index_list = index_dic[class_label.item()]
        pos = torch.randperm(len(index_list))[:shots].tolist()
        batch.append(index_list[pos])
    batch = torch.stack(batch).reshape(-1)

    if isinstance(labels, list):
        labels = [lb[batch] for lb in labels]
        return data[batch], labels
    else:
        return data[batch], labels[batch]


class Averager():

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.data = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def add(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.data.append(val)

    def item(self):
        return self.avg

    def obtain_data(self):
        return self.data

    def __len__(self):
        return len(self.data)


def one_hot(y, num_class):
    return torch.zeros((len(y), num_class)).cuda().scatter_(1, y.unsqueeze(1), 1)


class Timer():
    def __init__(self):
        self.o = time.time()

    def measure(self, p=1):
        x = (time.time() - self.o) / p
        x = int(x)
        if x >= 3600:
            return f'{x / 3600:.1f}h'
        if x >= 60:
            return f'{round(x / 60)}m'
        return f'{x}s'


class DiscreteUniform():
    def __init__(self, bound, shape, reduction_factor):
        self.bound = bound
        self.shape = shape
        b = int(reduction_factor / 2)
        ratio = torch.arange(-b, b + 1, device=torch.device('cuda'))
        self.choice = ratio / reduction_factor
        self.choice = self.choice * bound
        self.logger = get_root_logger()
        # for r in self.choice:
        #     self.logger.info(f'{r}')

    def sample(self):
        pos = self.generate_pos()
        noise = self.choice[pos]
        # for r in self.choice:
        #     sum = (noise == r).sum()
        #     self.logger.info(f'{r}: {sum/len(noise)}')
        return noise

    def generate_pos(self):
        return torch.randint(len(self.choice), self.shape, device=torch.device('cuda'))


class DiscreteUniform2():
    def __init__(self, bound, shape, reduction_factor):
        self.bound = bound
        self.shape = shape
        ratio1 = torch.tensor([-2, -2, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2], device=torch.device('cuda'))
        ratio2 = torch.tensor([-2, -2, -2, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 2, 2, 2],
                              device=torch.device('cuda'))
        self.choice = ratio1 / 4
        self.choice = self.choice * bound
        self.logger = get_root_logger()
        # for r in self.choice:
        #     self.logger.info(f'{r}')

    def sample(self):
        pos = self.generate_pos()
        noise = self.choice[pos]
        # str = ''
        # unique = torch.unique_consecutive(self.choice)
        # for r in unique:
        #     sum = (noise == r).sum()
        #     str += f'!!! {sum/len(noise):.5f} '
        # self.logger.info(str)
        return noise

    def generate_pos(self):
        return torch.randint(len(self.choice), self.shape, device=torch.device('cuda'))


class DiscreteUniform3():
    def __init__(self, bound, shape, reduction_factor):
        self.bound = bound
        self.shape = shape
        ratio1 = torch.tensor([-2, -2, -2, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 2, 2], device=torch.device('cuda'))

        self.choice = ratio1 / 4
        self.choice = self.choice * bound
        self.logger = get_root_logger()
        # for r in self.choice:
        #     self.logger.info(f'{r}')

    def sample(self):
        pos = self.generate_pos()
        noise = self.choice[pos]
        # str = ''
        # unique = torch.unique_consecutive(self.choice)
        # for r in unique:
        #     sum = (noise == r).sum()
        #     str += f'!!! {sum/len(noise):.5f} '
        # self.logger.info(str)
        return noise

    def generate_pos(self):
        return torch.randint(len(self.choice), self.shape, device=torch.device('cuda'))


class BetaDistribution():
    def __init__(self, alpha, beta, upper_bound, lower_bound):
        self.m = Beta(alpha, beta)
        self.upper_bound = upper_bound
        self.lower_bound = lower_bound
        self.length = self.upper_bound - self.lower_bound

    def sample(self):
        ans = self.m.sample()
        ans = ans * self.length + self.lower_bound
        return ans


class DiscreteBetaDistribution():
    def __init__(self, low, high, shape, bound, reduction_factor):
        self.low = low
        self.high = high
        self.shape = shape

        self.bound = bound
        b = int(reduction_factor / 2)
        ratio = torch.arange(-b, b + 1, device=torch.device('cuda'))
        self.choice = ratio / reduction_factor
        self.anchors = torch.arange(self.choice.shape[0] + 1) / (self.choice.shape[0])
        self.choice = self.choice * bound

        self.alpha = torch.tensor([0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0], device=torch.device('cuda'))
        self.beta = torch.tensor([0.3, 0.5, 1.0, 2.0, 3.0, 4.0, 5.0], device=torch.device('cuda'))

    def sample(self):
        a_p = np.random.randint(7)
        b_p = np.random.randint(7)
        alpha = self.alpha[a_p]
        beta = self.beta[b_p]
        alpha = torch.full(self.shape, alpha, device=torch.device('cuda'))
        beta = torch.full(self.shape, beta, device=torch.device('cuda'))

        m = Beta(alpha, beta)

        ans = m.sample()

        for i in range(self.anchors.shape[0] - 1):
            a1 = self.anchors[i]
            a2 = self.anchors[i + 1]
            p = torch.logical_and(ans >= a1, ans <= a2)
            ans[p] = i
        ans = ans.long()
        ans = self.choice[ans]

        return ans


class BoundNormal():
    def __init__(self, mean, sigma, bound_value):
        self.m = torch.distributions.normal.Normal(mean, sigma)
        self.mean = mean
        self.sigma = sigma
        self.bound_value = bound_value

    def sample(self):
        ans = self.m.sample()
        upper_bound = self.bound_value / 2
        lower_bound = - self.bound_value / 2
        upper_mask = torch.gt(ans, upper_bound)
        lower_mask = torch.lt(ans, lower_bound)

        # n_upper_pos = upper_mask.sum()
        # n_lower_pos = lower_mask.sum()
        # n_total_pos = torch.ones(ans.shape).sum()
        #
        # if n_lower_pos > 0 or n_lower_pos < 0:
        #     logger = get_root_logger()
        #     logger.debug(f'mean:{self.mean}, sigma:{self.sigma}, n_upper_pos: {n_upper_pos}, n_lower_pos: {n_lower_pos}, total_pos: {n_total_pos}')

        ans[upper_mask] = upper_bound[upper_mask]
        ans[lower_mask] = lower_bound[lower_mask]

        return ans


class BoundUniform():
    def __init__(self, low, high, bound_value):
        self.m = Uniform(low, high)
        self.low = low
        self.high = high
        self.bound_value = bound_value

    def sample(self):
        ans = self.m.sample()
        upper_bound = self.bound_value / 2
        lower_bound = - self.bound_value / 2
        upper_mask = torch.gt(ans, upper_bound)
        lower_mask = torch.lt(ans, lower_bound)

        # n_upper_pos = upper_mask.sum()
        # n_lower_pos = lower_mask.sum()
        # n_total_pos = torch.ones(ans.shape).sum()
        #
        # if n_lower_pos > 0 or n_lower_pos < 0:
        #     logger = get_root_logger()
        #     logger.debug(f'low:{self.low}, high:{self.high}, n_upper_pos: {n_upper_pos}, n_lower_pos: {n_lower_pos}, total_pos: {n_total_pos}')

        ans[upper_mask] = upper_bound[upper_mask]
        ans[lower_mask] = lower_bound[lower_mask]

        return ans


class UniformElements():
    def __init__(self, low, high, params_shape):
        self.m = Uniform(low, high)
        self.low = low
        self.high = high
        self.params_shape = params_shape

    def sample(self):
        # TODO finish element-wise Uniform distribution
        return


def pnorm(data, p):
    normB = torch.norm(data, 2, dim=1)
    for i in range(data.size(0)):
        data[i] = data[i] / torch.pow(normB[i], p)
    return data


class AvgDict():
    def __init__(self):
        self.dict = {}

    def add_dict(self, d):
        for key, value in d.items():
            if self.dict.get(key, None) is None:
                self.dict[key] = Averager()
                self.dict[key].add(value)
            else:
                self.dict[key].add(value)

    def get_ordinary_dict(self):
        d = {key: value.item() for key, value in self.dict.items()}
        return d


def assign_weights(m, weights):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    with torch.no_grad():
        for param in state_dict.keys():
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            # print(param, index)
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] = nn.Parameter(torch.from_numpy(weights[index:index + param_count].reshape(param_shape)))
            index += param_count
    m.load_state_dict(state_dict)
    return m


def flatten_params(m, numpy_output=True):
    total_params = []
    for param in m.parameters():
        total_params.append(param.view(-1))
    total_params = torch.cat(total_params)
    if numpy_output:
        return total_params.cpu().detach().numpy()
    return total_params


def get_cifar100_testloader(dataset_folder, transform, eval_batch_size):
    """
    Get CIFAR100 test loader.
    """
    testset = torchvision.datasets.CIFAR100(root=dataset_folder, train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=eval_batch_size, shuffle=False)
    return testloader


def eval_single_epoch(net, loader):
    net = net.to(DEVICE)
    net.eval()
    test_loss = 0
    correct = 0
    count = 0  # because of sampler
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    predictions = []
    ground_truths = []
    with torch.no_grad():
        for data, target in loader:
            data = data.to(DEVICE)
            target = target.to(DEVICE)[:, 0]
            count += len(target)
            output = net(data)
            test_loss += criterion(output, target).item() * len(target)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
            predictions.append(pred)
            ground_truths.append(target)
    ground_truths = torch.stack(ground_truths).cpu().view(-1).numpy()
    predictions = torch.stack(predictions).cpu().view(-1).numpy()
    conf_matrix = confusion_matrix(ground_truths, predictions)
    per_class_acc = conf_matrix.diagonal() / conf_matrix.sum(axis=0)
    test_loss /= count
    correct = correct.to('cpu')
    avg_acc = 100.0 * float(correct.numpy()) / count
    return {'accuracy': avg_acc, 'per_class_accuracy': per_class_acc}


def assign_weights(m, weights):
    state_dict = m.state_dict(keep_vars=True)
    index = 0
    with torch.no_grad():
        for param in state_dict.keys():
            if 'running_mean' in param or 'running_var' in param or 'num_batches_tracked' in param:
                continue
            # print(param, index)
            param_count = state_dict[param].numel()
            param_shape = state_dict[param].shape
            state_dict[param] = nn.Parameter(torch.from_numpy(weights[index:index + param_count].reshape(param_shape)))
            index += param_count
    m.load_state_dict(state_dict)
    return m


def flatten_params(m, numpy_output=True):
    total_params = []
    for param in m.parameters():
        total_params.append(param.view(-1))
    total_params = torch.cat(total_params)
    if numpy_output:
        return total_params.cpu().detach().numpy()
    return total_params


def evaluate(model, data_loader, debias_weight=1, bias_weight=1):
    target_attr_idx = 0
    bias_attr_idx = 1
    attr_dims = [10, 10]
    num_classes = 10
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model.eval()
    attrwise_acc_meter = MultiDimAverageMeter(attr_dims)
    for data, attr in tqdm(data_loader, leave=False):
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