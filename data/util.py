import os
import numpy as np
import torch
from torch.utils.data.dataset import Dataset, Subset
from torch.utils.data import Sampler
from torchvision import transforms as T
from torchvision.datasets.celeba import CelebA
from data.attr_dataset import AttributeDataset
from PIL import Image
from glob import glob


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


class ZippedDataset(Dataset):
    def __init__(self, datasets):
        super(ZippedDataset, self).__init__()
        self.dataset_sizes = [len(d) for d in datasets]
        self.datasets = datasets

    def __len__(self):
        return max(self.dataset_sizes)

    def __getitem__(self, idx):
        items = []
        for dataset_idx, dataset_size in enumerate(self.dataset_sizes):
            items.append(self.datasets[dataset_idx][idx % dataset_size])

        item = [torch.stack(tensors, dim=0) for tensors in zip(*items)]

        return item


class bFFHQDataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None):
        super(bFFHQDataset, self).__init__()
        self.transform = transform
        self.root = root

        self.image2pseudo = {}
        self.image_path_list = image_path_list

        if split == 'train':
            self.align = glob(os.path.join(root, split, 'align', "*", "*"))
            self.conflict = glob(os.path.join(root, split, 'conflict', "*", "*"))
            self.data = self.align + self.conflict

        elif split == 'valid':
            self.data = glob(os.path.join(root, split, "*"))

        elif split == 'test':
            self.data = glob(os.path.join(root, split, "*"))
            data_conflict = []
            for path in self.data:
                target_label = path.split('/')[-1].split('.')[0].split('_')[1]
                bias_label = path.split('/')[-1].split('.')[0].split('_')[2]
                if target_label != bias_label:
                    data_conflict.append(path)
            # self.data = data_conflict  ## for evaluating only on conflicting points ##TODO
        self.attr = torch.stack([torch.LongTensor([int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])]) for index in range(len(self.data))])
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        attr = self.attr[index]
        image = Image.open(self.data[index]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)

        return self.data[index], image, attr


transforms = {
    "ColoredMNIST": {
        "train": T.Compose([T.ToPILImage(), T.Resize((28, 28)), T.ToTensor()]),
        "eval": T.Compose([T.ToPILImage(), T.Resize((28, 28)), T.ToTensor()])
    },
    "CorruptedCIFAR10": {
        "train": T.Compose(
            [
                T.ToPILImage(),
                T.RandomCrop(32, padding=4),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
        "eval": T.Compose(
            [
                T.ToTensor(),
                T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
            ]
        ),
    },
    "bFFHQ": {
        "train": T.Compose([
            T.Resize((224, 224)),
            T.RandomCrop(224, padding=4),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        ),
        "valid": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        ),
        "eval": T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]
        )
    },
    "CelebA": {
        "train": T.Compose(
            [
                T.Resize((224, 224)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
        "eval": T.Compose(
            [
                T.Resize((224, 224)),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        ),
    },
}


def get_dataset(dataset_tag, data_dir, dataset_split, transform_split):
    dataset_category = dataset_tag.split("-")[0]
    root = os.path.join(data_dir, dataset_tag)
    transform = transforms[dataset_category][transform_split]
    if dataset_tag == "bFFHQ":
        dataset_split = "test" if (
                    dataset_split == "eval") else dataset_split  # different for bffhq and cmnist, ccifar10 ##todo
        dataset = bFFHQDataset(root=root, split=dataset_split, transform=transform)
    else:
        dataset_split = "valid" if (dataset_split == "eval") else dataset_split
        dataset = AttributeDataset(root=root, split=dataset_split, transform=transform)

    return dataset
