import os
from torch.utils.data.dataset import Dataset
from data.attr_dataset import AttributeDataset
from data.bffhq import bFFHQDataset
from data.transforms import transforms


class IdxDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return (idx, *self.dataset[idx])


def get_dataset(dataset_tag, data_dir, dataset_split, transform_split):
    dataset_category = dataset_tag.split("-")[0]
    root = os.path.join(data_dir, dataset_tag)
    transform = transforms[dataset_category][transform_split]
    if dataset_tag == "bFFHQ":
        dataset_split = "test" if (dataset_split == "eval") else dataset_split
        # different for bffhq and cmnist, ccifar10 ##todo
        dataset = bFFHQDataset(root=root, split=dataset_split, transform=transform)
    else:
        dataset_split = "valid" if (dataset_split == "eval") else dataset_split
        dataset = AttributeDataset(root=root, split=dataset_split, transform=transform)

    return dataset
