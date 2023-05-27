import os
import torch
from torch.utils.data.dataset import Dataset, Subset
from torchvision import transforms as T
from glob import glob
from PIL import Image


class CIFAR10Dataset(Dataset):
    def __init__(self, root, split, transform=None, image_path_list=None, use_type0=None, use_type1=None):
        super(CIFAR10Dataset, self).__init__()
        self.transform = transform
        self.root = root
        self.image2pseudo = {}
        self.image_path_list = image_path_list
        root = '/home/user/PycharmProjects/DebiasingStyleGAN2/data/cifar10c/0.5pct'
        if split == 'train':
            self.align = glob(os.path.join(root, 'align', "*", "*"))
            self.conflict = glob(os.path.join(root, 'conflict', "*", "*"))
            self.data = self.align + self.conflict

        elif split == 'valid':
            self.data = glob(os.path.join(root, split, "*", "*"))

        elif split == 'test':
            self.data = glob(os.path.join(root, '../test', "*", "*"))

        self.images = [Image.open(self.data[index]).convert('RGB') for index in range(len(self.data))]
        self.labels = torch.stack(
            [torch.tensor(int(self.data[index].split('_')[-2]), dtype=torch.long) for index, _ in enumerate(self.data)])
        self.bias = torch.stack(
            [torch.tensor(int(self.data[index].split('_')[-1].split('.')[0]), dtype=torch.long) for index, _ in
             enumerate(self.data)])
        self.attr = torch.cat((self.labels.view(-1, 1), self.bias.view(-1, 1)), dim=1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.transform is not None:
            image = self.transform(self.images[index])

        return image, self.attr[index]
