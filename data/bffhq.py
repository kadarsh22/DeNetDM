import torch
import os


from PIL import Image
from torch.utils.data.dataset import Dataset
from glob import glob


class bFFHQDataset(Dataset):
    base_folder = 'bffhq'
    target_attr_index = 0
    bias_attr_index = 1

    def __init__(self, root, split, type=None, transform=None):
        super(bFFHQDataset, self).__init__()
        self.transform = transform
        root = os.path.join(root, self.base_folder)

        self.root = root
        print(root)

        if split == 'train':
            self.align = glob(os.path.join(root, split, 'align', "*", "*"))
            self.conflict = glob(os.path.join(root, split, 'conflict', "*", "*"))
            self.data = self.align + self.conflict
            if type == 'align':
                self.data = self.align
            elif type == 'conflict':
                self.data = self.conflict
        
        elif split == 'valid':
            self.data = glob(os.path.join(root, split, "*"))

        elif split == 'test':
            self.data = glob(os.path.join(root, split, "*"))
        
        self.images = [self.transform(Image.open(self.data[index]).convert('RGB')) for index in range(len(self.data))]
        self.labels = torch.stack([torch.LongTensor([int(self.data[index].split('_')[-2]), int(self.data[index].split('_')[-1].split('.')[0])])  for index, _ in enumerate(self.data)])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        return self.images[index], self.labels[index]