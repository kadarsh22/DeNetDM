
from torch.utils.data.dataset import Dataset
from PIL import Image
from glob import glob
import os
import torch

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

        return image, attr