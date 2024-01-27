import os
import torch
from torch.utils.data.dataset import Dataset
from glob import glob
from PIL import Image
import pandas as pd
import numpy as np



class CelebADataset(Dataset):
    """
    CelebA dataset (already cropped and centered).
    NOTE: metadata_df is one-indexed.
    """
    def __init__(self, root, name='celebA', split='train', transform=None, conflict_pct=5):
        self.name = name
        self.transform = transform
        self.root = root
        self.split_dict = {
            "train": 0,
            "val": 1,
            "test": 2,
        }

        self.header_dir = root
        self.data_dir = os.path.join(self.header_dir, "img_align_celeba")

        print(f"Reading '{os.path.join(self.header_dir, 'metadata_blonde_subsampled.csv')}'")
        self.attrs_df = pd.read_csv(os.path.join(self.header_dir, "metadata_blonde_subsampled.csv"))
        self.filename_array = self.attrs_df["image_id"].values
        self.split_array = self.attrs_df["split"].values

        self.attrs_df = self.attrs_df.drop(labels="image_id", axis="columns")
        self.attr_names = self.attrs_df.columns.copy()
        self.attrs_df = self.attrs_df.values
        self.attrs_df[self.attrs_df == -1] = 0


        target_idx = self.attr_idx('Blond_Hair')
        self.y_array = self.attrs_df[:, target_idx]

        confounder_idx = self.attr_idx('Male')
        self.confounder_array = self.attrs_df[:, confounder_idx]

        self.split_token = 0 if split == "train" else 2
        mask = self.split_array == self.split_token

        num_split = np.sum(mask)
        indices = np.where(mask)[0]
        self.filename_array = self.filename_array[indices]
        self.y_array = torch.tensor(self.y_array[indices]).long()
        self.confounder_array = torch.tensor(self.confounder_array[indices]).long()
        self.attr = torch.stack([torch.LongTensor([int(self.y_array[index]), int(self.confounder_array[index])]) for index in range(len(self.filename_array))])
        self.indices = indices


    def attr_idx(self, attr_name):
        return self.attr_names.get_loc(attr_name)

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, index):
        # attr = torch.LongTensor([int(self.y_array[index]), int(self.confounder_array[index])])

        attr = self.attr[index]
        img_filename = os.path.join(self.data_dir,
                                    self.filename_array[index])
        image = Image.open(img_filename).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        return image, attr