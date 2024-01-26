import io
import torch
import numpy as np


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
        self.parameter[index] = self.alpha * self.parameter[index] + (1-self.alpha*self.updated[index]) * data
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

def make_borders_black(image_tensor):
    # Get the dimensions of the image tensor
    height, width = image_tensor.shape[-2], image_tensor.shape[-1]

    # Set the top border to black
    image_tensor[:, :, :5, :] = 0

    # Set the bottom border to black
    image_tensor[:, :, -5:, :] = 0

    # Set the left border to black
    image_tensor[:, :, :, :5] = 0

    # Set the right border to black
    image_tensor[:,: , :, -5:] = 0

    return image_tensor