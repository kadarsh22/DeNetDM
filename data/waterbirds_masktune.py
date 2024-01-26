from torchvision import transforms
import os
import torch
from data.waterbirds_dataset import WaterbirdsDataset

def get_waterbird_dataloader(data_dir, train_batch, test_batch, workers = 0):
    scale = 256.0/224.0
    target_resolution = (224, 224)
    transform_test = transforms.Compose([
            transforms.Resize(
                (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
            transforms.CenterCrop(target_resolution),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [
                                    0.229, 0.224, 0.225])
        ])
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(
            target_resolution,
            scale=(0.7, 1.0),
            ratio=(0.75, 1.3333333333333333),
            interpolation=2),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [
                                0.229, 0.224, 0.225])
    ])
    transform_data_to_mask = transforms.Compose([
        transforms.Resize(
            (int(target_resolution[0]*scale), int(target_resolution[1]*scale))),
        transforms.CenterCrop(target_resolution),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [
                                0.229, 0.224, 0.225])
    ])
    train_dataset = WaterbirdsDataset(raw_data_path=data_dir, split='train', transform=transform_train)
    print(len(train_dataset))
    print(len(train_dataset.places))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=train_batch, shuffle=True, num_workers=workers)
    
    val_dataset = WaterbirdsDataset(raw_data_path=data_dir, split='val', transform=transform_test)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=test_batch, shuffle=True, num_workers=workers,drop_last=True)
   
    test_dataset = WaterbirdsDataset(raw_data_path=data_dir, split='test', transform=transform_test, return_places=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=test_batch, shuffle=True, num_workers=workers,drop_last=True)
    return train_loader, val_loader , test_loader

