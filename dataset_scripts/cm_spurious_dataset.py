import os
import random
from mnistcifar_utils import get_mnist_cifar_env
import pdb
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset

class SubDataset(object):
    def __init__(self, x_array, y_array, env_array, transform, sp_array=None):
        self.x_array = x_array
        self.y_array = y_array[:, None]
        self.env_array = env_array[:, None]
        self.sp_array = sp_array[:, None]
        self.transform = transform
        assert len(self.x_array) == len(self.y_array)
        assert len(self.y_array) == len(self.env_array)

    def __len__(self):
        return len(self.x_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.env_array[idx]
        if self.sp_array is not None:
            sp = self.sp_array[idx]
        else:
            sp = None
        img = self.x_array[idx]
        img = (img *255).astype(np.uint8)
        img = img.transpose(1, 2, 0)
        img = Image.fromarray(img)
        x = self.transform(img)

        return x,y,g, sp

class SpuriousValDataset(Dataset):
    def __init__(self, val_dataset):
        self.val_dataset = val_dataset

    def __len__(self):
        return len(self.val_dataset)

    def __getitem__(self, idx):
        x, y, g, sp = self.val_dataset.__getitem__(idx)
        g = g * 0
        return x, y, g,sp

class CifarMnistSpuriousDataset(Dataset):
    def __init__(self, train_num,test_num,cons_ratios, cifar_classes=(1, 9),train_envs_ratio=None, label_noise_ratio=None, augment_data=True, color_spurious=False, transform_data_to_standard=1, oracle=0):
        self.cons_ratios=cons_ratios
        self.train_num = train_num
        self.test_num = test_num
        self.train_envs_ratio=train_envs_ratio
        self.augment_data = augment_data
        self.oracle = oracle
        self.x_array, self.y_array, self.env_array, self.sp_array= \
            get_mnist_cifar_env(
                train_num=train_num,
                test_num=test_num,
                cons_ratios=cons_ratios,
                train_envs_ratio=train_envs_ratio,
                label_noise_ratio=label_noise_ratio,
                cifar_classes=cifar_classes,
                color_spurious=color_spurious,
                oracle=oracle)
        self.feature_dim = self.x_array.reshape([self.x_array.shape[0], -1]).shape[1]
        self.transform_data_to_standard = transform_data_to_standard
        self.y_array = self.y_array.astype(np.int64)
        self.split_array = self.env_array
        self.n_train_envs = len(self.cons_ratios) - 1
        self.split_dict = {
            "train": range(len(self.cons_ratios) - 1),
            "val": [len(self.cons_ratios) - 1],
            "test": [len(self.cons_ratios) - 1]}
        self.n_classes = 2
        self.train_transform = get_transform_cub(transform_data_to_standard=self.transform_data_to_standard, train=True, augment_data=self.augment_data)
        self.eval_transform = get_transform_cub(transform_data_to_standard=self.transform_data_to_standard, train=False, augment_data=False)

    def __len__(self):
        return len(self.y_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.env_array[idx]

        img = self.x_array[idx]
        sp = self.sp_array[idx]
        # Figure out split and transform accordingly
        if self.split_array[idx] in self.split_dict['train']:
            img = self.train_transform(img)
        elif self.split_array[idx] in self.split_dict['val'] + self.split_dict['test']:
            img = self.eval_transform(img)
        x = img

        return x,y,g, sp

    def get_splits(self, splits, train_frac=1.0):
        subsets = []
        for split in splits:
            assert split in ('train','val','test'), split+' is not a valid split'
            mask = np.isin(self.split_array, self.split_dict[split])
            num_split = np.sum(mask)
            indices = np.where(mask)[0]
            if split == "train":
                subsets.append(
                    SubDataset(
                        x_array=self.x_array[indices],
                        y_array=self.y_array[indices],
                        env_array=self.env_array[indices],
                        sp_array=self.sp_array[indices],
                        transform=self.train_transform
                    ))
            else:
                subsets.append(
                    SpuriousValDataset(
                        SubDataset(
                            x_array=self.x_array[indices],
                            y_array=self.y_array[indices],
                            env_array=self.env_array[indices],
                            sp_array=self.sp_array[indices],
                            transform=self.train_transform
                        )))

        self.subsets = subsets
        return tuple(subsets)

def get_data_loader_cifarminst(batch_size, train_num, test_num, cons_ratios, train_envs_ratio, label_noise_ratio=None, augment_data=True, cifar_classes=(1, 9), color_spurious=False, transform_data_to_standard=1, oracle=0):
    np.random.seed(1)
    random.seed(1) # Fix the random seed of dataset
    spdc = CifarMnistSpuriousDataset(
        train_num=train_num,
        test_num=test_num,
        cons_ratios=cons_ratios,
        train_envs_ratio=train_envs_ratio,
        label_noise_ratio=label_noise_ratio,
        augment_data=augment_data,
        cifar_classes=cifar_classes,
        color_spurious=color_spurious,
        transform_data_to_standard=transform_data_to_standard,
        oracle=oracle)
    train_dataset, val_dataset, test_dataset = spdc.get_splits(
        splits=['train','val','test'])
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4)
    return spdc, train_loader, val_loader, test_loader, train_dataset, val_dataset, test_dataset

def get_transform_cub(transform_data_to_standard, train, augment_data):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    return transform

if __name__ == "__main__":
    spd, train_loader, val_loader, test_loader, train_data, val_data, test_data = \
    get_data_loader_cifarminst(
        batch_size=100,
        train_num=10000,
        test_num=1800,
        cons_ratios=[0.99,0.8,0.1],
        train_envs_ratio=[0.5,0.5],
        label_noise_ratio=0.1,
        augment_data=False,
        cifar_classes=(2,1),
        color_spurious=1,
        transform_data_to_standard=1)
    # spdc, train_loader, val_loader, test_loader, _, _, _ = get_data_loader_cifarminst(
    #     batch_size=100,
    #     train_num=10000,
    #     test_num=2000,
    #     cons_ratios=[0.99, 0.8, 0.1])
    # print(len(train_loader), len(val_loader), len(test_loader))
    # torch.manual_seed(0)
    loader_iter = iter(train_loader)
    x, y, g = loader_iter.__next__()
    print(y)
    # x, y, g = loader_iter.__next__()
    # print(g)
    # x, y, g = iter(test_loader).__next__()
    # print(g)
    # print(x.shape, y.shape, g.shape)
    # print("y", y)
    # print(g)

    # x, y, g = iter(val_loader).__next__()
    # print(x.shape, y.shape, g.shape)
    # print("y", y)
    # print(g)

    # x, y, g = iter(test_loader).__next__()
    # print(x.shape, y.shape, g.shape)
    # print("y", y)
    # print(g)
