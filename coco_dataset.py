import pdb
import torch
import numpy as np
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import Dataset, Subset
import os, sys, glob, time, subprocess
import h5py
from PIL import Image


def get_transform_coco(num_classes):
    if num_classes == 2:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.37493148, 0.21778074, 0.23026027],
                [0.10265636, 0.20582178, 0.21669184])
        ])
    elif num_classes == 10:
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                [0.34489644, 0.30505344, 0.3762387 ],
                [0.26109827, 0.2823534, 0.32291284])
        ])
    else:
        raise Exception
    return transform


class COCODataset(object):
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

def get_coco_handles(num_classes=2, sp_ratio_list=None, noise_ratio=0, dataset='colour', train_test=None, flags=None):
    data_dir = "data_dir/SPCOCO/coco"
    if dataset == 'places':
        dataset_name = 'cocoplaces_vf_{}_{}'.format(num_classes, confounder_strength)
        original_dirname = os.path.join(data_dir, dataset_name)
    elif dataset == 'colour':

        if flags.grayscale_model:
            dataset_name = 'cocogrey__class_{}_noise_{}_sz_{}'.format(
                num_classes,
                noise_ratio,
                flags.image_scale)
        else:
            dataset_name = 'cococolours_vf_num_class_{}_sp_{}_noise_{}_sz_{}'.format(
                num_classes,
                "_".join([str(x) for x in sp_ratio_list]),
                noise_ratio,
                flags.image_scale)
        original_dirname = os.path.join(data_dir, dataset_name)



    dirname = os.path.join(data_dir,  dataset_name)

    print('Copying data over, this will be worth it, be patient ...', end=' ')
    subprocess.call(['rsync', '-r', original_dirname, data_dir])
    print('Done!')

    if train_test == "train":
        train_file = h5py.File(dirname+'/train.h5py', mode='r')
        # print("what", dirname+'/train.h5py')
        return (train_file, None, None, None, None)
    elif train_test == "test":
        id_test_file = h5py.File(dirname+'/idtest.h5py', mode='r')
        return (id_test_file, None, None, None, None)
    else:
        raise Exception

def get_spcoco_dataset(sp_ratio_list=None, noise_ratio=None, num_classes=None, flags=None):
    coco_transform = get_transform_coco(2)
    train_data_handle, _, _, _, _ = get_coco_handles(
        num_classes=num_classes,
        sp_ratio_list=sp_ratio_list,
        noise_ratio=noise_ratio,
        dataset='colour', train_test="train", flags=flags)
    # shuffle train
    train_x_array = train_data_handle["images"].value
    train_y_array = train_data_handle["y"].value
    train_env_array = train_data_handle["e"].value
    train_sp_array = train_data_handle["g"].value
    perm = np.random.permutation(
        range(train_x_array.shape[0]))
    coco_dataset_train = COCODataset(
        x_array=train_x_array[perm],
        y_array=train_y_array[perm],
        env_array=train_env_array[perm],
        transform=coco_transform,
        sp_array=train_sp_array[perm])
    test_data_handle, _, _, _, _ = get_coco_handles(
        num_classes=num_classes,
        sp_ratio_list=sp_ratio_list,
        noise_ratio=noise_ratio,
        dataset='colour',
        train_test="test",
        flags=flags)
    coco_dataset_test = COCODataset(
        x_array=test_data_handle["images"].value,
        y_array=test_data_handle["y"].value,
        env_array=test_data_handle["e"].value,
        transform=coco_transform,
        sp_array=test_data_handle["g"].value)
    return coco_dataset_train, coco_dataset_test

