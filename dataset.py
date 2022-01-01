import pandas as pd
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import os
from PIL import Image

from RandAugment import RandAugment

IMG_SIZE = 384
transforms_train = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomResizedCrop(IMG_SIZE),
        transforms.ToTensor(),
        # ImageNet mean and std
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

# RandAugment
# transforms = ['identity','rotate','autocontrast','translateX']
# N,M
# N: number of augmentation transformations to apply sequentially (random choose from the list of transformations)
# M: magnitude for all the transformations [0,30] -> val = (float(self.m) / 30) * float(maxval - minval) + minval

transforms_train.transforms.insert(0, RandAugment(2,15))

transforms_test = transforms.Compose(
    [
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        # ImageNet mean and std
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ]
)

DATA_PATH_PATCHES = "../dataset/colon_big_data"

class PatchesDataset(torch.utils.data.Dataset):
    """
    Helper Class to create the pytorch dataset
    """

    def __init__(self, df, data_path=DATA_PATH_PATCHES, mode="train", transforms=None):
        super().__init__()
        self.df_data = df.values
        self.data_path = data_path
        self.transforms = transforms
        self.mode = mode
        if mode == "train":
            self.data_dir = "train_images"
            self.transforms = transforms_train
        elif mode == "valid":
            self.data_dir = "train_images"
            self.transforms = transforms_test
        else:
            self.data_dir = "test_images"
            self.transforms = transforms_test

    def __len__(self):
        return len(self.df_data)

    def __getitem__(self, index):
        img_name, label = self.df_data[index]
        img_path = os.path.join(self.data_path, self.data_dir, img_name)
        img = Image.open(img_path).convert("RGB")

        if self.transforms is not None:
            image = self.transforms(img)

        return image, label
