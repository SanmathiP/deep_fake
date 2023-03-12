"""Utility module to make Pytorch based Dataset and also 
a GPU version of the DataLoader. 
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torchvision.utils import make_grid
from torch.utils.data.dataloader import default_collate

import cv2
import random
import PIL
import json
import os
import numpy as np
import matplotlib.pyplot as plt

# from glob import glob

from copy import copy


def augmentations(val):
    """Augmentation settings for training and testing dataset.

    Parameters
    ----------
    val : bool
       Flag when set True, setups augmentation for validation set,
       else sets up for training set.

    Returns
    -------
    torchvision.transform.Compose()
        Augmentation setting for the dataset.
    """
    if val == True:
        aug = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
    else:
        aug = transforms.Compose(
            [
                transforms.RandomRotation(0.1),
                transforms.RandomHorizontalFlip(0.1),
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return aug


def make_train_val_dataset(root_dir, split_ratio=0.8, ensemble=False):
    """Creates training and validation Pytorch datasets.

    Parameters
    ----------
    root_dir : str
        Main dataset directory.

    split_ratio : float (Default:0.8)
        Percentage of the dataset to be used for training.

    ensemble : bool (Default:False)
        Flag when set to True, makes Ensemble Dataset, else make Normal Dataset.
    """
    if ensemble:
        dataset = EnsembleDataset(root_dir)
    else:
        dataset = ImageFolder(root=root_dir)
    dataset_length = len(dataset)
    all_indices = list(range(dataset_length))
    random.shuffle(all_indices)
    train_dataset_length = int(dataset_length * split_ratio)
    train_indices, val_indices = (
        all_indices[:train_dataset_length],
        all_indices[train_dataset_length:],
    )
    train_dataset, val_dataset = Subset(dataset, train_indices), Subset(
        dataset, val_indices
    )
    train_dataset.dataset.transform = augmentations(val=False)
    val_dataset.dataset = copy(dataset)
    val_dataset.dataset.transform = augmentations(val=True)
    return train_dataset, val_dataset


class EnsembleDataset(Dataset):
    """Dataset which takes into account all the modalities.

    Attributes
    ----------
    root_dir : str
            Main directory which holds the different modalities directory.


    Methods
    -------
    _img2tensor
            Given a path, applies augmention and returns a pytorch tensor.
    """

    def __init__(self, root_dir):
        """Cosntructor."""
        super().__init__()
        self.rgb_dir = root_dir + "/rgb"
        self.real_imgs = os.listdir(self.rgb_dir + "/real")
        self.fake_imgs = os.listdir(self.rgb_dir + "/fake")
        self.albedo_dir = root_dir + "/albedo"
        self.normal_map_dir = root_dir + "/normal_map"
        self.shadow_dir = root_dir + "/shadow"
        self.shading_dir = root_dir + "/shading"
        self.lighting_dir = root_dir + "/lighting"

    def _img2tensor(self, path):
        """Given an image path, loads the image, applies augmentation
        and returns a pytorch tensor.

        Parameters
        ----------
        path : str
            Path of an image

        Returns
        -------
        torch.Tensor
                Image tensor after augmentation.
        """
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = PIL.Image.fromarray(img)
        img = self.transform(img)
        img = img.unsqueeze(0)
        return img

    def __getitem__(self, idx):
        """Access a particular datapoint within the dataset.

        Parameters
        ----------
        idx : int
            Datapoint index.

        Returns
        -------
        tuple(torch.Tensor , torch.Tensor)
            A datapoint with the image transformed as tensor and its
            corresponding label.

        """
        if idx < len(self.real_imgs):
            label = 0
            label_name = "real"
            img_name = self.real_imgs[idx]
        else:
            label = 1
            label_name = "fake"
            idx -= len(self.real_imgs)
            img_name = self.fake_imgs[idx]

        rgb_tensor = self._img2tensor(self.rgb_dir + "/" + label_name + "/" + img_name)
        albedo_tensor = self._img2tensor(
            self.albedo_dir + "/" + label_name + "/" + img_name
        )
        normal_map_tensor = self._img2tensor(
            self.normal_map_dir + "/" + label_name + "/" + img_name
        )
        shadow_tensor = self._img2tensor(
            self.shadow_dir + "/" + label_name + "/" + img_name
        )
        shading_tensor = self._img2tensor(
            self.shading_dir + "/" + label_name + "/" + img_name
        )
        lighting_tensor = self._img2tensor(
            self.lighting_dir + "/" + label_name + "/" + img_name
        )

        combined_tensor = torch.cat(
            (
                rgb_tensor,
                albedo_tensor,
                normal_map_tensor,
                shadow_tensor,
                shading_tensor,
                lighting_tensor,
            ),
            dim=1,
        )

        return combined_tensor, label

    def __len__(self):
        return len(self.real_imgs) + len(self.fake_imgs)


def get_device():
    """
    Sets the default device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def shift_to_device(data, device):
    """
    Shifts data to device.
    """

    if isinstance(data, (list, tuple)):
        return [shift_to_device(each_data, device) for each_data in data]

    return data.to(device)


class GPUDataLoader:
    """
    Returns a GPU based DataLoader given a
    Pytorch Dataloader and a default device.
    """

    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __iter__(self):
        for batch in self.data:
            yield shift_to_device(batch, self.device)

    def __len__(self):
        return len(self.data)


def custom_collate(batch):
    """Collate function to alter the specification of the batch shape."""
    imgs, labels = default_collate(batch)
    imgs = imgs.reshape(
        imgs.shape[0] * imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4]
    )
    return [imgs, labels]
