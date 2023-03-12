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

import json
from copy import copy
import random

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
                transforms.RandomRotation(0.3),
                transforms.RandomHorizontalFlip(0.2),
                transforms.RandomAdjustSharpness(1.5),
                transforms.ToTensor(),
                transforms.Resize((224, 224)),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )

    return aug


def make_train_val_dataset(root_dir, metadata="deep_fake/metadata.json", split_ratio=0.8, ensemble=True):
    """Creates training and validation Pytorch datasets.

    Parameters
    ----------
    root_dir : str 
        Main dataset directory.
    
    metadata : str (Default : "deep_fake/metadata.json")
        Filepath to .json file which contains all the necessary meta-information 
        of the dataset. Necessary to capture the label information for each data.

    split_ratio : float (Default:0.8) 
        Percentage of the dataset to be used for training.

    ensemble : bool
        Flag when set to True, makes Ensemble Dataset, else make Normal Dataset. 
    """
    if ensemble:
        dataset = EnsembleDataset(root_dir, metadata)
    else:
        dataset = NormalDataset(root_dir , metadata)
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
    root_dir : list[str,str,str,str,str,str]
            List of directories which holds the different modalities.
            The orientation is:
            [albedo_dir,normal_map_dir,shadow_dir,shading_dir,lighting_dir,rgb_dir].

    all_imgs : list[...]
           List of image names for a balanced dataset. 

    Methods
    -------
    _img2tensor
            Given a path, applies augmention and returns a pytorch tensor.
    """
    def __init__(self, root_dir, metadata):
        """Constructor.
        """
        super().__init__()
        self.rgb_dir = root_dir[5]
        self.albedo_dir = root_dir[0]
        self.normal_map_dir = root_dir[1]
        self.shadow_dir = root_dir[2]
        self.shading_dir = root_dir[3]
        self.lighting_dir = root_dir[4]
        with open(metadata) as json_file:
            # .load creates a python dictionary
            self.metadata = json.load(json_file)
        real_imgs = []
        fake_imgs = []
        for each_img in os.listdir(self.lighting_dir):
            label = self.metadata[str(each_img[:-5]) + ".mp4"]["label"]
            if label == "FAKE":
                fake_imgs.append(each_img)
            else:
                real_imgs.append(each_img)

        num_real_data = len(real_imgs)
        fake_data = random.sample(fake_imgs, num_real_data)
        real_imgs.extend(fake_data)

        self.all_imgs = copy(real_imgs)
        random.shuffle(self.all_imgs)


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
        img_name = self.all_imgs[idx]

        label_name = self.metadata[str(img_name[:-5]) + ".mp4"]["label"]

        label = 0

        if label_name == "FAKE":
            label = 1

        rgb_tensor = self._img2tensor(self.rgb_dir + "/" + img_name)
        albedo_tensor = self._img2tensor(self.albedo_dir + "/" + img_name)
        normal_map_tensor = self._img2tensor(self.normal_map_dir + "/" + img_name)
        shadow_tensor = self._img2tensor(self.shadow_dir + "/" + img_name)
        shading_tensor = self._img2tensor(self.shading_dir + "/" + img_name)
        lighting_tensor = self._img2tensor(self.lighting_dir + "/" + img_name)

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
        """Returns the length of the dataset."""
        return len(self.all_imgs)

class NormalDataset(Dataset):
    """Dataset which takes into account a single modality.

    Attributes
    ----------
    root_dir : str
            Directories which holds the modality.

    all_imgs : list[...]
           List of image names for a balanced dataset. 

    Methods
    -------
    _img2tensor
            Given a path, applies augmention and returns a pytorch tensor.
    """
    def __init__(self, root_dir, metadata):
        super().__init__()
        self.root_dir = root_dir 
        with open(metadata) as json_file:
            # .load creates a python dictionary
            self.metadata = json.load(json_file)
        real_imgs = []
        fake_imgs = []
        for each_img in os.listdir(self.root_dir):
            label = self.metadata[str(each_img[:-5]) + ".mp4"]["label"]
            if label == "FAKE":
                fake_imgs.append(each_img)
            else:
                real_imgs.append(each_img)

        num_real_data = len(real_imgs)
        fake_data = random.sample(fake_imgs, num_real_data)
        real_imgs.extend(fake_data)

        self.all_imgs = copy(real_imgs)
        random.shuffle(self.all_imgs)


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
        img_name = self.all_imgs[idx]

        #print(f'Image name is : {img_name}')
        label_name = self.metadata[str(img_name[:-5]) + ".mp4"]["label"]

        label = 0

        if label_name == "FAKE":
            label = 1

        img_tensor = self._img2tensor(self.root_dir + "/" + img_name)

        return img_tensor, label

    def __len__(self):
        return len(self.all_imgs)


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
    """Collate function to alter the specification of the batch shape.
    """
    imgs, labels = default_collate(batch)
    imgs = imgs.reshape(
        imgs.shape[0] * imgs.shape[1], imgs.shape[2], imgs.shape[3], imgs.shape[4]
    )
    return [imgs, labels]