## Utility module to make dataset, dataloader .. ##
## and also a GPU version of the DataLoader . ##

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import transforms

import PIL
import json
import os
import numpy as np


## Defining an utility function for creating ##
## the training and validation filename array ##

def grab_train_val_names(root_dir, split_ratio=0.8):
    '''
    Returns a list of filenames for training data and a
    list of filenames for validation data given the
    root_dir and split_ratio.
    '''

    all_files = []

    for file in os.listdir(root_dir):
        if file.endswith('.jpg'):
            all_files.append(file[:-5])

    all_files = np.array(list(set(all_files)))

    train_size = int(len(all_files) * split_ratio)

    dataset_indices = np.arange(len(all_files))
    np.random.shuffle(dataset_indices)

    train_indices, val_indices = dataset_indices[:train_size], dataset_indices[train_size:]

    train_data = list(all_files[train_indices])
    val_data = list(all_files[val_indices])

    return train_data, val_data


## Pytorch custom dataset ##

class Custom_Dataset(Dataset):
    '''
    Creates a Pytorch based dataset given array of datafiles
    called data, the directory in which the lighting images
    are stored called data_dir and the location of the
    metadata .json file given by meta_file.
    '''

    def __init__(self, data, data_dir, meta_file):
        super().__init__()
        self.data = data
        self.data_dir = data_dir
        with open(meta_file) as json_file:
            # .load creates a python dictionary
            self.metadata = json.load(json_file)

    def __getitem__(self, idx):

        file_name = self.data[idx]
        label_name = self.metadata[file_name + '.mp4']['label']

        label = 0

        if label_name == 'FAKE':
            label = 1

        aug = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

        imgs = None

        for image_num in range(5):

            img_name = file_name + str(image_num) + '.jpg'

            img = PIL.Image.open(os.path.join(self.data_dir, img_name))
            img = aug(img)

            if image_num == 0:
                imgs = img

            else:
                imgs = torch.cat((imgs, img), 0)

        return imgs, label

    def __len__(self):

        return len(self.data)


## Creating the dataloader ##

def make_dataloader(dataset , batch_size , shiffle = True):
    '''
    Creates a Pytorch generic dataloder given a Pytorch dataset,
    batch_size and shuffle.
    '''

    dl = DataLoader(dataset = dataset , batch_size = batch_size , shuffle = True)

    return dl


## Creating GPU based dataloader ##
## For that we will define some small.. ##
## utility functions. ##

## Setting the default device ##

def get_device():
    '''
    Sets the default device.
    '''
    if torch.cuda.is_available():
        return torch.device('cuda')

    return torch.device('cpu')


## Setting the utility function for transfer ##
## of data to a specific device ##

def shift_to_device(data, device):
    '''
    Shifts data to device.
    '''

    if isinstance(data, (list, tuple)):
        return [shift_to_device(each_data, device) for each_data in data]

    return data.to(device)


## Creating GPUDataloader class ##

class GPUDataLoader:
    '''
    Returns a GPU based DataLoader given a
    Pytorch Dataloader and a default device.
    '''

    def __init__(self, data, device):
        self.data = data
        self.device = device

    def __iter__(self):
        for batch in self.data:
            yield shift_to_device(batch, self.device)

    def __len__(self):
        return len(self.data)
