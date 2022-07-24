## Importing the necessary libraries ##
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.models as models
from torchvision.transforms import transforms

from tensorboardX import SummaryWriter

import PIL
import json
import glob
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from make_dataset_dataloader import grab_train_val_names , get_device , shift_to_device , make_dataloader
from make_dataset_dataloader import Custom_Dataset , GPUDataLoader
from metrics import accuracy , precision_recall_f1
from create_model import create_model

MODALITY = 'Albedo'
DATA_PATH = 'modalities_output/albedo'

## Creation of the train and the val filenames ##
train_data , val_data = grab_train_val_names(DATA_PATH , split_ratio = 0.8)


## Creation of the training and validation datasets ##
train_dataset = Custom_Dataset(train_data , DATA_PATH , 'metadata.json')
val_dataset = Custom_Dataset(val_data , DATA_PATH , 'metadata.json')

## Creating our train and val dataloader instance ##
train_dataloader = make_dataloader(dataset = train_dataset , batch_size = 8 , shuffle = True)
val_dataloader = DataLoader(dataset = train_dataset , batch_size = 8 , shuffle = True)

## Setting default device ##
device = get_device()

## Creating gpu based dataloader instances ##
train_dl = GPUDataLoader(train_dataloader , device = device)
val_dl = GPUDataLoader(val_dataloader , device = device)

## Creating the model instance ##
model = create_model(model_type = 'alexnet')

## Setting the optimizer and loss ##
optim = torch.optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

## Setting up training phase ##
model = model.to(device)

## Setting the summarywriter ##
writer = SummaryWriter()

num_epochs = 100

loss_value = []
acc = []
precision = []
recall = []
f1 = []

val_loss_value = []
val_acc = []
val_precision = []
val_recall = []
val_f1 = []

for epoch in range(num_epochs):
    minibatch_loss = []
    minibatch_acc = []
    minibatch_precision = []
    minibatch_recall = []
    minibatch_f1 = []

    minibatch_val_loss = []
    minibatch_val_acc = []
    minibatch_val_precision = []
    minibatch_val_recall = []
    minibatch_val_f1 = []

    loop = tqdm(train_dl)

    model.train()

    for imgs, labels in loop:
        optim.zero_grad()

        preds = model(imgs)

        loss = criterion(preds, labels)

        minibatch_loss.append(loss.item())

        loss.backward()

        optim.step()

        prediction = torch.argmax(preds, 1)

        acc_ = accuracy(prediction.cpu(), labels.cpu())
        precision_, recall_, f1_ = precision_recall_f1(prediction, labels, confusion=False)


        minibatch_acc.append(acc_)
        minibatch_precision.append(precision_)
        minibatch_recall.append(recall_)
        minibatch_f1.append(f1_)

        loop.set_description('Epoch : {} / {}'.format(epoch + 1, num_epochs))

        loop.set_postfix(acc=acc_, precision=precision_, recall=recall_, f1=f1_)

    batch_loss = sum(minibatch_loss) / len(minibatch_loss)
    batch_acc = sum(minibatch_acc) / len(minibatch_acc)
    batch_precision = sum(minibatch_precision) / len(minibatch_precision)
    batch_recall = sum(minibatch_recall) / len(minibatch_recall)
    batch_f1 = sum(minibatch_f1) / len(minibatch_f1)

    loss_value.append(batch_loss)
    acc.append(batch_acc)
    precision.append(batch_precision)
    recall.append(batch_recall)
    f1.append(batch_f1)

    model.eval()

    for val_imgs, val_labels in val_dl:
        val_preds = model(val_imgs)
        val_loss = criterion(val_preds , val_labels)
        val_prediction = torch.argmax(val_preds, 1)

        val_acc_ = accuracy(val_prediction.cpu(), val_labels.cpu())
        val_precision_, val_recall_, val_f1_ = precision_recall_f1(val_prediction, val_labels, confusion=False)

        minibatch_val_loss.append(val_loss.item())
        minibatch_val_acc.append(val_acc_)
        minibatch_val_precision.append(val_precision_)
        minibatch_val_recall.append(val_recall_)
        minibatch_val_f1.append(val_f1_)

    val_batch_loss_value = sum(minibatch_val_loss) / len(minibatch_val_loss)
    val_batch_acc = sum(minibatch_val_acc) / len(minibatch_val_acc)
    val_batch_precision = sum(minibatch_val_precision) / len(minibatch_val_precision)
    val_batch_recall = sum(minibatch_val_recall) / len(minibatch_val_recall)
    val_batch_f1 = sum(minibatch_val_f1) / len(minibatch_val_f1)

    val_loss_value.append(val_batch_loss_value)
    val_acc.append(val_batch_acc)
    val_precision.append(val_batch_precision)
    val_recall.append(val_batch_recall)
    val_f1.append(val_batch_f1)

    print('################ Validation Results ############')
    print('Validation Loss :' , val_batch_loss_value)
    print('Validation Accuracy :', val_batch_acc)
    print('Validation Precision :', val_batch_precision)
    print('Validation Recall :', val_batch_recall)
    print('Validation F1 Score :', val_batch_f1)

    writer.add_scalar('Loss/train' , batch_loss , epoch)
    writer.add_scalar('Loss/test' , val_batch_loss_value , epoch)

    writer.add_scalar('Accuracy/train' , batch_acc , epoch)
    writer.add_scalar('Accuracy/test' , val_batch_acc , epoch)

    writer.add_scalar('Precision/train' , batch_precision , epoch)
    writer.add_scalar('Precision/test' , val_batch_precision , epoch)

    writer.add_scalar('Recall/train' , batch_recall , epoch)
    writer.add_scalar('Recall/test' , val_batch_recall , epoch)

    writer.add_scalar('F1/train' , batch_f1 , epoch)
    writer.add_scalar('F1/test' , val_batch_f1 , epoch)

data_df = pd.DataFrame(list(zip(loss_value ,
                                acc ,
                                precision ,
                                recall ,
                                f1,
                                val_loss_value ,
                                val_acc ,
                                val_precision ,
                                val_recall ,
                                val_f1)) ,
                       columns = ['loss_value' ,
                                'acc' ,
                                'precision' ,
                                'recall' ,
                                'f1',
                                'val_loss_value' ,
                                'val_acc' ,
                                'val_precision' ,
                                'val_recall' ,
                                'val_f1'])

pd.to_csv(data_df , index = False)
writer.close()


