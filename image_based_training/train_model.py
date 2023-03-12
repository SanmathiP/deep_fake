## Importing the necessary libraries ##
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchmetrics.classification import (
    BinaryAccuracy,
    BinaryPrecision,
    BinaryRecall,
    BinaryF1Score,
)
import torchvision
import torchvision.models as models
from torchvision.transforms import transforms
from torch.optim import Adam, RMSprop, SGD

from tensorboardX import SummaryWriter

import PIL
import json
import glob
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from make_dataset_dataloader import get_device, shift_to_device, make_train_val_dataset
from make_dataset_dataloader import GPUDataLoader

# from metrics import accuracy , precision_recall_f1
from create_model import create_model, EnsembleModel

MODALITY = "Normal"
DATA_PATH = "ffhq/albedo"

## Setting default device ##
device = get_device()

## Creating the model instance ##
model = create_model("resnet")
model = model.to(device)

## Optimizer and loss ##
optim = SGD(model.parameters(), lr=0.1, weight_decay=0.001)
criterion = nn.CrossEntropyLoss()

## LR Scheduler
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optim, "min", patience=7, cooldown=7, factor=0.9
)

## Making datasets ##
train_dataset, val_dataset = make_train_val_dataset(
    DATA_PATH, split_ratio=0.8, ensemble=False
)

## Creating dataloaders ##
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=True)

## Shifting dataloaders to GPU ##
train_dl = GPUDataLoader(train_dataloader, device=device)
val_dl = GPUDataLoader(val_dataloader, device=device)

## Setting the summarywriter ##
writer = SummaryWriter(f"runs/image_based/{MODALITY}")

num_epochs = 100

train_accuracy = BinaryAccuracy()
train_accuracy = train_accuracy.to(device)
train_precision = BinaryPrecision()
train_precision = train_precision.to(device)
train_recall = BinaryRecall()
train_recall = train_recall.to(device)
train_f1 = BinaryF1Score()
train_f1 = train_f1.to(device)

val_accuracy = BinaryAccuracy()
val_accuracy = val_accuracy.to(device)
val_precision = BinaryPrecision()
val_precision = val_precision.to(device)
val_recall = BinaryRecall()
val_recall = val_recall.to(device)
val_f1 = BinaryF1Score()
val_f1 = val_f1.to(device)

loss_value = []

val_loss_value = []


for epoch in range(num_epochs):
    minibatch_loss = []

    minibatch_val_loss = []

    loop = tqdm(train_dl)

    model.train()

    for imgs, labels in train_dl:
        optim.zero_grad()

        preds = model(imgs)

        prediction = torch.argmax(preds, 1)

        loss = criterion(preds, labels)

        minibatch_loss.append(loss.item())

        loss.backward()

        optim.step()

        train_accuracy.update(prediction, labels)
        train_precision.update(prediction, labels)
        train_recall.update(prediction, labels)
        train_f1.update(prediction, labels)

        loop.set_description("Epoch : {} / {}".format(epoch + 1, num_epochs))

        loop.set_postfix(loss=loss.item())

    batch_loss = sum(minibatch_loss) / len(minibatch_loss)
    batch_acc = train_accuracy.compute()
    batch_precision = train_precision.compute()
    batch_recall = train_recall.compute()
    batch_f1 = train_f1.compute()

    model.eval()

    for val_imgs, val_labels in val_dl:
        val_preds = model(val_imgs)
        val_loss = criterion(val_preds, val_labels)
        minibatch_val_loss.append(val_loss.item())
        val_prediction = torch.argmax(val_preds, 1)

        val_accuracy.update(val_prediction, val_labels)
        val_precision.update(val_prediction, val_labels)
        val_recall.update(val_prediction, val_labels)
        val_f1.update(val_prediction, val_labels)

    val_batch_loss_value = sum(minibatch_val_loss) / len(minibatch_val_loss)
    val_batch_acc = val_accuracy.compute()
    val_batch_precision = val_precision.compute()
    val_batch_recall = val_recall.compute()
    val_batch_f1 = val_f1.compute()

    scheduler.step(val_batch_loss_value)

    print("################ Validation Results ############")
    print("Validation Loss :", val_batch_loss_value)
    print("Validation Accuracy :", val_batch_acc)
    print("Validation Precision :", val_batch_precision)
    print("Validation Recall :", val_batch_recall)
    print("Validation F1 Score :", val_batch_f1)

    writer.add_scalar("Loss/train", batch_loss, epoch)
    writer.add_scalar("Loss/val", val_batch_loss_value, epoch)

    writer.add_scalar("Accuracy/train", batch_acc, epoch)
    writer.add_scalar("Accuracy/val", val_batch_acc, epoch)

    writer.add_scalar("Precision/train", batch_precision, epoch)
    writer.add_scalar("Precision/val", val_batch_precision, epoch)

    writer.add_scalar("Recall/train", batch_recall, epoch)
    writer.add_scalar("Recall/val", val_batch_recall, epoch)

    writer.add_scalar("F1/train", batch_f1, epoch)
    writer.add_scalar("F1/val", val_batch_f1, epoch)

    train_accuracy.reset()
    train_precision.reset()
    train_recall.reset()
    train_f1.reset()

    val_accuracy.reset()
    val_precision.reset()
    val_recall.reset()
    val_f1.reset()
torch.save(model.state_dict(), "best_model.pth")
writer.close()
