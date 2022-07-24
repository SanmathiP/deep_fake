## Evaluation strategies ##

## Evaluating accuracy ##

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy(pred_tensor, true_tensor):
    '''
    Calculates the accuracy given the pred_tensor and
    true_tensor.
    '''

    den = len(pred_tensor)

    num = sum(pred_tensor == true_tensor)

    acc = num / den

    return acc.item()


## Evaluating precision ##

def precision_recall_f1(pred_tensor, true_tensor, confusion=False):
    '''
    Calculates the precision , recall and f1 score
    given predicted and true tensors.
    Also shows the confusion matrix optionally.
    '''

    true_pos = sum((pred_tensor == 1) & (true_tensor == 1))
    false_pos = sum((pred_tensor == 1) & (true_tensor == 0))
    false_neg = sum((pred_tensor == 0) & (true_tensor == 1))
    true_neg = sum((pred_tensor == 0) & (true_tensor == 0))

    precision = (true_pos) / (true_pos + false_pos)
    recall = (true_pos) / (true_pos + false_neg)
    f1 = (2 * precision * recall) / (precision + recall)

    if confusion == True:
        confusion_matrix = np.array([[true_pos, false_pos],
                                     [false_neg, true_neg]])
        sns.heatmap(confusion_matrix, annot=True)
        plt.show()

    return precision.item(), recall.item(), f1.item()