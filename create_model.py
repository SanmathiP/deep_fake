## Creates various pretrained models ##

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

def create_model(model_type = 'alexnet'):
    '''
    Creates pretrained model of the given type.
    param:
    model_type (str) -- Type of the model.
    [Options : 'alexnet' , 'vgg' , 'resnet' ,  'densenet'].
    '''


    if model_type == 'alexnet':
        alexnet = models.alexnet(pretrained=True,
                                 progress=True)

        alexnet.features[0] = nn.Conv2d(15, 64, kernel_size=(11, 11), stride=(4, 4), padding=(2, 2))

        alexnet.classifier[4] = nn.Linear(4096, 1024, bias=True)

        alexnet.classifier[6] = nn.Linear(1024, 2, bias=True)

        return alexnet

    elif model_type == 'vgg':
        vgg19 = models.vgg19_bn(pretrained=True,
                                progress=True)

        vgg19.features[0] = nn.Conv2d(15, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        vgg19.classifier[3] = nn.Linear(4096, 1024, bias=True)

        vgg19.classifier[6] = nn.Linear(1024, 2, bias=True)

        return vgg19

    elif model_type == 'resnet':
        resnet50 = models.resnet50(pretrained=True,
                                   progress=True)

        resnet50.conv1 = nn.Conv2d(15, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        resnet50.fc = nn.Linear(in_features=2048, out_features=2, bias=True)

        return resnet50

    else:
        densenet121 = models.densenet121(pretrained=True,
                                         progress=True)

        densenet121.features[0] = nn.Conv2d(15, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

        densenet121.classifier = nn.Linear(in_features=1024, out_features=2, bias=True)

        return densenet121
