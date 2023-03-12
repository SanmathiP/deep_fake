"""Model for our Deepfake detection"""

import torch
import torch.nn as nn
import torchvision
import torchvision.models as models

def init_weights(layer):
    """Setting for initialization.
    """
    if (type(layer) == nn.Conv2d): 
        nn.init.kaiming_uniform_(layer.weight , mode = 'fan_in' , nonlinearity = 'relu')

        if layer.bias != None:
            nn.init.ones_(layer.bias)

    if (type(layer) == nn.Dropout):
        layer.p = 0.25

class ResNetModel(nn.Module):
    """Custom ensemble resnet model orchestrated for our project.
    """
    def __init__(self , path = None):
        super().__init__()
        full_model = models.resnet18(pretrained=False)
        if path != None:
            full_model.load_state_dict(torch.load(path))
        #else:
            #full_model = full_model.apply(init_weights)
        self.base_model = nn.Sequential(*list(full_model.children())[1:-1])

        self.inp_layer = nn.Sequential(
            nn.Conv2d(18 , 32 , kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32 , 32 , kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(32 , 64 , kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False),
        )
        self.classifier = nn.Linear(512,2,bias=True)

    def forward(self , img):
        out = self.inp_layer(img)
        out = self.base_model(out)
        out = out.squeeze(2).squeeze(2)
        return self.classifier(out)

class ResNetModelNormal(nn.Module):
    """Single modality resnet model.
    """
    def __init__(self , path = None):
        super().__init__()
        full_model = models.resnet18(pretrained=False)
        if path != None:
            full_model.load_state_dict(torch.load(path))
        #else:
            #full_model = full_model.apply(init_weights)
        self.base_model = nn.Sequential(*list(full_model.children())[:-1])

        self.classifier = nn.Linear(512,2,bias=True)

    def forward(self , img):
        out = self.base_model(img)
        out = out.squeeze(2).squeeze(2)
        return self.classifier(out)
    