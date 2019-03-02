from torchvision.models.alexnet import alexnet
import torch.nn as nn



def get_alexnet_model(pretrained=True, **kwargs):
    return alexnet(pretrained, **kwargs)

def get_layer(i, module):
    pass