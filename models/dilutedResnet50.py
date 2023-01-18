#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch.nn as nn
import torchvision

def MainModel(nOut=256, **kwargs):
    
    resnet = torchvision.models.resnet50(num_classes=nOut)
    resnet.layer4[1] = nn.Sequential()
    return resnet
