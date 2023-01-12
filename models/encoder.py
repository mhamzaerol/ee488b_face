#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch.nn as nn
import torchvision

def MainModel(**kwargs):
    resnet = torchvision.models.resnet18()
    in_feats = resnet.fc.in_features
    resnet.fc = nn.Identity()
    return resnet, in_feats
    
