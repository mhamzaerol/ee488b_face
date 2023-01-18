#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import models.ResNet18 as ResNet18
import models.GoogleNet as GoogleNet
import models.shuffleNet as shuffleNet

class EnsembleNet(nn.Module):

    def __init__(self, nOut=256, **kwargs):
        super(EnsembleNet, self).__init__()
        self.net1 = ResNet18.MainModel(nOut=nOut, **kwargs)
        self.net2 = shuffleNet.MainModel(nOut=nOut, **kwargs)
    
    def forward(self, x):
        x1 = self.net1(x)
        x2 = self.net2(x)
        return (x1 + x2) / 2

def MainModel(nOut=256, **kwargs):
        
    return EnsembleNet(nOut=nOut, **kwargs)
    