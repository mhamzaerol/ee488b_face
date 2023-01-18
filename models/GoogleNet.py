#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torchvision

def MainModel(nOut=256, **kwargs):
    
    return torchvision.models.googlenet(num_classes=nOut, init_weights=True)
