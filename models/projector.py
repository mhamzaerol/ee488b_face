#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch.nn as nn

def MainModel(n_feat=256, n_proj=64, bias=True, **kwargs):
    
    return nn.Sequential(
        nn.Linear(n_feat, n_feat, bias=bias),
        nn.ReLU(),
        nn.Linear(n_feat, n_proj, bias=bias),
    )
