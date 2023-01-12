#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch.nn as nn
import models.encoder as encoder
import models.projector as projector

class SimCLR(nn.Module):

    def __init__(self, n_proj):
        super(SimCLR, self).__init__()

        self.n_proj = n_proj

        self.encoder, n_feat = encoder.MainModel()
        self.projector = projector.MainModel(n_feat, n_proj, bias=False)

    def get_feat(self, x):
        return self.encoder(x)


    def forward(self, x_i, x_j):
        h_i = self.encoder(x_i)
        h_j = self.encoder(x_j)

        z_i = self.projector(h_i)
        z_j = self.projector(h_j)
        return h_i, h_j, z_i, z_j

def MainModel(n_proj=64, **kwargs):
    
    return SimCLR(n_proj)