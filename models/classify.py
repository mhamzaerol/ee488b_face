#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import models.encoder as encoder
import models.projector as projector

class Classifier(nn.Module):
    def __init__(self, n_class=256, pretrain_weights=None):
        super(Classifier, self).__init__()
        
        self.encoder, n_feat = encoder.MainModel()
        if pretrain_weights:
            self.encoder.load_state_dict(torch.load(pretrain_weights))

        self.classifiers = nn.ModuleList([ # right now it is one model
            # projector.MainModel(n_feat, n_class) # Keep this, also making it simpler could be better as the representataions could be well learnt!
            nn.Linear(n_feat, n_class)
            for _ in range(1)
        ])

    def get_feat(self, x):
        return self.encoder(x)

    def forward(self, x):
        x = self.encoder(x)
        outs = [classifier(x).unsqueeze(0) for classifier in self.classifiers]
        x = torch.cat(outs, dim=0)
        x = torch.mean(x, dim=0)
        return x

def MainModel(n_class=256, pretrain_weights=None, **kwargs):
    return Classifier(n_class, pretrain_weights)