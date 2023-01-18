#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn


class NT_Xent(nn.Module):
    def __init__(self, batch_size, temperature):
        super(NT_Xent, self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature

        self.mask = self.mask_correlated_samples(batch_size)
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        self.similarity_f = nn.CosineSimilarity(dim=2)

    def mask_correlated_samples(self, batch_size):
        N = 2 * batch_size
        mask = torch.ones((N, N), dtype=bool)
        mask = mask.fill_diagonal_(0)
        for i in range(batch_size):
            mask[i, batch_size + i] = 0
            mask[batch_size + i, i] = 0
        return mask

    def forward(self, z):
        """
        We do not sample negative examples explicitly.
        Instead, given a positive pair, similar to (Chen et al., 2017), we treat the other 2(N − 1) augmented examples within a minibatch as negative examples.
        """
        N = 2 * self.batch_size

        sim = self.similarity_f(z.unsqueeze(1), z.unsqueeze(0)) / self.temperature

        sim_i_j = torch.diag(sim, self.batch_size)
        sim_j_i = torch.diag(sim, -self.batch_size)

        positive_samples = torch.cat((sim_i_j, sim_j_i), dim=0).reshape(N, 1)
        negative_samples = sim[self.mask].reshape(N, -1)

        labels = torch.zeros(N).to(positive_samples.device).long()
        logits = torch.cat((positive_samples, negative_samples), dim=1)
        loss = self.criterion(logits, labels)
        loss /= N
        return loss

class LossFunction(nn.Module):
    def __init__(self, nOut, nClasses, batch_size, **kwargs):
        super(LossFunction, self).__init__()

        self.ssl_loss = NT_Xent(batch_size, 0.2)
        self.projector = nn.Sequential( # consider making it lighter too
            nn.Linear(nOut, nOut, bias=False),
            nn.ReLU(),
            nn.Linear(nOut, 64, bias=False)
        )

        self.criterion = nn.CrossEntropyLoss()
        self.fc = nn.Linear(nOut, nClasses)

        self.batch_size = batch_size

        print('Initialized softmax-nt_xent loss')

    def forward(self, x, label=None): # x consists of 2 * batch size elements. 
        nt_xent_loss = self.ssl_loss(self.projector(x))
        # x = x[:self.batch_size, :]
        x = self.fc(x)
        label = torch.cat([label, label], dim=0) # doing this since I could not really understand the data loader of the label
        nloss = self.criterion(x, label)
        
        return nloss * 0.9 + nt_xent_loss * 0.1
        
        # return nloss