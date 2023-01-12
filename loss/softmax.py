#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch
import torch.nn as nn
import torch.nn.functional as F
import time, pdb, numpy

class LossFunction(nn.Module):
	def __init__(self, n_class, **kwargs):
	    super(LossFunction, self).__init__()

	    self.test_normalize = True
	    
	    self.criterion  = torch.nn.CrossEntropyLoss()

	    print('Initialised Softmax Loss')

	def forward(self, x, label=None):

		nloss   = self.criterion(x, label)

		return nloss