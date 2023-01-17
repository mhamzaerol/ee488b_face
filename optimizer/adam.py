#! /usr/bin/python
# -*- encoding: utf-8 -*-

import torch

def Optimizer(param_list, lr, weight_decay, **kwargs):

	print('Initialised Adam optimizer')

	return torch.optim.Adam(
		[
			{'params': param, 'lr': lr, 'weight_decay': weight_decay}
			for param in param_list
		]
	)
