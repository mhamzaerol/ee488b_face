import numpy as np
import torch

class RandomBlock(object):

    def __init__(self, block_size, block_count):
        self.block_size = block_size
        self.block_count = block_count

    def __call__(self, img): 
        c, w, h = img.shape
        block_count = np.random.randint(self.block_count)
        for _ in range(block_count):
            block_size = int(np.random.rand() * self.block_size * min(w, h))
            x = np.random.randint(0, w - block_size)
            y = np.random.randint(0, h - block_size)
            img[:, x:x+block_size, y:y+block_size] = torch.rand(3, 1, 1)
        return img