import torch
from torch import nn
torch.manual_seed(0) # Set for testing purposes, please do not change!
import torch.nn.functional as F
from pkmn_constants import *

"""
Class to add gaussian noise to a layer, useful when training the discriminator so it doesn't overpower
the generator. Seems like a std of 0.1 works well
"""


class GaussianNoise(nn.Module):
    def __init__(self, std=0.1, decay_rate=0):
        super().__init__()
        self.std = std
        self.decay_rate = decay_rate

    def decay_step(self):
        self.std = max(self.std - self.decay_rate, 0)

    def forward(self, x):
        if self.training:
            return x + torch.empty_like(x).normal_(std=self.std)
        else:
            return x

# method to decay the std of a gaussian noise layer
def decay_gauss_std(dnn):
    std = 0
    for m in dnn.modules():
        if isinstance(m, GaussianNoise):
            m.decay_step()
            std = m.std            