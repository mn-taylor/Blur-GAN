import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from GAN.UnconditionalGen import UncondGenerator
from GAN.ConditonalGen import CondGenerator
from GAN import UnconditionalGen, ConditonalGen


class Gen1a(nn.Module):
    def __init__(self, latent_dim, blur_emb_dim):
        super(Gen1a, self).__init__()
        self.high_freq_model = CondGenerator(latent_dim, blur_emb_dim)
    
    def forward(self, z, blur):
        blur = blur.view(-1, 1, 1, 1)
        img = self.high_freq_model(z, blur)# (1-blur) * self.low_freq_model(z, blur) +  blur * self.mid_freq_model(z, blur)  # + blur * self.low_freq_model(z, blur) # ((1-blur) * self.mid_freq_model(z, blur) +  (blur * self.low_freq_model(z, blur)))
        return img
