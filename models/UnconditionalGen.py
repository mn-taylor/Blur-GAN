import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np




# Generator Network
class UncondGenerator(nn.Module):
    def __init__(self, latent_dim):
        super(UncondGenerator, self).__init__()
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(latent_dim, 256),
            nn.LeakyReLU(0.2),
            
            # Hidden layers
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            
            # Output layer
            nn.Linear(1024, 28*28),
            nn.Tanh()
        )
    
    def forward(self, z):
        img = self.model(z)
        img = img.view(img.size(0), 1, 28, 28)
        return img