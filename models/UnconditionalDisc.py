import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Discriminator Network
class UncondDiscriminator(nn.Module):
    def __init__(self, blur_emb_dim):
        super(UncondDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(28*28 + blur_emb_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Hidden layers
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Output layer
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    
    def forward(self, img):
        img_flat = img.view(img.size(0), -1)
        validity = self.model(img_flat)
        return validity