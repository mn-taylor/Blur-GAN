import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class NewCondGenerator(nn.Module):
    def __init__(self, latent_dim=100, blur_embed_dim=16):
        super(NewCondGenerator, self).__init__()
        
        self.blur_embed_dim = blur_embed_dim
        
        # Blur parameter embedding
        self.blur_embedding = nn.Sequential(
            nn.Linear(1, blur_embed_dim),
            nn.ReLU(True),
            nn.Linear(blur_embed_dim, blur_embed_dim)
        )
        
        # Initial dense layer (noise + blur embedding)
        self.fc = nn.Linear(latent_dim + blur_embed_dim, 7*7*256)
        
        # Transpose convolution layers
        self.conv_transpose = nn.Sequential(
            # 7x7x256 -> 14x14x128
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            
            # 14x14x128 -> 28x28x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            
            # 28x28x64 -> 28x28x1
            nn.ConvTranspose2d(64, 1, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )
    
    def forward(self, noise, blur_param):
        # Embed blur parameter
        blur_embed = self.blur_embedding(blur_param.unsqueeze(1)).squeeze(1).squeeze(1).squeeze(1)
        
        # Concatenate noise and blur embedding
        x = torch.cat([noise, blur_embed], dim=1)
        
        # Generate image
        x = self.fc(x)
        x = x.view(x.size(0), 256, 7, 7)
        x = self.conv_transpose(x)
        return x
    

# Generator Network
class CondGenerator(nn.Module):
    def __init__(self, latent_dim, blur_emb_dim):
        super(CondGenerator, self).__init__()
        self.fc1 = nn.Sequential(
            # Input layer
            nn.Linear(latent_dim + blur_emb_dim, 256),
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

        self.blur_emb_low_freq = nn.Sequential(
            nn.Linear(1, blur_emb_dim),
            nn.ReLU(True),
            nn.Linear(blur_emb_dim, blur_emb_dim),
        )

    
    def forward(self, z, blur):
        blur_emb_low_freq = self.blur_emb_low_freq(blur).squeeze(1).squeeze(1)
        img = self.fc1(torch.cat([z, blur_emb_low_freq], dim=1))
        img = img.view(img.size(0), 1, 28, 28)
        return img