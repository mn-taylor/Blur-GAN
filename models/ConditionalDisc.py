import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

class NewCondDiscriminator(nn.Module):
    def __init__(self, blur_embed_dim=16):
        super(NewCondDiscriminator, self).__init__()
        
        self.blur_embed_dim = blur_embed_dim
        
        # Blur parameter embedding
        self.blur_embedding = nn.Sequential(
            nn.Linear(1, blur_embed_dim),
            nn.ReLU(True),
            nn.Linear(blur_embed_dim, blur_embed_dim)  # Same size as flattened image # was 28x28
        )

        # Convolutional layers for image processing
        self.conv = nn.Sequential(
            # 28x28x1 -> 14x14x64
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            # 14x14x64 -> 7x7x128
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
            
            # 7x7x128 -> 4x4x256
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout2d(0.3),
        )
        
        # Final classification layers (image features + blur embedding)
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4 + blur_embed_dim, 512), # Was 28x28
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )
    
    def forward(self, image, blur_param):
        # Process image through conv layers
        img_features = self.conv(image.float())
        img_features = img_features.view(img_features.size(0), -1)
        
        # Embed blur parameter
        blur_embed = self.blur_embedding(blur_param.unsqueeze(1))
        
        # Concatenate image features and blur embedding
        x = torch.cat([img_features, blur_embed], dim=1)
        
        # Final classification
        x = self.fc(x)
        return x

# Discriminator Network
class CondDiscriminator(nn.Module):
    def __init__(self, blur_emb_dim):
        super(CondDiscriminator, self).__init__()
        self.model = nn.Sequential(
            # Input layer
            nn.Linear(28*28 + blur_emb_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Hidden layers
            nn.Linear(512, 256),
            # nn.LeakyReLU(0.2),
            # nn.Dropout(0.3),
        
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            # Output layer
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

        self.blur_embd = nn.Sequential(
            nn.Linear(1, blur_emb_dim),
            nn.ReLU(),
            nn.Linear(blur_emb_dim, blur_emb_dim),
        )

    
    def forward(self, img, blur):
        blur = blur.view(-1, 1, 1, 1)
        blur_emb = self.blur_embd(blur).squeeze(1).squeeze(1)
        img_flat = img.view(img.size(0), -1)
    
        validity = self.model(torch.cat([img_flat, blur_emb], dim=1))
        return validity