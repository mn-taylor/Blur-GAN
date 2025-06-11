import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
import numpy as np
from GAN.Gen1b import Gen1b
from GAN.Gen1a import Gen1a
from GAN.ConditionalDisc import CondDiscriminator
import kornia
import time

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set random seed for reproducibility
torch.manual_seed(42)

# Hyperparameters
LATENT_DIM = 100
BLUR_EMB_DIM = 100
BATCH_SIZE = 32
EPOCHS = 50
# WARM_UP = 1/2
LEARNING_RATE = 0.0003
BETA1 = 0.5
OUTPUT_DIR = './gan_outputs/1a_overfit'
EPSILON = 0.02

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)


class BlurredImageSampler(torch.utils.data.Dataset):
    def __init__(self, blurred_tensor_path, labels_path="./data"):
        self.all_blurred = torch.load(blurred_tensor_path)[0:1]  # shape [N, B, 1, 28, 28]
        self.labels = datasets.MNIST(root=labels_path, train=True, download=False).targets  # [N]
        self.N, self.B = self.all_blurred.shape[:2]

    def __len__(self):
        return 1024

    def __getitem__(self, idx):
        blur_idx = torch.randint(0, self.B, (1,)).item()
        img = self.all_blurred[0, blur_idx]
        label = self.labels[idx]
        return img, label, torch.tensor(blur_idx / (self.B - 1), dtype=torch.float32)


# Data preparation
def prepare_data():
    blurred_dataset = BlurredImageSampler("data/MNIST/blurred/precomputed_blurred_mnist.pt")

    # dataset = RepeatedDataset(full_dataset, index=0, repeat=1024)
    
    dataloader = DataLoader(
        blurred_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True
    )
    
    return dataloader

def apply_gaussian_blur(img, radius):
    if radius == 0:
        return img
    kernel_size = int(2 * radius + 1)  # rough rule of thumb: ~2 stddevs
    return kornia.filters.gaussian_blur2d(img, (kernel_size, kernel_size), (radius, radius))

# Weight initialization
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        nn.init.zeros_(m.bias.data)

# Writer func
def get_writer(filename):
    def write(x):
        with open(filename, 'a') as f:
            f.write(x + "\n")

    # clear out file
    with open(filename, "w") as f:
        f.write("")

    return write

write = get_writer(OUTPUT_DIR + "/losses.txt")

# Training function
def train_gan():
    # Prepare data
    write(f"device: {device}")
    write("Preparing dataset ... ")

    dataloader = prepare_data()

    write("Initializing ")
    # Initialize networks and move to device
    generator = Gen1a(LATENT_DIM, BLUR_EMB_DIM).to(device)
    discriminator = CondDiscriminator(BLUR_EMB_DIM).to(device)
    
    # Initialize weights
    generator.apply(weights_init)
    discriminator.apply(weights_init)
    
    # Loss and optimizers
    adversarial_loss = nn.BCELoss()
    continuity_loss  = nn.MSELoss()
    
    optimizer_G = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=(BETA1, 0.999))
    
    # Tracking lists for loss
    g_losses = []
    d_losses = []
    
    # Training loop
    for epoch in range(EPOCHS):
        epoch_g_losses = []
        epoch_d_losses = []
        
        start = time.time()
        for i, (imgs, _, blur_levels) in enumerate(dataloader):
            # Move images to device
            imgs = imgs.to(device)
            blur_levels = blur_levels.to(device)
            
            # Adversarial ground truths
            valid = torch.ones(imgs.size(0), 1).to(device)
            fake = torch.zeros(imgs.size(0), 1).to(device)
            
            # Train Generator
            for _ in range(1):
                optimizer_G.zero_grad()
                
                # Sample noise
                z = torch.randn(imgs.size(0), LATENT_DIM).to(device)
                blur_radii = torch.randint(0, 28, (imgs.size(0),)).to(device) / 27.0 # Uniform distribution
                
                # Generate images
                generated_imgs = generator(z, blur_radii)
                similar_generated_imgs = generator(z, blur_radii + EPSILON)
                
                # Generator loss
                g_loss = adversarial_loss(discriminator(generated_imgs, blur_radii), valid)
                g_loss += continuity_loss(generated_imgs, similar_generated_imgs)
                
                g_loss.backward()
                torch.nn.utils.clip_grad_norm_(generator.parameters(), max_norm=1.0)
                optimizer_G.step()
            
            # Train Discriminator
            optimizer_D.zero_grad()

            real_loss = adversarial_loss(discriminator(imgs, blur_levels), valid)# (min([( epoch / EPOCHS / WARM_UP), 1])) * adversarial_loss(discriminator(imgs, blur_levels), valid)
            real_loss +=  adversarial_loss(discriminator(imgs, blur_radii), fake) # randomly associated images and blur levels should be assigned fake
            


            # Fake images loss
            z = torch.randn(imgs.size(0), LATENT_DIM).to(device)
            blur_radii = torch.randint(0, 28, (imgs.size(0),)).to(device) /27
            generated_imgs = generator(z, blur_radii).detach()
            fake_loss = adversarial_loss(discriminator(generated_imgs, blur_radii), fake)
            
            # Total discriminator loss
            a = 0.5
            d_loss = ((1 - a) * real_loss + a * fake_loss) # (real_loss + fake_loss) / 2
            
            d_loss.backward()
            torch.nn.utils.clip_grad_norm_(discriminator.parameters(), max_norm=1.0)
            optimizer_D.step()
            
            # Store losses
            epoch_g_losses.append(g_loss.item())
            epoch_d_losses.append(d_loss.item())
        
        # Average epoch losses
        avg_g_loss = np.mean(epoch_g_losses)
        avg_d_loss = np.mean(epoch_d_losses)
        g_losses.append(avg_g_loss)
        d_losses.append(avg_d_loss)
        
        # Print progress
        if epoch % 5 == 0:
            write(f'Epoch [{epoch}/{EPOCHS}], '
                  f'D Loss: {avg_d_loss:.4f}, '
                  f'G Loss: {avg_g_loss:.4f}')
            
            # Generate and save images periodically
            generate_and_plot_blur_process(generator, epoch)
            generate_and_plot_zero(generator, epoch)
    
            # Plot and save loss curves
            plt.figure(figsize=(10, 5))
            plt.plot(g_losses, label='Generator Loss')
            plt.plot(d_losses, label='Discriminator Loss')
            plt.title('GAN Training Losses')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.savefig(os.path.join(OUTPUT_DIR, 'training_losses.png'))
            plt.close()

        end = time.time()
        write(f"Completed Epoch in {end-start} seconds")
    
    return generator


# Generate and save images
def generate_and_plot_blur_process(generator, epoch, num_images=100):
    # Set generator to evaluation mode
    generator.eval()
    
    # Generate images
    with torch.no_grad():
        z = torch.randn(num_images, LATENT_DIM).to(device)
        blurs = torch.tensor(np.linspace(0, 1, num_images)).float().to(device)
        generated_imgs = generator(z,blurs).to(device).cpu()
    
    # Prepare output path
    output_path = os.path.join(OUTPUT_DIR, f'generated_images_range_epoch_{epoch}.png')
    
    # Plot generated images
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(10, 10, i+1)
        plt.imshow(generated_imgs[i].squeeze().numpy(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    write(f'Saved generated images to {output_path}')

def generate_and_plot_zero(generator, epoch, num_images=100):
    # Set generator to evaluation mode
    generator.eval()
    
    # Generate images
    with torch.no_grad():
        z = torch.randn(num_images, LATENT_DIM).to(device)
        blurs = torch.zeros(num_images).to(device)
        generated_imgs = generator(z,blurs).to(device).cpu()
    
    # Prepare output path
    output_path = os.path.join(OUTPUT_DIR, f'generated_image_zeros_epoch_{epoch}.png')
    
    # Plot generated images
    plt.figure(figsize=(10, 10))
    for i in range(num_images):
        plt.subplot(10, 10, i+1)
        plt.imshow(generated_imgs[i].squeeze().numpy(), cmap='gray')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    
    write(f'Saved generated images to {output_path}')



def main():
    # Train the GAN
    trained_generator = train_gan()

if __name__ == '__main__':
    main()