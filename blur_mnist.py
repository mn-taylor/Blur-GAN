import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import kornia
import os

# Parameters
blur_levels = 28
batch_size = 512  # Adjust based on your memory
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load MNIST
transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
])
mnist = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
loader = DataLoader(mnist, batch_size=batch_size, shuffle=False)

# Create storage tensor: [num_images, blur_levels, 1, 28, 28]
all_blurred = torch.zeros(len(mnist), blur_levels, 1, 28, 28, dtype=torch.float16)

os.makedirs("data/MNIST/blurred", exist_ok=True)

print("Precomputing blurred images...")
with torch.no_grad():
    for batch_idx, (images, _) in enumerate(loader):
        images = images.to(device)  # [B, 1, 28, 28]
        for b in range(blur_levels):
            sigma = b//2
            if sigma == 0:
                blurred = images  # no blur
            else:
                kernel_size = sigma  + (1 if sigma%2==0 else 0)
                blurred = kornia.filters.gaussian_blur2d(images, (kernel_size, kernel_size), (sigma, sigma))
            all_blurred[batch_idx * batch_size:batch_idx * batch_size + images.size(0), b] = blurred.cpu()

# Save the tensor
torch.save(all_blurred, "data/MNIST/blurred/precomputed_blurred_mnist.pt")
