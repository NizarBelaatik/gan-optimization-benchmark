from torchvision import datasets, transforms
from torchvision.utils import save_image
import os
from config import config


# Same transform as your training
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
])

# Load CIFAR10 training set
dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)

# Create directory to save real images
real_dir = config.dirs['real_images']
os.makedirs(real_dir, exist_ok=True)

# Save 1000 real images from CIFAR10
for i in range(1000):
    image_tensor, _ = dataset[i]
    save_image(image_tensor, os.path.join(real_dir, f"{i:04d}.png"), normalize=True)

print("✅ Saved 1000 real CIFAR10 images to real_images/")
