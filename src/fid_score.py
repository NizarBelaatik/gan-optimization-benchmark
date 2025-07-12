from PIL import Image
import numpy as np
import torch_fidelity
import os 
from config import config
def ensure_dir(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)

def split_grid_to_individual_images(grid_path, output_dir, nrow=8):
    """Split a grid image into individual images"""
    grid = Image.open(grid_path)
    w, h = grid.size
    single_w, single_h = w // nrow, h // nrow
    
    for i in range(nrow):
        for j in range(nrow):
            left = j * single_w
            top = i * single_h
            single_img = grid.crop((left, top, left + single_w, top + single_h))
            single_img.save(os.path.join(output_dir, f"img_{i*nrow + j}.png"))

# Usage:
for opt in ["Adam", "RMSprop", "SGD", "Lookahead"]:
    fake_dir = config.dirs['samples'] / opt
    temp_dir = fake_dir / "temp_split"
    ensure_dir(temp_dir)
    
    # Split all grid images in the directory
    for grid_img in fake_dir.glob("*.png"):
        split_grid_to_individual_images(grid_img, temp_dir, nrow=8)
    
    # Compute FID on the split images
    metrics = torch_fidelity.calculate_metrics(
        input1="cifar10-train",  # Use built-in CIFAR-10
        input2=str(temp_dir),
        fid=True,
        cuda=True 
    )
    print(f"{opt}: FID = {metrics['frechet_inception_distance']:.2f}")
