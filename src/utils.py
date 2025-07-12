import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np
from config import config

from io import BytesIO
import re
import os
from pathlib import Path
from PIL import Image
import shutil

def ensure_dir(path):
    """Ensure directory exists"""
    os.makedirs(path, exist_ok=True)

def gradient_penalty(D, real, fake, device):
    """Compute gradient penalty for WGAN-GP"""
    alpha = torch.rand(real.size(0), 1, 1, 1).to(device)
    interpolates = (alpha * real + ((1 - alpha) * fake)).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones(d_interpolates.size()).to(device),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return penalty

def save_checkpoint(epoch, G, D, opt_G, opt_D, optimizer_name, losses):
    """Save training checkpoint"""
    ensure_dir(config.dirs['checkpoints'])
    checkpoint = {
        'epoch': epoch + 1,  # Save NEXT epoch to resume
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'opt_G_state_dict': opt_G.state_dict(),
        'opt_D_state_dict': opt_D.state_dict(),
        'G_losses': losses["G_losses"],
        'D_losses': losses["D_losses"]
    }
    torch.save(checkpoint, 
              os.path.join(config.dirs['checkpoints'], 
                         f"{optimizer_name}_checkpoint.pth"))

def load_checkpoint(optimizer_name):
    """Load training checkpoint"""
    path = os.path.join(config.dirs['checkpoints'], 
                       f"{optimizer_name}_checkpoint.pth")
    if os.path.exists(path):
        return torch.load(path)
    return None


def save_samples(G, noise, optimizer_name, epoch, batch_idx=None):
    """Save generated samples as individual images for FID calculation"""
    samples_dir = os.path.join(config.dirs['samples'], optimizer_name)
    ensure_dir(samples_dir)
    
    with torch.no_grad():
        samples = G(noise)  # shape: (batch_size, 3, 64, 64)
        
        for i in range(samples.shape[0]):
            img_path = os.path.join(
                samples_dir,
                f"epoch{epoch}_batch{batch_idx}_{i}.png" if batch_idx is not None else f"epoch{epoch}_{i}.png"
            )
            save_image(samples[i], img_path, normalize=True)
            
def save_metrics(metrics, optimizer_name):
    """Save training metrics"""
    ensure_dir(config.dirs['metrics'])
    metric_path = os.path.join(
        config.dirs['metrics'],
        f"{optimizer_name}_metrics.pkl"
    )
    torch.save(metrics, metric_path)

def save_all_samples(G, noise, optimizer_name, epoch, batch_idx):
    """Save all generated samples in optimizer-specific folders"""
    samples_dir = os.path.join(config.dirs['samples'], optimizer_name)
    os.makedirs(samples_dir, exist_ok=True)
    
    samples_path = os.path.join(
        samples_dir,
        f"epoch{epoch}_batch{batch_idx}.png"
    )
    with torch.no_grad():
        samples = G(noise)
        save_image(samples, samples_path, nrow=8, normalize=True)



def ensure_clean_dir(path: Path):
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)

def split_grid_to_individual_images(grid_path, output_dir, nrow=8):
    #grid = Image.open(grid_path)
    with Image.open(grid_path) as grid:
        w, h = grid.size
        single_w, single_h = w // nrow, h // nrow

        # Extract grid basename for unique naming
        base_name = grid_path.stem

        for i in range(nrow):
            for j in range(nrow):
                left = j * single_w
                top = i * single_h
                single_img = grid.crop((left, top, left + single_w, top + single_h))
                img_path = output_dir / f"{base_name}_img_{i * nrow + j}.png"
                single_img.save(img_path)
                single_img.close()  

        grid.close()


def extract_indices(filename):
    match = re.search(r'epoch(\d+)_batch(\d+)_(\d+)\.png', filename)
    if match:
        return int(match.group(1)), int(match.group(2)), int(match.group(3))
    return float('inf'), float('inf'), float('inf')

def create_image_grid(images, title=None):
    fig, axes = plt.subplots(8, 8, figsize=(8, 8))
    if title:
        fig.suptitle(title, fontsize=14)
    for ax, img in zip(axes.flatten(), images):
        ax.imshow(img)
        ax.axis('off')
    for ax in axes.flatten()[len(images):]:
        ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)

    # Save to memory
    buf = BytesIO()
    plt.savefig(buf, format='png')
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)