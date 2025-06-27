import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.utils import make_grid
from PIL import Image
import pandas as pd
from config import config

def plot_losses():
    """Plot and compare loss curves for all optimizers"""
    plt.figure(figsize=(12, 8))
    
    optimizers = ['Adam', 'RMSprop', 'SGD', 'Lookahead']
    colors = ['blue', 'green', 'red', 'purple']
    
    for opt, color in zip(optimizers, colors):
        try:
            metrics_path = os.path.join(config.dirs['metrics'], f"{opt}_metrics.pkl")
            metrics = torch.load(metrics_path)
            
            # Smooth the curves for better visualization
            smooth_window = 50
            g_loss = pd.Series(metrics['G_losses']).rolling(smooth_window).mean()
            d_loss = pd.Series(metrics['D_losses']).rolling(smooth_window).mean()
            
            plt.plot(g_loss, label=f'{opt} Generator', color=color, linestyle='-')
            plt.plot(d_loss, label=f'{opt} Discriminator', color=color, linestyle='--')
        except FileNotFoundError:
            print(f"Metrics not found for {opt}")
    
    plt.title('Generator and Discriminator Losses by Optimizer')
    plt.xlabel('Iterations')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(config.dirs['metrics'], 'all_losses_comparison.png'))
    plt.show()

def display_sample_progression(optimizer):
    """Display sample progression for a specific optimizer"""
    samples_dir = os.path.join(config.dirs['samples'], optimizer)
    if not os.path.exists(samples_dir):
        print(f"No samples found for {optimizer}")
        return
    
    # Get all sample files and sort them
    sample_files = sorted(
        [f for f in os.listdir(samples_dir) if f.endswith('.png')],
        key=lambda x: (int(x.split('_')[0][5:]), int(x.split('_')[1][5:-4]))
    )
    
    # Select a subset to display (first, middle, last)
    selected_indices = [0, len(sample_files)//2, -1]
    selected_files = [sample_files[i] for i in selected_indices]
    
    # Display the samples
    plt.figure(figsize=(15, 5))
    for i, sample_file in enumerate(selected_files):
        img = Image.open(os.path.join(samples_dir, sample_file))
        plt.subplot(1, 3, i+1)
        plt.imshow(img)
        plt.title(f"{optimizer}\n{sample_file[:-4]}")
        plt.axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(config.dirs['metrics'], f'{optimizer}_sample_progression.png'))
    plt.show()

def compare_final_samples():
    """Compare final samples from all optimizers"""
    optimizers = ['Adam', 'RMSprop', 'SGD', 'Lookahead']
    
    plt.figure(figsize=(15, 10))
    for i, opt in enumerate(optimizers):
        samples_dir = os.path.join(config.dirs['samples'], opt)
        if not os.path.exists(samples_dir):
            continue
            
        # Get the last sample
        sample_files = sorted(
            [f for f in os.listdir(samples_dir) if f.endswith('.png')],
            key=lambda x: (int(x.split('_')[0][5:]), int(x.split('_')[1][5:-4]))
        )
        
        if sample_files:
            img = Image.open(os.path.join(samples_dir, sample_files[-1]))
            plt.subplot(2, 2, i+1)
            plt.imshow(img)
            plt.title(opt)
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(config.dirs['metrics'], 'final_samples_comparison.png'))
    plt.show()

def calculate_fid_scores():
    """Calculate FID scores (placeholder - requires real implementation)"""
    print("Note: FID calculation requires additional implementation")
    print("You'll need to install torch-fidelity or use a pretrained Inception network")
    
    # This is just a placeholder structure
    optimizers = ['Adam', 'RMSprop', 'SGD', 'Lookahead']
    fid_scores = {opt: np.random.uniform(10, 100) for opt in optimizers}  # Random values for demo
    
    # Save to CSV
    pd.DataFrame.from_dict(fid_scores, orient='index', columns=['FID Score']).to_csv(
        os.path.join(config.dirs['metrics'], "fid_scores.csv")
    )
    
    return fid_scores

if __name__ == "__main__":
    # Create all visualizations
    plot_losses()
    
    for optimizer in ['Adam', 'RMSprop', 'SGD', 'Lookahead']:
        display_sample_progression(optimizer)
    
    compare_final_samples()
    
    # Calculate FID scores (requires proper implementation)
    calculate_fid_scores()