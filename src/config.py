import torch
from pathlib import Path
import os
import glob

class Config:
    version = "1.0"  # Track code versions
    git_hash = os.getenv("GIT_HASH", "dev")  # For reproducibility

    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 100
    batch_size = 64
    epochs = 15
    lr_G = 0.0002
    lr_D = 0.0002
    gp_weight = 10  # Gradient penalty weight
    n_critic = 5    # Number of D updates per G update
    sample_interval = 5
    sample_dir = "samples"
    dataset = "cifar10"  # or "mnist" 

    base_dir = Path("outputs")
    dirs = {
        'models': base_dir/"models",
        'samples': base_dir/"samples",
        'checkpoints': base_dir/"checkpoints", 
        'metrics': base_dir/"metrics",
        'logs': base_dir/"logs",
        'real_images':base_dir/"real_images"
    }
    
    @property
    def resume2(self):
        """Check for ANY optimizer's checkpoint"""
        checkpoint_path = os.path.join(self.dirs['checkpoints'], "*_checkpoint.pth")
        return len(glob.glob(checkpoint_path)) > 0
    
    @property
    def resume(self):
        checkpoint_dir = self.dirs['checkpoints']
        return any(
            os.path.exists(os.path.join(checkpoint_dir, f"{opt}_checkpoint.pth"))
            for opt in ['Adam', 'RMSprop', 'SGD', 'Lookahead']
        )
    
    def __init__(self):
        # Create all directories
        for dir_path in self.dirs.values():
            os.makedirs(dir_path, exist_ok=True)
            
        # Create data directory if not exists
        os.makedirs("data", exist_ok=True)
config = Config()