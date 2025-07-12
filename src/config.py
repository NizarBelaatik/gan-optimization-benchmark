import torch
from pathlib import Path
import os
import glob

class Config:
    version = "1.2"
    git_hash = os.getenv("GIT_HASH", "dev")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    latent_dim = 100
    batch_size = 64
    epochs = 15
    lr_G = 0.0002
    lr_D = 0.0002
    gp_weight = 10
    n_critic = 5
    sample_interval = 5
    checkpoint_interval = 5
    sample_dir = "samples"
    dataset = "cifar10"
    resume=False
    base_dir = Path("outputs")
    dirs = {
        'models': base_dir/"models",
        'samples': base_dir/"samples",
        'checkpoints': base_dir/"checkpoints", 
        'metrics': base_dir/"metrics",
        'results': base_dir/"results",
    }
    
    

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