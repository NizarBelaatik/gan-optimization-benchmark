import torch
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import pandas as pd
import os
from config import config
from pathlib import Path

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
    checkpoint_path = os.path.join(
        config.dirs['checkpoints'],
        f"{optimizer_name}_checkpoint.pth"  # Consistent naming
    )
    torch.save({
        'epoch': epoch,
        'G_state_dict': G.state_dict(),
        'D_state_dict': D.state_dict(),
        'opt_G_state_dict': opt_G.state_dict(),
        'opt_D_state_dict': opt_D.state_dict(),
        'G_losses': losses["G_losses"],
        'D_losses': losses["D_losses"]
    }, checkpoint_path)

def load_checkpoint(optimizer_name):
    """Load training checkpoint"""
    checkpoint_path = os.path.join(
        config.dirs['checkpoints'],
        f"{optimizer_name}_checkpoint.pth"  # Match this with save_checkpoint()
    )
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path)
        return (
            checkpoint['epoch'],
            checkpoint['G_state_dict'],
            checkpoint['D_state_dict'],
            checkpoint['opt_G_state_dict'],
            checkpoint['opt_D_state_dict'],
            {'G_losses': checkpoint['G_losses'], 
             'D_losses': checkpoint['D_losses']}
        )
    return None

def save_samples(G, noise, optimizer_name, epoch):
    """Save generated samples"""
    ensure_dir(config.dirs['samples'])
    samples_path = os.path.join(
        config.dirs['samples'],
        f"{optimizer_name}_epoch{epoch}.png"
    )
    with torch.no_grad():
        samples = G(noise)
        save_image(samples, samples_path, nrow=8, normalize=True)

def save_metrics(metrics, optimizer_name):
    """Save training metrics"""
    ensure_dir(config.dirs['metrics'])
    metric_path = os.path.join(
        config.dirs['metrics'],
        f"{optimizer_name}_metrics.pkl"
    )
    torch.save(metrics, metric_path)

def save_results(results, optimizers):
    """Save final results"""
    ensure_dir(config.dirs['metrics'])
    
    # Save FID scores
    fid_data = {opt: results[opt].get("FID", "Not calculated") for opt in optimizers}
    pd.DataFrame.from_dict(fid_data, orient='index', columns=['FID Score']).to_csv(
        os.path.join(config.dirs['metrics'], "fid_scores.csv")
    )
    
    # Save loss curves
    for opt in optimizers:
        loss_df = pd.DataFrame({
            'Generator Loss': results[opt]["G_losses"],
            'Discriminator Loss': results[opt]["D_losses"]
        })
        loss_df.to_csv(
            os.path.join(config.dirs['metrics'], f"{opt}_losses.csv")
        )
        
        plt.figure(figsize=(10, 5))
        plt.plot(results[opt]["G_losses"], label='Generator Loss')
        plt.plot(results[opt]["D_losses"], label='Discriminator Loss')
        plt.title(f'Training Losses - {opt}')
        plt.xlabel('Iterations')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(
            os.path.join(config.dirs['metrics'], f"{opt}_losses.png")
        )
        plt.close()