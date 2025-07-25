import torch
import torch.optim as optim
import os
from pathlib import Path
from models import Generator, Discriminator,Lookahead
from utils import (
    save_checkpoint, 
    load_checkpoint, 
    gradient_penalty, 
    save_samples,
    save_metrics
)
from config import config
from data import get_dataloader

def train_gan(optimizer_name, resume=False):
    """Train GAN with specified optimizer"""
    # Initialize
    device = torch.device(config.device)
    dataloader = get_dataloader(dataset_name=config.dataset, batch_size=config.batch_size)
    
    # Models
    G = Generator(config.latent_dim).to(device)
    D = Discriminator().to(device)
    
    # Optimizer setup
    optimizers = {
        'Adam': (optim.Adam, {'lr': config.lr_G, 'betas': (0.5, 0.999)}),
        'RMSprop': (optim.RMSprop, {'lr': config.lr_G}),
        'SGD': (optim.SGD, {'lr': config.lr_G, 'momentum': 0.9}),
        'Lookahead': (optim.Adam, {'lr': config.lr_G, 'betas': (0.5, 0.999)})
    }
    
    opt_class, opt_params = optimizers[optimizer_name]
    opt_G = opt_class(G.parameters(), **opt_params)
    opt_D = optim.Adam(D.parameters(), lr=config.lr_D, betas=(0.5, 0.999))
    
    if optimizer_name == 'Lookahead':
        opt_G = Lookahead(opt_G, k=5, alpha=0.5)

    # Resume logic
    start_epoch = 0
    losses = {'G_losses': [], 'D_losses': [], 'iter_count': 0}
    
    if resume:
        checkpoint = load_checkpoint(optimizer_name)
        if checkpoint:
            start_epoch = checkpoint['epoch']
            G.load_state_dict(checkpoint['G_state_dict'])
            D.load_state_dict(checkpoint['D_state_dict'])
            opt_G.load_state_dict(checkpoint['opt_G_state_dict'])
            opt_D.load_state_dict(checkpoint['opt_D_state_dict'])
            losses = checkpoint['losses']

    # Training loop
    fixed_noise = torch.randn(64, config.latent_dim, 1, 1, device=device)
    try:
        for epoch in range(start_epoch, config.epochs):
            for batch_idx, (real_imgs, _) in enumerate(dataloader):

                real_imgs = real_imgs.to(device)
                batch_size = real_imgs.size(0)
                
                # --- Discriminator Update ---
                D.zero_grad()
                
                # Real images
                real_pred = D(real_imgs)
                real_loss = -real_pred.mean()
                
                # Fake images
                noise = torch.randn(batch_size, config.latent_dim, 1, 1, device=device)
                fake_imgs = G(noise).detach()
                fake_pred = D(fake_imgs)
                fake_loss = fake_pred.mean()
                
                # Gradient penalty
                gp = gradient_penalty(D, real_imgs, fake_imgs, device)
                
                d_loss = real_loss + fake_loss + gp * config.gp_weight
                d_loss.backward()
                opt_D.step()
                
                # Store discriminator loss for every batch
                losses['D_losses'].append(d_loss.item())
                
                # --- Generator Update ---
                # Only update generator every n_critic steps
                if losses['iter_count'] % config.n_critic == 0:
                    G.zero_grad()
                    noise = torch.randn(batch_size, config.latent_dim, 1, 1, device=device)
                    gen_pred = D(G(noise))
                    g_loss = -gen_pred.mean()
                    g_loss.backward()
                    opt_G.step()
                    
                    # Store generator loss only when it updates
                    losses['G_losses'].append(g_loss.item())
                
                losses['iter_count'] += 1

                # --- Save samples ---
                #if batch_idx % config.sample_interval == 0:
                #    save_all_samples(G, fixed_noise, optimizer_name, epoch, batch_idx)
                if batch_idx % config.sample_interval == 0:
                    noise = torch.randn(config.batch_size, config.latent_dim, 1, 1, device=device)
                    save_samples(G, noise, optimizer_name, epoch, batch_idx)
                # Print progress
                print(f"[{optimizer_name}] Epoch {epoch+1}/{config.epochs} | Batch {batch_idx}/{len(dataloader)} | "
                      f"G Loss: {g_loss.item() if 'g_loss' in locals() else 'N/A':.4f} | "
                      f"D Loss: {d_loss.item():.4f} | Iter: {losses['iter_count']}")

            # Save checkpoint at end of epoch
            if epoch % config.checkpoint_interval == 0:
                save_checkpoint(epoch, G, D, opt_G, opt_D, optimizer_name, losses)
                
    except KeyboardInterrupt:
        print("\nInterrupt detected - saving checkpoint...")
        save_checkpoint(epoch, G, D, opt_G, opt_D, optimizer_name, losses)
        return losses
    
    # Final save
    torch.save(G.state_dict(), os.path.join(config.dirs['models'], f"{optimizer_name}_G.pth"))
    torch.save(D.state_dict(), os.path.join(config.dirs['models'], f"{optimizer_name}_D.pth"))
    save_samples(G, fixed_noise, optimizer_name, 'final')
    save_metrics(losses, optimizer_name)
    
    return losses

if __name__ == "__main__":
    # Verify all directories exist
    for dir_path in config.dirs.values():
        os.makedirs(dir_path, exist_ok=True)
    if "cuda" in config.device and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        config.device = "cpu"
        print("Device is using CPU")
    else:
        print("Device is using Cuda (GPU)")

    print(f"Auto-resume: {config.resume}")
    # Train with all optimizers
    for optimizer in ['Adam', 'RMSprop', 'SGD', 'Lookahead']:
        print(f"\n{'='*40}")
        print(f"Starting training with {optimizer}")
        train_gan(optimizer, resume=config.resume)