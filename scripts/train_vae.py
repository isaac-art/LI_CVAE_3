import os
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import sys
sys.path.append('.')

from models.vae import VAE
from utils.data_utils import get_unsupervised_data_loader


def save_images(images, filename, nrow=8):
    """Save a grid of images."""
    # Convert to numpy and denormalize
    images = images.cpu().detach().numpy()
    # Reshape to [n, h, w]
    images = images.squeeze(1)
    
    # Create a grid of images
    n = min(images.shape[0], nrow * nrow)
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))
    
    # Create figure
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2 * rows))
    axes = axes.flatten()
    
    # Plot each image
    for i in range(n):
        axes[i].imshow(images[i], cmap='gray')
        axes[i].axis('off')
    
    # Hide empty subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def train(args):
    """Train the VAE model."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loader
    data_loader = get_unsupervised_data_loader(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    # Create model
    model = VAE(latent_dim=args.latent_dim).to(device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Create scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=5, 
        verbose=True
    )
    
    # Create checkpoint directory
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_kld_loss = 0
        
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, data in enumerate(progress_bar):
            # Move data to device
            data = data.to(device)
            
            # Forward pass
            recon_batch, mu, logvar = model(data)
            
            # Compute loss
            loss, recon_loss, kld_loss = model.loss_function(recon_batch, data, mu, logvar)
            
            # Backward pass
            optimizer.zero_grad()
            loss = loss / len(data)  # Normalize by batch size
            loss.backward()
            optimizer.step()
            
            # Update metrics
            global_step += 1
            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item() / len(data)
            epoch_kld_loss += kld_loss.item() / len(data)
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': loss.item(),
                'recon_loss': recon_loss.item() / len(data),
                'kld_loss': kld_loss.item() / len(data)
            })
            
            # Generate samples
            if batch_idx % args.sample_every == 0:
                model.eval()
                with torch.no_grad():
                    # Reconstruct some training examples
                    n = min(8, data.size(0))
                    comparison = torch.cat([
                        data[:n],
                        recon_batch[:n]
                    ])
                    save_images(
                        comparison,
                        os.path.join(args.sample_dir, f'reconstruction_epoch{epoch+1}_step{global_step}.png'),
                        nrow=n
                    )
                    
                    # Generate random samples
                    sample = model.sample(64, device=device)
                    save_images(
                        sample,
                        os.path.join(args.sample_dir, f'sample_epoch{epoch+1}_step{global_step}.png')
                    )
                    
                    # Create latent space interpolation
                    z1 = torch.randn(1, args.latent_dim, device=device)
                    z2 = torch.randn(1, args.latent_dim, device=device)
                    interpolation = model.interpolate(z1, z2, steps=8)
                    save_images(
                        interpolation,
                        os.path.join(args.sample_dir, f'interpolation_epoch{epoch+1}_step{global_step}.png')
                    )
                model.train()
        
        # Compute epoch metrics
        epoch_loss /= len(data_loader)
        epoch_recon_loss /= len(data_loader)
        epoch_kld_loss /= len(data_loader)
        
        # Update scheduler
        scheduler.step(epoch_loss)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}, "
              f"Recon Loss: {epoch_recon_loss:.4f}, KLD Loss: {epoch_kld_loss:.4f}")
        
        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(args.checkpoint_dir, 'best_model.pt'))
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(args.checkpoint_dir, f'model_epoch{epoch+1}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train VAE model')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='images',
                        help='Path to image directory')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size')
    
    # Model parameters
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of latent space')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Saving parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/vae',
                        help='Directory to save checkpoints')
    parser.add_argument('--sample_dir', type=str, default='samples/vae',
                        help='Directory to save samples')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--sample_every', type=int, default=50,
                        help='Generate samples every N batches')
    
    args = parser.parse_args()
    
    train(args) 