import os
import argparse
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time

import sys
sys.path.append('.')

from models.neural_ca import CAModel
from utils.ca_data_utils import get_ca_data_loader


def save_images(images, filename, titles=None):
    """Save a grid of images."""
    # Convert to numpy and denormalize
    images = images.cpu().detach().numpy()
    # Reshape to [n, h, w]
    images = images.squeeze(1)
    
    # Create figure with subplots
    n = images.shape[0]
    rows = int(np.sqrt(n))
    cols = int(np.ceil(n / rows))
    
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 2, rows * 2))
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
    # Plot each image
    for i in range(n):
        axes[i].imshow(images[i], cmap='gray')
        if titles is not None and i < len(titles):
            axes[i].set_title(titles[i])
        axes[i].axis('off')
    
    # Hide remaining empty subplots
    for i in range(n, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()


def train_growing(args):
    """Train the Neural CA model to grow from a seed to target images."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loader
    data_loader = get_ca_data_loader(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    # Create model
    model = CAModel(hidden_channels=args.hidden_channels, fire_rate=args.fire_rate).to(device)
    
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
    
    # Create checkpoint and sample directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    # Create pool of target images (a subset of the data loader)
    target_pool = []
    for batch in data_loader:
        target_pool.append(batch)
        if len(target_pool) * args.batch_size >= args.pool_size:
            break
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(range(args.iters_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")
        for _ in progress_bar:
            # Randomly select a target batch
            target_batch = target_pool[np.random.randint(len(target_pool))].to(device)
            
            # Initialize cell states
            x = model.init_state(target_batch.size(0), device)
            
            # Run for a random number of steps (to make the model work for variable steps)
            n_steps = np.random.randint(args.min_steps, args.max_steps + 1)
            
            # Simulate for n_steps
            x = model(x, steps=n_steps)
            
            # Get the image channel from the state
            pred_img = model.get_image(x)
            
            # Loss computation
            loss = F.mse_loss(pred_img, target_batch)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Generate samples
            if global_step % args.sample_every == 0:
                # Select a fixed target for visualization
                target = target_batch[0:1]
                
                # Visualize growth process
                x = model.init_state(1, device)
                images = [model.get_image(x)]
                
                for step in range(0, args.max_steps + 1, 10):
                    x = model(x, steps=10)
                    images.append(model.get_image(x))
                
                # Add target
                images.append(target)
                
                # Concatenate and save
                images = torch.cat(images, dim=0)
                titles = ['Seed'] + [f'Step {s}' for s in range(10, args.max_steps + 1, 10)] + ['Target']
                save_images(
                    images,
                    os.path.join(args.sample_dir, f'growing_epoch{epoch+1}_step{global_step}.png'),
                    titles=titles
                )
        
        # Compute epoch metrics
        epoch_loss /= batch_count
        
        # Update scheduler
        scheduler.step(epoch_loss)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(args.checkpoint_dir, 'best_model_growing.pt'))
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(args.checkpoint_dir, f'model_growing_epoch{epoch+1}.pt'))


def train_persistent(args):
    """Train the Neural CA model to persist and maintain patterns."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loader
    data_loader = get_ca_data_loader(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    # Create model
    model = CAModel(hidden_channels=args.hidden_channels, fire_rate=args.fire_rate).to(device)
    
    # Load from growing model if specified
    if args.load_growing:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_model_growing.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded pre-trained growing model")
    
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
    
    # Create checkpoint and sample directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    # Create pool of target images (a subset of the data loader)
    target_pool = []
    for batch in data_loader:
        target_pool.append(batch)
        if len(target_pool) * args.batch_size >= args.pool_size:
            break
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(range(args.iters_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")
        for _ in progress_bar:
            # Randomly select a target batch
            target_batch = target_pool[np.random.randint(len(target_pool))].to(device)
            
            # First grow the pattern
            x = model.init_state(target_batch.size(0), device)
            x = model(x, steps=args.grow_steps)
            
            # Then check if it persists
            initial_images = model.get_image(x)
            
            # Run for persistence steps
            x = model(x, steps=args.persist_steps)
            final_images = model.get_image(x)
            
            # Loss computation - a combination of:
            # 1. Final image should match target
            # 2. Final image should match initial image (persistence)
            loss_target = F.mse_loss(final_images, target_batch)
            loss_persist = F.mse_loss(final_images, initial_images)
            loss = loss_target + loss_persist
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Generate samples
            if global_step % args.sample_every == 0:
                # Select a fixed target for visualization
                target = target_batch[0:1]
                
                # Visualize growth and persistence
                x = model.init_state(1, device)
                
                # Grow phase
                grow_images = [model.get_image(x)]
                for _ in range(args.grow_steps // 10):
                    x = model(x, steps=10)
                    grow_images.append(model.get_image(x))
                
                # After growth state
                post_grow_state = x.clone()
                
                # Persistence phase
                persist_images = []
                for _ in range(args.persist_steps // 20):
                    x = model(x, steps=20)
                    persist_images.append(model.get_image(x))
                
                # Add target
                all_images = torch.cat(grow_images + persist_images + [target], dim=0)
                
                # Titles
                grow_titles = ['Seed'] + [f'Grow {s}' for s in range(10, args.grow_steps + 1, 10)]
                persist_titles = [f'Persist {s}' for s in range(20, args.persist_steps + 1, 20)]
                titles = grow_titles + persist_titles + ['Target']
                
                save_images(
                    all_images,
                    os.path.join(args.sample_dir, f'persistent_epoch{epoch+1}_step{global_step}.png'),
                    titles=titles
                )
        
        # Compute epoch metrics
        epoch_loss /= batch_count
        
        # Update scheduler
        scheduler.step(epoch_loss)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(args.checkpoint_dir, 'best_model_persistent.pt'))
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(args.checkpoint_dir, f'model_persistent_epoch{epoch+1}.pt'))


def train_regeneration(args):
    """Train the Neural CA model to regenerate damaged patterns."""
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create data loader
    data_loader = get_ca_data_loader(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        shuffle=True
    )
    
    # Create model
    model = CAModel(hidden_channels=args.hidden_channels, fire_rate=args.fire_rate).to(device)
    
    # Load from persistent model if specified
    if args.load_persistent:
        checkpoint = torch.load(os.path.join(args.checkpoint_dir, 'best_model_persistent.pt'))
        model.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded pre-trained persistent model")
    
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
    
    # Create checkpoint and sample directories
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.sample_dir, exist_ok=True)
    
    # Training loop
    global_step = 0
    best_loss = float('inf')
    
    # Create pool of target images (a subset of the data loader)
    target_pool = []
    for batch in data_loader:
        target_pool.append(batch)
        if len(target_pool) * args.batch_size >= args.pool_size:
            break
    
    for epoch in range(args.epochs):
        epoch_loss = 0
        batch_count = 0
        
        progress_bar = tqdm(range(args.iters_per_epoch), desc=f"Epoch {epoch+1}/{args.epochs}")
        for _ in progress_bar:
            # Randomly select a target batch
            target_batch = target_pool[np.random.randint(len(target_pool))].to(device)
            
            # First grow the pattern
            x = model.init_state(target_batch.size(0), device)
            x = model(x, steps=args.grow_steps)
            
            # Damage the pattern
            x_damaged = model.damage_cells(x, damage_size=args.damage_size)
            
            # Run regeneration steps
            x_regen = model(x_damaged, steps=args.regen_steps)
            
            # Get images
            initial_images = model.get_image(x)
            damaged_images = model.get_image(x_damaged)
            final_images = model.get_image(x_regen)
            
            # Loss computation - regenerated pattern should match original pattern
            loss = F.mse_loss(final_images, initial_images)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss += loss.item()
            batch_count += 1
            global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Generate samples
            if global_step % args.sample_every == 0:
                # Select a single example
                target = target_batch[0:1]
                
                # Visualize growth, damage, and regeneration
                x = model.init_state(1, device)
                
                # Grow phase - fast forward
                x = model(x, steps=args.grow_steps)
                grown_image = model.get_image(x)
                
                # Apply damage
                x_damaged = model.damage_cells(x, damage_size=args.damage_size)
                damaged_image = model.get_image(x_damaged)
                
                # Regeneration phase
                regen_images = [damaged_image]
                x_regen = x_damaged.clone()
                for _ in range(args.regen_steps // 10):
                    x_regen = model(x_regen, steps=10)
                    regen_images.append(model.get_image(x_regen))
                
                # Concatenate images
                all_images = torch.cat([grown_image] + regen_images + [target], dim=0)
                
                # Titles
                titles = ['Grown'] + ['Damaged'] + [f'Regen {s}' for s in range(10, args.regen_steps, 10)] + ['Target']
                
                save_images(
                    all_images,
                    os.path.join(args.sample_dir, f'regeneration_epoch{epoch+1}_step{global_step}.png'),
                    titles=titles
                )
        
        # Compute epoch metrics
        epoch_loss /= batch_count
        
        # Update scheduler
        scheduler.step(epoch_loss)
        
        # Print metrics
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {epoch_loss:.4f}")
        
        # Save checkpoint
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(args.checkpoint_dir, 'best_model_regeneration.pt'))
        
        # Save regular checkpoint
        if (epoch + 1) % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, os.path.join(args.checkpoint_dir, f'model_regeneration_epoch{epoch+1}.pt'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Neural CA model')
    
    # Training mode
    parser.add_argument('--mode', type=str, default='growing',
                        choices=['growing', 'persistent', 'regeneration'],
                        help='Training mode')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='images',
                        help='Path to image directory')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--pool_size', type=int, default=32,
                        help='Number of target images to use for training')
    
    # Model parameters
    parser.add_argument('--hidden_channels', type=int, default=16,
                        help='Number of hidden channels')
    parser.add_argument('--fire_rate', type=float, default=0.5,
                        help='Cell update probability')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=8,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--iters_per_epoch', type=int, default=100,
                        help='Number of iterations per epoch')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Steps parameters
    parser.add_argument('--min_steps', type=int, default=32,
                        help='Minimum number of steps for growing')
    parser.add_argument('--max_steps', type=int, default=64,
                        help='Maximum number of steps for growing')
    parser.add_argument('--grow_steps', type=int, default=64,
                        help='Number of steps for growth phase')
    parser.add_argument('--persist_steps', type=int, default=128,
                        help='Number of steps for persistence phase')
    parser.add_argument('--regen_steps', type=int, default=64,
                        help='Number of steps for regeneration phase')
    parser.add_argument('--damage_size', type=int, default=32,
                        help='Size of damage area')
    
    # Saving parameters
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/neural_ca',
                        help='Directory to save checkpoints')
    parser.add_argument('--sample_dir', type=str, default='samples/neural_ca',
                        help='Directory to save samples')
    parser.add_argument('--save_every', type=int, default=5,
                        help='Save checkpoint every N epochs')
    parser.add_argument('--sample_every', type=int, default=20,
                        help='Generate samples every N batches')
    
    # Loading parameters
    parser.add_argument('--load_growing', action='store_true',
                        help='Load pretrained growing model for persistent training')
    parser.add_argument('--load_persistent', action='store_true',
                        help='Load pretrained persistent model for regeneration training')
    
    args = parser.parse_args()
    
    # Run the appropriate training mode
    if args.mode == 'growing':
        train_growing(args)
    elif args.mode == 'persistent':
        train_persistent(args)
    elif args.mode == 'regeneration':
        train_regeneration(args) 