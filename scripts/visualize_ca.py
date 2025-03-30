import os
import argparse
import torch
import json
import numpy as np
from PIL import Image

import sys
sys.path.append('.')

from models.neural_ca import CAModel


def export_model_for_browser(model, save_dir, n_steps=100, img_size=128, step_size=1):
    """Export model states for visualization in browser."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Export model parameters
    model_info = {
        "channels": model.total_channels,
        "hidden_channels": model.hidden_channels,
        "img_size": img_size,
        "step_size": step_size
    }
    
    with open(os.path.join(save_dir, 'model_info.json'), 'w') as f:
        json.dump(model_info, f)
    
    # Create initial state and run simulation
    device = next(model.parameters()).device
    x = model.init_state(batch_size=1, device=device)
    
    # Export states at each step
    for i in range(0, n_steps, step_size):
        # Extract visible channel
        visible = model.get_image(x)[0, 0].cpu().detach().numpy()
        
        # Save as PNG
        img = Image.fromarray((visible * 255).astype(np.uint8))
        img.save(os.path.join(save_dir, f'step_{i:04d}.png'))
        
        # Run next steps
        x = model(x, steps=step_size)
    
    print(f"Exported {n_steps // step_size} states to {save_dir}")


def export_regeneration_sequence(model, save_dir, damage_size=32, n_steps=100, img_size=128, step_size=1):
    """Export regeneration sequence for visualization in browser."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Get device from model
    device = next(model.parameters()).device
    
    # First grow the pattern
    x = model.init_state(batch_size=1, device=device)
    x = model(x, steps=64)  # Grow steps
    
    # Save fully grown pattern
    grown_img = model.get_image(x)[0, 0].cpu().detach().numpy()
    img = Image.fromarray((grown_img * 255).astype(np.uint8))
    img.save(os.path.join(save_dir, 'grown.png'))
    
    # Damage the pattern
    x_damaged = model.damage_cells(x, damage_size=damage_size)
    
    # Save damaged pattern
    damaged_img = model.get_image(x_damaged)[0, 0].cpu().detach().numpy()
    img = Image.fromarray((damaged_img * 255).astype(np.uint8))
    img.save(os.path.join(save_dir, 'damaged.png'))
    
    # Run regeneration steps and save at intervals
    x_regen = x_damaged.clone()
    for i in range(0, n_steps, step_size):
        # Extract visible channel
        visible = model.get_image(x_regen)[0, 0].cpu().detach().numpy()
        
        # Save as PNG
        img = Image.fromarray((visible * 255).astype(np.uint8))
        img.save(os.path.join(save_dir, f'regen_{i:04d}.png'))
        
        # Run next steps
        x_regen = model(x_regen, steps=step_size)
    
    print(f"Exported regeneration sequence with {n_steps // step_size} steps to {save_dir}")


def main(args):
    """Main function to export model for visualization."""
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create model
    model = CAModel(hidden_channels=args.hidden_channels).to(device)
    
    # Load checkpoint
    print(f"Loading model from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Export model
    if args.mode == 'growing':
        export_model_for_browser(
            model,
            os.path.join(args.output_dir, 'growing'),
            n_steps=args.n_steps,
            img_size=args.img_size,
            step_size=args.step_size
        )
    elif args.mode == 'regeneration':
        export_regeneration_sequence(
            model,
            os.path.join(args.output_dir, 'regeneration'),
            damage_size=args.damage_size,
            n_steps=args.n_steps,
            img_size=args.img_size,
            step_size=args.step_size
        )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Export Neural CA for browser visualization')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--hidden_channels', type=int, default=16,
                        help='Number of hidden channels in the model')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size')
    
    # Export parameters
    parser.add_argument('--mode', type=str, default='growing',
                        choices=['growing', 'regeneration'],
                        help='Mode to visualize')
    parser.add_argument('--output_dir', type=str, default='web',
                        help='Directory to save exports')
    parser.add_argument('--n_steps', type=int, default=100,
                        help='Number of steps to export')
    parser.add_argument('--step_size', type=int, default=1,
                        help='Number of steps to simulate between exports')
    parser.add_argument('--damage_size', type=int, default=32,
                        help='Size of damage area for regeneration')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    main(args) 