#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
import sys
sys.path.append('.')

from models.vae import VAE
from utils.data_utils import get_unsupervised_data_loader
from tqdm import tqdm

def main(args):
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    model = VAE(latent_dim=args.latent_dim).to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Loaded model from {args.checkpoint}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Create data loader
    data_loader = get_unsupervised_data_loader(
        args.data_dir,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        shuffle=False  # No need to shuffle for encoding
    )
    print(f"Created data loader from {args.data_dir}")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process all images and extract latent vectors
    all_latents = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            # Get images from batch
            images = batch[0] if isinstance(batch, tuple) else batch
            images = images.to(device)
            
            # Encode images to get latent vectors
            mu, logvar = model.encode(images)
            z = model.reparameterize(mu, logvar)
            
            # Save individual latents if requested
            if args.save_individual:
                for j, latent in enumerate(z):
                    latent_np = latent.cpu().numpy()
                    np.save(os.path.join(args.output_dir, f"latent_{i * args.batch_size + j}.npy"), latent_np)
            
            # Collect all latents
            all_latents.append(z.cpu().numpy())
    
    # Combine all latents
    all_latents = np.vstack(all_latents)
    print(f"Encoded {all_latents.shape[0]} images to latent vectors of dimension {all_latents.shape[1]}")
    
    # Save combined latents
    np.save(os.path.join(args.output_dir, "all_latents.npy"), all_latents)
    print(f"Saved all latents to {os.path.join(args.output_dir, 'all_latents.npy')}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Precompute latent vectors from images')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of latent space')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to image directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Directory to save latent vectors')
    parser.add_argument('--save_individual', action='store_true',
                        help='Save individual latent vectors in addition to combined')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    main(args) 