import os
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms

import sys
sys.path.append('.')

from models.vae import VAE


def save_image(image, filename):
    """Save a single image."""
    # Convert to numpy and denormalize
    image = image.cpu().detach().numpy()
    # Reshape to [h, w]
    image = image.squeeze()
    
    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def save_grid(images, filename, nrow=8):
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
    if rows * cols > 1:
        axes = axes.flatten()
    else:
        axes = [axes]
    
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


def sample_random(model, args):
    """Generate random samples from the model."""
    model.eval()
    with torch.no_grad():
        samples = model.sample(args.num_samples, device=args.device)
        save_grid(samples, os.path.join(args.output_dir, 'random_samples.png'))
        
        # Save individual samples if requested
        if args.save_individual:
            for i in range(args.num_samples):
                save_image(samples[i], os.path.join(args.output_dir, f'sample_{i}.png'))


def interpolate_latent(model, args):
    """Generate interpolations in latent space."""
    model.eval()
    with torch.no_grad():
        # Generate pairs of random latent vectors
        for i in range(args.num_interpolations):
            z1 = torch.randn(1, model.latent_dim, device=args.device)
            z2 = torch.randn(1, model.latent_dim, device=args.device)
            
            # Interpolate between them
            interpolations = model.interpolate(z1, z2, steps=args.steps)
            
            # Save as grid
            save_grid(
                interpolations, 
                os.path.join(args.output_dir, f'interpolation_{i}.png'),
                nrow=args.steps
            )


def load_and_preprocess_image(image_path, img_size=128):
    """Load and preprocess an image for inpainting."""
    # Load image and convert to grayscale
    image = Image.open(image_path).convert('L')
    
    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    return transform(image).unsqueeze(0)  # Add batch dimension


def inpaint_image(model, image, mask, args):
    """Inpaint the masked regions of the image."""
    model.eval()
    with torch.no_grad():
        # Move to device
        image = image.to(args.device)
        mask = mask.to(args.device)
        
        # Calculate masked image
        masked_image = image * mask
        
        # Encode masked image
        mu, logvar = model.encode(masked_image)
        
        # Generate multiple samples
        samples = []
        for _ in range(args.num_inpaint_samples):
            # Sample from latent distribution
            z = model.reparameterize(mu, logvar)
            
            # Decode
            reconstructed = model.decode(z)
            
            # Combine original (unmasked) and reconstructed (masked) parts
            inpainted = image * mask + reconstructed * (1 - mask)
            samples.append(inpainted)
        
        # Concatenate samples
        samples = torch.cat(samples, dim=0)
        
        # Add original and masked images for comparison
        comparison = torch.cat([image, masked_image, samples], dim=0)
        
        # Save as grid
        save_grid(comparison, os.path.join(args.output_dir, 'inpainting.png'))


def create_mask(img_size=128, mask_type='center'):
    """Create a binary mask for inpainting.
    mask_type: 'center', 'random', 'left', 'right', 'top', 'bottom'
    """
    mask = torch.ones(1, 1, img_size, img_size)
    
    if mask_type == 'center':
        # Center square mask
        s = img_size // 4
        mask[:, :, img_size//2-s:img_size//2+s, img_size//2-s:img_size//2+s] = 0
    elif mask_type == 'random':
        # Random mask with random holes
        mask = torch.ones(1, 1, img_size, img_size)
        n_holes = np.random.randint(1, 5)
        for _ in range(n_holes):
            y = np.random.randint(0, img_size - img_size//4)
            x = np.random.randint(0, img_size - img_size//4)
            h = np.random.randint(img_size//8, img_size//4)
            w = np.random.randint(img_size//8, img_size//4)
            mask[:, :, y:y+h, x:x+w] = 0
    elif mask_type == 'left':
        # Left half mask
        mask[:, :, :, :img_size//2] = 0
    elif mask_type == 'right':
        # Right half mask
        mask[:, :, :, img_size//2:] = 0
    elif mask_type == 'top':
        # Top half mask
        mask[:, :, :img_size//2, :] = 0
    elif mask_type == 'bottom':
        # Bottom half mask
        mask[:, :, img_size//2:, :] = 0
    
    return mask


def main(args):
    """Main function."""
    # Set device
    args.device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    model = VAE(latent_dim=args.latent_dim).to(args.device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded model from {args.checkpoint}")
    
    # Generate random samples
    if args.random_samples:
        print("Generating random samples...")
        sample_random(model, args)
    
    # Generate latent space interpolations
    if args.interpolate:
        print("Generating latent space interpolations...")
        interpolate_latent(model, args)
    
    # Inpaint image
    if args.inpaint:
        print("Inpainting image...")
        if args.image_path is None:
            print("No image provided for inpainting, using a random sample")
            # Generate a random sample to inpaint
            with torch.no_grad():
                image = model.sample(1, device=args.device)
        else:
            # Load and preprocess image
            image = load_and_preprocess_image(args.image_path, img_size=args.img_size)
        
        # Create mask
        mask = create_mask(img_size=args.img_size, mask_type=args.mask_type)
        
        # Inpaint
        inpaint_image(model, image, mask, args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample and inpaint with VAE model')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of latent space')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size')
    
    # Device
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    # Operation flags
    parser.add_argument('--random_samples', action='store_true',
                        help='Generate random samples')
    parser.add_argument('--interpolate', action='store_true',
                        help='Generate latent space interpolations')
    parser.add_argument('--inpaint', action='store_true',
                        help='Inpaint an image')
    
    # Random sampling parameters
    parser.add_argument('--num_samples', type=int, default=64,
                        help='Number of random samples to generate')
    parser.add_argument('--save_individual', action='store_true',
                        help='Save individual samples')
    
    # Interpolation parameters
    parser.add_argument('--num_interpolations', type=int, default=5,
                        help='Number of interpolation sequences to generate')
    parser.add_argument('--steps', type=int, default=10,
                        help='Number of steps in interpolation')
    
    # Inpainting parameters
    parser.add_argument('--image_path', type=str, default=None,
                        help='Path to image to inpaint')
    parser.add_argument('--mask_type', type=str, default='center',
                        choices=['center', 'random', 'left', 'right', 'top', 'bottom'],
                        help='Type of mask to use for inpainting')
    parser.add_argument('--num_inpaint_samples', type=int, default=4,
                        help='Number of inpainting samples to generate')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='samples/vae',
                        help='Directory to save samples')
    
    args = parser.parse_args()
    
    main(args) 