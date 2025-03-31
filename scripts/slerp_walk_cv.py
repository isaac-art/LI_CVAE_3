#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
import cv2
import random
import sys
import time
sys.path.append('.')

from models.vae import VAE
from utils.data_utils import get_unsupervised_data_loader

# Global variables
model = None
device = None
latent_dim = None
img_size = None
current_z = None
target_z = None
step_size = 0.01  # How much to move toward target in each step
proximity_threshold = 5  # Threshold for considering "close enough" to target
data_loader = None
precomputed_latents = None

def get_random_dataset_latent():
    """Get a random latent vector from encoding a dataset image or from precomputed latents."""
    global data_loader, precomputed_latents
    
    # Check if precomputed latents exist
    if precomputed_latents is not None:
        # Sample random latent from precomputed ones
        idx = random.randint(0, len(precomputed_latents) - 1)
        latent = precomputed_latents[idx]
        return torch.tensor(latent, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Check if data loader exists
    if data_loader is None:
        # Random latent if no dataset
        print("No dataset or precomputed latents found, returning random latent")
        return torch.randn(1, latent_dim).to(device) * 2.0
    
    try:
        # Get a random image from dataset
        data_iter = iter(data_loader)
        batch = next(data_iter)
        data_image = batch[0] if isinstance(batch, tuple) else batch
        print(f"Got dataset latent")
        # Take first image from batch
        seed_image = data_image[0:1].to(device)
        
        # Encode to get latent vector
        with torch.no_grad():
            mu, logvar = model.encode(seed_image)
            z = model.reparameterize(mu, logvar)
        
        return z
    except Exception as e:
        print(f"Error getting dataset latent: {e}")
        # Fall back to random latent
        return torch.randn(1, latent_dim).to(device) * 2.0

def slerp_step(z1, z2, step_size):
    """Take a step in SLERP interpolation from z1 toward z2."""
    # Normalize vectors for SLERP
    z1_norm = z1 / torch.norm(z1, dim=-1, keepdim=True)
    z2_norm = z2 / torch.norm(z2, dim=-1, keepdim=True)
    
    # Compute the cosine of the angle between the vectors
    cos_omega = torch.sum(z1_norm * z2_norm, dim=-1, keepdim=True).clamp(-1, 1)
    omega = torch.acos(cos_omega)
    sin_omega = torch.sin(omega)
    
    # Create interpolated vector
    if sin_omega.item() < 1e-6:
        # Linear interpolation if vectors are very close or opposite
        interp_z = z1 * (1 - step_size) + z2 * step_size
    else:
        # Actual SLERP
        s1 = torch.sin((1 - step_size) * omega) / sin_omega
        s2 = torch.sin(step_size * omega) / sin_omega
        interp_z = (z1_norm * s1 + z2_norm * s2)
        
        # Preserve magnitude
        z1_mag = torch.norm(z1, dim=-1, keepdim=True)
        z2_mag = torch.norm(z2, dim=-1, keepdim=True)
        interp_mag = z1_mag * (1 - step_size) + z2_mag * step_size
        interp_z = interp_z * interp_mag / torch.norm(interp_z, dim=-1, keepdim=True)
    
    return interp_z

def is_close_enough(z1, z2, threshold):
    """Check if two latent vectors are close enough."""
    # Compute Euclidean distance
    distance = torch.norm(z1 - z2, dim=1).item()
    return distance < threshold

def get_new_target():
    """Get a new random target, either from dataset or random latent space."""
    return get_random_dataset_latent()
    if random.random() < 0.7 and (data_loader is not None or precomputed_latents is not None):
        # Use dataset image as target most of the time if available
        return get_random_dataset_latent()
    else:
        # Sometimes use random latent for diversity
        return torch.randn(1, latent_dim).to(device) * 2.0

def tensor_to_image(tensor):
    """Convert tensor to numpy image for display."""
    # Convert tensor to numpy and scale to 0-255
    img_np = tensor.cpu().detach().numpy().squeeze() * 255
    img_np = img_np.astype(np.uint8)
    
    # If grayscale, convert to RGB
    if len(img_np.shape) == 2:
        img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
    
    return img_np

def initialize_model(args):
    """Initialize the model and latent vectors."""
    global model, device, latent_dim, img_size, current_z, target_z, data_loader, proximity_threshold, precomputed_latents
    
    # Set device
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'mps':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Set parameters
    latent_dim = args.latent_dim
    img_size = args.img_size
    proximity_threshold = args.proximity_threshold
    
    # Load model
    model = VAE(latent_dim=latent_dim).to(device)
    
    # Load checkpoint
    try:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        print(f"Loaded model from {args.checkpoint}")
    except Exception as e:
        print(f"Error loading model: {e}")
        raise
    
    # Set random seed
    seed = args.seed if args.seed is not None else random.randint(0, 10000)
    torch.manual_seed(seed)
    random.seed(seed)
    
    # Load precomputed latents if provided
    if args.latents_file:
        try:
            precomputed_latents = np.load(args.latents_file)
            print(f"Loaded {len(precomputed_latents)} precomputed latents from {args.latents_file}")
        except Exception as e:
            print(f"Error loading precomputed latents: {e}")
            precomputed_latents = None
    
    # Create data loader if data directory is provided and no precomputed latents
    if args.data_dir and precomputed_latents is None:
        try:
            data_loader = get_unsupervised_data_loader(
                args.data_dir,
                batch_size=args.batch_size,
                img_size=img_size,
                num_workers=args.num_workers,
                shuffle=True
            )
            print(f"Created data loader from {args.data_dir}")
        except Exception as e:
            print(f"Error creating data loader: {e}")
            data_loader = None
    
    # Initialize latent vectors
    if args.use_data_image and (data_loader is not None or precomputed_latents is not None):
        current_z = get_random_dataset_latent()
    else:
        current_z = torch.randn(1, latent_dim).to(device)
    
    # Set a random target
    target_z = get_new_target()
    
    return seed

def main(args):
    """Main function."""
    global current_z, target_z, proximity_threshold, step_size
    
    # Initialize model and variables
    seed = initialize_model(args)
    print(f"Using random seed: {seed}")
    
    # Set step size
    step_size = args.step_size
    
    # Create OpenCV window
    window_name = "SLERP Latent Space Walker"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    
    # Resize window
    display_size = args.display_size
    cv2.resizeWindow(window_name, display_size, display_size)
    
    paused = False
    use_dataset_targets = True
    frames_since_target_change = 0
    min_frames_between_targets = 10  # Prevent target changes too quickly
    
    print("Controls:")
    print("  ESC: Quit")
    print("  SPACE: Pause/Resume")
    print("  N: New random target")
    print("  D: Toggle dataset/random targets")
    print("  UP/DOWN: Adjust step size")
    print("  LEFT/RIGHT: Adjust proximity threshold")
    
    while True:
        # Generate the image at the current position
        with torch.no_grad():
            decoded_image = model.decode(current_z)
        
        # Convert to numpy for OpenCV
        display_img = tensor_to_image(decoded_image)
        
        # Add distance info to the image
        distance = torch.norm(current_z - target_z, dim=1).item()
        # cv2.putText(
        #     display_img, 
        #     f"Distance: {distance:.3f} (Threshold: {proximity_threshold:.3f})", 
        #     (10, 30), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 
        #     0.7, 
        #     (255, 255, 255), 
        #     2
        # )
        # cv2.putText(
        #     display_img, 
        #     f"Step size: {step_size:.3f}", 
        #     (10, 60), 
        #     cv2.FONT_HERSHEY_SIMPLEX, 
        #     0.7, 
        #     (255, 255, 255), 
        #     2
        # )
        
        # Display the image
        display_img = cv2.resize(display_img, (128,128))
        canvas = np.zeros((128,128*12,3), dtype=np.uint8)
        # Fill canvas with 12 alternating images
        for i in range(12):
            x_offset = i * 128
            if i % 2 == 0:
                # Normal orientation
                canvas[:, x_offset:x_offset+128] = display_img
            else:
                # Flipped horizontally
                canvas[:, x_offset:x_offset+128] = cv2.flip(display_img, 1)
        
        # Update display_img to show the full canvas
        display_img = canvas
        cv2.imshow(window_name, display_img)
        # print(f"Distance: {distance:.3f} (Threshold: {proximity_threshold:.3f})")
        
        # Process keyboard input (30ms wait)
        key = cv2.waitKey(30) & 0xFF
        
        if key == 27:  # ESC key
            break
        elif key == ord(' '):  # SPACE key
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}")
        elif key == ord('n'):  # N key
            target_z = get_new_target()
            frames_since_target_change = 0
            print("New target generated manually")
        elif key == ord('d'):  # D key
            use_dataset_targets = not use_dataset_targets
            print(f"Using {'dataset' if use_dataset_targets else 'random'} targets")
        elif key == 82:  # UP key
            step_size = min(step_size * 1.2, 0.5)
            print(f"Step size increased to {step_size:.4f}")
        elif key == 84:  # DOWN key
            step_size = max(step_size / 1.2, 0.001)
            print(f"Step size decreased to {step_size:.4f}")
        elif key == 81:  # LEFT key
            proximity_threshold = max(proximity_threshold / 1.2, 0.01)
            print(f"Proximity threshold decreased to {proximity_threshold:.4f}")
        elif key == 83:  # RIGHT key
            proximity_threshold = min(proximity_threshold * 1.2, 5.0)
            print(f"Proximity threshold increased to {proximity_threshold:.4f}")
        
        # Skip the rest if paused
        if paused:
            continue
        
        # Take a step toward the target
        current_z = slerp_step(current_z, target_z, step_size)
        frames_since_target_change += 1
        
        # Check if we're close enough to the target and enough frames have passed
        if frames_since_target_change > min_frames_between_targets:
            if is_close_enough(current_z, target_z, proximity_threshold):
                # Get new target based on preference
                if use_dataset_targets and (precomputed_latents is not None or data_loader is not None):
                    print("Generating dataset target")
                    target_z = get_random_dataset_latent()
                else:
                    print("Generating random target")
                    target_z = torch.randn(1, latent_dim).to(device) * 2.0
                
                frames_since_target_change = 0
                print(f"New target generated (reached proximity threshold {proximity_threshold:.3f})")
        
        # Slight delay to control frame rate
        time.sleep(0.01)
    
    # Cleanup
    cv2.destroyAllWindows()
    print("Visualization stopped.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='OpenCV SLERP Walk Visualizer')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of latent space')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size')
    parser.add_argument('--seed', type=int, default=None,
                        help='Random seed for reproducibility')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Path to image directory')
    parser.add_argument('--use_data_image', action='store_true',
                        help='Initialize with data image')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    parser.add_argument('--latents_file', type=str, default=None,
                        help='Path to precomputed latents file (.npy)')
    
    # Visualization parameters
    parser.add_argument('--display_size', type=int, default=512,
                        help='Size of display window')
    parser.add_argument('--step_size', type=float, default=0.01,
                        help='Step size for SLERP interpolation')
    parser.add_argument('--proximity_threshold', type=float, default=5.0,
                        help='Distance threshold for reaching target')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    main(args) 