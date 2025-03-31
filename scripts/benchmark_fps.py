#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
import time
import sys
sys.path.append('.')

from models.vae import VAE

def benchmark_decoding(model, latents, device, num_frames=1000, warmup_frames=100):
    """Benchmark the FPS of decoding latent vectors."""
    model.eval()
    
    # Warmup
    for i in range(warmup_frames):
        idx = i % len(latents)
        z = torch.tensor(latents[idx], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model.decode(z)
    
    # Benchmark
    start_time = time.time()
    for i in range(num_frames):
        idx = i % len(latents)
        z = torch.tensor(latents[idx], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model.decode(z)
    
    end_time = time.time()
    elapsed = end_time - start_time
    fps = num_frames / elapsed
    
    return fps, elapsed

def benchmark_interpolation(model, latents, device, num_frames=1000, warmup_frames=100, step_size=0.05):
    """Benchmark the FPS of interpolating between latent vectors."""
    model.eval()
    
    # Create pairs of latents for interpolation
    num_pairs = min(100, len(latents) // 2)
    pairs = []
    for i in range(num_pairs):
        idx1 = i * 2
        idx2 = i * 2 + 1
        if idx2 < len(latents):
            pairs.append((latents[idx1], latents[idx2]))
    
    # Warmup
    current_z = torch.tensor(pairs[0][0], dtype=torch.float32).unsqueeze(0).to(device)
    for i in range(warmup_frames):
        pair_idx = (i // 20) % len(pairs)  # Change target every 20 frames
        target_z = torch.tensor(pairs[pair_idx][1], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Interpolate
        current_z = slerp_step(current_z, target_z, step_size)
        
        # Decode
        with torch.no_grad():
            _ = model.decode(current_z)
    
    # Benchmark
    current_z = torch.tensor(pairs[0][0], dtype=torch.float32).unsqueeze(0).to(device)
    start_time = time.time()
    for i in range(num_frames):
        pair_idx = (i // 20) % len(pairs)  # Change target every 20 frames
        target_z = torch.tensor(pairs[pair_idx][1], dtype=torch.float32).unsqueeze(0).to(device)
        
        # Interpolate
        current_z = slerp_step(current_z, target_z, step_size)
        
        # Decode
        with torch.no_grad():
            _ = model.decode(current_z)
    
    end_time = time.time()
    elapsed = end_time - start_time
    fps = num_frames / elapsed
    
    return fps, elapsed

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

def main(args):
    # Set device
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'mps':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
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
    
    # Load precomputed latents
    try:
        latents = np.load(args.latents_file)
        print(f"Loaded {len(latents)} precomputed latents from {args.latents_file}")
    except Exception as e:
        print(f"Error loading precomputed latents: {e}")
        raise
    
    print(f"\nRunning benchmarks for {args.num_frames} frames (with {args.warmup_frames} warmup frames)...\n")
    
    # Benchmark decoding
    decode_fps, decode_time = benchmark_decoding(
        model, 
        latents, 
        device, 
        num_frames=args.num_frames, 
        warmup_frames=args.warmup_frames
    )
    print(f"Decoding-only FPS: {decode_fps:.2f} frames/second ({decode_time:.2f} seconds for {args.num_frames} frames)")
    
    # Benchmark interpolation
    interp_fps, interp_time = benchmark_interpolation(
        model, 
        latents, 
        device, 
        num_frames=args.num_frames, 
        warmup_frames=args.warmup_frames,
        step_size=args.step_size
    )
    print(f"Interpolation+Decoding FPS: {interp_fps:.2f} frames/second ({interp_time:.2f} seconds for {args.num_frames} frames)")
    
    # Run with different batch sizes if requested
    if args.batch_benchmark:
        print("\nBenchmarking different batch sizes:")
        batch_sizes = [1, 2, 4, 8, 16]
        
        for batch_size in batch_sizes:
            # Skip if not enough latents
            if batch_size > len(latents):
                print(f"Skipping batch size {batch_size} (not enough latents)")
                continue
                
            print(f"\nBatch size: {batch_size}")
            
            # Prepare batched latents
            batched_latents = []
            for i in range(0, len(latents) - batch_size + 1, batch_size):
                batch = latents[i:i+batch_size]
                batched_latents.append(batch)
            
            # Warmup
            for i in range(min(args.warmup_frames, len(batched_latents))):
                idx = i % len(batched_latents)
                z = torch.tensor(batched_latents[idx], dtype=torch.float32).to(device)
                with torch.no_grad():
                    _ = model.decode(z)
            
            # Benchmark
            frames_to_process = min(args.num_frames, len(batched_latents) * batch_size)
            batches_to_process = frames_to_process // batch_size
            
            start_time = time.time()
            for i in range(batches_to_process):
                idx = i % len(batched_latents)
                z = torch.tensor(batched_latents[idx], dtype=torch.float32).to(device)
                with torch.no_grad():
                    _ = model.decode(z)
            
            end_time = time.time()
            elapsed = end_time - start_time
            fps = (batches_to_process * batch_size) / elapsed
            
            print(f"FPS: {fps:.2f} frames/second ({elapsed:.2f} seconds for {batches_to_process * batch_size} frames)")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='FPS Benchmark for VAE Decoding')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of latent space')
    parser.add_argument('--latents_file', type=str, required=True,
                        help='Path to precomputed latents file (.npy)')
    
    # Benchmark parameters
    parser.add_argument('--num_frames', type=int, default=1000,
                        help='Number of frames to benchmark')
    parser.add_argument('--warmup_frames', type=int, default=100,
                        help='Number of warmup frames')
    parser.add_argument('--step_size', type=float, default=0.05,
                        help='Step size for SLERP interpolation')
    parser.add_argument('--batch_benchmark', action='store_true',
                        help='Benchmark different batch sizes')
    parser.add_argument('--device', type=str, default='cpu',
                        help='Device to use (cuda/mps/cpu)')
    
    args = parser.parse_args()
    
    main(args) 