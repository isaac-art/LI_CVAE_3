#!/usr/bin/env python
import os
import argparse
import torch
import numpy as np
import time
import sys
import threading
import queue
from concurrent.futures import ThreadPoolExecutor
sys.path.append('.')

from models.vae import VAE

def worker_function(model_id, model, latents, device, num_frames, result_queue):
    """Worker function for each model thread"""
    model.eval()
    frames_processed = 0
    start_time = time.time()
    
    for i in range(num_frames):
        idx = i % len(latents)
        z = torch.tensor(latents[idx], dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            _ = model.decode(z)
        frames_processed += 1
    
    end_time = time.time()
    elapsed = end_time - start_time
    fps = frames_processed / elapsed
    
    result_queue.put((model_id, fps, elapsed))
    return fps, elapsed

def benchmark_parallel_models(models, latents, device, num_frames=1000, warmup_frames=100):
    """Benchmark multiple models running in parallel"""
    # Warmup phase
    print("Warming up models...")
    for model in models:
        model.eval()
        for i in range(warmup_frames):
            idx = i % len(latents)
            z = torch.tensor(latents[idx], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = model.decode(z)
    
    # Create a queue for results
    result_queue = queue.Queue()
    
    # Start benchmarking with threads
    print(f"Running benchmark with {len(models)} models in parallel...")
    start_time = time.time()
    
    with ThreadPoolExecutor(max_workers=len(models)) as executor:
        futures = []
        for i, model in enumerate(models):
            future = executor.submit(
                worker_function, 
                i, 
                model, 
                latents, 
                device, 
                num_frames, 
                result_queue
            )
            futures.append(future)
        
        # Wait for all futures to complete
        for future in futures:
            future.result()
    
    # Collect results
    end_time = time.time()
    total_elapsed = end_time - start_time
    
    # Calculate total frames processed
    total_frames = num_frames * len(models)
    total_fps = total_frames / total_elapsed
    
    # Get individual model results
    individual_results = []
    while not result_queue.empty():
        individual_results.append(result_queue.get())
    
    # Sort by model ID
    individual_results.sort(key=lambda x: x[0])
    
    return total_fps, total_elapsed, individual_results

def main(args):
    # Set device
    if args.device == 'cuda':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    elif args.device == 'mps':
        device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    
    # Load precomputed latents
    try:
        latents = np.load(args.latents_file)
        print(f"Loaded {len(latents)} precomputed latents from {args.latents_file}")
    except Exception as e:
        print(f"Error loading precomputed latents: {e}")
        raise
    
    # Create multiple models
    models = []
    for i in range(args.num_models):
        model = VAE(latent_dim=args.latent_dim).to(device)
        
        # Load checkpoint
        try:
            checkpoint = torch.load(args.checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
        except Exception as e:
            print(f"Error loading model {i}: {e}")
            raise
        
        models.append(model)
    
    print(f"Created {len(models)} model instances")
    print(f"\nRunning benchmark with {args.num_frames} frames per model (with {args.warmup_frames} warmup frames)...\n")
    
    # Benchmark all models in parallel
    total_fps, total_elapsed, individual_results = benchmark_parallel_models(
        models,
        latents,
        device,
        num_frames=args.num_frames,
        warmup_frames=args.warmup_frames
    )
    
    # Print results
    print("\nResults:")
    print(f"Total FPS across all models: {total_fps:.2f} frames/second")
    print(f"Total time elapsed: {total_elapsed:.2f} seconds")
    print(f"Total frames processed: {args.num_frames * args.num_models}")
    print(f"Average FPS per model: {total_fps / args.num_models:.2f}")
    
    print("\nIndividual model performance:")
    for model_id, fps, elapsed in individual_results:
        print(f"Model {model_id}: {fps:.2f} FPS ({elapsed:.2f} seconds)")
    
    # Calculate and report GPU memory usage if available
    if torch.cuda.is_available():
        memory_allocated = torch.cuda.memory_allocated(device) / (1024 ** 2)  # MB
        memory_reserved = torch.cuda.memory_reserved(device) / (1024 ** 2)    # MB
        print(f"\nGPU Memory:")
        print(f"  Allocated: {memory_allocated:.2f} MB")
        print(f"  Reserved:  {memory_reserved:.2f} MB")
    
    # If batch processing is requested
    if args.batch_benchmark:
        print("\n\nRunning batch processing benchmark...")
        # Only use one model to test batch processing
        batch_model = models[0]
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
                    _ = batch_model.decode(z)
            
            # Benchmark
            frames_to_process = min(args.num_frames, len(batched_latents) * batch_size)
            batches_to_process = frames_to_process // batch_size
            
            start_time = time.time()
            for i in range(batches_to_process):
                idx = i % len(batched_latents)
                z = torch.tensor(batched_latents[idx], dtype=torch.float32).to(device)
                with torch.no_grad():
                    _ = batch_model.decode(z)
            
            end_time = time.time()
            elapsed = end_time - start_time
            fps = (batches_to_process * batch_size) / elapsed
            
            print(f"  FPS: {fps:.2f} frames/second ({elapsed:.2f} seconds for {batches_to_process * batch_size} frames)")
            print(f"  Equivalent to {fps / batch_size:.2f} FPS per single sample")
    
    # Compare with single model inference if requested
    if args.compare_single:
        print("\n\nComparing with single model inference...")
        single_model = models[0]
        
        # Warmup
        for i in range(args.warmup_frames):
            idx = i % len(latents)
            z = torch.tensor(latents[idx], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = single_model.decode(z)
        
        # Benchmark
        start_time = time.time()
        for i in range(args.num_frames):
            idx = i % len(latents)
            z = torch.tensor(latents[idx], dtype=torch.float32).unsqueeze(0).to(device)
            with torch.no_grad():
                _ = single_model.decode(z)
        
        end_time = time.time()
        single_elapsed = end_time - start_time
        single_fps = args.num_frames / single_elapsed
        
        print(f"Single model FPS: {single_fps:.2f} frames/second")
        print(f"Multi-model efficiency: {(total_fps / args.num_models) / single_fps * 100:.2f}% of single model performance")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Multi-Model FPS Benchmark')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of latent space')
    parser.add_argument('--latents_file', type=str, required=True,
                        help='Path to precomputed latents file (.npy)')
    parser.add_argument('--num_models', type=int, default=4,
                        help='Number of models to run in parallel')
    
    # Benchmark parameters
    parser.add_argument('--num_frames', type=int, default=1000,
                        help='Number of frames to benchmark per model')
    parser.add_argument('--warmup_frames', type=int, default=100,
                        help='Number of warmup frames')
    parser.add_argument('--batch_benchmark', action='store_true',
                        help='Benchmark different batch sizes with a single model')
    parser.add_argument('--compare_single', action='store_true',
                        help='Compare with single model inference')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    main(args) 