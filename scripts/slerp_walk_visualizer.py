import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64
from io import BytesIO
from PIL import Image
import random
import sys
import threading
import time
import atexit
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
last_image_b64 = None
step_size = 0.05  # How much to move toward target in each step
request_counter = 0  # Counter for tracking requests
request_threshold = 100  # Change target every N requests
walker_thread = None  # Background thread for continuous walking
thread_running = False  # Flag to control the walker thread
image_lock = threading.Lock()  # Lock for thread-safe image access

# Create FastAPI app
app = FastAPI(title="SLERP Latent Space Walker")

# Setup CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files directory
app.mount("/static", StaticFiles(directory="web/slerp_walker/static"), name="static")

# Function to stop the walker thread on shutdown
def stop_walker_thread():
    global thread_running, walker_thread
    if thread_running and walker_thread and walker_thread.is_alive():
        print("Stopping walker thread...")
        thread_running = False
        walker_thread.join(timeout=2.0)
        print("Walker thread stopped.")

# Register shutdown handlers
@app.on_event("shutdown")
async def shutdown_event():
    stop_walker_thread()

# Register with atexit to ensure cleanup on abnormal termination
atexit.register(stop_walker_thread)

class InitModelRequest(BaseModel):
    """Request model for initializing the VAE model."""
    seed: Optional[int] = None
    use_data_image: bool = False

def tensor_to_b64(tensor):
    """Convert tensor to base64 encoded image string."""
    # Convert tensor to numpy and scale to 0-255
    img_np = tensor.cpu().detach().numpy().squeeze() * 255
    img_np = img_np.astype(np.uint8)
    
    # Convert to PIL image
    img = Image.fromarray(img_np)
    
    # Convert to base64
    buffer = BytesIO()
    img.save(buffer, format="PNG")
    img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
    
    return img_str

def get_random_dataset_latent():
    """Get a random latent vector from encoding a dataset image."""
    global data_loader
    
    # Check if data loader exists
    if 'data_loader' not in globals() or data_loader is None:
        # Random latent if no dataset
        return torch.randn(1, latent_dim).to(device) * 2.0
    
    try:
        # Get a random image from dataset
        data_iter = iter(data_loader)
        batch = next(data_iter)
        data_image = batch[0] if isinstance(batch, tuple) else batch
        
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

def get_new_target():
    """Get a new random target, either from dataset or random latent space."""
    if random.random() < 0.7 and 'data_loader' in globals() and data_loader is not None:
        # Use dataset image as target most of the time if available
        return get_random_dataset_latent()
    else:
        # Sometimes use random latent for diversity
        return torch.randn(1, latent_dim).to(device) * 2.0

def continuous_walker():
    """Background thread that continuously walks through latent space."""
    global current_z, target_z, last_image_b64, request_counter, thread_running
    
    local_counter = 0  # Local counter for this thread
    
    print("Starting continuous walker thread")
    
    while thread_running:
        try:
            # Increment local counter
            local_counter += 1
            
            # Check if we should change target
            if local_counter >= request_threshold:
                # Reset counter and get new target
                local_counter = 0
                new_target = get_new_target()
                
                # Update global target under lock
                with image_lock:
                    target_z = new_target
                    # Also update global counter to keep them in sync
                    request_counter = 0
                
                print(f"Walker thread: New target generated")
            
            # Take a step toward the target using SLERP
            with image_lock:
                # Get current copy of vectors under lock
                current_z_copy = current_z.clone()
                target_z_copy = target_z.clone()
            
            # Compute new position (outside lock for better concurrency)
            new_position = slerp_step(current_z_copy, target_z_copy, step_size)
            
            # Generate the image at the new position
            with torch.no_grad():
                decoded_image = model.decode(new_position)
            
            # Convert to base64
            new_image_b64 = tensor_to_b64(decoded_image)
            
            # Update global state under lock
            with image_lock:
                current_z = new_position
                last_image_b64 = new_image_b64
                request_counter = local_counter
        
        except Exception as e:
            print(f"Error in walker thread: {e}")
        
        # Sleep to control frame rate
        time.sleep(0.02)  # 50 FPS target

    print("Walker thread stopped")

@app.get("/", response_class=HTMLResponse)
async def index():
    """Serve the main HTML page."""
    # Read the HTML file directly
    try:
        with open("web/slerp_walker/index.html", "r") as f:
            html_content = f.read()
        return HTMLResponse(content=html_content)
    except Exception as e:
        return HTMLResponse(content=f"<html><body><h1>Error</h1><p>{str(e)}</p></body></html>")

@app.post("/init_model")
async def init_model(request: Optional[InitModelRequest] = Body(default=None)):
    """Initialize or reset the model and latent vectors."""
    global current_z, target_z, last_image_b64, request_counter, thread_running, walker_thread
    
    # Default values if request is None
    if request is None:
        request = InitModelRequest()
        
    # Get random seed from request or generate one
    seed = request.seed if request.seed is not None else random.randint(0, 10000)
    torch.manual_seed(seed)
    
    # Stop existing walker thread if running
    if thread_running and walker_thread and walker_thread.is_alive():
        thread_running = False
        walker_thread.join(timeout=2.0)
    
    with image_lock:
        # Reset request counter
        request_counter = 0
        
        # Initialize from dataset or randomly
        if request.use_data_image and 'data_loader' in globals() and data_loader is not None:
            try:
                # Get a random image from dataset
                current_z = get_random_dataset_latent()
            except Exception as e:
                print(f"Error initializing with data image: {e}")
                # Fall back to random
                current_z = torch.randn(1, latent_dim).to(device)
        else:
            # Random initialization
            current_z = torch.randn(1, latent_dim).to(device)
        
        # Set a random target
        target_z = get_new_target()
        
        # Generate initial image
        with torch.no_grad():
            decoded_image = model.decode(current_z)
        
        # Convert to base64
        last_image_b64 = tensor_to_b64(decoded_image)
    
    # Start the continuous walker thread
    thread_running = True
    walker_thread = threading.Thread(target=continuous_walker, daemon=True)
    walker_thread.start()
    
    return {
        'status': 'success',
        'message': 'Model initialized',
        'seed': seed,
        'image': last_image_b64
    }

@app.get("/get_current_image")
async def get_current_image():
    """Get the current pre-computed image."""
    global last_image_b64
    
    with image_lock:
        if last_image_b64 is None:
            return {
                'status': 'error',
                'message': 'Model not initialized'
            }
        
        # Just return the latest pre-computed image
        return {
            'status': 'success',
            'image': last_image_b64
        }

@app.post("/random_target")
async def random_target():
    """Generate a new random target latent vector."""
    global target_z, request_counter
    
    # Get new target
    new_target = get_new_target()
    
    # Update globals under lock
    with image_lock:
        target_z = new_target
        request_counter = 0
    
    return {
        'status': 'success',
        'message': 'New target generated'
    }

def main(args):
    """Main function."""
    global model, device, latent_dim, img_size, data_loader
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Set parameters
    latent_dim = args.latent_dim
    img_size = args.img_size
    
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
    
    # Create data loader if data directory is provided
    if args.data_dir:
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
    else:
        data_loader = None
    
    # Make sure the template directory exists
    os.makedirs("web/slerp_walker/static", exist_ok=True)
    
    # Start server with uvicorn
    import uvicorn
    uvicorn.run(
        app, 
        host=args.host, 
        port=args.port,
        log_level="info"
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SLERP Walk Visualizer')
    
    # Model parameters
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to checkpoint file')
    parser.add_argument('--latent_dim', type=int, default=64,
                        help='Dimension of latent space')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size')
    
    # Data parameters
    parser.add_argument('--data_dir', type=str, default='images',
                        help='Path to image directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for data loading')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of workers for data loading')
    
    # Server parameters
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host to run the server on')
    parser.add_argument('--port', type=int, default=8080,
                        help='Port to run the server on')
    parser.add_argument('--debug', action="store_true",
                        help='Run in debug mode')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    main(args) 