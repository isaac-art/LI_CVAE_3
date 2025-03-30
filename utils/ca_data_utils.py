import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

class GrayscaleImageDataset(Dataset):
    """Dataset for loading grayscale images for Neural CA training."""
    
    def __init__(self, root_dir, img_size=128):
        self.root_dir = root_dir
        self.img_size = img_size
        
        # Collect all image paths from root_dir and subdirectories
        self.img_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.img_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        
        # Transforms
        self.transform = transforms.Compose([
            transforms.Grayscale(),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
        ])
    
    def __len__(self):
        return len(self.img_paths)
    
    def crop_to_black_edges(self, img_array):
        """Crop the image to the first black pixel on each side."""
        # Find first black pixel on each side (threshold close to 0 for almost black)
        threshold = 10  # Pixel values below this are considered black
        
        # Find first black pixel from left
        left_crop = 0
        for i in range(img_array.shape[1]):
            if np.min(img_array[:, i]) < threshold:
                left_crop = i
                break
                
        # Find first black pixel from right
        right_crop = img_array.shape[1] - 1
        for i in range(img_array.shape[1] - 1, -1, -1):
            if np.min(img_array[:, i]) < threshold:
                right_crop = i
                break
                
        # Find first black pixel from top
        top_crop = 0
        for i in range(img_array.shape[0]):
            if np.min(img_array[i, :]) < threshold:
                top_crop = i
                break
                
        # Find first black pixel from bottom
        bottom_crop = img_array.shape[0] - 1
        for i in range(img_array.shape[0] - 1, -1, -1):
            if np.min(img_array[i, :]) < threshold:
                bottom_crop = i
                break
        
        # Crop the image
        return img_array[top_crop:bottom_crop+1, left_crop:right_crop+1]
    
    def resize_and_pad(self, img_array):
        """Resize to have longest dimension = img_size, pad other dimension with white."""
        # Get current dimensions
        h, w = img_array.shape
        
        # Calculate new dimensions
        if h > w:
            new_h = self.img_size
            new_w = int(w * (self.img_size / h))
        else:
            new_w = self.img_size
            new_h = int(h * (self.img_size / w))
        
        # Resize
        img_resized = np.array(Image.fromarray(img_array).resize((new_w, new_h)))
        
        # Create white canvas
        canvas = np.ones((self.img_size, self.img_size), dtype=np.uint8) * 255
        
        # Calculate position to center the image
        y_offset = (self.img_size - new_h) // 2
        x_offset = (self.img_size - new_w) // 2
        
        # Paste the resized image onto the canvas
        canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = img_resized
        
        return canvas
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        
        # Load image as numpy array in grayscale
        image = np.array(Image.open(img_path).convert('L'))
        
        # Apply preprocessing
        # 1. Crop to black edges
        image = self.crop_to_black_edges(image)
        
        # 2. Resize and pad
        image = self.resize_and_pad(image)
        
        # Convert to tensor (normalize to [0, 1])
        image_tensor = torch.from_numpy(image).float() / 255.0
        
        # Add channel dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor

def get_ca_data_loader(root_dir, batch_size=8, img_size=128, num_workers=4, shuffle=True):
    """Creates a data loader for Neural CA training."""
    dataset = GrayscaleImageDataset(root_dir, img_size=img_size)
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return data_loader 