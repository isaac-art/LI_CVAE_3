import os
import glob
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

def get_class_mapping(root_dir):
    """Creates a mapping between class folder names and numeric indices."""
    class_dirs = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_dirs)}
    return class_to_idx, class_dirs

# Custom augmentation functions
def mixup(img1, img2, label1, label2, alpha=0.2):
    """Performs mixup augmentation between two images."""
    # Generate mixup coefficient
    lam = np.random.beta(alpha, alpha)
    # Mix images and labels
    mixed_img = lam * img1 + (1 - lam) * img2
    return mixed_img, label1, label2, lam

def cutmix(img1, img2, label1, label2, alpha=1.0):
    """Performs cutmix augmentation between two images."""
    # Generate random box parameters
    lam = np.random.beta(alpha, alpha)
    
    # Get image dimensions (assuming square images)
    _, h, w = img1.shape
    
    # Calculate box size
    cut_rat = np.sqrt(1. - lam)
    cut_w = int(w * cut_rat)
    cut_h = int(h * cut_rat)
    
    # Get random center of box
    cx = np.random.randint(w)
    cy = np.random.randint(h)
    
    # Calculate box boundaries
    bbx1 = np.clip(cx - cut_w // 2, 0, w)
    bby1 = np.clip(cy - cut_h // 2, 0, h)
    bbx2 = np.clip(cx + cut_w // 2, 0, w)
    bby2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Create mask
    img1_copy = img1.clone()
    img1_copy[:, bby1:bby2, bbx1:bbx2] = img2[:, bby1:bby2, bbx1:bbx2]
    
    # Adjust lambda based on actual box size
    lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (w * h)
    
    return img1_copy, label1, label2, lam

class AugmentedGrayscaleDataset(Dataset):
    """Dataset with advanced augmentation for grayscale images."""
    def __init__(self, root_dir, transform=None, img_size=128, use_strong_aug=True):
        self.root_dir = root_dir
        self.img_size = img_size
        self.use_strong_aug = use_strong_aug
        
        # Get class mapping
        self.class_to_idx, self.classes = get_class_mapping(root_dir)
        
        # Basic transform
        self.basic_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        
        # Advanced augmentation for training
        self.strong_aug = transforms.Compose([
            transforms.Resize((int(img_size * 1.2), int(img_size * 1.2))),  # Resize larger for cropping
            transforms.RandomCrop(img_size),  # Random crop for translation effect
            transforms.RandomHorizontalFlip(p=0.5),  # Horizontal flip
            transforms.RandomAffine(
                degrees=0,  # No rotation as requested
                translate=(0.2, 0.2),  # Random translation
                scale=(0.9, 1.1),  # Random scaling
                shear=0  # No shear as requested
            ),
            transforms.Grayscale(),
            transforms.ToTensor(),
        ])
        
        # Custom transform or use default
        self.transform = transform if transform is not None else self.basic_transform
        
        # Collect all image paths and their corresponding labels
        self.img_paths = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            class_idx = self.class_to_idx[class_name]
            
            # Find all images in this class directory
            for ext in ['*.jpg', '*.jpeg', '*.png']:
                paths = glob.glob(os.path.join(class_dir, ext))
                self.img_paths.extend(paths)
                self.labels.extend([class_idx] * len(paths))
    
    def __len__(self):
        return len(self.img_paths)
    
    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        image = Image.open(img_path).convert('L')  # Ensure grayscale
        label = self.labels[idx]
        
        # Apply transforms
        if self.use_strong_aug:
            image = self.strong_aug(image)
        else:
            image = self.transform(image)
            
        return image, label

class MixupCutmixBatchSampler:
    """Applies mixup or cutmix to a batch of images."""
    def __init__(self, dataset, mixup_prob=0.3, cutmix_prob=0.3, mixup_alpha=0.2, cutmix_alpha=1.0):
        self.dataset = dataset
        self.mixup_prob = mixup_prob
        self.cutmix_prob = cutmix_prob
        self.mixup_alpha = mixup_alpha
        self.cutmix_alpha = cutmix_alpha
    
    def apply_mixing(self, images, labels):
        batch_size = images.size(0)
        mixing_type = np.random.random()
        
        # Apply mixup
        if mixing_type < self.mixup_prob:
            # Create random permutation for mixup pairs
            indices = torch.randperm(batch_size)
            images_permuted = images[indices]
            labels_permuted = labels[indices]
            
            # Apply mixup to all images in batch
            mixed_images = []
            mixed_labels1 = []
            mixed_labels2 = []
            lambdas = []
            
            for i in range(batch_size):
                mixed_img, label1, label2, lam = mixup(
                    images[i], images_permuted[i], 
                    labels[i], labels_permuted[i], 
                    alpha=self.mixup_alpha
                )
                mixed_images.append(mixed_img)
                mixed_labels1.append(label1)
                mixed_labels2.append(label2)
                lambdas.append(lam)
            
            mixed_images = torch.stack(mixed_images)
            mixed_labels1 = torch.tensor(mixed_labels1)
            mixed_labels2 = torch.tensor(mixed_labels2)
            lambdas = torch.tensor(lambdas)
            
            return mixed_images, mixed_labels1, mixed_labels2, lambdas, 'mixup'
        
        # Apply cutmix
        elif mixing_type < (self.mixup_prob + self.cutmix_prob):
            # Create random permutation for cutmix pairs
            indices = torch.randperm(batch_size)
            images_permuted = images[indices]
            labels_permuted = labels[indices]
            
            # Apply cutmix to all images in batch
            mixed_images = []
            mixed_labels1 = []
            mixed_labels2 = []
            lambdas = []
            
            for i in range(batch_size):
                mixed_img, label1, label2, lam = cutmix(
                    images[i], images_permuted[i], 
                    labels[i], labels_permuted[i], 
                    alpha=self.cutmix_alpha
                )
                mixed_images.append(mixed_img)
                mixed_labels1.append(label1)
                mixed_labels2.append(label2)
                lambdas.append(lam)
            
            mixed_images = torch.stack(mixed_images)
            mixed_labels1 = torch.tensor(mixed_labels1)
            mixed_labels2 = torch.tensor(mixed_labels2)
            lambdas = torch.tensor(lambdas)
            
            return mixed_images, mixed_labels1, mixed_labels2, lambdas, 'cutmix'
        
        # No mixing
        else:
            return images, labels, None, None, 'none'

def get_data_loaders(data_dir, batch_size=32, img_size=128, num_workers=4, use_augmentation=True):
    """Creates data loaders for training and validation with augmentation."""
    # Define transforms
    basic_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(),
        transforms.ToTensor(),
    ])
    
    # Create datasets
    train_dataset = AugmentedGrayscaleDataset(
        data_dir, 
        transform=basic_transform, 
        img_size=img_size, 
        use_strong_aug=use_augmentation
    )
    
    # Create a validation dataset without augmentation
    val_dataset = AugmentedGrayscaleDataset(
        data_dir, 
        transform=basic_transform, 
        img_size=img_size, 
        use_strong_aug=False
    )
    
    # Split into train/val (95/5)
    # Calculate indices for a stratified split to ensure each class is represented in validation
    train_indices = []
    val_indices = []
    
    # Group indices by class
    indices_by_class = {}
    for idx, (_, label) in enumerate(train_dataset):
        if label not in indices_by_class:
            indices_by_class[label] = []
        indices_by_class[label].append(idx)
    
    # Split each class proportionally
    for class_label, indices in indices_by_class.items():
        np.random.shuffle(indices)
        split = int(0.95 * len(indices))
        train_indices.extend(indices[:split])
        val_indices.extend(indices[split:])
    
    # Create subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create data loaders with custom collate function
    train_loader = DataLoader(
        train_subset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=mixup_cutmix_collate_fn if use_augmentation else None
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    # Get the original full dataset for class names and indices
    full_dataset = AugmentedGrayscaleDataset(data_dir, use_strong_aug=False)
    
    return train_loader, val_loader, full_dataset.class_to_idx, full_dataset.classes

# Custom collate function for applying mixup/cutmix during batch loading
def mixup_cutmix_collate_fn(batch):
    """Custom collate function that applies mixup or cutmix to batches."""
    # Separate images and labels
    images = torch.stack([item[0] for item in batch])
    labels = torch.tensor([item[1] for item in batch])
    
    # Determine if we should apply mixup or cutmix or neither
    mixing = np.random.choice(['none', 'mixup', 'cutmix'], p=[0.4, 0.3, 0.3])
    
    if mixing == 'none':
        return images, labels
    
    elif mixing == 'mixup':
        # Random pairing
        indices = torch.randperm(len(batch))
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]
        
        # Apply mixup
        lam = np.random.beta(0.2, 0.2)
        # Convert to tensor
        lam = torch.tensor(lam, dtype=torch.float32)
        mixed_images = lam * images + (1 - lam) * shuffled_images
        
        return mixed_images, labels, shuffled_labels, lam, mixing
    
    elif mixing == 'cutmix':
        # Random pairing
        indices = torch.randperm(len(batch))
        shuffled_images = images[indices]
        shuffled_labels = labels[indices]
        
        # Apply cutmix - single mask for the whole batch
        lam = np.random.beta(1.0, 1.0)
        
        _, h, w = images[0].shape
        
        # Calculate box dimensions
        cut_rat = np.sqrt(1. - lam)
        cut_w = int(w * cut_rat)
        cut_h = int(h * cut_rat)
        
        # Get random center of box
        cx = np.random.randint(w)
        cy = np.random.randint(h)
        
        # Calculate box boundaries
        bbx1 = np.clip(cx - cut_w // 2, 0, w)
        bby1 = np.clip(cy - cut_h // 2, 0, h)
        bbx2 = np.clip(cx + cut_w // 2, 0, w)
        bby2 = np.clip(cy + cut_h // 2, 0, h)
        
        # Apply cutmix to all images
        mixed_images = images.clone()
        mixed_images[:, :, bby1:bby2, bbx1:bbx2] = shuffled_images[:, :, bby1:bby2, bbx1:bbx2]
        
        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1)) / (w * h)
        # Convert to tensor
        lam = torch.tensor(lam, dtype=torch.float32)
        
        return mixed_images, labels, shuffled_labels, lam, mixing

def visualize_batch(images, grid_size=None):
    """Visualize a batch of images."""
    import matplotlib.pyplot as plt
    
    # If no grid size is provided, try to make a square grid
    if grid_size is None:
        size = int(np.sqrt(len(images)))
        grid_size = (size, size)
    
    # Create the figure
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
    axes = axes.flatten()
    
    # Plot each image
    for i, ax in enumerate(axes):
        if i < len(images):
            # Convert from tensor to numpy if needed
            if isinstance(images[i], torch.Tensor):
                img = images[i].detach().cpu().numpy().squeeze()
            else:
                img = images[i]
            
            ax.imshow(img, cmap='gray')
            ax.axis('off')
    
    plt.tight_layout()
    return fig

def class_to_onehot(labels, num_classes):
    """Convert class indices to one-hot vectors."""
    return torch.nn.functional.one_hot(labels, num_classes).float()

class UnsupervisedGrayscaleDataset(Dataset):
    """Dataset for unsupervised learning with grayscale images.
    Preprocessing steps:
    1. Crop sides to first black pixel
    2. Scale to 128px on the largest dimension
    3. Pad with white to make it 128x128
    """
    def __init__(self, root_dir, img_size=128):
        self.root_dir = root_dir
        self.img_size = img_size
        
        # Collect all image paths
        self.img_paths = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            # Find all images in root_dir and all subdirectories
            self.img_paths.extend(glob.glob(os.path.join(root_dir, '**', ext), recursive=True))
        
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
        # Open image and convert to grayscale
        image = np.array(Image.open(img_path).convert('L'))
        
        # Apply preprocessing
        # 1. Crop to black edges
        image = self.crop_to_black_edges(image)
        
        # 2. Resize and pad
        image = self.resize_and_pad(image)
        
        # Convert to tensor and normalize
        image_tensor = torch.from_numpy(image).float() / 255.0
        
        # Add channel dimension
        image_tensor = image_tensor.unsqueeze(0)
        
        return image_tensor

def get_unsupervised_data_loader(root_dir, batch_size=32, img_size=128, num_workers=4, shuffle=True):
    """Creates data loader for unsupervised learning."""
    dataset = UnsupervisedGrayscaleDataset(root_dir, img_size=img_size)
    
    # Create data loader
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return data_loader 