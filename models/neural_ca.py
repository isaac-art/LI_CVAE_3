import torch
import torch.nn as nn
import torch.nn.functional as F


class CAModel(nn.Module):
    """Neural Cellular Automata model for 128x128 grayscale images."""
    
    def __init__(self, hidden_channels=16, fire_rate=0.5):
        super().__init__()
        self.hidden_channels = hidden_channels
        self.fire_rate = fire_rate
        
        # For grayscale images, we have:
        # 1 channel for the visible state (grayscale value)
        # 1 channel for cell "alive" state
        # hidden_channels for the hidden state
        self.total_channels = hidden_channels + 2
        
        # Perception kernel for sensing the neighborhood
        self.perception = nn.Conv2d(
            in_channels=self.total_channels,
            out_channels=self.total_channels * 3,  # 3 kernels: identity, sobel_x, sobel_y
            kernel_size=3,
            padding=1,
            bias=False
        )
        
        # Initialize perception kernels with Sobel filters and identity
        self._init_perception_kernels()
        
        # Update model - a simple MLP
        self.update_model = nn.Sequential(
            nn.Conv2d(self.total_channels * 3, 128, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(128, self.total_channels, kernel_size=1),
        )
    
    def _init_perception_kernels(self):
        """Initialize perception kernels with Sobel filters and identity."""
        with torch.no_grad():
            # Create identity, Sobel X, and Sobel Y kernels
            identity = torch.zeros(self.total_channels, self.total_channels, 3, 3)
            sobel_x = torch.zeros(self.total_channels, self.total_channels, 3, 3)
            sobel_y = torch.zeros(self.total_channels, self.total_channels, 3, 3)
            
            # Set identity kernel (center pixel only)
            for i in range(self.total_channels):
                identity[i, i, 1, 1] = 1.0
                
                # Sobel X: [-1, 0, 1; -2, 0, 2; -1, 0, 1]
                sobel_x[i, i, 0, 0] = -1.0
                sobel_x[i, i, 0, 2] = 1.0
                sobel_x[i, i, 1, 0] = -2.0
                sobel_x[i, i, 1, 2] = 2.0
                sobel_x[i, i, 2, 0] = -1.0
                sobel_x[i, i, 2, 2] = 1.0
                
                # Sobel Y: [-1, -2, -1; 0, 0, 0; 1, 2, 1]
                sobel_y[i, i, 0, 0] = -1.0
                sobel_y[i, i, 0, 1] = -2.0
                sobel_y[i, i, 0, 2] = -1.0
                sobel_y[i, i, 2, 0] = 1.0
                sobel_y[i, i, 2, 1] = 2.0
                sobel_y[i, i, 2, 2] = 1.0
            
            # Combine kernels into perception weights
            perception_weights = torch.cat([identity, sobel_x, sobel_y], dim=0)
            self.perception.weight.data = perception_weights
            
            # Freeze the perception weights
            self.perception.weight.requires_grad = False
    
    def perceive(self, x):
        """Apply perception kernel to the state."""
        return self.perception(x)
    
    def update(self, x):
        """Update cell states based on perception."""
        # Get perception
        y = self.perceive(x)
        # Apply update model
        dx = self.update_model(y)
        # Apply stochastic update
        if self.fire_rate < 1.0 and self.training:
            mask = torch.rand_like(x[:, :1, :, :]) < self.fire_rate
            mask = mask.repeat(1, self.total_channels, 1, 1)
            dx = dx * mask
        # Add update to current state
        x = x + dx
        return x
    
    def alive_mask(self, x):
        """Get mask of cells that are alive (has alive channel > 0.1)."""
        return x[:, 1:2, :, :] > 0.1
    
    def forward(self, x, steps=1):
        """Run cellular automaton for multiple steps."""
        for _ in range(steps):
            # Only update living cells
            pre_alive_mask = self.alive_mask(x)
            # Update
            x = self.update(x)
            # Constrain visible channel to [0, 1]
            clamped_channel = torch.clamp(x[:, 0:1, :, :], 0.0, 1.0)
            x = torch.cat([clamped_channel, x[:, 1:]], dim=1)
            
            # Apply alive_mask: when a cell dies, it cannot be revived
            # and its values are reset to 0
            post_alive_mask = self.alive_mask(x)
            alive_mask = pre_alive_mask | post_alive_mask
            x = x * alive_mask.float()
        
        return x
    
    def init_state(self, batch_size, device, seed=None):
        """Initialize state with a seed cell in the center."""
        # Create empty grid
        x = torch.zeros(
            batch_size, 
            self.total_channels, 
            128, 
            128, 
            device=device
        )
        
        if seed is None:
            # Place a single cell in the center
            x[:, :, 64, 64] = 1.0
            # Ensure alive channel is set
            x[:, 1, 64, 64] = 1.0
        else:
            # Use provided seed
            x = seed
            
        return x

    def get_image(self, x):
        """Extract the visible grayscale channel from state."""
        return x[:, 0:1, :, :]
    
    def damage_cells(self, x, damage_size=32):
        """Apply damage to cells by setting a square region to 0."""
        damaged = x.clone()
        
        # Calculate center of the grid
        center_h, center_w = 64, 64
        
        # Define damage region (centered square)
        h_start = center_h - damage_size // 2
        h_end = center_h + damage_size // 2
        w_start = center_w - damage_size // 2
        w_end = center_w + damage_size // 2
        
        # Apply damage
        damaged[:, :, h_start:h_end, w_start:w_end] = 0.0
        
        return damaged 