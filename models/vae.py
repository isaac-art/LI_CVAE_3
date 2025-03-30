import torch
import torch.nn as nn
import torch.nn.functional as F

class VAE(nn.Module):
    """Variational Autoencoder for 128x128 grayscale images."""
    def __init__(self, latent_dim=64):
        super(VAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder
        self.encoder = nn.Sequential(
            # Input: 1 x 128 x 128
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),  # 32 x 64 x 64
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 512 x 4 x 4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True)
        )
        
        # Latent space
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 4)
        
        self.decoder = nn.Sequential(
            # Input: 512 x 4 x 4
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 256 x 8 x 8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 128 x 16 x 16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x 32 x 32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 32 x 64 x 64
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            
            nn.ConvTranspose2d(32, 1, kernel_size=4, stride=2, padding=1),  # 1 x 128 x 128
            nn.Sigmoid()  # Output in range [0, 1]
        )
        
    def encode(self, x):
        """Encode input into latent space."""
        x = self.encoder(x)
        x = x.view(x.size(0), -1)  # Flatten
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar
    
    def decode(self, z):
        """Decode latent vector into image."""
        z = self.decoder_input(z)
        z = z.view(z.size(0), 512, 4, 4)
        return self.decoder(z)
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def sample(self, num_samples, device='cuda'):
        """Sample from latent space and decode."""
        z = torch.randn(num_samples, self.latent_dim).to(device)
        return self.decode(z)
    
    def interpolate(self, z1, z2, steps=10):
        """Interpolate between two latent vectors."""
        z = torch.zeros(steps, self.latent_dim, device=z1.device)
        for i in range(steps):
            alpha = i / (steps - 1)
            z[i] = z1 * (1 - alpha) + z2 * alpha
        return self.decode(z)
    
    def slerp(self, z1, z2, steps=10):
        """Spherical linear interpolation between two latent vectors.
        
        Args:
            z1: First latent vector
            z2: Second latent vector
            steps: Number of interpolation steps
            
        Returns:
            Decoded images from the interpolated latent vectors
        """
        # Normalize the vectors to unit sphere
        z1_norm = z1 / torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = z2 / torch.norm(z2, dim=-1, keepdim=True)
        
        # Compute the cosine of the angle between the vectors
        cos_omega = torch.sum(z1_norm * z2_norm, dim=-1, keepdim=True).clamp(-1, 1)
        omega = torch.acos(cos_omega)
        sin_omega = torch.sin(omega)
        
        # Initialize result tensor
        z = torch.zeros(steps, self.latent_dim, device=z1.device)
        
        # Handle special case when vectors are very close or opposite
        if sin_omega.item() < 1e-6:
            # Linear interpolation if vectors are very close or opposite
            for i in range(steps):
                alpha = i / (steps - 1)
                z[i] = z1 * (1 - alpha) + z2 * alpha
        else:
            # Actual SLERP
            for i in range(steps):
                alpha = i / (steps - 1)
                s1 = torch.sin((1 - alpha) * omega) / sin_omega
                s2 = torch.sin(alpha * omega) / sin_omega
                z[i] = z1 * s1 + z2 * s2
                
        # Preserve the original magnitudes with spherical interpolation directions
        z1_mag = torch.norm(z1, dim=-1, keepdim=True)
        z2_mag = torch.norm(z2, dim=-1, keepdim=True)
        
        for i in range(steps):
            alpha = i / (steps - 1)
            mag = z1_mag * (1 - alpha) + z2_mag * alpha
            z[i] = z[i] * mag / torch.norm(z[i], dim=-1, keepdim=True)
                
        return self.decode(z)
    
    def forward(self, x):
        """Forward pass."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar
    
    def loss_function(self, recon_x, x, mu, logvar, kld_weight=0.00025):
        """VAE loss function combining reconstruction loss with KL divergence."""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence
        kld_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        loss = recon_loss + kld_weight * kld_loss
        
        return loss, recon_loss, kld_loss 