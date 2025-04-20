import torch
import torch.nn as nn

from ndlinear import NdLinear


# Traditional Sparse Autoencoder with nn.Linear
class SparseAutoencoderLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # Encoder
        self.encoder = nn.Linear(input_dim, latent_dim)
        # Decoder
        self.decoder = nn.Linear(latent_dim, hidden_dim)
        
    def encode(self, x):
        return torch.relu(self.encoder(x))
        
    def decode(self, z):
        return self.decoder(z)
        
    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z

# Sparse Autoencoder with NdLinear
class SparseAutoencoderNdLinear(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()

        # Use 1D representations to avoid dimension issues, but split input_dim into two dimensions
        dim1 = int(input_dim**0.5)  # For MNIST, this should be close to 28
        self.input_reshape = (dim1, dim1)
        
        # For latent space, use a structure that maintains approximately the same capacity
        latent_dim1 = int(latent_dim**0.5)
        self.latent_reshape = (latent_dim1, latent_dim1)
        
        # Encoder: takes reshaped input and encodes to latent space
        self.encoder = NdLinear(input_dims=self.input_reshape, hidden_size=self.latent_reshape)
        # Decoder: reconstructs from latent space back to original dims
        self.decoder = NdLinear(input_dims=self.latent_reshape, hidden_size=self.input_reshape)
        
        
    def encode(self, x):
        # Reshape from [batch, input_dim] to [batch, dim1, dim1]
        batch_size = x.size(0)
        x_reshaped = x.view(batch_size, *self.input_reshape)
        return torch.relu(self.encoder(x_reshaped))
        
    def decode(self, z):
        recon = self.decoder(z)
        # Flatten the output to match the original input dimensions
        batch_size = recon.size(0)
        return recon.view(batch_size, -1)
        
    def forward(self, x):
        z = self.encode(x)
        recon = self.decode(z)
        return recon, z