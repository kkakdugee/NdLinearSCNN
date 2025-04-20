import torch
import os
import time

# Model Configuration
input_dim = 784  # for MNIST
hidden_dim = 784  # output dimension (same as input for autoencoder)
latent_dim = 256  # bottleneck/code dimension

# Training Configuration
batch_size = 128
num_epochs = 15
learning_rate = 1e-3
weight_decay = 1e-5

# Sparsity Configuration
sparsity_weight = 5e-2  # Weight for KL divergence sparsity penalty
kl_target_sparsity = 0.01  # Target activation level (aim for 1% active neurons)

# System Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
results_dir = 'benchmark_results'

# Create a unique experiment ID based on timestamp
experiment_id = f"experiment_{int(time.time())}"