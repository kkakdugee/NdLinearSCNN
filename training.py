# training.py

import torch
import torch.nn as nn
import torch.optim as optim
import time
import os
import json
import numpy as np
from config import device, learning_rate, weight_decay, sparsity_weight, num_epochs, results_dir
from data_loader import get_data_loaders

def kl_divergence_sparsity(activations, kl_target_sparsity):
    """
    KL divergence sparsity penalty.
    
    Args:
        activations: The hidden layer activations
        target_sparsity: The desired average activation value (typically small, e.g., 0.05)
    
    Returns:
        KL divergence between average activations and target sparsity
    """
    # Calculate average activation for each hidden unit over the batch
    avg_activations = torch.mean(activations, dim=0)
    
    # Add small epsilon to prevent log(0)
    epsilon = 1e-10
    avg_activations = torch.clamp(avg_activations, epsilon, 1.0 - epsilon)
    
    # Calculate KL divergence between average activation and target sparsity
    kl_div = kl_target_sparsity * torch.log(kl_target_sparsity / avg_activations) + \
             (1 - kl_target_sparsity) * torch.log((1 - kl_target_sparsity) / (1 - avg_activations))
    
    # Sum over all hidden units
    return torch.sum(kl_div)

def count_parameters(model):
    """Count the number of trainable parameters in a model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def train_model(model, train_loader, optimizer, epoch, model_name, sparsity_weight, kl_target_sparsity):
    """
    Train the model for one epoch using KL divergence sparsity penalty
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        optimizer: The optimizer to use
        epoch: Current epoch number
        model_name: Name identifier for the model (e.g., "Linear" or "NdLinear")
        sparsity_weight: Weight for the sparsity penalty
        target_sparsity: Target activation level for KL divergence
        
    Returns:
        train_loss: Average reconstruction loss for the epoch
        sparsity: Average sparsity level (percentage of inactive neurons)
        train_time: Time taken for the epoch
    """
    model.train()
    train_loss = 0
    active_neurons_pct = 0  # Percentage of active neurons
    sparsity = 0  # True sparsity (percentage of inactive neurons)
    start_time = time.time()
    
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(data.size(0), -1).to(device)
        optimizer.zero_grad()
        
        # Forward pass
        recon_batch, activations = model(data)
        
        # Reshape activations if needed (for NdLinear)
        if activations.dim() > 2:
            batch_size = activations.size(0)
            activations_flat = activations.reshape(batch_size, -1)
        else:
            activations_flat = activations
        
        # Calculate reconstruction loss
        recon_loss = nn.MSELoss()(recon_batch, data)
        
        # Calculate KL divergence sparsity penalty
        sparse_loss = sparsity_weight * kl_divergence_sparsity(activations_flat, kl_target_sparsity)
        
        # Total loss
        loss = recon_loss + sparse_loss
        
        # Backward pass and optimize
        loss.backward()
        optimizer.step()
        
        train_loss += recon_loss.item()
        
        # Calculate percentage of active neurons
        batch_active_pct = torch.mean(torch.count_nonzero(activations_flat > 0.01, dim=1).float() / 
                                    activations_flat.size(1)).item()
        active_neurons_pct += batch_active_pct
        
        # Calculate true sparsity (percentage of inactive neurons)
        sparsity += 1.0 - batch_active_pct
    
    train_time = time.time() - start_time
    train_loss /= len(train_loader)
    active_neurons_pct /= len(train_loader)
    sparsity /= len(train_loader)
    
    print(f'Epoch: {epoch}, {model_name}:')
    print(f'  Loss: {train_loss:.6f}, Active Neurons: {active_neurons_pct:.4f}, Sparsity: {sparsity:.4f}, Train Time: {train_time:.2f}s')
    
    return train_loss, sparsity, train_time

def test_model(model, test_loader, model_name):
    """
    Test the model on the test dataset
    
    Args:
        model: The model to test
        test_loader: DataLoader for test data
        model_name: Name identifier for the model
        
    Returns:
        test_loss: Average reconstruction loss on test data
        sparsity: Average sparsity level (percentage of inactive neurons)
        inference_time: Time taken for inference
    """
    model.eval()
    test_loss = 0
    active_neurons_pct = 0
    sparsity = 0
    start_time = time.time()
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1).to(device)
            recon_batch, activations = model(data)
            
            # Reshape activations if needed (for NdLinear)
            if activations.dim() > 2:
                batch_size = activations.size(0)
                activations_flat = activations.reshape(batch_size, -1)
            else:
                activations_flat = activations
            
            # Calculate loss
            test_loss += nn.MSELoss()(recon_batch, data).item()
            
            # Calculate percentage of active neurons
            batch_active_pct = torch.mean(torch.count_nonzero(activations_flat > 0.01, dim=1).float() / 
                                        activations_flat.size(1)).item()
            active_neurons_pct += batch_active_pct
            
            # Calculate true sparsity (percentage of inactive neurons)
            sparsity += 1.0 - batch_active_pct
    
    inference_time = time.time() - start_time
    test_loss /= len(test_loader)
    active_neurons_pct /= len(test_loader)
    sparsity /= len(test_loader)
    
    print(f'Test {model_name}:')
    print(f'  Loss: {test_loss:.6f}, Active Neurons: {active_neurons_pct:.4f}, Sparsity: {sparsity:.4f}, Inference Time: {inference_time:.2f}s')
    
    return test_loss, sparsity, inference_time

def calculate_detailed_losses(model, test_loader, sparsity_weight, kl_target_sparsity):
    """
    Calculate detailed loss metrics for a model on the test set using KL divergence
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        sparsity_weight: Weight for the sparsity penalty
        target_sparsity: Target activation level for KL divergence
        
    Returns:
        A dictionary containing reconstruction_loss, sparsity_loss, and total_loss
    """
    model.eval()
    reconstruction_loss = 0
    sparsity_loss = 0
    total_loss = 0
    
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.view(data.size(0), -1).to(device)
            
            # Forward pass
            recon_batch, activations = model(data)
            
            # Reshape activations if needed (for NdLinear)
            if activations.dim() > 2:
                batch_size = activations.size(0)
                activations_flat = activations.reshape(batch_size, -1)
            else:
                activations_flat = activations
            
            # Calculate reconstruction loss
            batch_recon_loss = nn.MSELoss()(recon_batch, data)
            reconstruction_loss += batch_recon_loss.item()
            
            # Calculate KL divergence sparsity penalty
            batch_sparsity_loss = sparsity_weight * kl_divergence_sparsity(activations_flat, kl_target_sparsity)
            sparsity_loss += batch_sparsity_loss.item()
            
            # Calculate total loss
            batch_total_loss = batch_recon_loss + batch_sparsity_loss
            total_loss += batch_total_loss.item()
    
    # Average over all batches
    num_batches = len(test_loader)
    return {
        'reconstruction_loss': reconstruction_loss / num_batches,
        'sparsity_loss': sparsity_loss / num_batches,
        'total_loss': total_loss / num_batches
    }

def run_training(linear_model, ndlinear_model, save_path, sparsity_weight, kl_target_sparsity):
    """
    Run the complete training process for both models and save metrics
    
    Args:
        linear_model: The standard linear autoencoder model
        ndlinear_model: The NdLinear autoencoder model
        save_path: Directory to save checkpoints and metrics
        sparsity_weight: Weight for the sparsity penalty
        target_sparsity: Target activation level for KL divergence
        
    Returns:
        linear_metrics: Dictionary of metrics for the linear model
        ndlinear_metrics: Dictionary of metrics for the NdLinear model
    """
    os.makedirs(save_path, exist_ok=True)
    
    # Log the sparsity settings
    print(f"Using KL divergence sparsity penalty with weight {sparsity_weight}")
    print(f"Target sparsity level: {kl_target_sparsity}")
    
    # Load data
    train_loader, test_loader = get_data_loaders()
    
    # Count parameters
    linear_params = count_parameters(linear_model)
    ndlinear_params = count_parameters(ndlinear_model)
    
    print(f'Linear Model Parameters: {linear_params}')
    print(f'NdLinear Model Parameters: {ndlinear_params}')
    
    # Setup optimizers
    linear_optimizer = optim.Adam(linear_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    ndlinear_optimizer = optim.Adam(ndlinear_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Metrics storage
    linear_metrics = {
        'train_loss': [], 'test_loss': [], 
        'train_sparsity': [], 'test_sparsity': [],
        'train_time': [], 'inference_time': []
    }
    
    ndlinear_metrics = {
        'train_loss': [], 'test_loss': [], 
        'train_sparsity': [], 'test_sparsity': [],
        'train_time': [], 'inference_time': []
    }
    
    # Training loop
    for epoch in range(1, num_epochs + 1):
        # Train and test Linear model
        train_loss, train_sparsity, train_time = train_model(
            linear_model, train_loader, linear_optimizer, epoch, "Linear",
            sparsity_weight=sparsity_weight, kl_target_sparsity=kl_target_sparsity
        )
        test_loss, test_sparsity, inference_time = test_model(
            linear_model, test_loader, "Linear"
        )
        
        linear_metrics['train_loss'].append(train_loss)
        linear_metrics['test_loss'].append(test_loss)
        linear_metrics['train_sparsity'].append(train_sparsity)
        linear_metrics['test_sparsity'].append(test_sparsity)
        linear_metrics['train_time'].append(train_time)
        linear_metrics['inference_time'].append(inference_time)
        
        # Train and test NdLinear model
        train_loss, train_sparsity, train_time = train_model(
            ndlinear_model, train_loader, ndlinear_optimizer, epoch, "NdLinear",
            sparsity_weight=sparsity_weight, kl_target_sparsity=kl_target_sparsity
        )
        test_loss, test_sparsity, inference_time = test_model(
            ndlinear_model, test_loader, "NdLinear"
        )
        
        ndlinear_metrics['train_loss'].append(train_loss)
        ndlinear_metrics['test_loss'].append(test_loss)
        ndlinear_metrics['train_sparsity'].append(train_sparsity)
        ndlinear_metrics['test_sparsity'].append(test_sparsity)
        ndlinear_metrics['train_time'].append(train_time)
        ndlinear_metrics['inference_time'].append(inference_time)
        
        # Save model checkpoints periodically
        if epoch % 10 == 0 or epoch == num_epochs:
            linear_checkpoint = {
                'model_state_dict': linear_model.state_dict(),
                'optimizer_state_dict': linear_optimizer.state_dict(),
                'epoch': epoch,
                'metrics': linear_metrics
            }
            
            ndlinear_checkpoint = {
                'model_state_dict': ndlinear_model.state_dict(),
                'optimizer_state_dict': ndlinear_optimizer.state_dict(),
                'epoch': epoch,
                'metrics': ndlinear_metrics
            }
            
            torch.save(linear_checkpoint, os.path.join(save_path, f'linear_model_epoch_{epoch}.pt'))
            torch.save(ndlinear_checkpoint, os.path.join(save_path, f'ndlinear_model_epoch_{epoch}.pt'))
            
            # Save current metrics to JSON
            with open(os.path.join(save_path, 'linear_metrics.json'), 'w') as f:
                json.dump(linear_metrics, f)
            
            with open(os.path.join(save_path, 'ndlinear_metrics.json'), 'w') as f:
                json.dump(ndlinear_metrics, f)
    
    # Save final models
    torch.save(linear_model.state_dict(), os.path.join(save_path, 'linear_model_final.pt'))
    torch.save(ndlinear_model.state_dict(), os.path.join(save_path, 'ndlinear_model_final.pt'))
    
    # Save model metrics
    model_info = {
        'linear_params': linear_params,
        'ndlinear_params': ndlinear_params,
        'final_train_loss_linear': linear_metrics['train_loss'][-1],
        'final_test_loss_linear': linear_metrics['test_loss'][-1],
        'final_train_loss_ndlinear': ndlinear_metrics['train_loss'][-1],
        'final_test_loss_ndlinear': ndlinear_metrics['test_loss'][-1],
        'experiment_config': {
            'sparsity_weight': sparsity_weight,
            'target_sparsity': kl_target_sparsity,
            'learning_rate': learning_rate,
            'weight_decay': weight_decay,
            'num_epochs': num_epochs
        }
    }
    
    with open(os.path.join(save_path, 'model_info.json'), 'w') as f:
        json.dump(model_info, f)
    
    return linear_metrics, ndlinear_metrics