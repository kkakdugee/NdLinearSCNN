# visualization.py

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import json
import pandas as pd
from config import device

def calculate_detailed_losses(model, test_loader, sparsity_weight, kl_target_sparsity):
    """
    Calculate detailed loss metrics for a model on the test set
    
    Args:
        model: The model to evaluate
        test_loader: DataLoader for test data
        sparsity_type: Type of sparsity penalty ('l1' or 'kl')
        sparsity_weight: Weight for the sparsity penalty
        kl_target_sparsity: Target activation level for KL divergence
        
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
            
            # Calculate sparsity penalty

            from training import kl_divergence_sparsity
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

def evaluate_feature_quality(model, test_loader, name, save_path):
    """Evaluate feature quality by visualizing sample reconstructions"""
    model.eval()
    with torch.no_grad():
        # Get a batch of test images
        data, _ = next(iter(test_loader))
        samples = data[:10].view(10, -1).to(device)
        
        # Get reconstructions
        recon, _ = model(samples)
        
        # Reshape back to images
        samples = samples.view(10, 1, 28, 28).cpu()
        recon = recon.view(10, 1, 28, 28).cpu()
        
        # Create plot
        plt.figure(figsize=(12, 6))
        for i in range(10):
            # Original images
            plt.subplot(2, 10, i + 1)
            plt.imshow(samples[i][0], cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Original')
            
            # Reconstructed images
            plt.subplot(2, 10, i + 11)
            plt.imshow(recon[i][0], cmap='gray')
            plt.axis('off')
            if i == 0:
                plt.title('Reconstructed')
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f'{name}_reconstructions.png'))
        plt.close()
        
        # Try to visualize features, but skip if not possible
        try:
            if name == "linear" and hasattr(model.decoder, 'weight'):
                decoder_weights = model.decoder.weight.data.cpu()
                n_features = min(64, decoder_weights.shape[1])
                reshaped_weights = decoder_weights[:, :n_features].t().view(n_features, 28, 28)
                
                plt.figure(figsize=(8, 8))
                for i in range(min(64, n_features)):
                    plt.subplot(8, 8, i + 1)
                    plt.imshow(reshaped_weights[i], cmap='viridis')
                    plt.axis('off')
                
                plt.tight_layout()
                plt.savefig(os.path.join(save_path, f'{name}_features.png'))
                plt.close()
            else:
                print(f"Skipping feature visualization for {name} model")
        except Exception as e:
            print(f"Error visualizing features for {name} model: {e}")

def visualize_sparsity_patterns(model, test_loader, name, save_path):
    """Visualize the sparsity patterns in the latent space"""
    model.eval()
    activations_list = []
    
    with torch.no_grad():
        for data, _ in test_loader:
            if len(activations_list) >= 10:  # Limit to 10 batches
                break
                
            data = data.view(data.size(0), -1).to(device)
            _, activations = model(data)
            
            # Handle multi-dimensional activations by flattening them for visualization
            if activations.dim() > 2:
                # For NdLinear, reshape from [batch, dim1, dim2] to [batch, dim1*dim2]
                batch_size = activations.size(0)
                activations = activations.reshape(batch_size, -1)
            
            activations_list.append(activations.cpu())
    
    # Concatenate all activations
    all_activations = torch.cat(activations_list, dim=0)
    
    # Calculate average activation per neuron
    avg_activations = torch.mean(all_activations, dim=0)
    
    # Plot histogram of average activations
    plt.figure(figsize=(10, 6))
    plt.hist(avg_activations.numpy(), bins=50)
    plt.title(f'Histogram of Average Neuron Activations - {name}')
    plt.xlabel('Average Activation')
    plt.ylabel('Count')
    plt.savefig(os.path.join(save_path, f'{name}_activation_histogram.png'))
    plt.close()
    
    # Plot heatmap of sample activations
    plt.figure(figsize=(12, 8))
    plt.imshow(all_activations[:100].permute(1, 0), aspect='auto', cmap='viridis')  # Use permute instead of T
    plt.colorbar(label='Activation Value')
    plt.title(f'Activation Patterns for 100 Samples - {name}')
    plt.xlabel('Sample Index')
    plt.ylabel('Neuron Index')
    plt.savefig(os.path.join(save_path, f'{name}_activation_patterns.png'))
    plt.close()

def plot_training_metrics(linear_metrics, ndlinear_metrics, save_path):
    """Plot and compare training metrics between the two models"""
    plt.figure(figsize=(12, 8))
    
    # Loss comparison
    plt.subplot(2, 2, 1)
    plt.plot(linear_metrics['train_loss'], label='Linear Train')
    plt.plot(linear_metrics['test_loss'], label='Linear Test')
    plt.plot(ndlinear_metrics['train_loss'], label='NdLinear Train')
    plt.plot(ndlinear_metrics['test_loss'], label='NdLinear Test')
    plt.title('Loss Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    
    # Sparsity comparison
    plt.subplot(2, 2, 2)
    plt.plot(linear_metrics['train_sparsity'], label='Linear Train')
    plt.plot(linear_metrics['test_sparsity'], label='Linear Test')
    plt.plot(ndlinear_metrics['train_sparsity'], label='NdLinear Train')
    plt.plot(ndlinear_metrics['test_sparsity'], label='NdLinear Test')
    plt.title('True Sparsity Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('True Sparsity (% Inactive Neurons)')
    plt.legend()
    
    # Training time comparison
    plt.subplot(2, 2, 3)
    plt.plot(linear_metrics['train_time'], label='Linear')
    plt.plot(ndlinear_metrics['train_time'], label='NdLinear')
    plt.title('Training Time Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()
    
    # Inference time comparison
    plt.subplot(2, 2, 4)
    plt.plot(linear_metrics['inference_time'], label='Linear')
    plt.plot(ndlinear_metrics['inference_time'], label='NdLinear')
    plt.title('Inference Time Comparison')
    plt.xlabel('Epoch')
    plt.ylabel('Time (s)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'metrics_comparison.png'))
    plt.close()
    
    # Save metrics to CSV
    linear_df = pd.DataFrame(linear_metrics)
    ndlinear_df = pd.DataFrame(ndlinear_metrics)
    
    linear_df.to_csv(os.path.join(save_path, 'linear_metrics.csv'), index=False)
    ndlinear_df.to_csv(os.path.join(save_path, 'ndlinear_metrics.csv'), index=False)

def calculate_convergence_rate(loss_history, threshold=0.9):
    """
    Calculate at which epoch the model reached a specified threshold of its final performance
    
    Args:
        loss_history: List of loss values over training epochs
        threshold: Percentage of final performance to measure (default: 90%)
        
    Returns:
        epochs_to_converge: How many epochs it took to reach threshold*final_performance
    """
    if not loss_history:
        return None
    
    # Get final loss (best performance)
    final_loss = loss_history[-1]
    
    # For loss metrics, lower is better, so we're looking for the first epoch
    # where loss <= final_loss / threshold
    target_loss = final_loss / threshold
    
    for epoch, loss in enumerate(loss_history, 1):
        if loss <= target_loss:
            return epoch
    
    # If never reached threshold
    return len(loss_history)

def print_benchmark_summary(linear_model, ndlinear_model, test_loader, linear_metrics, ndlinear_metrics, save_path):
    """Print and save a summary of the benchmark results including detailed loss metrics"""
    try:
        with open(os.path.join(save_path, 'model_info.json'), 'r') as f:
            model_info = json.load(f)
            
        linear_params = model_info['linear_params']
        ndlinear_params = model_info['ndlinear_params']
        
        # Get experiment configuration
        experiment_config = model_info.get('experiment_config', {})
        sparsity_weight = experiment_config.get('sparsity_weight', 1e-2)
        kl_target_sparsity = experiment_config.get('kl_target_sparsity', 0.05)
        
        # Calculate detailed losses
        print("Calculating detailed loss metrics...")
        linear_losses = calculate_detailed_losses(
            linear_model, test_loader, 
            sparsity_weight=sparsity_weight, 
            kl_target_sparsity=kl_target_sparsity
        )
        
        ndlinear_losses = calculate_detailed_losses(
            ndlinear_model, test_loader, 
            sparsity_weight=sparsity_weight, 
            kl_target_sparsity=kl_target_sparsity
        )
        
        # Calculate convergence rates
        linear_convergence = calculate_convergence_rate(linear_metrics['train_loss'])
        ndlinear_convergence = calculate_convergence_rate(ndlinear_metrics['train_loss'])
        
        # Build summary
        summary = "===== BENCHMARK SUMMARY =====\n"
        summary += f"Model Parameters: Linear = {linear_params}, NdLinear = {ndlinear_params}\n"
        summary += f"Parameter Reduction: {(1 - ndlinear_params/linear_params) * 100:.2f}%\n"
        
        # Training metrics
        summary += f"\n----- TRAINING METRICS -----\n"
        summary += f"Final Train Loss: Linear = {linear_metrics['train_loss'][-1]:.6f}, NdLinear = {ndlinear_metrics['train_loss'][-1]:.6f}\n"
        summary += f"Final Test Loss: Linear = {linear_metrics['test_loss'][-1]:.6f}, NdLinear = {ndlinear_metrics['test_loss'][-1]:.6f}\n"
        summary += f"Convergence Rate (epochs to 90% performance): Linear = {linear_convergence}, NdLinear = {ndlinear_convergence}\n"
        
        # Detailed loss metrics
        summary += f"\n----- DETAILED LOSS METRICS -----\n"
        summary += f"Reconstruction Loss: Linear = {linear_losses['reconstruction_loss']:.6f}, NdLinear = {ndlinear_losses['reconstruction_loss']:.6f}\n"
        summary += f"KL Divergence Sparsity Loss: Linear = {linear_losses['sparsity_loss']:.6f}, NdLinear = {ndlinear_losses['sparsity_loss']:.6f}\n"
        summary += f"Total Loss: Linear = {linear_losses['total_loss']:.6f}, NdLinear = {ndlinear_losses['total_loss']:.6f}\n"
        
        # Performance metrics
        summary += f"\n----- PERFORMANCE METRICS -----\n"
        summary += f"Average Training Time: Linear = {np.mean(linear_metrics['train_time']):.2f}s, NdLinear = {np.mean(ndlinear_metrics['train_time']):.2f}s\n"
        summary += f"Average Inference Time: Linear = {np.mean(linear_metrics['inference_time']):.2f}s, NdLinear = {np.mean(ndlinear_metrics['inference_time']):.2f}s\n"
        
        # Sparsity metrics
        summary += f"\n----- SPARSITY METRICS -----\n"
        summary += f"Active Neurons: Linear = {1.0 - linear_metrics['test_sparsity'][-1]:.4f}, NdLinear = {1.0 - ndlinear_metrics['test_sparsity'][-1]:.4f}\n"
        summary += f"True Sparsity (% inactive): Linear = {linear_metrics['test_sparsity'][-1]:.4f}, NdLinear = {ndlinear_metrics['test_sparsity'][-1]:.4f}\n"
        summary += f"Target Sparsity Level: {kl_target_sparsity:.4f}\n"
        
        print(summary)
        
        # Save summary to file
        with open(os.path.join(save_path, 'benchmark_summary.txt'), 'w') as f:
            f.write(summary)
            
        # Save detailed metrics to JSON for future reference
        detailed_metrics = {
            'linear': {
                'losses': linear_losses,
                'convergence_rate': linear_convergence,
                'parameters': linear_params
            },
            'ndlinear': {
                'losses': ndlinear_losses,
                'convergence_rate': ndlinear_convergence,
                'parameters': ndlinear_params
            }
        }
        with open(os.path.join(save_path, 'detailed_metrics.json'), 'w') as f:
            json.dump(detailed_metrics, f)
            
    except Exception as e:
        print(f"Error generating benchmark summary: {e}")