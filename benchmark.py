# benchmark.py
import torch.nn as nn
from ndlinear import NdLinear
import os
import torch
import json
import argparse
from autoencoders import SparseAutoencoderLinear, SparseAutoencoderNdLinear
from training import run_training
from visualization import evaluate_feature_quality, visualize_sparsity_patterns, plot_training_metrics, print_benchmark_summary
from data_loader import get_data_loaders
from config import input_dim, hidden_dim, latent_dim, device, results_dir, experiment_id

def inspect_model_attributes(model, name):
    """Print the attributes and structure of the model for debugging"""
    print(f"\n{name} Model Attributes:")
    
    for param_name, param in model.named_parameters():
        print(f"Parameter: {param_name}, Shape: {param.shape}")
    
    print("\nModel Structure:")
    print(model)
    
    if name == "ndlinear":
        print("\nNdLinear Decoder Attributes:")
        for attr_name in dir(model.decoder):
            if not attr_name.startswith('_') and not callable(getattr(model.decoder, attr_name)):
                try:
                    attr_value = getattr(model.decoder, attr_name)
                    print(f"  {attr_name}: {type(attr_value)}")
                    if isinstance(attr_value, torch.Tensor):
                        print(f"    Shape: {attr_value.shape}")
                except Exception as e:
                    print(f"  Error accessing {attr_name}: {e}")

def compare_model_architectures(linear_model, ndlinear_model):
    """Compare detailed architecture between Linear and NdLinear models"""
    print("\n===== ARCHITECTURE COMPARISON =====")
    
    # Log model structures
    print("\nLinear Model Structure:")
    for name, module in linear_model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"  {name}: in_features={module.in_features}, out_features={module.out_features}")
            print(f"    Parameters: weight={module.weight.shape}, bias={module.bias.shape if module.bias is not None else None}")
    
    print("\nNdLinear Model Structure:")
    for name, module in ndlinear_model.named_modules():
        if isinstance(module, NdLinear):
            print(f"  {name}: input_dims={module.input_dims}, hidden_size={module.hidden_size}")
            # Print parameter shapes - these will depend on NdLinear's implementation
            print(f"    Parameters:")
            for param_name, param in module.named_parameters():
                print(f"      {param_name}: {param.shape}")
    
    # Compare activations
    print("\nActivation Shapes Comparison:")
    # Create sample input
    sample_input = torch.randn(1, 784).to(device)
    
    # Linear model
    with torch.no_grad():
        _, linear_activations = linear_model(sample_input)
        print(f"  Linear model activation shape: {linear_activations.shape}")
    
    # NdLinear model
    with torch.no_grad():
        _, ndlinear_activations = ndlinear_model(sample_input)
        print(f"  NdLinear model activation shape: {ndlinear_activations.shape}")
        print(f"  NdLinear model flattened activation shape: {ndlinear_activations.reshape(1, -1).shape}")

def run_benchmark(mode='all', checkpoint_path=None):
    """
    Run the benchmark in the specified mode
    
    Args:
        mode (str): One of 'all', 'train', 'evaluate', 'visualize'
        checkpoint_path (str): Path to load checkpoints from (for evaluate/visualize modes)
    """
    # Create results directory
    save_path = os.path.join(results_dir, experiment_id)
    os.makedirs(save_path, exist_ok=True)
    
    # Initialize models
    linear_model = SparseAutoencoderLinear(input_dim, hidden_dim, latent_dim).to(device)
    ndlinear_model = SparseAutoencoderNdLinear(input_dim, hidden_dim, latent_dim).to(device)
    
    # Inspect model structure
    inspect_model_attributes(linear_model, "Linear")
    inspect_model_attributes(ndlinear_model, "NdLinear")

    # Compare architecture details
    compare_model_architectures(linear_model, ndlinear_model)
    
    if mode in ['all', 'train']:
        print("Running training phase...")
        linear_metrics, ndlinear_metrics = run_training(
            linear_model, ndlinear_model, save_path,
            sparsity_weight=config.sparsity_weight, 
            kl_target_sparsity=config.kl_target_sparsity
        )
    else:
        # Load metrics from checkpoint
        if checkpoint_path is None:
            raise ValueError("checkpoint_path must be provided for evaluate/visualize modes")
        
        # Load linear model and metrics
        try:
            linear_checkpoint = torch.load(os.path.join(checkpoint_path, 'linear_model_final.pt'))
            linear_model.load_state_dict(linear_checkpoint)
            
            with open(os.path.join(checkpoint_path, 'linear_metrics.json'), 'r') as f:
                linear_metrics = json.load(f)
                
            # Load ndlinear model and metrics
            ndlinear_checkpoint = torch.load(os.path.join(checkpoint_path, 'ndlinear_model_final.pt'))
            ndlinear_model.load_state_dict(ndlinear_checkpoint)
            
            with open(os.path.join(checkpoint_path, 'ndlinear_metrics.json'), 'r') as f:
                ndlinear_metrics = json.load(f)
                
            print("Successfully loaded models and metrics from checkpoint")
        except Exception as e:
            print(f"Error loading models/metrics from checkpoint: {e}")
            if mode != 'train':
                return
    
    if mode in ['all', 'evaluate', 'visualize']:
        print("Running evaluation phase...")
        train_loader, test_loader = get_data_loaders()
        
        # Feature quality evaluation
        evaluate_feature_quality(linear_model, test_loader, "linear", save_path)
        evaluate_feature_quality(ndlinear_model, test_loader, "ndlinear", save_path)
        
        # Sparsity patterns
        visualize_sparsity_patterns(linear_model, test_loader, "linear", save_path)
        visualize_sparsity_patterns(ndlinear_model, test_loader, "ndlinear", save_path)
        
        # Plot training metrics
        plot_training_metrics(linear_metrics, ndlinear_metrics, save_path)
        
        # Print benchmark summary - updated call with models and test_loader
        print_benchmark_summary(linear_model, ndlinear_model, test_loader, linear_metrics, ndlinear_metrics, save_path)
        
        print(f"All results saved to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run benchmarks for nn.Linear vs NdLinear in Sparse Autoencoders')
    parser.add_argument('--mode', type=str, default='all', choices=['all', 'train', 'evaluate', 'visualize'], 
                        help='Benchmark mode: all, train, evaluate, or visualize')
    parser.add_argument('--checkpoint', type=str, default=None, 
                        help='Path to checkpoint directory for evaluation/visualization')
    parser.add_argument('--sparsity_weight', type=float, default=5e-2,
                        help='Weight for KL divergence sparsity penalty')
    parser.add_argument('--target_sparsity', type=float, default=0.01,
                        help='Target activation level for KL divergence')
    
    args = parser.parse_args()
    
    # Update config values from arguments
    import config
    config.sparsity_weight = args.sparsity_weight
    config.kl_target_sparsity = args.target_sparsity
    
    torch.manual_seed(3)
    run_benchmark(mode=args.mode, checkpoint_path=args.checkpoint)