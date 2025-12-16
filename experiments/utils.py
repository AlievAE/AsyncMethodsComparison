"""
Shared utilities for experiments: data loading, plotting, result saving.
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes, fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import os
import json
from datetime import datetime
import yaml


def load_config(config_path):
    """Load YAML config file."""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def save_results(results, experiment_name, plots_dir='../plots'):
    """Save experiment results to JSON."""
    os.makedirs(plots_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filepath = os.path.join(plots_dir, f'{experiment_name}_{timestamp}.json')
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, dict):
            serializable_results[key] = {
                k: v.tolist() if isinstance(v, np.ndarray) else v 
                for k, v in value.items()
            }
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    print(f"Results saved to {filepath}")
    return filepath


# =============================================================================
# Data Loading
# =============================================================================

def load_diabetes_data(test_size=0.2, random_state=42):
    """
    Load and preprocess Diabetes dataset for Linear Regression.
    
    Returns:
        tuple: (X, y) normalized feature matrix and targets
    """
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Normalize y for better convergence
    y = (y - y.mean()) / y.std()
    
    print(f"Diabetes dataset loaded: X shape = {X.shape}, y shape = {y.shape}")
    return X, y


def load_diabetes_binary_data(threshold='median'):
    """
    Load and preprocess Diabetes dataset for *binary* classification (Logistic Regression).

    We binarize the continuous diabetes target into {0,1} using a threshold.

    Args:
        threshold: 'median' or a numeric value. If 'median', uses median(y).

    Returns:
        tuple: (X, y_bin) with standardized X and binary labels in {0,1}.
    """
    diabetes = load_diabetes()
    X, y = diabetes.data, diabetes.target

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    if threshold == 'median':
        thr = float(np.median(y))
    else:
        thr = float(threshold)

    y_bin = (y > thr).astype(float)

    # Sanity print
    pos_rate = float(y_bin.mean())
    print(f"Diabetes (binary) loaded: X shape = {X.shape}, y pos-rate = {pos_rate:.3f}, threshold = {thr:.3f}")
    return X, y_bin


def load_a9a_data(data_dir='../data'):
    """
    Load and preprocess a9a dataset for Logistic Regression.
    Downloads if not present.
    
    Returns:
        tuple: (X, y) feature matrix and binary labels (0/1)
    """
    a9a_path = os.path.join(data_dir, 'a9a.npz')
    
    if os.path.exists(a9a_path):
        # Load cached version
        data = np.load(a9a_path)
        X, y = data['X'], data['y']
        print(f"a9a dataset loaded from cache: X shape = {X.shape}")
    else:
        # Download from OpenML
        print("Downloading a9a dataset from OpenML...")
        a9a = fetch_openml(name='a9a', version=1, as_frame=False, parser='auto')
        X, y = a9a.data, a9a.target
        
        # Convert to dense if sparse
        if hasattr(X, 'toarray'):
            X = X.toarray()
        
        # Convert labels to 0/1
        y = (y.astype(float) > 0).astype(float)
        
        # Standardize features
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        
        # Cache the dataset
        os.makedirs(data_dir, exist_ok=True)
        np.savez(a9a_path, X=X, y=y)
        print(f"a9a dataset downloaded and cached: X shape = {X.shape}")
    
    return X, y


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_loss_vs_time(results_dict, title, save_path=None, figsize=(10, 6)):
    """
    Plot relative suboptimality |f(x) - f*| / f* vs simulated time for multiple methods.
    f* is computed as the minimum loss achieved across all methods.
    
    Args:
        results_dict: {method_name: {'times': [...], 'losses': [...]}}
        title: Plot title
        save_path: Path to save figure (optional)
    """
    # Compute f* = minimum loss across all methods
    f_star = min(min(data['losses']) for data in results_dict.values())
    # Avoid division by zero; use small epsilon if f_star is 0
    f_star = max(f_star, 1e-15)
    
    plt.figure(figsize=figsize)
    
    for method_name, data in results_dict.items():
        losses = np.array(data['losses'])
        rel_subopt = np.abs(losses - f_star) / f_star
        plt.plot(data['times'], rel_subopt, label=method_name, linewidth=2)
    
    plt.xlabel('Simulated Time (based on worker configs)', fontsize=12)
    plt.ylabel(r'Relative Suboptimality $|f(x) - f^*| / f^*$', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_loss_vs_iterations(results_dict, title, save_path=None, figsize=(10, 6)):
    """
    Plot relative suboptimality |f(x) - f*| / f* vs iterations for multiple methods.
    f* is computed as the minimum loss achieved across all methods.
    """
    # Compute f* = minimum loss across all methods
    f_star = min(min(data['losses']) for data in results_dict.values())
    # Avoid division by zero; use small epsilon if f_star is 0
    f_star = max(f_star, 1e-15)
    
    plt.figure(figsize=figsize)
    
    for method_name, data in results_dict.items():
        losses = np.array(data['losses'])
        rel_subopt = np.abs(losses - f_star) / f_star
        plt.plot(rel_subopt, label=method_name, linewidth=2)
    
    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel(r'Relative Suboptimality $|f(x) - f^*| / f^*$', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()

def plot_grad_norm_vs_time(results_dict, title, save_path=None, figsize=(10, 6)):
    """
    Plot gradient norm vs simulated time for multiple methods.

    Expects: results_dict[method] contains 'times' and 'grad_norms'.
    """
    plt.figure(figsize=figsize)

    for method_name, data in results_dict.items():
        if 'grad_norms' not in data:
            continue
        plt.plot(data['times'], data['grad_norms'], label=method_name, linewidth=2)

    plt.xlabel('Simulated Time (based on worker configs)', fontsize=12)
    plt.ylabel('Gradient Norm ||g||', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_grad_norm_vs_iterations(results_dict, title, save_path=None, figsize=(10, 6)):
    """
    Plot gradient norm vs iterations for multiple methods.
    """
    plt.figure(figsize=figsize)

    for method_name, data in results_dict.items():
        if 'grad_norms' not in data:
            continue
        plt.plot(data['grad_norms'], label=method_name, linewidth=2)

    plt.xlabel('Iterations', fontsize=12)
    plt.ylabel('Gradient Norm ||g||', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")

    plt.close()


def plot_compression_comparison(results_dict, title, save_path=None, figsize=(12, 5)):
    """
    Plot compression methods comparison (loss vs time and vs iterations).
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Loss vs Time
    for method_name, data in results_dict.items():
        axes[0].plot(data['times'], data['losses'], label=method_name, linewidth=2)
    axes[0].set_xlabel('Simulated Time', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Loss vs Simulated Time', fontsize=12)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_yscale('log')
    
    # Loss vs Iterations
    for method_name, data in results_dict.items():
        axes[1].plot(data['losses'], label=method_name, linewidth=2)
    axes[1].set_xlabel('Iterations', fontsize=12)
    axes[1].set_ylabel('Loss', fontsize=12)
    axes[1].set_title('Loss vs Iterations', fontsize=12)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.3)
    axes[1].set_yscale('log')
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def plot_batch_size_sweep(results_dict, title, save_path=None, figsize=(10, 6)):
    """
    Plot final loss vs batch size for RennalaSGD experiments.
    """
    plt.figure(figsize=figsize)
    
    batch_sizes = sorted(results_dict.keys())
    final_losses = [results_dict[bs]['losses'][-1] for bs in batch_sizes]
    final_times = [results_dict[bs]['times'][-1] for bs in batch_sizes]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].plot(batch_sizes, final_losses, 'o-', linewidth=2, markersize=8)
    axes[0].set_xlabel('Batch Size', fontsize=12)
    axes[0].set_ylabel('Final Loss', fontsize=12)
    axes[0].set_title('Final Loss vs Batch Size', fontsize=12)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(batch_sizes, final_times, 's-', linewidth=2, markersize=8, color='orange')
    axes[1].set_xlabel('Batch Size', fontsize=12)
    axes[1].set_ylabel('Total Simulated Time', fontsize=12)
    axes[1].set_title('Simulated Time vs Batch Size', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    
    plt.close()


def create_time_distributions(config):
    """
    Create scipy distributions from config specification.
    
    Args:
        config: dict with 'type' and parameters
        
    Returns:
        scipy distribution or float
    """
    from scipy import stats
    
    dist_type = config.get('type', 'constant')
    
    if dist_type == 'constant':
        return config.get('value', 1.0)
    elif dist_type == 'exponential':
        scale = config.get('scale', 1.0)
        return stats.expon(scale=scale)
    elif dist_type == 'lognormal':
        s = config.get('s', 1.0)
        scale = config.get('scale', 1.0)
        return stats.lognorm(s=s, scale=scale)
    elif dist_type == 'pareto':
        b = config.get('b', 2.0)
        return stats.pareto(b=b)
    elif dist_type == 'uniform':
        low = config.get('low', 0.5)
        high = config.get('high', 1.5)
        return stats.uniform(loc=low, scale=high-low)
    else:
        raise ValueError(f"Unknown distribution type: {dist_type}")

