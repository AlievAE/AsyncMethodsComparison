"""
Linear Regression Experiments on Diabetes Dataset
Run all experiments defined in configs/linreg_diabetes.yaml
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from functools import partial
import argparse

from experiments.utils import (
    load_config, load_diabetes_data, save_results,
    plot_loss_vs_time, plot_loss_vs_iterations,
    plot_grad_norm_vs_time, plot_grad_norm_vs_iterations,
    plot_compression_comparison, plot_batch_size_sweep,
    create_time_distributions
)
from methods import MinibatchSGD, AsynchronousGD, RennalaSGD, AsynchronousNAG
from LinearRegression import linear_regression_loss, linear_regression_gradient


def get_method_class(method_name):
    """Get method class by name."""
    methods = {
        'MinibatchSGD': MinibatchSGD,
        'AsynchronousGD': AsynchronousGD,
        'AsynchronousNAG': AsynchronousNAG,
        'RennalaSGD': RennalaSGD,
    }
    return methods[method_name]


def run_method(method_name, data, initial_x, time_distributions, loss_fn, gradient_fns, 
               lr, max_time=None, num_steps=None, compression_flag='none', compression_size=100,
               batch_size=None, momentum=None):
    """Run a single method and return results.
    
    Args:
        max_time: Run until simulated time exceeds this value (preferred for fair comparison)
        num_steps: Run for fixed number of iterations (legacy mode)
        momentum: Momentum for NAG (only used when method_name == 'AsynchronousNAG')
    """
    MethodClass = get_method_class(method_name)
    
    # Build kwargs depending on method type
    kwargs = dict(
        initial_x=initial_x.copy(),
        data=data,
        time_distributions=time_distributions,
        loss_fn=loss_fn,
        gradient_fns=gradient_fns,
        learning_rate=lr,
        compression_flag=compression_flag,
        compression_size=compression_size,
    )
    if method_name == 'AsynchronousNAG' and momentum is not None:
        kwargs['momentum'] = momentum
    
    method = MethodClass(**kwargs)
    
    if method_name == 'RennalaSGD' and batch_size is not None:
        method.set_batch_size(batch_size)
    elif method_name == 'RennalaSGD':
        method.set_batch_size(len(time_distributions))  # Default: wait for all
    
    if max_time is not None:
        _, losses, times, _ = method.run_until_time(max_time)
    elif num_steps is not None:
        _, losses, times, _ = method.run_steps(num_steps)
    else:
        raise ValueError("Must specify either max_time or num_steps")
    
    return {'losses': losses, 'times': times, 'grad_norms': method.grad_norm_history}


def experiment_homogeneous(config, data, initial_x, loss_fn, lambda_reg, plots_dir):
    """Experiment 1: Sync vs Async under Homogeneous Workers"""
    exp_config = config['experiment_homogeneous']
    if not exp_config['enabled']:
        print("Skipping homogeneous experiment (disabled)")
        return
    
    print("\n" + "="*60)
    print("Experiment 1: Homogeneous Workers")
    print("="*60)
    
    opt = config['optimization']
    n_workers = exp_config['n_workers']
    time_dist = create_time_distributions(exp_config['time_distribution'])
    time_distributions = [time_dist] * n_workers
    
    gradient_fns = [
        partial(linear_regression_gradient, batch_size=len(data[1])//n_workers, lambda_reg=lambda_reg)
        for _ in range(n_workers)
    ]
    
    results = {}
    nag_momentum = exp_config.get('nag_momentum')
    for method_name in exp_config['methods']:
        print(f"  Running {method_name}...")
        batch_size = n_workers if method_name == 'RennalaSGD' else None
        momentum = nag_momentum if method_name == 'AsynchronousNAG' else None
        results[method_name] = run_method(
            method_name, data, initial_x, time_distributions, loss_fn, gradient_fns,
            opt['learning_rate'], max_time=opt['max_time'], batch_size=batch_size, momentum=momentum
        )
    
    # Plot results
    plot_loss_vs_time(results, 'LinReg Diabetes: Homogeneous Workers - Loss vs Time',
                      os.path.join(plots_dir, 'linreg_homogeneous_time.png'))
    plot_loss_vs_iterations(results, 'LinReg Diabetes: Homogeneous Workers - Loss vs Iterations',
                            os.path.join(plots_dir, 'linreg_homogeneous_iter.png'))
    plot_grad_norm_vs_time(results, 'LinReg Diabetes: Homogeneous Workers - ||g|| vs Time',
                           os.path.join(plots_dir, 'linreg_homogeneous_gradnorm_time.png'))
    plot_grad_norm_vs_iterations(results, 'LinReg Diabetes: Homogeneous Workers - ||g|| vs Iterations',
                                 os.path.join(plots_dir, 'linreg_homogeneous_gradnorm_iter.png'))
    save_results(results, 'linreg_homogeneous', plots_dir)


def experiment_heterogeneous(config, data, initial_x, loss_fn, lambda_reg, plots_dir):
    """Experiment 2: Heterogeneous Workers (Stragglers)"""
    exp_config = config['experiment_heterogeneous']
    if not exp_config['enabled']:
        print("Skipping heterogeneous experiment (disabled)")
        return
    
    print("\n" + "="*60)
    print("Experiment 2: Heterogeneous Workers (Stragglers)")
    print("="*60)
    
    opt = config['optimization']
    n_workers = exp_config['n_workers']
    time_distributions = [create_time_distributions(t) for t in exp_config['worker_times']]
    
    gradient_fns = [
        partial(linear_regression_gradient, batch_size=len(data[1])//n_workers, lambda_reg=lambda_reg)
        for _ in range(n_workers)
    ]
    
    results = {}
    nag_momentum = exp_config.get('nag_momentum')
    for method_name in exp_config['methods']:
        print(f"  Running {method_name}...")
        batch_size = n_workers if method_name == 'RennalaSGD' else None
        momentum = nag_momentum if method_name == 'AsynchronousNAG' else None
        results[method_name] = run_method(
            method_name, data, initial_x, time_distributions, loss_fn, gradient_fns,
            opt['learning_rate'], max_time=opt['max_time'], batch_size=batch_size, momentum=momentum
        )
    
    plot_loss_vs_time(results, 'LinReg Diabetes: Heterogeneous Workers - Loss vs Time',
                      os.path.join(plots_dir, 'linreg_heterogeneous_time.png'))
    plot_loss_vs_iterations(results, 'LinReg Diabetes: Heterogeneous Workers - Loss vs Iterations',
                            os.path.join(plots_dir, 'linreg_heterogeneous_iter.png'))
    plot_grad_norm_vs_time(results, 'LinReg Diabetes: Heterogeneous Workers - ||g|| vs Time',
                           os.path.join(plots_dir, 'linreg_heterogeneous_gradnorm_time.png'))
    plot_grad_norm_vs_iterations(results, 'LinReg Diabetes: Heterogeneous Workers - ||g|| vs Iterations',
                                 os.path.join(plots_dir, 'linreg_heterogeneous_gradnorm_iter.png'))
    save_results(results, 'linreg_heterogeneous', plots_dir)


def experiment_stochastic(config, data, initial_x, loss_fn, lambda_reg, plots_dir):
    """Experiment 3: Stochastic Heterogeneity"""
    exp_config = config['experiment_stochastic']
    if not exp_config['enabled']:
        print("Skipping stochastic experiment (disabled)")
        return
    
    print("\n" + "="*60)
    print("Experiment 3: Stochastic Heterogeneity")
    print("="*60)
    
    opt = config['optimization']
    n_workers = exp_config['n_workers']
    time_distributions = [create_time_distributions(t) for t in exp_config['worker_times']]
    
    gradient_fns = [
        partial(linear_regression_gradient, batch_size=len(data[1])//n_workers, lambda_reg=lambda_reg)
        for _ in range(n_workers)
    ]
    
    results = {}
    nag_momentum = exp_config.get('nag_momentum')
    for method_name in exp_config['methods']:
        print(f"  Running {method_name}...")
        batch_size = n_workers if method_name == 'RennalaSGD' else None
        momentum = nag_momentum if method_name == 'AsynchronousNAG' else None
        results[method_name] = run_method(
            method_name, data, initial_x, time_distributions, loss_fn, gradient_fns,
            opt['learning_rate'], max_time=opt['max_time'], batch_size=batch_size, momentum=momentum
        )
    
    plot_loss_vs_time(results, 'LinReg Diabetes: Stochastic Heterogeneity - Loss vs Time',
                      os.path.join(plots_dir, 'linreg_stochastic_time.png'))
    plot_loss_vs_iterations(results, 'LinReg Diabetes: Stochastic Heterogeneity - Loss vs Iterations',
                            os.path.join(plots_dir, 'linreg_stochastic_iter.png'))
    plot_grad_norm_vs_time(results, 'LinReg Diabetes: Stochastic Heterogeneity - ||g|| vs Time',
                           os.path.join(plots_dir, 'linreg_stochastic_gradnorm_time.png'))
    plot_grad_norm_vs_iterations(results, 'LinReg Diabetes: Stochastic Heterogeneity - ||g|| vs Iterations',
                                 os.path.join(plots_dir, 'linreg_stochastic_gradnorm_iter.png'))
    save_results(results, 'linreg_stochastic', plots_dir)


def experiment_batch_size(config, data, initial_x, loss_fn, lambda_reg, plots_dir):
    """Experiment 4: RennalaSGD Batch Size Sweep"""
    exp_config = config['experiment_batch_size']
    if not exp_config['enabled']:
        print("Skipping batch size experiment (disabled)")
        return
    
    print("\n" + "="*60)
    print("Experiment 4: RennalaSGD Batch Size Sweep")
    print("="*60)
    
    opt = config['optimization']
    n_workers = exp_config['n_workers']
    time_dist = create_time_distributions(exp_config['time_distribution'])
    time_distributions = [time_dist] * n_workers
    
    gradient_fns = [
        partial(linear_regression_gradient, batch_size=len(data[1])//n_workers, lambda_reg=lambda_reg)
        for _ in range(n_workers)
    ]
    
    results = {}
    for batch_size in exp_config['batch_sizes']:
        print(f"  Running RennalaSGD with batch_size={batch_size}...")
        results[batch_size] = run_method(
            'RennalaSGD', data, initial_x, time_distributions, loss_fn, gradient_fns,
            opt['learning_rate'], max_time=opt['max_time'], batch_size=batch_size
        )
    
    plot_batch_size_sweep(results, 'LinReg Diabetes: RennalaSGD Batch Size Sweep',
                          os.path.join(plots_dir, 'linreg_batch_size_sweep.png'))
    
    # Also plot all curves together
    results_named = {f'batch_size={k}': v for k, v in results.items()}
    plot_loss_vs_time(results_named, 'LinReg Diabetes: RennalaSGD Batch Sizes - Loss vs Time',
                      os.path.join(plots_dir, 'linreg_batch_size_curves.png'))
    plot_grad_norm_vs_time(results_named, 'LinReg Diabetes: RennalaSGD Batch Sizes - ||g|| vs Time',
                           os.path.join(plots_dir, 'linreg_batch_size_gradnorm_curves.png'))
    save_results(results_named, 'linreg_batch_size', plots_dir)


def experiment_compression(config, data, initial_x, loss_fn, lambda_reg, plots_dir):
    """Experiment 5: Compression Comparison"""
    exp_config = config['experiment_compression']
    if not exp_config['enabled']:
        print("Skipping compression experiment (disabled)")
        return
    
    print("\n" + "="*60)
    print("Experiment 5: Compression Comparison")
    print("="*60)
    
    opt = config['optimization']
    n_workers = exp_config['n_workers']
    time_dist = create_time_distributions(exp_config['time_distribution'])
    time_distributions = [time_dist] * n_workers
    
    gradient_fns = [
        partial(linear_regression_gradient, batch_size=len(data[1])//n_workers, lambda_reg=lambda_reg)
        for _ in range(n_workers)
    ]
    
    results = {}
    for comp in exp_config['compression_types']:
        flag = comp['flag']
        k = comp['k'] if comp['k'] else initial_x.shape[0]
        name = f"{flag}" if flag == 'none' else f"{flag}_k={k}"
        print(f"  Running {exp_config['method']} with {name}...")
        
        results[name] = run_method(
            exp_config['method'], data, initial_x, time_distributions, loss_fn, gradient_fns,
            opt['learning_rate'], max_time=opt['max_time'], 
            compression_flag=flag, compression_size=k
        )
    
    plot_compression_comparison(results, 'LinReg Diabetes: Compression Comparison',
                                os.path.join(plots_dir, 'linreg_compression.png'))
    save_results(results, 'linreg_compression', plots_dir)


def experiment_compression_hetero(config, data, initial_x, loss_fn, lambda_reg, plots_dir):
    """Experiment 6: Compression under Heterogeneity"""
    exp_config = config['experiment_compression_hetero']
    if not exp_config['enabled']:
        print("Skipping compression+heterogeneity experiment (disabled)")
        return
    
    print("\n" + "="*60)
    print("Experiment 6: Compression under Heterogeneity")
    print("="*60)
    
    opt = config['optimization']
    n_workers = exp_config['n_workers']
    time_distributions = [create_time_distributions(t) for t in exp_config['worker_times']]
    
    gradient_fns = [
        partial(linear_regression_gradient, batch_size=len(data[1])//n_workers, lambda_reg=lambda_reg)
        for _ in range(n_workers)
    ]
    
    results = {}
    for comp in exp_config['compression_types']:
        flag = comp['flag']
        k = comp['k'] if comp['k'] else initial_x.shape[0]
        name = f"{flag}" if flag == 'none' else f"{flag}_k={k}"
        print(f"  Running {exp_config['method']} with {name}...")
        
        results[name] = run_method(
            exp_config['method'], data, initial_x, time_distributions, loss_fn, gradient_fns,
            opt['learning_rate'], opt['num_steps'],
            compression_flag=flag, compression_size=k
        )
    
    plot_compression_comparison(results, 'LinReg Diabetes: Compression under Heterogeneity',
                                os.path.join(plots_dir, 'linreg_compression_hetero.png'))
    save_results(results, 'linreg_compression_hetero', plots_dir)


def main():
    parser = argparse.ArgumentParser(description='Run Linear Regression experiments')
    parser.add_argument('--config', type=str, default='configs/linreg_diabetes.yaml',
                        help='Path to config file')
    parser.add_argument('--plots-dir', type=str, default='../plots',
                        help='Directory to save plots')
    parser.add_argument('--experiment', type=str, default='all',
                        choices=['all', 'homogeneous', 'heterogeneous', 'stochastic', 
                                 'batch_size', 'compression', 'compression_hetero'],
                        help='Which experiment to run')
    args = parser.parse_args()
    
    # Load config
    config_path = os.path.join(os.path.dirname(__file__), args.config)
    config = load_config(config_path)
    
    # Create plots directory
    plots_dir = os.path.join(os.path.dirname(__file__), args.plots_dir)
    os.makedirs(plots_dir, exist_ok=True)
    
    # Load data
    print("Loading Diabetes dataset...")
    X, y = load_diabetes_data()
    data = (X, y)
    
    # Setup
    lambda_reg = config['optimization']['lambda_reg']
    loss_fn = partial(linear_regression_loss, lambda_reg=lambda_reg)
    initial_x = np.zeros(X.shape[1])
    
    print(f"Feature dimension: {X.shape[1]}")
    print(f"Number of samples: {X.shape[0]}")
    print(f"Learning rate: {config['optimization']['learning_rate']}")
    print(f"Max simulated time: {config['optimization']['max_time']}")
    
    # Run experiments
    experiments = {
        'homogeneous': experiment_homogeneous,
        'heterogeneous': experiment_heterogeneous,
        'stochastic': experiment_stochastic,
        'batch_size': experiment_batch_size,
        'compression': experiment_compression,
        'compression_hetero': experiment_compression_hetero,
    }
    
    if args.experiment == 'all':
        for name, exp_fn in experiments.items():
            exp_fn(config, data, initial_x, loss_fn, lambda_reg, plots_dir)
    else:
        experiments[args.experiment](config, data, initial_x, loss_fn, lambda_reg, plots_dir)
    
    print("\n" + "="*60)
    print("All experiments completed!")
    print(f"Plots saved to: {plots_dir}")
    print("="*60)


if __name__ == '__main__':
    main()

