import numpy as np
import json
import glob
import matplotlib.pyplot as plt
import os
import re


def plot_loss_and_error_vs_mem(mem_sizes=[4, 16, 64], figsize_loss=(12, 8), figsize_err=(10, 6), error_metric='MAE'):
    """
    Parse saved loss curves and error metrics, and plot in two separate figures:
    1. Loss curves for multiple memory sizes
    2. Error metrics vs memory size
    
    Args:
        mem_sizes: List of memory sizes to analyze and plot
        figsize_loss: Figure size for loss curves plot
        figsize_err: Figure size for error metrics plot
        error_metric: Which error metric to use for mean error plot ('MAE', 'RMSE', 'MSE')
    """
    loss_files = sorted(glob.glob("ising_tfim_disorder/loss_curves_nsteps*_disorder*.npy"))
    metrics_files = sorted(glob.glob("ising_tfim_disorder/error_metrics_nsteps*_disorder*.json"))
    
    if not loss_files or not metrics_files:
        print("Missing loss or metrics files.")
        print("Directory contents:", sorted(glob.glob("ising_tfim_disorder/*")))
        return
    
    loss_data = {}
    all_metrics = {}
    pattern = re.compile(r"nsteps(\d+)_disorder(\d+)")
    
    # Load loss data
    for loss_file in loss_files:
        basename = os.path.basename(loss_file)
        match = pattern.search(basename)
        if not match:
            continue
        n_steps = int(match.group(1))
        disorder = float(match.group(2))
        loss_curves = np.load(loss_file, allow_pickle=True).item()
        for mem_size in mem_sizes:
            if mem_size in loss_curves:
                loss_data.setdefault(mem_size, {})[(n_steps, disorder)] = loss_curves[mem_size]

    # Load error metrics - handle string keys like "1", "2", "4", etc.
    for metrics_file in metrics_files:
        basename = os.path.basename(metrics_file)
        match = pattern.search(basename)
        if not match:
            continue
        n_steps = int(match.group(1))
        disorder = float(match.group(2))
        
        with open(metrics_file, 'r') as f:
            error_metrics = json.load(f)
        
        all_metrics[(n_steps, disorder)] = error_metrics
    
    if not loss_data or not all_metrics:
        print("No data found.")
        return
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(mem_sizes)))
    linestyles = ['-', '--', '-.', ':']
    
    # Figure 1: Loss curves
    plt.figure(figsize=figsize_loss)
    for idx, mem_size in enumerate(mem_sizes):
        if mem_size in loss_data:
            for (n_steps, disorder), loss_curve in loss_data[mem_size].items():
                epochs = np.arange(len(loss_curve))
                plt.plot(epochs, loss_curve, 'o-', linewidth=1.5,
                         color=colors[idx],
                         linestyle=linestyles[idx % len(linestyles)],
                         label=f'mem={mem_size}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves by Memory Size')
    plt.legend(fontsize='small')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    
    # Figure 2: Error vs Memory Size (mean over all nsteps/disorder)
    mem_size_ints = []
    mean_errors = []
    
    plt.figure(figsize=figsize_err)
    for mem_size in mem_sizes:
        errors_for_mem = []
        for (_, _), metrics in all_metrics.items():
            mem_str = str(mem_size)
            if mem_str in metrics:
                if error_metric in metrics[mem_str]:
                    errors_for_mem.append(metrics[mem_str][error_metric])
        
        if errors_for_mem:
            mean_err = np.mean(errors_for_mem)
            mem_size_ints.append(mem_size)
            mean_errors.append(mean_err)
            print(f"mem_size {mem_size}: {len(errors_for_mem)} points, mean {error_metric} = {mean_err:.4f}")
        else:
            print(f"No {error_metric} data for mem_size {mem_size}")
    
    if mem_size_ints:
        plt.plot(mem_size_ints, mean_errors, 'o-', markersize=8, linewidth=3, color='tab:red')
        plt.xlabel('Memory Size')
        plt.ylabel(f'Mean {error_metric}')
        plt.title(f'Mean {error_metric} vs Memory Size')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()


# Example usage:
if __name__ == "__main__":
    plot_loss_and_error_vs_mem(mem_sizes=[1, 2, 3, 4, 5, 6, 7, 8], error_metric='MAE')
