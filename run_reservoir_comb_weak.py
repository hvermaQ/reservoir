"""
Quantum Reservoir Computing Pipeline for Option Pricing Deviations
Updated to use full per-window post-washout time series features.
Supports XXZ, NNN (chaotic/localized), IAA (chaotic/localized) models.
"""

import multiprocessing
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score

# Core modules
from data_gen import generate_data
from reservoir_gen import reservoir_results_per_window_ancilla, DEFAULT_DET_INTERVENTIONS
from feature_engineering import create_lagged_binary_features, features_from_results
from config import CONFIG

print("=== QUANTUM RESERVOIR COMPUTING PIPELINE ===")
print(f"Washout length: {CONFIG['washout_length']}")

def get_deviation_timeseries(df, strike, option_type, expiry):
    """Extract deviation time series for specific strike/expiry."""
    ts = df[(df['strike'] == strike) & 
            (df['type'] == option_type) & 
            (df['expiration'] == expiry)].sort_values('date')
    return ts['date'].values, ts['Deviation'].values

def prepare_data_pipeline():
    """Full data preparation: load → binarize → add washout."""
    # 1. Load raw option data
    print("1. Loading data...")
    raw_data = generate_data(CONFIG['target'])
    dates, deviations = get_deviation_timeseries(
        raw_data, CONFIG['strike'], CONFIG['option_type'], CONFIG['expiry']
    )
    print(f"   Dataset: {len(deviations)} points, μ={deviations.mean():.4f}, σ={deviations.std():.4f}")
    
    # 2. Binarized lagged windows
    print("2. Binarization + lagged features...")
    X_windows, y_targets = create_lagged_binary_features(
        deviations, 
        lag_window=CONFIG['window_lags'],
        sigma_threshold=1.0,
        two_sigma_threshold=2.0
    )
    print(f"   X_windows: {X_windows.shape}, targets: {y_targets.shape}")
    print(f"   Labels: {np.unique(X_windows)}")
    
    # 3. Prepend washout prefix (handled by reservoir_gen)
    num_windows = X_windows.shape[0]
    print(f"   {num_windows} windows prepared")
    
    return X_windows, y_targets

def run_single_config(args):
    """Execute ancilla-assisted weak measurement reservoir + MLP for one config."""
    model_key, mem_size, X_windows, y_targets = args
    model_kwargs = {'use_random': True} if model_key.startswith('NNN') else {}

    results_list = reservoir_results_per_window_ancilla(
        X_windows=X_windows,
        model_key=model_key,
        num_memory=mem_size,
        shots=CONFIG['shots'],
        dt=CONFIG['dt'],
        n_steps=CONFIG['n_steps'],
        det_basis=DEFAULT_DET_INTERVENTIONS,
        model_kwargs=model_kwargs,
        washout_length=CONFIG['washout_length'],
        epsilon=CONFIG['epsilon'],            # weak measurement angle
        final_strong_measure=False            # keep only ancilla trajectories
    )

    # Extract full post-washout ancilla <Z> time series
    feature_vectors = features_from_results(
        results_list, 
        washout_length=CONFIG['washout_length'], 
        compress_mode="full"
    )
    feature_vectors = feature_vectors[:len(y_targets)]  # align with targets

    # Train/test split
    split_idx = int(CONFIG['train_split'] * len(y_targets))
    X_train, X_test = feature_vectors[:split_idx], feature_vectors[split_idx:]
    y_train, y_test = y_targets[:split_idx], y_targets[split_idx:]

    # MLP readout
    mlp = MLPRegressor(
        hidden_layer_sizes=CONFIG['mlp_layers'],
        max_iter=CONFIG['mlp_max_iter'],
        random_state=0
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    # Metrics
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred),
        'RMSE': root_mean_squared_error(y_test, y_pred),
        'R2': r2_score(y_test, y_pred)
    }

    print(f"   {model_key}_M{mem_size}: MAE={metrics['MAE']:.4f}, "
          f"RMSE={metrics['RMSE']:.4f}, R2={metrics['R2']:.4f}")

    return {
        'model_key': model_key,
        'mem_size': mem_size,
        **metrics,
        'loss_curve': mlp.loss_curve_
    }


def visualize_results(all_results):
    """Generate loss curves and MAE heatmap."""
    
    # Loss curves by model
    grouped = {}
    for res in all_results:
        if res is None: continue
        key = res['model_key']
        grouped.setdefault(key, []).append(res)
    
    for model_key, res_list in grouped.items():
        plt.figure(figsize=(8, 6))
        for res in sorted(res_list, key=lambda r: r['mem_size']):
            plt.plot(res['loss_curve'], label=f'M{res["mem_size"]}', 
                     marker='o', markersize=3)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'{model_key} MLP Loss (dt={CONFIG["dt"]:.2f})')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f'loss_{model_key}_dt{CONFIG["dt"]:.2f}.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # MAE heatmap
    models = sorted(set(res['model_key'] for res in all_results if res))
    mem_sizes = sorted(set(res['mem_size'] for res in all_results if res))
    mae_matrix = np.full((len(models), len(mem_sizes)), np.nan)
    
    for i, model in enumerate(models):
        for j, mem in enumerate(mem_sizes):
            for res in all_results:
                if res and res['model_key'] == model and res['mem_size'] == mem:
                    mae_matrix[i, j] = res['MAE']
                    break
    
    plt.figure(figsize=(10, 6))
    im = plt.imshow(mae_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='MAE')
    plt.xticks(range(len(mem_sizes)), mem_sizes)
    plt.yticks(range(len(models)), models)
    plt.xlabel('Memory Size')
    plt.ylabel('Model')
    plt.title(f'MAE Heatmap (dt={CONFIG["dt"]:.2f})')
    plt.tight_layout()
    plt.savefig(f'mae_heatmap_dt{CONFIG["dt"]:.2f}.png', dpi=300, bbox_inches='tight')
    plt.close()

def save_results(all_results):
    """Save metrics and loss curves."""
    metrics, loss_curves = {}, {}
    for res in all_results:
        if res is None: continue
        key = res['model_key']
        mem = res['mem_size']
        metrics.setdefault(key, {})[mem] = {
            'MAE': res['MAE'], 'RMSE': res['RMSE'], 'R2': res['R2']
        }
        loss_curves.setdefault(key, {})[mem] = res['loss_curve']
    
    base = f'dt{CONFIG["dt"]:.2f}_w{CONFIG["washout_length"]}'
    np.save(f'{base}_loss.npy', loss_curves)
    with open(f'{base}_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2, default=float)

if __name__ == '__main__':
    # Prepare data once
    X_windows, y_targets = prepare_data_pipeline()
    
    # Create tasks
    tasks = [(model, mem, X_windows, y_targets) 
             for model in CONFIG['model_keys'] 
             for mem in CONFIG['memory_sizes']]
    
    print(f"3. Launching {len(tasks)} configurations in parallel...")
    
    # Multiprocessing execution
    with multiprocessing.Pool(processes=10) as pool:
        all_results = pool.map(run_single_config, tasks)

    # Analysis and visualization
    print("4. Generating visualizations...")
    visualize_results(all_results)
    save_results(all_results)
    
    print("5. COMPLETE!")
    print("   PNGs: loss curves, MAE heatmap")
    print("   Files: metrics.json, loss.npy")
