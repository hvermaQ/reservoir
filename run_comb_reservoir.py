# run_comb_reservoir.py
"""
Main script for time series prediction using spin-chain reservoirs.
Computes binarized input once for all model-memory configs.
"""

import multiprocessing
import json
import numpy as np
import matplotlib.pyplot as plt

from sklearn.neural_network import MLPRegressor
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)

from gen_dat import generate_data
from reservoir_gen import (
    reservoir_from_binary_sequence,
    DEFAULT_DET_INTERVENTIONS,
)
from feature_engineering import (
    create_lagged_binary_features,
    extract_sigmaz_reset_with_washout,
)

# ---------------------------------------------------------------------
# Data loading and deviation extraction
# ---------------------------------------------------------------------

target = "AAPL"
raw_data = generate_data(target)

def get_deviation_timeseries(df, strike, option_type, expiry):
    ts = df[
        (df["strike"] == strike)
        & (df["type"] == option_type)
        & (df["expiration"] == expiry)
    ].sort_values("date")
    return ts["date"].values, ts["Deviation"].values

dates, deviations = get_deviation_timeseries(
    raw_data, strike=500, option_type="call", expiry="2013-01-19"
)

# ---------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------

T_INTERVAL = 1.75
WASHOUT_LENGTH = 0
WINDOW_LAGS = 5

MEMORY_SIZES = [1, 2]
MODEL_KEYS = [
    "XXZ",
    "NNN_CHAOTIC",
    "NNN_LOCALIZED",
    "IAA_CHAOTIC",
    "IAA_LOCALIZED",
]

# ---------------------------------------------------------------------
# Precompute binarization + lagged features once
# ---------------------------------------------------------------------

X, y = create_lagged_binary_features(
    deviations,
    lag_window=WINDOW_LAGS,
    sigma_threshold=1.0,
    two_sigma_threshold=2.0,
)

# Use the last label in each lag window as reservoir input per time step
reservoir_input = X[:, -1]

# Prepend washout prefix
extended_labels = np.concatenate(
    [np.zeros(WASHOUT_LENGTH, dtype=int), reservoir_input]
)

# Corresponding trimmed targets after washout
trimmed_y = y

# ---------------------------------------------------------------------
# Core worker function (accepts precomputed inputs)
# ---------------------------------------------------------------------

def run_single_config(args):
    model_key, mem_size, x_seq, targets = args

    model_kwargs = {}
    if model_key.startswith("NNN"):
        model_kwargs["use_random"] = True

    # Run reservoir on precomputed binarized input sequence
    result = reservoir_from_binary_sequence(
        x_seq=x_seq,
        model_key=model_key,
        num_memory=mem_size,
        shots=256,
        dt=T_INTERVAL,
        n_steps=1,
        det_basis=DEFAULT_DET_INTERVENTIONS,
        model_kwargs=model_kwargs,
    )

    # Extract reservoir features from intermediate measurements
    reservoir_features = extract_sigmaz_reset_with_washout(
        result,
        num_timesteps=len(x_seq),
        washout_length=WASHOUT_LENGTH,
    )
    reservoir_features = reservoir_features[-len(targets):]

    # Train/test split
    split = int(0.8 * len(targets))
    X_train, X_test = reservoir_features[:split].reshape(-1, 1), reservoir_features[split:].reshape(-1, 1)
    y_train, y_test = targets[:split], targets[split:]

    # Classical readout (MLP)
    mlp = MLPRegressor(
        hidden_layer_sizes=(2,),
        max_iter=500,
        random_state=0,
    )
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "model_key": model_key,
        "mem_size": mem_size,
        "MAE": mae,
        "RMSE": rmse,
        "MSE": mse,
        "R2": r2,
        "loss_curve": mlp.loss_curve_,
    }

# ---------------------------------------------------------------------
# Saving and plotting utilities (unchanged)
# ---------------------------------------------------------------------

def save_results(all_results, t_interval):
    metrics = {}
    loss_curves = {}
    for res in all_results:
        key = res["model_key"]
        mem = res["mem_size"]
        if key not in metrics:
            metrics[key] = {}
            loss_curves[key] = {}
        metrics[key][mem] = {
            "MAE": res["MAE"],
            "RMSE": res["RMSE"],
            "MSE": res["MSE"],
            "R2": res["R2"],
        }
        loss_curves[key][mem] = res["loss_curve"]

    base = f"t{t_interval:.2f}"
    np.save(f"loss_curves_{base}.npy", loss_curves)
    with open(f"error_metrics_{base}.json", "w") as f:
        json.dump(metrics, f, indent=2)

def plot_loss_curves(all_results):
    grouped = {}
    for res in all_results:
        grouped.setdefault(res["model_key"], []).append(res)

    for model_key, res_list in grouped.items():
        plt.figure(figsize=(8, 6))
        for res in sorted(res_list, key=lambda r: r["mem_size"]):
            plt.plot(
                res["loss_curve"],
                label=f"mem={res['mem_size']}",
                marker='o',
                markersize=3
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"MLP Training Loss – {model_key} (t={T_INTERVAL:.2f})")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(f"loss_{model_key}_t{T_INTERVAL:.2f}.png", dpi=300)
        plt.show()

def plot_mae_heatmap(all_results):
    models = sorted(set(res["model_key"] for res in all_results))
    mem_sizes = sorted(set(res["mem_size"] for res in all_results))
    
    mae_matrix = np.zeros((len(models), len(mem_sizes)))
    for i, model in enumerate(models):
        for j, mem in enumerate(mem_sizes):
            for res in all_results:
                if res["model_key"] == model and res["mem_size"] == mem:
                    mae_matrix[i, j] = res["MAE"]
                    break
    
    plt.figure(figsize=(10, 6))
    im = plt.imshow(mae_matrix, cmap='viridis', aspect='auto')
    plt.colorbar(im, label='MAE')
    plt.xticks(range(len(mem_sizes)), mem_sizes)
    plt.yticks(range(len(models)), models)
    plt.xlabel('Memory Size')
    plt.ylabel('Model')
    plt.title(f'MAE Heatmap (t={T_INTERVAL:.2f})')
    plt.tight_layout()
    plt.savefig(f"mae_heatmap_t{T_INTERVAL:.2f}.png", dpi=300)
    plt.show()

# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    tasks = [
        (model, mem, extended_labels, trimmed_y) for model in MODEL_KEYS for mem in MEMORY_SIZES
    ]

    print(f"Running {len(tasks)} configurations...")
    with multiprocessing.Pool(processes=4) as pool:
        all_results = pool.map(run_single_config, tasks)

    print("Done! Plotting results...")

    plot_loss_curves(all_results)
    plot_mae_heatmap(all_results)
    save_results(all_results, T_INTERVAL)
    print("Results saved!")


"""
result = reservoir_from_binary_sequence(x_seq=X[0], model_key="XXZ", num_memory=2, dt=1.75, n_steps=1)
print(result)

reservoir_features = extract_sigmaz_reset_with_washout(
    result,
    n_steps=len(X[0]),
    washout_length=0, 
)
"""