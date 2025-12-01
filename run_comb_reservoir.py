# run_comb_reservoir.py
"""
Main script for time series prediction using spin-chain reservoirs.
Compares different Hamiltonians across memory sizes.
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
from reservoir_gen import (  # Correct import: new time-series reservoir interface
    reservoir_from_binary_sequence,
    DEFAULT_DET_INTERVENTIONS,
)
from feature_engineering import (
    binarize_deviation_4_levels_windowed,
    extract_sigmaz_reset_with_washout,
    make_lagged_features,
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

T_INTERVAL = 1.75  # physical time between interventions
WASHOUT_LENGTH = 5
WINDOW_LAGS = 5

MEMORY_SIZES = [1, 2, 3, 4]
MODEL_KEYS = [
    "XXZ",
    "NNN_CHAOTIC",
    "NNN_LOCALIZED",
    "IAA_CHAOTIC",
    "IAA_LOCALIZED",
]

# ---------------------------------------------------------------------
# Core worker function
# ---------------------------------------------------------------------

def run_single_config(args):
    model_key, mem_size = args

    # 1) Binarize deviations locally
    x_seq = binarize_deviation_4_levels_windowed(
        deviations, window=WINDOW_LAGS, k_sigma=2.0
    )

    # 2) Add washout prefix mapped to 'I' (label 0)
    extended_labels = np.concatenate(
        [np.zeros(WASHOUT_LENGTH, dtype=int), x_seq]
    )

    # 3) Run reservoir: single result with intermediate measurements
    model_kwargs = {}
    if model_key.startswith("NNN"):
        model_kwargs["use_random"] = True

    result = reservoir_from_binary_sequence(
        x_seq=extended_labels,
        model_key=model_key,
        num_memory=mem_size,
        shots=1024,
        dt=T_INTERVAL,
        n_steps=1,
        det_basis=DEFAULT_DET_INTERVENTIONS,
        model_kwargs=model_kwargs,
    )

    # 4) Extract reservoir features as <σ_z> time series, discard washout
    reservoir_features = extract_sigmaz_reset_with_washout(
        result,
        n_steps=len(extended_labels),
        washout_length=WASHOUT_LENGTH,
    )
    # Align with original deviations length
    reservoir_features = reservoir_features[-len(deviations) :]

    # 5) Create lagged dataset for supervised learning
    X, y = make_lagged_features(
        reservoir_features,
        target_series=deviations,
        window=WINDOW_LAGS,
    )

    # 6) Train-test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 7) Train MLP readout
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
# Saving and plotting functions (unchanged)
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
# Main entry
# ---------------------------------------------------------------------

if __name__ == "__main__":
    tasks = [(model, mem) for model in MODEL_KEYS for mem in MEMORY_SIZES]

    print(f"Running {len(tasks)} configurations...")
    with multiprocessing.Pool(processes=4) as pool:
        all_results = pool.map(run_single_config, tasks)

    print("Done! Plotting results...")

    plot_loss_curves(all_results)
    plot_mae_heatmap(all_results)
    save_results(all_results, T_INTERVAL)
    print("Results saved!")
