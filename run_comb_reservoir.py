# run_ppe_reservoir.py

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
    reservoir_comb_from_binary_sequence,
    DEFAULT_DET_INTERVENTIONS,
)
from feature_engineering import (
    binarize_deviation_4_levels,
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

# Physical time between interventions t (paper uses values like 1.75)[file:22]
T_INTERVAL = 1.75
WASHOUT_LENGTH = 5
WINDOW_LAGS = 10

MEMORY_SIZES = [1, 2, 3, 4, 5, 6, 7, 8]
# Hamiltonians to compare (keys from ham_gen.MODEL_BLOCKS)
MODEL_KEYS = [
    "XXZ",
    "NNN_CHAOTIC",
    "NNN_LOCALIZED",
    "IAA_CHAOTIC",
    "IAA_LOCALIZED",
]


# ---------------------------------------------------------------------
# Core worker: one (model_key, mem_size) pair
# ---------------------------------------------------------------------

def run_single_config(args):
    model_key, mem_size = args

    # 1) Pre-binarize deviations into 0/1/2/3 once
    x_seq = binarize_deviation_4_levels(deviations)

    # 2) Add washout prefix (all zeros → 'I' intervention)
    extended_labels = np.concatenate(
        [np.zeros(WASHOUT_LENGTH, dtype=int), x_seq]
    )

    # 3) Build and run reservoir comb
    model_kwargs = {}
    if model_key.startswith("NNN"):
        # Allow random NNN fields if desired
        model_kwargs["use_random"] = True

    raw_result = reservoir_comb_from_binary_sequence(
        x_seq=extended_labels,
        model_key=model_key,
        num_memory=mem_size,
        shots=1024,
        dt=T_INTERVAL,      # t between interventions
        n_steps=1,          # single-step evolution of duration t
        det_basis=DEFAULT_DET_INTERVENTIONS,
        model_kwargs=model_kwargs,
    )

    # 4) Extract features, discard washout, align with original deviations
    features_full = extract_sigmaz_reset_with_washout(
        raw_result,
        data_length=len(extended_labels),
        washout_length=WASHOUT_LENGTH,
    )
    # Ensure length matches base deviations
    reservoir_features = features_full[-len(deviations):]

    # 5) Build lagged dataset
    X, y = make_lagged_features(
        reservoir_features,
        target_series=deviations,
        window=WINDOW_LAGS,
    )

    # 6) Train/test split
    split = int(0.8 * len(X))
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    # 7) Classical readout (MLP)
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
# Saving utilities
# ---------------------------------------------------------------------

def save_results(all_results, t_interval):
    # Organize by model_key
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

    # Save JSON + NPY per t
    base = f"t{t_interval:.2f}"
    np.save(f"loss_curves_{base}.npy", loss_curves)
    with open(f"error_metrics_{base}.json", "w") as f:
        json.dump(metrics, f, indent=2)


# ---------------------------------------------------------------------
# Plotting: loss curves grouped by Hamiltonian
# ---------------------------------------------------------------------

def plot_loss_curves(all_results):
    # Group results by model_key
    grouped = {}
    for res in all_results:
        grouped.setdefault(res["model_key"], []).append(res)

    for model_key, res_list in grouped.items():
        plt.figure()
        for res in sorted(res_list, key=lambda r: r["mem_size"]):
            plt.plot(
                res["loss_curve"],
                label=f"mem={res['mem_size']}",
            )
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title(f"MLP Training Loss – {model_key}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

if __name__ == "__main__":
    # All (model, memory) combinations
    tasks = [(m, mem) for m in MODEL_KEYS for mem in MEMORY_SIZES]

    with multiprocessing.Pool() as pool:
        all_results = pool.map(run_single_config, tasks)

    # Plot loss curves per Hamiltonian
    plot_loss_curves(all_results)

    # Save to disk
    save_results(all_results, T_INTERVAL)
