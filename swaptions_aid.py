import numpy as np
from sklearn.neural_network import MLPRegressor
import pandas as pd

from reserve_mem import (
    reservoir_with_qubit_reuse,
    extract_sigmaz_reset_with_washout,
)

# Load from your source file
df = pd.read_excel("opt_data/sample_Simulated_Swaption_Price.xlsx", sheet_name=0)

# -------------------------
# Column selection utilities
# -------------------------

def select_timeseries(df: pd.DataFrame, col_index: int) -> pd.Series:
    col_name = df.columns[col_index]
    return df[col_name].astype(float)


def first_nan_index(series: pd.Series):
    nan_mask = series.isna().values
    if not nan_mask.any():
        return None
    return np.argmax(nan_mask)


def build_last5_window(series: pd.Series, idx_nan: int, window=5):
    start = idx_nan - window
    if start < 0:
        return None
    window_vals = series.iloc[start:idx_nan].values
    if np.any(np.isnan(window_vals)):
        return None
    return window_vals


# -------------------------
# Main imputation function
# -------------------------

def impute_nans_reservoir_stepwise(
    series,
    w=5,
    mem_size=2,
    washout_length=5,
    max_train_windows=10,
    n_steps=8,                  # FIXED ansatz repetitions
):
    """
    Strict NaN imputation using reservoir computing.
    - Uses ONLY clean windows for training
    - Performs stepwise forward imputation
    - Reservoir depth (n_steps) is NOT window length
    """

    vals = np.asarray(series, dtype=float).copy()
    T = len(vals)

    # -----------------------------
    # Step 1 — collect clean windows
    # -----------------------------

    X_train, y_train = [], []

    for t in range(w, T):
        if len(X_train) >= max_train_windows:
            break

        window = vals[t - w : t]
        target = vals[t]

        if np.isnan(target) or np.isnan(window).any():
            continue

        # Normalize window
        mean = window.mean()
        std = window.std() if window.std() > 0 else 1.0
        window_norm = (window - mean) / std

        extended = np.concatenate([np.zeros(washout_length), window_norm])

        raw = reservoir_with_qubit_reuse(
            extended,
            num_memory=mem_size,
            shots=1024,
            J=1.0,
            dt=1,
            n_steps=n_steps,           # FIXED depth
            disorder_scale=1.0,
        )

        # Extract exactly n_steps features
        feats = extract_sigmaz_reset_with_washout(
            raw, n_steps=n_steps, washout_length=washout_length
        )

        X_train.append(feats)
        y_train.append(target)

    if len(X_train) == 0:
        return vals

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # -----------------------------
    # Step 2 — train readout MLP
    # -----------------------------

    mlp = MLPRegressor(
        hidden_layer_sizes=(n_steps,),
        max_iter=500,
        random_state=0,
    )
    mlp.fit(X_train, y_train)

    # -----------------------------
    # Step 3 — stepwise imputation
    # -----------------------------

    for i in range(T):
        if not np.isnan(vals[i]):
            continue

        if i < w or np.isnan(vals[i - w : i]).any():
            continue

        window = vals[i - w : i]

        # Normalize
        mean = window.mean()
        std = window.std() if window.std() > 0 else 1.0
        window_norm = (window - mean) / std

        extended = np.concatenate([np.zeros(washout_length), window_norm])

        raw = reservoir_with_qubit_reuse(
            extended,
            num_memory=mem_size,
            shots=1024,
            J=1.0,
            dt=1,
            n_steps=n_steps,
            disorder_scale=1.0,
        )

        feats = extract_sigmaz_reset_with_washout(
            raw, n_steps=n_steps, washout_length=washout_length
        )

        vals[i] = mlp.predict([feats])[0]

    return vals
