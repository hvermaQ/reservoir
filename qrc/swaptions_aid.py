# swaptions_aid.py (Deviation-based, Dynamic Caching, Reservoir Imputation)

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from reserve_mem import (
    reservoir_with_qubit_reuse,
    extract_sigmaz_reset_with_washout,
)

# ---------------------------------------------------------------------
# Load Excel file once (as in your original version)
# ---------------------------------------------------------------------
df = pd.read_excel("opt_data/sample_Simulated_Swaption_Price.xlsx", sheet_name=0)


# ---------------------------------------------------------------------
# Column selection utilities (unchanged API)
# ---------------------------------------------------------------------
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


# ---------------------------------------------------------------------
# Main reservoir-based NaN imputation function (deviation modeling)
# ---------------------------------------------------------------------
def impute_nans_reservoir_stepwise(
    series,
    w: int = 5,
    mem_size: int = 2,
    washout_length: int = 10,
    max_train_windows: int = 20,
):
    """
    Deviation-based reservoir imputation with dynamic caching.

    Instead of predicting v_t directly, we predict:
        deviation_t = v_t - mean(window)

    Then reconstruct:
        v_t = mean(window) + deviation_pred

    Reservoir features are cached based on a rounded, normalized window
    (Option B), so repeated or similar windows reuse the same QPU result.
    """

    # ---- Reservoir hyperparameters ----
    shots = 128
    ansatz_steps = 4

    vals = np.asarray(series, dtype=float).copy()
    T = len(vals)

    # Cache: key (rounded normalized window) -> reservoir features
    reservoir_cache = {}

    def window_to_norm_and_key(window: np.ndarray):
        """
        Center + normalize a window and compute its cache key
        (rounded normalized values).
        """
        mean = window.mean()
        centered = window - mean
        std = centered.std()
        if std < 1e-8:
            std = 1.0
        normed = centered / std
        key = tuple(np.round(normed, 3))  # Option B: rounding for more cache hits
        return normed, key

    def get_reservoir_features(window: np.ndarray) -> np.ndarray:
        """
        Return reservoir features for a given window, using dynamic caching.
        If the key is new, run the reservoir once and store the result.
        """
        window_norm, key = window_to_norm_and_key(window)

        if key in reservoir_cache:
            return reservoir_cache[key]

        # Build extended input: washout zeros + normalized window
        extended = np.concatenate([np.zeros(washout_length), window_norm])

        raw = reservoir_with_qubit_reuse(
            extended,
            num_memory=mem_size,
            shots=shots,
            J=1.0,
            dt=1.0,
            n_steps=ansatz_steps,
            disorder_scale=1.0,
        )

        feats = extract_sigmaz_reset_with_washout(
            raw,
            n_steps=w,
            washout_length=washout_length
        )

        reservoir_cache[key] = feats
        return feats

    # -----------------------------------------------------------------
    # Step 1: Build training set (predict deviations)
    # -----------------------------------------------------------------
    X_train, y_train = [], []

    for t in range(w, T):
        if len(X_train) >= max_train_windows:
            break

        window = vals[t - w:t]
        v_t = vals[t]

        if np.isnan(v_t) or np.isnan(window).any():
            continue

        w_mean = window.mean()
        target_deviation = v_t - w_mean

        feats = get_reservoir_features(window)
        X_train.append(feats)
        y_train.append(target_deviation)

    if not X_train:
        return vals  # nothing to learn

    X_train = np.vstack(X_train)
    y_train = np.array(y_train)

    # -----------------------------------------------------------------
    # Step 2: Train readout on DEVIATIONS
    # -----------------------------------------------------------------
    readout = Ridge(alpha=1.0)
    readout.fit(X_train, y_train)

    # -----------------------------------------------------------------
    # Step 3: Stepwise imputation (autoregressive)
    # -----------------------------------------------------------------
    imputed = vals.copy()

    while True:
        nan_idxs = np.where(np.isnan(imputed))[0]
        if len(nan_idxs) == 0:
            break

        changed = False

        for i in nan_idxs:
            if i < w:
                continue

            window = imputed[i - w: i]
            if np.isnan(window).any():
                continue

            w_mean = window.mean()
            feats = get_reservoir_features(window)
            deviation_pred = readout.predict(feats.reshape(1, -1))[0]

            # reconstruct original value: window mean + deviation
            v_pred = w_mean + deviation_pred

            # enforce non-negative volatility
            imputed[i] = max(v_pred, 0.0)
            changed = True

        if not changed:
            break

    return imputed
