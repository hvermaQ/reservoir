# feature_engineering.py
"""
Preprocessing and feature-extraction utilities for PPE reservoirs.

Responsibilities:
  - Map real-valued deviation time series to discrete lagged-window features,
    combining binarization and lagged feature construction.
  - Extract σ_z-like features from reservoir results (with washout support).
"""

import numpy as np


# ---------------------------------------------------------------------
# Merged lagged binarization + lagged feature construction
# ---------------------------------------------------------------------

def create_lagged_binary_features(
    values, lag_window, sigma_threshold=1.0, two_sigma_threshold=2.0
):
    """
    Lagged binarization for PREDICTION: window predicts NEXT value.
    
    X[t] = binarized labels [t-lag_window+1 : t+1] → y[t] = values[t+1]
    """
    vals = np.asarray(values, dtype=float)
    T = len(vals)
    
    # Binarize full sequence first
    binary_seq = np.zeros(T, dtype=int)
    for t in range(T):
        start = max(0, t - lag_window + 1)
        context = vals[start : t + 1]
        mu_t = context.mean()
        sigma_t = context.std()
        x_t = vals[t]
        
        if sigma_t == 0:
            binary_seq[t] = 1 if x_t < mu_t else 2
            continue
        
        neg_large = mu_t - two_sigma_threshold * sigma_t
        neg_small = mu_t - sigma_threshold * sigma_t
        pos_small = mu_t + sigma_threshold * sigma_t
        
        if x_t < neg_large:
            binary_seq[t] = 0
        elif x_t < neg_small:
            binary_seq[t] = 1
        elif x_t < pos_small:
            binary_seq[t] = 2
        else:
            binary_seq[t] = 3
    
    # Build lagged windows predicting NEXT value
    X_list, y_list = [], []
    for t in range(lag_window - 1, T - 1):  # Stop 1 early for y[t+1]
        X_list.append(binary_seq[t - lag_window + 1 : t + 1])
        y_list.append(vals[t + 1])  # NEXT timestep!
    
    X = np.array(X_list)
    y = np.array(y_list)
    return X, y

# ---------------------------------------------------------------------
# Extract <σ_z> from reservoir result with washout support
# -------------------------------------------------------------------

def _sigmaz_from_counts(counts: dict, n_bits: int) -> np.ndarray:
    """
    Compute ⟨σ_z⟩ for each of n_bits classical bits from Qiskit counts.

    Qiskit bitstrings are little-endian: rightmost char = classical bit 0.
    """
    total_shots = sum(counts.values())
    bit1_count = np.zeros(n_bits)
    for bitstring, count in counts.items():
        bs = bitstring.replace(' ', '')
        for t in range(n_bits):
            idx = len(bs) - 1 - t
            if idx >= 0 and bs[idx] == '1':
                bit1_count[t] += count
    return 1 - 2 * (bit1_count / total_shots) if total_shots > 0 else np.ones(n_bits)


def extract_features_from_results(results_list, washout_length=5, discard_washout=True):
    """
    Extract a scalar feature per window from Qiskit reservoir results.

    Parameters
    ----------
    results_list : list of qiskit.result.Result
    washout_length : int
        Number of initial timesteps per circuit to discard.
    discard_washout : bool
        If True, return the last post-washout ⟨σ_z⟩ value per window.
        If False, return the final timestep value across the full sequence.

    Returns
    -------
    features : np.ndarray, shape (num_windows,)
    """
    features = []
    for result in results_list:
        counts = result.get_counts()
        n_total = len(next(iter(counts)).replace(' ', ''))
        sigmaz = _sigmaz_from_counts(counts, n_total)

        if discard_washout:
            feature = sigmaz[washout_length:][-1]
        else:
            feature = sigmaz[-1]

        features.append(feature)

    return np.array(features)


def create_lagged_quantile_features(values, lag_window):
    """
    Quantile-based symbolic encoding into 4 bins.
    Consistent, stationary, suitable for quantum reservoirs.
    """
    vals = np.asarray(values, dtype=float)
    T = len(vals)
    
    # Compute global quantiles once
    q1, q2, q3 = np.quantile(vals, [0.25, 0.5, 0.75])

    # Encode into symbolic alphabet {0,1,2,3}
    encoded = np.digitize(vals, bins=[q1, q2, q3])  # returns 0,1,2,3
    
    # Build lagged windows predicting next value
    X_list, y_list = [], []
    for t in range(lag_window - 1, T - 1):
        X_list.append(encoded[t - lag_window + 1 : t + 1])
        y_list.append(vals[t + 1])

    return np.array(X_list), np.array(y_list)


# -------------------------------------------------------------
# 1) Extract ⟨Z⟩(t) from intermediate measurements after washout
# -------------------------------------------------------------

def extract_features_weak(results_list, washout_length=5, ancilla_cbit=0):
    """
    Convert Qiskit reservoir results (ancilla weak measurements) into
    per-timestep ⟨Z⟩ arrays for each window, discarding the washout.

    Each classical bit t holds the ancilla outcome at timestep t.

    Returns
    -------
    np.ndarray of shape (num_windows, window_length - washout_length)
    """
    all_features = []
    for result in results_list:
        counts = result.get_counts()
        n_total = len(next(iter(counts)).replace(' ', ''))
        sigmaz = _sigmaz_from_counts(counts, n_total)
        all_features.append(sigmaz[washout_length:])
    return np.vstack(all_features)


def features_from_results(results_list, washout_length=5, compress_mode="full"):
    """
    Extract feature vectors from Qiskit reservoir results.

    Parameters
    ----------
    results_list : list of qiskit.result.Result
    washout_length : int
        Initial timesteps to discard per circuit.
    compress_mode : str
        "full"  — return the entire post-washout ⟨σ_z⟩ sequence per window.
                  Shape: (num_windows, window_size).
        "last"  — return the final post-washout value per window.
                  Shape: (num_windows,).
        "mean"  — return the mean post-washout value per window.
                  Shape: (num_windows,).

    Returns
    -------
    np.ndarray of appropriate shape.
    """
    rows = []
    for result in results_list:
        counts = result.get_counts()
        n_total = len(next(iter(counts)).replace(' ', ''))
        sigmaz = _sigmaz_from_counts(counts, n_total)
        post = sigmaz[washout_length:]

        if compress_mode == "full":
            rows.append(post)
        elif compress_mode == "last":
            rows.append(post[-1])
        elif compress_mode == "mean":
            rows.append(post.mean())
        else:
            raise ValueError(f"Unknown compress_mode: {compress_mode!r}")

    return np.array(rows)
