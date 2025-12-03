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

def extract_features_from_results(results_list, washout_length=5, discard_washout=True):
    """
    Extract features from list of Results, discarding initial washout timesteps.
    
    Parameters
    ----------
    results_list : list of qat.core.Result
    washout_length : int
        Number of initial timesteps per circuit to discard.
    discard_washout : bool
        If True, return features only after washout period.
        If False, return all timesteps (for debugging).
        
    Returns
    -------
    features : np.ndarray, shape (num_windows,) or (num_windows, window_size-washout)
    """
    features = []
    
    for result in results_list:
        bit1_probs = np.zeros(len(result.raw_data[0].intermediate_measurements))
        total_prob = 0.0

        for sample in result.raw_data:
            prob = sample.probability if hasattr(sample, "probability") else sample["probability"]
            total_prob += prob
            
            for t, meas in enumerate(sample.intermediate_measurements):
                cbit_val = meas.cbits[0] if hasattr(meas, "cbits") else meas["cbits"][0]
                if cbit_val == 1:
                    bit1_probs[t] += prob

        sigmaz = 1 - 2 * (bit1_probs / total_prob) if total_prob > 0 else np.ones_like(bit1_probs)
        
        if discard_washout:
            # Discard first washout_length timesteps, keep window timesteps only
            window_features = sigmaz[washout_length:]
            # Use final feature of window or mean
            feature = window_features[-1]  # Last timestep after washout
            # feature = np.mean(window_features)  # Alternative: average
        else:
            feature = sigmaz[-1]  # Full sequence final timestep
            
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

def extract_features(results_list, washout_length=5):
    """
    Convert QPU results (with intermediate weak measurements)
    into per-timestep expectation values <Z_t> for each window.

    Returns
    -------
    np.ndarray of shape (num_windows, window_length)
    """
    all_features = []

    for result in results_list:
        # Number of timesteps = number of weak measurements
        num_steps = len(result.raw_data[0].intermediate_measurements)

        p1 = np.zeros(num_steps)
        total_prob = 0.0

        # Accumulate probabilities of ancilla=1 at each timestep
        for sample in result.raw_data:
            prob = sample.probability
            total_prob += prob

            for t, meas in enumerate(sample.intermediate_measurements):
                bit = meas.cbits[0]     # ancilla measurement outcome
                if bit == 1:
                    p1[t] += prob

        # Compute <Z> = 1 - 2 P(ancilla=1)
        ez = 1 - 2 * (p1 / total_prob)

        # Remove washout, keep only effective timesteps
        post_ez = ez[washout_length:]

        all_features.append(post_ez)

    return np.vstack(all_features)


# -------------------------------------------------------------
# 2) Compress per-timestep feature vectors into scalars
# -------------------------------------------------------------

def postprocess_features(feature_vectors, mode="full"):
    """
    Convert each per-window feature vector (length = window_size)
    into a final scalar feature.

    Options:
        "last"  -> final timestep after washout (recommended)
        "mean"  -> average over all timesteps
        "sum"   -> sum of <Z_t>
        "pca1"  -> first principal component
    """

    if mode == "last":
        return feature_vectors[:, -1]

    elif mode == "full":
        return feature_vectors

    elif mode == "mean":
        return np.mean(feature_vectors, axis=1)

    elif mode == "sum":
        return np.sum(feature_vectors, axis=1)

    elif mode == "pca1":
        from sklearn.decomposition import PCA
        return PCA(n_components=1).fit_transform(feature_vectors).flatten()

    else:
        raise ValueError(f"Unknown compression mode: {mode}")


# -------------------------------------------------------------
# 3) Simple wrapper: from raw reservoir results → final features
# -------------------------------------------------------------

def features_from_results(results_list, washout_length=5, compress_mode="last"):
    """
    Full post-processing pipeline:
       1. Extract <Z_t> per timestep after washout
       2. Compress to scalar per window
    """
    feature_vectors = extract_features(results_list, washout_length=washout_length)

    final_features = postprocess_features(feature_vectors, mode=compress_mode)

    return final_features