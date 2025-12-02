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
