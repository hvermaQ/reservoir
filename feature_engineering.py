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
# ---------------------------------------------------------------------

def extract_sigmaz_reset_with_washout(result, num_timesteps, washout_length=10):
    """
    Extract <σ_z> per timestep from a Result with intermediate measurements,
    discarding the initial washout timesteps.

    Parameters
    ----------
    result : qat.core.Result
        Result object from reservoir circuit with intermediate measurements.
    num_timesteps : int
        Number of intermediate measurements to extract (including washout).
    washout_length : int
        Number of initial timesteps to discard.

    Returns
    -------
    sigmaz : np.ndarray, shape (num_timesteps - washout_length,)
        Expectation values <σ_z> of system qubit for each timestep.
    """
    bit1_prob = np.zeros(num_timesteps)
    total_prob = 0.0

    for sample in result.raw_data:
        prob = sample.probability if hasattr(sample, "probability") else sample["probability"]
        total_prob += prob

        for t in range(num_timesteps):
            meas_idx = t  # intermediate measurement index
            int_meas = sample.intermediate_measurements[meas_idx]
            cbit_val = int_meas.cbits[0] if hasattr(int_meas, "cbits") else int_meas["cbits"][0]
            if cbit_val == 1:
                bit1_prob[t] += prob

    if total_prob == 0:
        return np.ones(num_timesteps - washout_length)

    sigmaz = 1 - 2 * (bit1_prob / total_prob)
    return sigmaz[washout_length:]