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

def extract_features_weak(results_list, washout_length=5, ancilla_cbit=0):
    """ß
    Convert QPU results (with intermediate weak measurements)
    into per-timestep expectation values <Z_t> for each window.
    Parameters
    ----------
    results_list : list
        List of Result objects from the QPU.
    washout_length : int
        Number of initial timesteps to discard (washout).
    ancilla_cbit : int
        Classical bit index corresponding to the ancilla qubit.

    Returns
    -------
    np.ndarray of shape (num_windows, window_length)
        Each row corresponds to one input window, containing <Z> per timestep.
    """
    all_features = []

    for result in results_list:
        # Dictionary: timestep -> accumulated P(ancilla=1)
        p1_dict = {}
        total_prob_dict = {}

        for sample in result.raw_data:
            prob = sample.probability

            for meas in sample.intermediate_measurements:
                # Only consider the ancilla classical bit
                if ancilla_cbit in meas.cbits:
                    t_idx = meas.gate_pos  # use gate_pos as unique timestep identifier
                    # Accumulate probability of ancilla=1
                    bit = meas.cbits[0]  # assuming single ancilla bit
                    if t_idx not in p1_dict:
                        p1_dict[t_idx] = 0.0
                        total_prob_dict[t_idx] = 0.0
                    if bit == 1:
                        p1_dict[t_idx] += prob
                    total_prob_dict[t_idx] += prob

        # Sort timesteps by gate_pos to get chronological order
        sorted_steps = sorted(p1_dict.keys())
        ez_list = []
        for t in sorted_steps:
            # Avoid division by zero
            if total_prob_dict[t] == 0:
                ez_list.append(1.0)  # default <Z>=1 if no data
            else:
                ez_list.append(1 - 2 * (p1_dict[t] / total_prob_dict[t]))

        ez_array = np.array(ez_list)
        # Remove washout steps
        ez_post = ez_array[washout_length:]
        all_features.append(ez_post)

    return np.vstack(all_features)
