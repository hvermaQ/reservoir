# feature_engineering.py
"""
Preprocessing and feature-extraction utilities for PPE reservoirs.

Responsibilities:
  - Map real-valued deviation time series to discrete intervention labels
    using a *local* (windowed) mean / variance.
  - Extract σ_z-like features from reservoir results (with washout support).
  - Build lagged feature matrices for classical readout models.
"""

import numpy as np


# ---------------------------------------------------------------------
# Deviation → 0/1/2/3 binning using windowed ±kσ bands
# ---------------------------------------------------------------------

def binarize_deviation_4_levels_windowed(values, window, k_sigma=2.0):
    """
    Map deviations to 4 labels {0,1,2,3} using a local (windowed) mean and
    ±k_sigma * σ thresholds.

    For each time t, consider the trailing window [t-window+1, t] (clipped
    at 0), compute local mean μ_t and standard deviation σ_t, and encode
    the current value x_t as:

      label 0: very negative, x_t < μ_t - kσ_t
      label 1: mildly negative, μ_t - kσ_t <= x_t < μ_t
      label 2: mildly positive, μ_t <= x_t < μ_t + kσ_t
      label 3: very positive, x_t >= μ_t + kσ_t

    If σ_t = 0 (flat window), all points in that window are mapped via
    the central thresholds μ_t only.

    Parameters
    ----------
    values : array-like of float
        Deviation time series of length T.
    window : int
        Window length W >= 1. If W >= T, this reduces to a global
        (full-series) encoding.
    k_sigma : float
        Number of standard deviations defining the outer bands.

    Returns
    -------
    x_seq : np.ndarray, shape (T,), dtype=int
        Integer labels in {0,1,2,3}.
    """
    vals = np.asarray(values, dtype=float)
    T = len(vals)
    if window is None or window < 1:
        raise ValueError("window must be a positive integer")

    x_seq = np.zeros(T, dtype=int)

    for t in range(T):
        start = max(0, t - window + 1)
        segment = vals[start : t + 1]
        mu_t = segment.mean()
        sigma_t = segment.std()

        x_t = vals[t]
        if sigma_t == 0.0:
            # Degenerate case: all values in window equal.
            # Use μ_t as central threshold.
            if x_t < mu_t:
                x_seq[t] = 1  # mildly negative
            elif x_t > mu_t:
                x_seq[t] = 2  # mildly positive
            else:
                x_seq[t] = 2  # exactly at mean → treat as mildly positive
            continue

        lower = mu_t - k_sigma * sigma_t
        upper = mu_t + k_sigma * sigma_t

        if x_t < lower:
            x_seq[t] = 0
        elif x_t < mu_t:
            x_seq[t] = 1
        elif x_t < upper:
            x_seq[t] = 2
        else:
            x_seq[t] = 3

    return x_seq


# ---------------------------------------------------------------------
# Label → intervention mapping (optional utility)
# ---------------------------------------------------------------------

def map_labels_to_interventions(labels, mapping):
    """
    Map integer labels to intervention identifiers (e.g. 'I', 'Z', 'X', 'Y').

    Parameters
    ----------
    labels : array-like of int
        Discrete labels (e.g. from binarize_deviation_4_levels_windowed).
    mapping : dict
        Dictionary {int_label: any_identifier}.

    Returns
    -------
    mapped : list
        Sequence of mapped identifiers.
    """
    labels = np.asarray(labels, dtype=int)
    return [mapping[int(l)] for l in labels]


# ---------------------------------------------------------------------
# Reservoir feature extraction (σ_z) with washout
# ---------------------------------------------------------------------

def extract_sigmaz_reset_with_washout(raw_result, data_length, washout_length=0):
    """
    Extract σ_z-like scalar features per intervention, discarding washout.

    This is a backend-dependent placeholder: adapt to your QLM result
    structure. The interface is kept stable so the run script does not
    need to change.

    Parameters
    ----------
    raw_result : qat.core.Result
        Output of reservoir_comb_from_binary_sequence.
    data_length : int
        Total number of intervention steps used in the run (including washout).
    washout_length : int
        Number of initial interventions to discard.

    Returns
    -------
    features : np.ndarray, shape (data_length - washout_length,)
        One scalar feature per *kept* intervention.
    """
    # Example sketch for single-qubit σ_z from bitstrings:
    #   - For each shot: bit 0 corresponds to measuring |0> or |1> on the system.
    #   - σ_z = +1 for |0>, -1 for |1>.
    # Adjust this to your actual QLM Result format.
    shots = raw_result.get_samples()  # or appropriate accessor
    bits = np.array([s.state.bits[0] for s in shots], dtype=int)
    sz = 1.0 - 2.0 * bits.mean()  # <σ_z> over shots

    # Here we assume a single σ_z value per run; in a more detailed
    # implementation, you might reconstruct a time-resolved feature.
    features = np.full(data_length, sz, dtype=float)

    if washout_length > 0:
        return features[washout_length:]
    return features


# ---------------------------------------------------------------------
# Lagged feature construction
# ---------------------------------------------------------------------

def make_lagged_features(reservoir_features, target_series, window=10):
    """
    Construct lagged features from reservoir output for a prediction task.

    For t >= window-1:
      X_t = [f_{t-window+1}, ..., f_t]
      y_t = target_series[t]

    Parameters
    ----------
    reservoir_features : array-like of float
        1D array of reservoir features (length T).
    target_series : array-like of float
        1D array of target values (same length T).
    window : int
        Number of past steps to include per feature vector.

    Returns
    -------
    X : np.ndarray, shape (T-window+1, window)
    y : np.ndarray, shape (T-window+1,)
    """
    f = np.asarray(reservoir_features, dtype=float)
    y = np.asarray(target_series, dtype=float)
    assert len(f) == len(y)
    T = len(f)

    X_list = []
    y_list = []
    for t in range(window - 1, T):
        X_list.append(f[t - window + 1 : t + 1])
        y_list.append(y[t])

    X = np.array(X_list)
    y_out = np.array(y_list)
    return X, y_out


# feature_engineering.py (updated extract function)

def extract_sigmaz_reset_with_washout(result, n_steps, washout_length=10):
    """
    Compute <sigma_z> per timestep from intermediate measurements, skipping 
    first washout_length measurements.

    Assumes the reservoir circuit measures after EVERY intervention+evolution
    step, storing results in intermediate_measurements.

    Parameters
    ----------
    result : qat.core.Result
        Single result from reservoir circuit with intermediate measurements.
    n_steps : int
        Number of timesteps to extract (after washout).
    washout_length : int
        Number of initial measurements to discard.

    Returns
    -------
    sigmaz : np.ndarray, shape (n_steps,)
        <σ_z>(t) for t = washout_length, ..., washout_length + n_steps - 1
    """
    bit1_prob = np.zeros(n_steps)
    total_prob = 0.0
    
    for sample in result.raw_data:
        prob = sample.probability if hasattr(sample, 'probability') else sample['probability']
        total_prob += prob
        
        # Skip first washout_length timesteps, extract next n_steps
        for t in range(n_steps):
            meas_idx = washout_length + t  # Start AFTER washout
            int_meas = sample.intermediate_measurements[meas_idx]
            cbit_val = int_meas.cbits[0] if hasattr(int_meas, 'cbits') else int_meas['cbits'][0]
            if cbit_val == 1:
                bit1_prob[t] += prob
    
    sigmaz = 1 - 2 * (bit1_prob / total_prob) if total_prob else np.ones(n_steps)
    return sigmaz
