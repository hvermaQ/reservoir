# preprocess_and_features.py
"""
Preprocessing and feature-extraction utilities for PPE reservoirs.

Responsibilities:
  - Map real-valued deviation time series to discrete intervention labels.
  - Extract σ_z-like features from reservoir results (with washout support).
  - Build lagged feature matrices for classical readout models.
"""

import numpy as np


# ---------------------------------------------------------------------
# Deviation → 0/1/2/3 binning using ±2σ bands
# ---------------------------------------------------------------------

def binarize_deviation_4_levels(values):
    """
    Map deviations to 4 labels {0,1,2,3} using ±2σ and 0 as thresholds.

    Example scheme:
      label 0: very negative, x < -2σ
      label 1: mildly negative, -2σ <= x < 0
      label 2: mildly positive, 0 <= x < +2σ
      label 3: very positive, x >= +2σ

    Parameters
    ----------
    values : array-like of float
        Deviation time series.

    Returns
    -------
    x_seq : np.ndarray, dtype=int
        Integer labels in {0,1,2,3}.
    """
    vals = np.asarray(values, dtype=float)
    sigma = np.std(vals)
    x_seq = np.zeros_like(vals, dtype=int)

    x_seq[vals < -2 * sigma] = 0
    x_seq[(vals >= -2 * sigma) & (vals < 0.0)] = 1
    x_seq[(vals >= 0.0) & (vals < 2 * sigma)] = 2
    x_seq[vals >= 2 * sigma] = 3
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
        Discrete labels (e.g. from binarize_deviation_4_levels).
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
    #
    #   - For each shot: bit 0 corresponds to measuring |0> or |1> on the system.
    #   - σ_z = +1 for |0>, -1 for |1>.
    #
    # Adjust this to your actual QLM Result format.
    shots = raw_result.get_samples()  # or appropriate accessor
    # Expect shots to be an array of bitstrings; adapt as needed
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
