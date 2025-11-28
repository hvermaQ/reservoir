#import multiprocessing
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
#from gen_dat import generate_data
from reserve_mem import reservoir_with_qubit_reuse, extract_sigmaz_reset_with_washout, make_lagged_features
#from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

#import json
import numpy as np
import pandas as pd

# Load from your source file
df = pd.read_excel("opt_data/sample_Simulated_Swaption_Price.xlsx", sheet_name=0)

#print(df.columns.values)

def select_timeseries(df: pd.DataFrame, col_index: int) -> pd.Series:
    col_name = df.columns[col_index]
    if col_name not in df.columns:
        raise ValueError(f"{col_name} not in DataFrame columns")
    return df[col_name].astype(float)  # Excel NAs → NaN

def first_nan_index(series: pd.Series):
    nan_mask = series.isna().values
    if not nan_mask.any():
        return None
    return np.argmax(nan_mask)  # index of first True[web:109][web:110]

def build_last5_window(series: pd.Series, idx_nan: int, window=5):
    start = idx_nan - window
    if start < 0:
        return None  # not enough history
    window_vals = series.iloc[start:idx_nan].values
    if np.any(np.isnan(window_vals)):
        return None  # window must be fully observed
    return window_vals

def impute_nans_reservoir_stepwise(series, w=5, mem_size=2, washout_length=10, max_train_windows=10):
    vals = np.asarray(series, dtype=float).copy()
    T = len(vals)

    #Collect at most `max_train_windows` clean windows and their targets ---
    X_train_feat, y_train = [], []
    for t in range(w, T):
        if len(X_train_feat) >= max_train_windows:
            break
        window = vals[t-w:t]
        y = vals[t]
        if np.isnan(y) or np.any(np.isnan(window)):
            continue

        extended = np.concatenate([np.zeros(washout_length), window])
        #learn from these features
        raw = reservoir_with_qubit_reuse(
            extended,
            num_memory=mem_size,
            shots=1024,
            J=1.0,
            dt=1,
            n_steps=8,
            disorder_scale=1.0,
        )
        feats = extract_sigmaz_reset_with_washout(
            raw, n_steps=len(window), washout_length=washout_length
        )
        X_train_feat.append(feats[-1])
        y_train.append(y)

    if not X_train_feat:
        return vals  # nothing to learn from

    X_train_feat = np.array(X_train_feat).reshape(-1, 1)
    y_train = np.array(y_train)

    mlp = MLPRegressor(hidden_layer_sizes=(2,), max_iter=500, random_state=0)
    mlp.fit(X_train_feat, y_train)

    # --- 2) Stepwise imputation for all NaNs using last w points only ---
    imputed = vals.copy()
    while True:
        nan_idx = np.where(np.isnan(imputed))[0]
        if len(nan_idx) == 0:
            break
        i = nan_idx[0]
        if i < w:
            break

        window = imputed[i-w:i]
        if np.any(np.isnan(window)):
            break

        extended = np.concatenate([np.zeros(washout_length), window])
        raw = reservoir_with_qubit_reuse(
            extended,
            num_memory=mem_size,
            shots=1024,
            J=1.0,
            dt=1,
            n_steps=8,
            disorder_scale=1.0,
        )
        feats = extract_sigmaz_reset_with_washout(
            raw, n_steps=len(window), washout_length=washout_length
        )
        x_feat = np.array([[feats[-1]]])
        imputed[i] = mlp.predict(x_feat)[0]

    return imputed