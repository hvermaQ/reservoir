import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# End-to-end pipeline for quantum reservoir computing + option-deviation prediction.
# This script is intentionally modular so future selection, model, and reservoir
# choices can be changed without rewriting the whole flow.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error
from qrc.data_gen import generate_data
from qrc.reserve_mem import reservoir_with_qubit_reuse, extract_sigmaz_reset_with_washout

# --------------------------------------------------------------------------------
# Configuration placeholders for future changes and experiment selection
# --------------------------------------------------------------------------------
TARGET = 'AAPL'  # current ticker — change this for a different dataset
SELECTED_STRIKE = 500
SELECTED_EXPIRY = None  # None = automatically select the longest valid series
OPTION_TYPE = 'call'
FORECAST_HORIZON = 1  # 1 = next-step prediction, >1 = delayed target
WINDOW = 10  # number of past reservoir outputs used for prediction
WASHOUT = 10  # initial transient steps discarded from the reservoir output
# If the dataset is shorter than WINDOW + FORECAST_HORIZON, the code will
# choose the longest available series and reduce the effective window.

MEMORY_SIZES = [2, 5, 8, 10, 12, 15]

# Reservoir hyperparameters — these are intentionally exposed for sensitivity testing.
RESERVOIR_CONFIG = {
    'shots': 10000,
    'J': -1.0,
    'dt': 0.1,
    'n_steps': 8,
    # TODO: add `disorder_scale` and `g_scale` as separate config options.
}

# Readout architecture, currently simple but can be extended to deeper layers.
READOUT_CONFIG = {
    'hidden_layer_sizes': (32, 16),
    'max_iter': 15000,
    'random_state': 0,
    # TODO: add regularization / validation split to prevent overfitting.
}

# --------------------------------------------------------------------------------
# Data selection helpers
# --------------------------------------------------------------------------------
raw_data = generate_data(TARGET)


def get_deviation_timeseries(df, strike, option_type, expiry):
    """Extract one option series by strike, type, and expiry."""
    ts = df[
        (df['strike'] == strike) &
        (df['type'] == option_type) &
        (df['expiration'] == expiry)
    ].sort_values('date')
    return ts['date'].values, ts['Deviation'].values


def find_longest_series(df, option_type='call', min_length=None):
    """Find the longest available option series in the dataset."""
    calls = df[df['type'] == option_type]
    if calls.empty:
        raise ValueError(f"No {option_type} option data available to select from.")

    counts = (
        calls.groupby(['expiration', 'strike'])
        .size()
        .reset_index(name='count')
        .sort_values('count', ascending=False)
    )

    best = counts.iloc[0]
    if min_length is None or best['count'] >= min_length:
        return best['strike'], best['expiration'], int(best['count'])

    # No series meets the requested minimum length, so choose the longest available
    print(
        f"Warning: desired minimum series length {min_length} is unavailable. "
        f"Using longest available series with {best['count']} points instead."
    )
    return best['strike'], best['expiration'], int(best['count'])


def select_valid_timeseries(df, strike, option_type, expiry, window):
    """Select a valid series and avoid analyzing empty or too-short time series."""
    if strike is None or expiry is None:
        strike, expiry, count = find_longest_series(
            df, option_type=option_type, min_length=window + FORECAST_HORIZON
        )
        print(
            f"No explicit strike/expiry selected. "
            f"Using longest available {option_type} series: strike={strike}, "
            f"expiry={expiry}, count={count}."
        )

    dates, deviations = get_deviation_timeseries(df, strike, option_type, expiry)
    if len(deviations) == 0:
        raise ValueError(
            f"Selected series is empty: strike={strike}, expiry={expiry}. "
            "Check the available data in the dataset."
        )

    effective_window = min(window, len(deviations) - FORECAST_HORIZON)
    if effective_window < 1:
        raise ValueError(
            f"Selected series length {len(deviations)} is too short for forecast horizon "
            f"{FORECAST_HORIZON}. Reduce FORECAST_HORIZON or use a longer dataset."
        )

    if effective_window < window:
        print(
            f"Warning: requested WINDOW={window} is larger than available data. "
            f"Using effective WINDOW={effective_window} instead."
        )

    if len(deviations) < window + FORECAST_HORIZON:
        print(
            f"Warning: selected series length {len(deviations)} is shorter than "
            f"WINDOW+FORECAST_HORIZON ({window + FORECAST_HORIZON})."
        )

    return dates, deviations, effective_window

# --------------------------------------------------------------------------------
# Reservoir / feature helpers
# --------------------------------------------------------------------------------

def extract_reservoir_features(result, n_steps, washout_length):
    """Extract reservoir observables from the circuit result.

    Currently only data-qubit σ_z outputs are returned, but this is the
    central extension point for richer memory-qubit readouts.
    """
    return extract_sigmaz_reset_with_washout(result, n_steps, washout_length)


def make_lagged_features(features, targets, window, horizon=1):
    """Create lagged input/output pairs with optional forecast horizon."""
    X, y = [], []
    for i in range(window, len(features) - horizon + 1):
        X.append(features[i - window:i])
        y.append(targets[i + horizon - 1])
    return np.array(X), np.array(y)

# Placeholder for future memory-capacity benchmark
def synthetic_memory_capacity_task():
    """TODO: Add a synthetic k-step delay task to isolate reservoir memory."""
    pass

# --------------------------------------------------------------------------------
# Main experiment flow
# --------------------------------------------------------------------------------

def main():
    dates, deviations, effective_window = select_valid_timeseries(
        raw_data, SELECTED_STRIKE, OPTION_TYPE, SELECTED_EXPIRY, WINDOW
    )

    print(f"Using target={TARGET}, strike={SELECTED_STRIKE}, expiry={SELECTED_EXPIRY}")
    print(f"Series length: {len(deviations)}")
    if effective_window != WINDOW:
        print(f"Effective WINDOW reduced from {WINDOW} to {effective_window}.")

    error_metrics = {}
    for mem_size in MEMORY_SIZES:
        print('=== memory size', mem_size)
        extended = np.concatenate([np.zeros(WASHOUT), deviations])

        raw_result = reservoir_with_qubit_reuse(
            extended,
            num_memory=mem_size,
            shots=RESERVOIR_CONFIG['shots'],
            J=RESERVOIR_CONFIG['J'],
            dt=RESERVOIR_CONFIG['dt'],
            n_steps=RESERVOIR_CONFIG['n_steps'],
        )

        rev_features = extract_reservoir_features(
            raw_result, len(deviations), washout_length=WASHOUT
        )

        # TODO: include memory-qubit features here once the reservoir output is extended.
        readout_features = rev_features

        final_data, final_features = make_lagged_features(
            readout_features, deviations, effective_window, horizon=FORECAST_HORIZON
        )
        if final_data.ndim == 1 or len(final_data) == 0:
            raise ValueError(
                f"Not enough data to create lagged features. "
                f"Received {len(final_data)} samples from features length {len(rev_features)}."
            )

        n_samples = len(final_data)
        if n_samples < 2:
            print(
                f"Skipping memory size {mem_size}: only {n_samples} lagged sample(s) available. "
                "Need at least 2 samples for train/test split."
            )
            continue

        split = min(max(1, int(0.8 * n_samples)), n_samples - 1)
        X_train, X_test = final_data[:split], final_data[split:]
        y_train, y_test = final_features[:split], final_features[split:]

        if len(X_train) == 0 or len(X_test) == 0:
            raise ValueError(
                f"Train/test split failed: split={split}, n_samples={n_samples}."
            )

        mlp = MLPRegressor(
            hidden_layer_sizes=READOUT_CONFIG['hidden_layer_sizes'],
            max_iter=READOUT_CONFIG['max_iter'],
            random_state=READOUT_CONFIG['random_state'],
        )
        mlp.fit(X_train, y_train)

        y_pred = mlp.predict(X_test)
        error_metrics[mem_size] = {
            'MAE': mean_absolute_error(y_test, y_pred),
            'RMSE': root_mean_squared_error(y_test, y_pred),
            'MSE': mean_squared_error(y_test, y_pred),
            'R2': r2_score(y_test, y_pred),
        }

        plt.plot(mlp.loss_curve_, label=f'memory size {mem_size}')

    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MLPRegressor Training Loss')
    plt.legend()
    plt.show()

    memory_keys = sorted(error_metrics.keys())
    rmse_vals = [error_metrics[k]['RMSE'] for k in memory_keys]
    r2_vals = [error_metrics[k]['R2'] for k in memory_keys]

    plt.figure()
    plt.plot(memory_keys, rmse_vals, marker='o', label='RMSE')
    plt.xlabel('Memory size')
    plt.ylabel('Error')
    plt.title('Error metrics vs Memory size')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(memory_keys, r2_vals, marker='o', label='R2')
    plt.xlabel('Memory size')
    plt.ylabel('R2 score')
    plt.title('R2 vs Memory size')
    plt.legend()
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    main()

