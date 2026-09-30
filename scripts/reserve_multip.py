import multiprocessing
from sklearn.neural_network import MLPRegressor
import matplotlib.pyplot as plt
from data_gen import generate_data
from reserve_mem import reservoir_with_qubit_reuse, extract_sigmaz_reset_with_washout, make_lagged_features
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

import json
import numpy as np

target = 'AAPL'
raw_data = generate_data(target)

def get_deviation_timeseries(df, strike, option_type, expiry):
    ts = df[
        (df['strike'] == strike) &
        (df['type'] == option_type) &
        (df['expiration'] == expiry)
    ].sort_values('date')
    return ts['date'].values, ts['Deviation'].values

dates, deviations = get_deviation_timeseries(raw_data, 500, 'call', '2013-01-19')

#implemented washout also
def process_memory_size(mem_size):
    # Append zeros to input to implement washout via preprocessing
    washout_length = 5
    extended_deviations = np.concatenate([np.zeros(washout_length), deviations])
    # Pass extended input to reservoir
    raw_features = reservoir_with_qubit_reuse(extended_deviations, 
                                            num_memory=mem_size,
                                            shots=1024,
                                            J=1.0,
                                            dt=1,
                                            n_steps=8,
                                            disorder_scale=1)

    # Extract features only for original data length (discard washout output)
    rev_features = extract_sigmaz_reset_with_washout(raw_features, len(deviations), washout_length=washout_length)
    #make reservoir
    #raw_features = reservoir_with_qubit_reuse(deviations, num_memory=mem_size, shots=1024, J=1.0, dt=1, n_steps=8, disorder_scale=10.0)
    #get features
    #rev_features = extract_sigmaz_reset(raw_features, len(deviations))
    #lag and make dataset for train and test
    final_data, final_features = make_lagged_features(rev_features, deviations, window=10)
    # Split into train/test (e.g., last 20% for test)
    split = int(0.8 * len(final_data))
    X_train, X_test = final_data[:split], final_data[split:]
    y_train, y_test = final_features[:split], final_features[split:]
    #last layer is classical inference layer
    mlp = MLPRegressor(hidden_layer_sizes=(2,), max_iter=500, random_state=0)
    mlp.fit(X_train, y_train)
    y_pred = mlp.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return {
        'mem_size': mem_size,
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'R2': r2,
        'loss_curve': mlp.loss_curve_
    }

def save_results(error_metrics, results, n_steps, disorder_scale):
    # Save loss curves as .npy (numpy array)
    loss_curves = {res['mem_size']: res['loss_curve'] for res in results}
    loss_curve_filename = f"loss_curves_nsteps{n_steps}_disorder{disorder_scale}.npy"
    metrics_filename = f"error_metrics_nsteps{n_steps}_disorder{disorder_scale}.json"
    np.save(loss_curve_filename, loss_curves)
    # Save error metrics as JSON
    with open(metrics_filename, "w") as f:
        json.dump(error_metrics, f, indent=2)

if __name__ == '__main__':
    memory_sizes = [1, 2, 3, 4, 5, 6, 7, 8]
    n_steps = 8
    disorder_scale = 1
    with multiprocessing.Pool(processes=len(memory_sizes)) as pool:
        results = pool.map(process_memory_size, memory_sizes)

    error_metrics = {res['mem_size']: {k: res[k] for k in ['MAE', 'RMSE', 'MSE', 'R2']} for res in results}

    for res in results:
        plt.plot(res['loss_curve'], label=f'memory size {res["mem_size"]}')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('MLPRegressor Training Loss')
    plt.legend()
    plt.show()

    # Plot MAE vs memory size
    mem_sizes_sorted = sorted(error_metrics.keys())
    maes = [error_metrics[mem]['MAE'] for mem in mem_sizes_sorted]
    plt.figure()
    plt.plot(mem_sizes_sorted, maes, marker='o')
    plt.xlabel('Memory Size')
    plt.ylabel('MAE')
    plt.title('MAE vs Memory Size')
    plt.grid(True)
    plt.show()

    # Save results to file with metadata in filename
    save_results(error_metrics, results, n_steps, disorder_scale)