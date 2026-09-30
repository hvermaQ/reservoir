#import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from reservoir.gen_dat import generate_data
from reserve_mem import reservoir_with_qubit_reuse, extract_sigmaz_reset, make_lagged_features
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

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

#def process_memory_size(mem_size):
#make reservoir
mem_size = 2
raw_features = reservoir_with_qubit_reuse(deviations, num_memory=mem_size, shots=1024, J=1.0, dt=0.1, n_steps=1)
#get features
rev_features = extract_sigmaz_reset(raw_features, len(deviations))
#lag and make dataset for train and test
final_data, final_features = make_lagged_features(rev_features, deviations, lag=10)
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
"""
    return {
        'mem_size': mem_size,
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'R2': r2,
        'loss_curve': mlp.loss_curve_
    }
"""
#memory_sizes = [1, 2]
#results = []
#for mem_size in memory_sizes:
#res = process_memory_size(mem_size)
#    results.append(res)

#error_metrics = {res['mem_size']: {k: res[k] for k in ['MAE', 'RMSE', 'MSE', 'R2']} for res in results}
"""for res in results:
    plt.plot(res['loss_curve'], label=f'memory size {res["mem_size"]}')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLPRegressor Training Loss')
plt.legend()
plt.show()"""