#end-to-end
#workflow:
#1. calculating baseline through black-scholes formula
#2. Calculating deviations in the prices from this baseline
#3. reservoir encoding with deviations
#4. using processed time series features (deviation based) in NN
#5. evaluating performance as a function of memory size in reservoir

#imports
from sklearn.neural_network import MLPRegressor
#from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from reservoir.gen_dat import generate_data
from reserve_mem import reservoir_with_data_interactions, extract_sigmaz_reset, make_lagged_features
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, root_mean_squared_error

#data prep, e.g. selection of option chain and calculation of baseline deviation
target = 'AAPL'
raw_data = generate_data(target)

#deviation and preprocessing

def get_deviation_timeseries(df, strike, option_type, expiry):
    ts = df[
        (df['strike'] == strike) &
        (df['type'] == option_type) &
        (df['expiration'] == expiry)
    ].sort_values('date')
    return ts['date'].values, ts['Deviation'].values

#filter a type of call based on strike price and expiry
# Usage:
dates, deviations = get_deviation_timeseries(raw_data, 500, 'call', '2013-01-19')

#print(dates)
#print(deviations)

#memory loop with qubit reuse

memory_sizes = [2, 3, 4, 5]
error_metrics = {}
for mem_size in memory_sizes:
    print(mem_size)
    #make reservoir
    raw_features = reservoir_with_data_interactions(deviations, num_memory=mem_size, shots=1024, J=1.0, dt=0.1, n_steps=1)
    #get features
    rev_features = extract_sigmaz_reset(raw_features, len(deviations), mem_size)
    #lag and make dataset for train and test
    final_data, final_features = make_lagged_features(rev_features, deviations, lag=10)
    #print
    print(final_data)
    print(final_features)
    # Split into train/test (e.g., last 20% for test)
    split = int(0.8 * len(deviations))
    X_train, X_test = final_data[:split], final_data[split:]
    y_train, y_test = final_features[:split], final_features[split:]
    #last layer is classical inference layer
    # MLP regressor (simple NN)
    mlp = MLPRegressor(hidden_layer_sizes=(1,), max_iter=500, random_state=0)
    mlp.fit(X_train, y_train)
    # Predictions and error logging
    y_pred = mlp.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = root_mean_squared_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    error_metrics[mem_size] = {
        'MAE': mae,
        'RMSE': rmse,
        'MSE': mse,
        'R2': r2
    }
    #loss curves for each memory size
    plt.plot(mlp.loss_curve_, label='memory size {}'.format(mem_size))

plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MLPRegressor Training Loss')
plt.legend()
plt.show()
#accuracy as function of memory size