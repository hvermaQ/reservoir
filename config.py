# Configuration
CONFIG = {
    'target': 'AAPL',
    'strike': 500,
    'option_type': 'call',
    'expiry': '2013-01-19',
    'dt': 1.75,
    'washout_length': 4,  # Set >0 to test initialization effects
    'window_lags': 5,
    'memory_sizes': [2, 3, 4, 5],
    'model_keys': ['XXZ', 'NNN_CHAOTIC', 'NNN_LOCALIZED', 'IAA_CHAOTIC', 'IAA_LOCALIZED'],
    #'model_keys': ['NNN_CHAOTIC', 'NNN_LOCALIZED'],
    'shots': 1024,
    'n_steps': 1,
    'train_split': 0.8,
    'mlp_layers': (2,),
    'mlp_max_iter': 500
}