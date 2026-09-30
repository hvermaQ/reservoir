"""
Experiment configuration.

Two corrections relative to the previous version:

  * strike/expiry now resolve to a contract that exists in data/2013-06/. The
    previous values (strike 500, expiry 2013-01-19) matched zero rows, so the
    deviation series was empty and the pipeline died unpacking an empty array.

  * `dt` is the TROTTER STEP, not the time between interventions: the physical
    interval is n_steps * dt. The old dt=1.75 with n_steps=5 gave a total time of
    8.75, five times deeper than the delta-t = 1.75 of the source paper and far
    past the point where the reservoir retains any input information. Measured
    retention peaks near total_time = 0.2, hence dt = 0.04 with n_steps = 5.
"""

N_STEPS = 5
TOTAL_TIME = 0.2                 # physical time between interventions

CONFIG = {
    'target': 'AAPL',
    'strike': 455,               # 20 quotes, and survives generate_data's +/-20%-of-spot filter
    'option_type': 'call',
    'expiry': '2013-07-20',
    'dt': TOTAL_TIME / N_STEPS,  # Trotter step; total interval = n_steps * dt
    'n_steps': N_STEPS,
    'total_time': TOTAL_TIME,
    'washout_length': 4,
    'window_lags': 10,           # was 5; series here are 20 points, so 10 is the practical max
    'memory_sizes': [2, 4, 6, 8],
    'model_keys': ['XXZ', 'NNN_CHAOTIC', 'NNN_LOCALIZED', 'IAA_CHAOTIC', 'IAA_LOCALIZED'],
    'shots': 1024,               # only used by the shot-based circuits in reservoir_gen
    'train_split': 0.8,
    'mlp_layers': (2,),
    'mlp_max_iter': 500,
    'epsilon': 0.25,             # weak measurement angle
    'initial_state': 'neel',
    'encoding': 'continuous',
}
