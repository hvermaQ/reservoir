# diagnose_pipeline.py
"""
Comprehensive diagnosis script for quantum reservoir computing pipeline.
Tests each step individually to isolate failures.
"""

import numpy as np
from gen_dat import generate_data
from reservoir_gen import (
    reservoir_from_binary_sequence,
    DEFAULT_DET_INTERVENTIONS,
)
from feature_engineering import (
    create_lagged_binary_features,
    extract_sigmaz_reset_with_washout,
)

# Load data
target = "AAPL"
raw_data = generate_data(target)

def get_deviation_timeseries(df, strike, option_type, expiry):
    ts = df[
        (df["strike"] == strike)
        & (df["type"] == option_type)
        & (df["expiration"] == expiry)
    ].sort_values("date")
    return ts["date"].values, ts["Deviation"].values

dates, deviations = get_deviation_timeseries(
    raw_data, strike=500, option_type="call", expiry="2013-01-19"
)

print("=== PIPELINE DIAGNOSIS REPORT ===\n")

# ---------------------------------------------------------------------
# STEP 1: DATA LOADING
print("1. DATA LOADING")
print(f"   Dataset length: {len(deviations)}")
print(f"   Data range: {deviations.min():.4f} to {deviations.max():.4f}")
print(f"   Data mean/std: {deviations.mean():.4f} ± {deviations.std():.4f}")
print(f"   First 10 values: {deviations[:10]}")
print("   ✓ PASS" if len(deviations) > 0 else "   ✗ EMPTY DATASET")
print()

# ---------------------------------------------------------------------
# STEP 2: BINARIZATION + LAGGED FEATURES
print("2. BINARIZATION + LAGGED FEATURES")
WINDOW_LAGS = 5
try:
    X, y = create_lagged_binary_features(
        deviations, lag_window=WINDOW_LAGS, sigma_threshold=1.0, two_sigma_threshold=2.0
    )
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Unique labels in X: {np.unique(X)}")
    print(f"   Label distribution: {np.bincount(X.flatten())}")
    print(f"   Sample X[0]: {X[0]}")
    print("   ✓ BINARIZATION OK")
except Exception as e:
    print(f"   ✗ BINARIZATION FAILED: {e}")
    raise
print()

# ---------------------------------------------------------------------
# STEP 4: SINGLE RESERVOIR CALL (ALL MODELS)
print("4. SINGLE RESERVOIR CALL (ALL MODELS)")
MODEL_KEYS = ["XXZ", "NNN_CHAOTIC", "NNN_LOCALIZED", "IAA_CHAOTIC", "IAA_LOCALIZED"]

for model_key in MODEL_KEYS:
    try:
        print(f"   Testing {model_key} model...")
        model_kwargs = {"use_random": True} if model_key.startswith("NNN") else {}
        
        result = reservoir_from_binary_sequence(
            x_seq=X,  # Short sequence for speed
            model_key=model_key,
            num_memory=2,
            shots=64,  # Reduced shots for diagnosis
            dt=1.75,
            n_steps=1,
            det_basis=DEFAULT_DET_INTERVENTIONS,
            model_kwargs=model_kwargs,
        )
        print(f"   ✓ {model_key} RESERVOIR OK: {type(result)}")
        
        # Test feature extraction immediately for this result
        features = extract_sigmaz_reset_with_washout(
            result,
            num_timesteps=len(X),  # Updated parameter name
            washout_length=0,
        )
        print(features)
        print(f"      Features shape: {features.shape}, range: [{features.min():.3f}, {features.max():.3f}]")
        
    except Exception as e:
        print(f"   ✗ {model_key} RESERVOIR FAILED: {e}")
        import traceback
        traceback.print_exc()
        break  # Stop on first failure
print()

# ---------------------------------------------------------------------
# STEP 7: TEST SERIALIZATION (multiprocessing proxy)
print("7. SERIALIZATION TEST")
try:
    import pickle
    test_args = ("XXZ", 2, reservoir_input[:10], y[:10])
    pickled = pickle.dumps(test_args)
    unpickled = pickle.loads(pickled)
    print("   Task args serialize/deserialize: OK")
    print("   ✓ SERIALIZATION OK")
except Exception as e:
    print(f"   ✗ SERIALIZATION FAILED: {e}")
print()

# ---------------------------------------------------------------------
# STEP 8: FULL TASK EXECUTION TEST
print("8. FULL TASK EXECUTION")
def test_worker(args):
    model_key, mem_size, x_seq, targets = args
    model_kwargs = {"use_random": True} if model_key.startswith("NNN") else {}
    
    result = reservoir_from_binary_sequence(
        x_seq=x_seq[:10],  # Short test
        model_key=model_key,
        num_memory=mem_size,
        shots=32,
        dt=1.75,
        n_steps=1,
        det_basis=DEFAULT_DET_INTERVENTIONS,
        model_kwargs=model_kwargs,
    )
    
    features = extract_sigmaz_reset_with_washout(result, n_steps=10, washout_length=0)
    return {"model_key": model_key, "features_shape": features.shape}

try:
    test_task = ("XXZ", 2, reservoir_input[:10], y[:10])
    result = test_worker(test_task)
    print(f"   Full task completed: {result}")
    print("   ✓ FULL PIPELINE OK")
except Exception as e:
    print(f"   ✗ FULL TASK FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DIAGNOSIS COMPLETE ===")
print("If all steps pass, multiprocessing issue is elsewhere.")
print("Focus on the step that fails for root cause.")