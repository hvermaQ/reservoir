# diagnose_pipeline.py
"""
Comprehensive diagnosis script using reservoir_results_per_window.
Tests each step individually to isolate failures.
"""

import numpy as np
from data_gen import generate_data
from reservoir_gen import (
    reservoir_results_per_window,  # Updated to per-window version
    DEFAULT_DET_INTERVENTIONS,
)
from feature_engineering import (
    create_lagged_binary_features,
    extract_features_from_results,  # Updated extraction for list of results
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
WASHOUT_LENGTH = 0  # For diagnosis
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
# STEP 3: SMALL SUBSET FOR QUICK TESTS
print("3. SMALL SUBSET FOR TESTS")
#test_X = X[:3]  # First 3 windows only
test_X = X
print(f"   Test X shape: {test_X.shape}")
print("   ✓ SUBSET OK")
print()


# ---------------------------------------------------------------------
# STEP 4: SINGLE RESERVOIR CALL (ALL MODELS) - PER WINDOW
print("4. SINGLE RESERVOIR CALL (ALL MODELS) - PER WINDOW")
MODEL_KEYS = ["XXZ", "NNN_CHAOTIC", "NNN_LOCALIZED", "IAA_CHAOTIC", "IAA_LOCALIZED"]

for model_key in MODEL_KEYS:
    try:
        print(f"   Testing {model_key} model on {test_X.shape[0]} windows...")
        model_kwargs = {"use_random": True} if model_key.startswith("NNN") else {}
        
        # Run reservoir per window
        results_list = reservoir_results_per_window(
            X_windows=test_X,  # 3 windows
            model_key=model_key,
            num_memory=2,
            shots=32,  # Minimal shots for diagnosis
            dt=1.75,
            n_steps=1,
            det_basis=DEFAULT_DET_INTERVENTIONS,
            model_kwargs=model_kwargs,
            washout_length=WASHOUT_LENGTH,
        )
        print(f"   ✓ {model_key}: {len(results_list)} Results obtained")
        
        # Extract features
        features = extract_features_from_results(results_list, washout_length=WASHOUT_LENGTH)
        print(f"      Features shape: {features.shape}, range: [{features.min():.3f}, {features.max():.3f}]")
        print(features)
    except Exception as e:
        print(f"   ✗ {model_key} RESERVOIR FAILED: {e}")
        import traceback
        traceback.print_exc()
        break
print()


# ---------------------------------------------------------------------
# STEP 7: TEST SERIALIZATION (multiprocessing proxy)
print("7. SERIALIZATION TEST")
try:
    import pickle
    test_args = ("XXZ", 2, test_X[:2], y[:2])  # Small X_windows subset
    pickled = pickle.dumps(test_args)
    unpickled = pickle.loads(pickled)
    print("   Task args (X_windows) serialize/deserialize: OK")
    print(f"   Unpickled X shape: {unpickled[2].shape}")
    print("   ✓ SERIALIZATION OK")
except Exception as e:
    print(f"   ✗ SERIALIZATION FAILED: {e}")
print()


# ---------------------------------------------------------------------
# STEP 8: FULL TASK EXECUTION TEST
print("8. FULL TASK EXECUTION")
def test_worker(args):
    model_key, mem_size, X_windows, targets = args
    model_kwargs = {"use_random": True} if model_key.startswith("NNN") else {}
    
    # Run reservoir on small subset
    results_list = reservoir_results_per_window(
        X_windows=X_windows[:2],  # 2 windows only
        model_key=model_key,
        num_memory=mem_size,
        shots=16,  # Minimal for test
        dt=1.75,
        n_steps=1,
        det_basis=DEFAULT_DET_INTERVENTIONS,
        model_kwargs=model_kwargs,
        washout_length=0,
    )
    
    features = extract_features_from_results(results_list, washout_length=0)
    return {"model_key": model_key, "features_shape": features.shape, "num_results": len(results_list)}

try:
    test_task = ("XXZ", 2, test_X[:2], y[:2])
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
