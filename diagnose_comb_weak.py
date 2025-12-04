"""
DIAGNOSIS SCRIPT — QUANTILE ENCODING + ANCILLA WEAK MEASUREMENT
Checks each pipeline stage individually.
"""

import numpy as np

from data_gen import generate_data

from reservoir_gen import (
    reservoir_results_per_window_ancilla,
    DEFAULT_DET_INTERVENTIONS,
)

from feature_engineering import (
    create_lagged_quantile_features,
    extract_features_weak,
)


# ============================================================
# STEP 0 — Load data
# ============================================================
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

print("\n=== PIPELINE DIAGNOSIS (QUANTILE + WEAK MEAS.) ===\n")

# ============================================================
# STEP 1 — Data loading
# ============================================================
print("1. DATA LOADING")
print(f"   N = {len(deviations)}")
print(f"   Range: {deviations.min():.4f} → {deviations.max():.4f}")
print(f"   Mean/std: {deviations.mean():.4f} ± {deviations.std():.4f}")
print("   ✓ PASS\n")


# ============================================================
# STEP 2 — Quantile encoding + lag windows
# ============================================================
print("2. QUANTILE ENCODING + LAG WINDOWS")

WINDOW_LAGS = 8
WASHOUT = 7

try:
    X, y = create_lagged_quantile_features(
        deviations,
        lag_window=WINDOW_LAGS,
    )
    print(f"   X shape: {X.shape}")
    print(f"   y shape: {y.shape}")
    print(f"   Unique quantile labels in X: {np.unique(X)}")
    print(f"   Sample X[0]: {X[0]}")
    print("   ✓ QUANTILE ENCODING OK")
except Exception as e:
    print(f"   ✗ FAILED: {e}")
    raise

# ============================================================
# STEP 3 — Subset for fast tests
# ============================================================
print("3. SMALL SUBSET FOR QUICK TESTS")
test_X = X       # five windows
test_y = y
print(f"   test_X shape: {test_X.shape}")
print("   ✓ SUBSET READY\n")

# ============================================================
# STEP 4 — Weak-measurement reservoir test
# ============================================================
print("4. RESERVOIR TEST — WEAK MEASUREMENT")

MODEL_KEYS = [
    "XXZ",
    "NNN_CHAOTIC",
    "NNN_LOCALIZED",
    "IAA_CHAOTIC",
    "IAA_LOCALIZED",
]

for model_key in MODEL_KEYS:
    print(f"   Testing {model_key}...")

    model_kwargs = {"use_random": True} if model_key.startswith("NNN") else {}

    try:
        results = reservoir_results_per_window_ancilla(
            X_windows=test_X,
            model_key=model_key,
            num_memory=2,
            shots=1024,
            dt=1.75,
            n_steps=2,
            det_basis=DEFAULT_DET_INTERVENTIONS,
            model_kwargs=model_kwargs,
            washout_length=WASHOUT,
            epsilon=0.12,
            final_strong_measure=False,
        )

        print(f"      ✓ reservoir output (list length) = {len(results)}")

        features = extract_features_weak(
            results,
            washout_length=WASHOUT,
            ancilla_cbit=0,
        )
        print(features[WASHOUT:])
        print(f"      Feature tensor shape: {features.shape}")
        print(f"      Sample feature vector norm: {np.linalg.norm(features[0]):.3f}")

    except Exception as e:
        print(f"      ✗ FAILURE: {e}")
        import traceback
        traceback.print_exc()
        break

# ============================================================
# STEP 5 — Serialization test (MP compatibility)
# ============================================================
print("5. SERIALIZATION TEST")
try:
    import pickle
    test_args = ("XXZ", 2, test_X[:2], test_y[:2])
    enc = pickle.dumps(test_args)
    dec = pickle.loads(enc)
    print(f"   ✓ Pickle OK — recovered shape {dec[2].shape}")
except Exception as e:
    print(f"   ✗ SERIALIZATION FAILED: {e}")

print()


# ============================================================
# STEP 6 — Mini full pipeline test
# ============================================================
print("6. FULL TASK EXECUTION TEST")

def test_worker(args):
    model_key, mem_size, X_w, targets = args
    model_kwargs = {"use_random": True} if model_key.startswith("NNN") else {}

    results = reservoir_results_per_window_ancilla(
        X_windows=X_w,
        model_key=model_key,
        num_memory=mem_size,
        shots=32,
        dt=1.75,
        n_steps=1,
        det_basis=DEFAULT_DET_INTERVENTIONS,
        model_kwargs=model_kwargs,
        washout_length=0,
        epsilon=0.12,
    )
    feats = extract_features_from_results_ancilla(results, washout_length=0)
    return {"model": model_key, "feat_shape": feats.shape, "n_res": len(results)}


try:
    result = test_worker(("XXZ", 2, test_X[:2], test_y[:2]))
    print(f"   ✓ FULL TASK OK → {result}")
except Exception as e:
    print(f"   ✗ FULL TASK FAILED: {e}")
    import traceback
    traceback.print_exc()

print("\n=== DIAGNOSIS COMPLETE ===")
