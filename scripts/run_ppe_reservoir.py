"""
Main experiment: PPE-readout quantum reservoir vs classical baselines.

Sweeps inter-intervention time, Hamiltonian model and encoding across five
datasets, and reports every reservoir score alongside three baselines fitted on
identical splits.

Run:  python3 scripts/run_ppe_reservoir.py
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from qrc.datasets import DATASETS
from qrc.ppe import reservoir_features, ppe_diagnostics
from qrc.evaluate import (windowize, make_split, baselines, evaluate_features,
                          window_memory_profile)

W = 10
NUM_MEMORY = 4
N_STEPS = 5
WASHOUT = 4
MODELS = ["XXZ", "NNN_CHAOTIC", "NNN_LOCALIZED", "IAA_CHAOTIC", "IAA_LOCALIZED"]
TIMES = [0.1, 0.2, 0.3, 0.5, 1.75, 8.75]       # 1.75 = paper delta-t, 8.75 = old CONFIG
DATA_KW = {
    "narma10": dict(n=3000), "mackey_glass": dict(n=3000), "lorenz": dict(n=3000),
    "stocks": dict(max_series=1200), "options": dict(max_series=600),
}
OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "ppe")


def normalize_for_encoding(X_raw, tr):
    """Map raw window values into [-1,1] using TRAIN statistics only."""
    mu, sd = X_raw[tr].mean(), X_raw[tr].std()
    sd = sd if sd > 1e-12 else 1.0
    return np.clip((X_raw - mu) / (3.0 * sd), -1.0, 1.0)


def load_all():
    out = {}
    for name, kw in DATA_KW.items():
        t0 = time.time()
        series = DATASETS[name](**kw)
        Xs, Xr, y, sid = windowize(series, W)
        tr, te = make_split(sid)
        out[name] = dict(Xs=Xs, Xr=Xr, y=y, sid=sid, tr=tr, te=te)
        print(f"  {name:13s} {len(series):5d} series -> {len(y):6d} windows "
              f"(train {tr.sum()}, test {te.sum()})  [{time.time()-t0:.1f}s]")
    return out


def main():
    os.makedirs(OUT, exist_ok=True)
    print("=" * 78)
    print("PPE RESERVOIR EXPERIMENT")
    print(f"W={W}  L={1+NUM_MEMORY}  n_steps={N_STEPS}  washout={WASHOUT}")
    print("=" * 78)

    print("\n[1] Loading datasets")
    data = load_all()

    print("\n[2] Baselines (identical splits)")
    base = {}
    for name, d in data.items():
        base[name] = baselines(d["Xs"], d["Xr"], d["y"], d["tr"], d["te"])
        row = "  ".join(f"{k}={v['NRMSE']:.3f}" for k, v in base[name].items())
        print(f"  {name:13s} NRMSE  {row}")

    print("\n[3] Time sweep (XXZ, continuous encoding)")
    sweep = {}
    for name, d in data.items():
        u = normalize_for_encoding(d["Xr"], d["tr"])
        line = []
        for T in TIMES:
            F = reservoir_features(u, "XXZ", num_memory=NUM_MEMORY, dt=T / N_STEPS,
                                   n_steps=N_STEPS, washout_length=WASHOUT,
                                   encoding="continuous", initial_state="neel")
            s = evaluate_features(F, d["y"], d["tr"], d["te"])
            sweep[(name, T)] = s
            line.append(f"T={T:<5g} {s['NRMSE']:.3f}")
        print(f"  {name:13s} NRMSE  " + "  ".join(line))

    best_T = {n: min(TIMES, key=lambda T: sweep[(n, T)]["NRMSE"]) for n in data}
    print(f"\n  best T per dataset: {best_T}")

    print("\n[4] Model + encoding comparison at each dataset's best T")
    grid = {}
    for name, d in data.items():
        T = best_T[name]
        u = normalize_for_encoding(d["Xr"], d["tr"])
        print(f"\n  --- {name} (T={T}) ---")
        for model in MODELS:
            kw = {"use_random": True, "seed": 0} if model.startswith("NNN") else {}
            row = {}
            for enc, Xin in (("continuous", u), ("symbolic", d["Xs"])):
                F = reservoir_features(Xin, model, num_memory=NUM_MEMORY, dt=T / N_STEPS,
                                       n_steps=N_STEPS, washout_length=WASHOUT,
                                       encoding=enc, initial_state="neel", model_kwargs=kw)
                row[enc] = evaluate_features(F, d["y"], d["tr"], d["te"])
            grid[(name, model)] = row
            print(f"    {model:15s} NRMSE cont={row['continuous']['NRMSE']:.4f} "
                  f"sym={row['symbolic']['NRMSE']:.4f}   "
                  f"R2 cont={row['continuous']['R2']:+.4f}")

    print("\n[5] Initial-state ablation (Neel vs single excitation), NARMA-10")
    d = data["narma10"]; u = normalize_for_encoding(d["Xr"], d["tr"]); T = best_T["narma10"]
    init_ab = {}
    for init in ("neel", "single"):
        F = reservoir_features(u, "XXZ", num_memory=NUM_MEMORY, dt=T / N_STEPS,
                               n_steps=N_STEPS, washout_length=WASHOUT,
                               encoding="continuous", initial_state=init)
        init_ab[init] = evaluate_features(F, d["y"], d["tr"], d["te"])
        print(f"  {init:8s} NRMSE={init_ab[init]['NRMSE']:.4f}  R2={init_ab[init]['R2']:+.4f}")

    print("\n[6] System-size scaling (NARMA-10, XXZ)")
    size_scan = {}
    for nm in (2, 4, 6, 8):
        F = reservoir_features(u, "XXZ", num_memory=nm, dt=T / N_STEPS, n_steps=N_STEPS,
                               washout_length=WASHOUT, encoding="continuous",
                               initial_state="neel")
        size_scan[nm] = evaluate_features(F, d["y"], d["tr"], d["te"])
        print(f"  L={1+nm:2d}  n_features={size_scan[nm]['n_features']:4d}  "
              f"NRMSE={size_scan[nm]['NRMSE']:.4f}  R2={size_scan[nm]['R2']:+.4f}")

    print("\n[7] PPE diagnostics for this setup (exact, 4^5 = 1024 sequences)")
    diags = []
    for model in MODELS:
        kw = {"use_random": True, "seed": 0} if model.startswith("NNN") else {}
        dg = ppe_diagnostics(model, num_memory=NUM_MEMORY, n_interventions=5,
                             dt=0.2 / N_STEPS, n_steps=N_STEPS, model_kwargs=kw)
        diags.append(dg)
        print(f"  {model:15s} mean_S={dg['mean_S']:.4f}  std_S={dg['std_S']:.4f}  "
              f"CoV={dg['coeff_var']:.4f}  (max S={dg['max_S']:.4f})")

    print("\n[8] Incremental value: does the reservoir add anything over its own raw input?")
    incr = {}
    for name, d in data.items():
        T = best_T[name]
        u = normalize_for_encoding(d["Xr"], d["tr"])
        F = reservoir_features(u, "XXZ", num_memory=NUM_MEMORY, dt=T / N_STEPS,
                               n_steps=N_STEPS, washout_length=WASHOUT,
                               encoding="continuous", initial_state="neel")
        raw = evaluate_features(d["Xr"], d["y"], d["tr"], d["te"])["NRMSE"]
        rsv = evaluate_features(F, d["y"], d["tr"], d["te"])["NRMSE"]
        both = evaluate_features(np.hstack([d["Xr"], F]), d["y"], d["tr"], d["te"])["NRMSE"]
        incr[name] = dict(raw_only=raw, reservoir_only=rsv, raw_plus_reservoir=both)
        verdict = ("helps" if both < raw - 0.005 else "no gain" if both < raw + 0.005 else "HURTS")
        print(f"  {name:13s} raw={raw:.4f}  resv={rsv:.4f}  raw+resv={both:.4f}   {verdict}")

    payload = {
        "config": dict(W=W, L=1 + NUM_MEMORY, n_steps=N_STEPS, washout=WASHOUT,
                       times=TIMES, models=MODELS),
        "baselines": base,
        "sweep": {f"{n}|{T}": v for (n, T), v in sweep.items()},
        "best_T": best_T,
        "grid": {f"{n}|{m}": v for (n, m), v in grid.items()},
        "initial_state_ablation": init_ab,
        "size_scan": {str(k): v for k, v in size_scan.items()},
        "ppe_diagnostics": diags,
        "incremental_value": incr,
    }
    with open(os.path.join(OUT, "results.json"), "w") as f:
        json.dump(payload, f, indent=2, default=float)
    print(f"\nWrote {os.path.join(OUT, 'results.json')}")


if __name__ == "__main__":
    main()
