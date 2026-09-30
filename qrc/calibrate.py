"""
calibrate.py — choose the interaction time that puts a model at a target entropy.

Motivation. Comparing a chaotic model against a localised one at a COMMON
interaction time confounds two things: the models differ in how much they
entangle, and they differ in dynamics. "Localised does as well as chaotic" is
then ambiguous -- it could mean entanglement is irrelevant, or it could mean the
particular T happened to suit both.

The clean design is an entropy-MATCHED comparison: pick T per model so that the
task-conditional mean Renyi-2 entropy is equal across models, then compare
accuracy. Any residual difference is attributable to the dynamics rather than to
the amount of entanglement, and a null difference becomes a much sharper claim.

Sweeping T within one model gives the complementary dose-response axis: entropy
varies continuously while the Hamiltonian is fixed.

Caveat this measurement exposes. The achievable entropy range is model
dependent. XXZ sits above half the Page value even at the shortest times tested,
so low-entropy targets are simply unreachable for it, and the calibration
reports the achieved value rather than pretending otherwise. Never assume a
requested target was met -- read `achieved_frac`.
"""
from __future__ import annotations

import numpy as np

from qrc import cache
from qrc.metrics import entropy_stats
from qrc.ppe import reservoir_features

# Log-spaced coarse scan, then local refinement. A pure bisection is unsafe here:
# the entropy-vs-T curve is not monotone at short times.
T_GRID = np.geomspace(0.005, 8.0, 24)


def measure_entropy(model_key: str, num_memory: int, total_time: float,
                    n_steps: int = 5, n_probe: int = 256, W: int = 8,
                    encoding: str = "continuous", initial_state: str = "neel",
                    model_kwargs: dict | None = None, seed: int = 0) -> float:
    """Mean Renyi-2 entropy as a fraction of the Page value, under random drive."""
    rng = np.random.default_rng(seed)
    U = (rng.integers(0, 4, size=(n_probe, W)) if encoding == "symbolic"
         else rng.uniform(-1.0, 1.0, size=(n_probe, W)))
    L = 1 + num_memory
    F = reservoir_features(U, model_key, num_memory=num_memory,
                           dt=total_time / n_steps, n_steps=n_steps,
                           washout_length=0, encoding=encoding,
                           initial_state=initial_state, model_kwargs=model_kwargs)
    st = entropy_stats(F, L)
    return float(st.get("S_frac_max", float("nan")))


def calibrate(model_key: str, num_memory: int, target_frac: float,
              n_steps: int = 5, n_probe: int = 256, W: int = 8,
              encoding: str = "continuous", initial_state: str = "neel",
              model_kwargs: dict | None = None, seed: int = 0,
              refine: int = 12) -> dict:
    """
    Find total_time whose entropy fraction is closest to `target_frac`.

    Returns the chosen T, the achieved fraction, and whether the target was
    actually bracketed. Cached on disk: the scan costs ~24 small reservoir runs
    and is reused by every unit that asks for the same target.
    """
    params = {"cal_version": 2,
              "model": model_key, "num_memory": num_memory, "target": round(float(target_frac), 6),
              "n_steps": n_steps, "n_probe": n_probe, "W": W, "encoding": encoding,
              "initial_state": initial_state, "seed": seed,
              "model_kwargs": sorted((model_kwargs or {}).items())}

    def build():
        fracs = np.array([measure_entropy(model_key, num_memory, T, n_steps, n_probe, W,
                                          encoding, initial_state, model_kwargs, seed)
                          for T in T_GRID])
        err = np.abs(fracs - target_frac)
        i = int(np.nanargmin(err))
        best_T, best_f = float(T_GRID[i]), float(fracs[i])

        # local refinement between the neighbours of the coarse optimum
        lo = T_GRID[max(0, i - 1)]
        hi = T_GRID[min(len(T_GRID) - 1, i + 1)]
        for T in np.geomspace(lo, hi, refine):
            f = measure_entropy(model_key, num_memory, float(T), n_steps, n_probe, W,
                                encoding, initial_state, model_kwargs, seed)
            if abs(f - target_frac) < abs(best_f - target_frac):
                best_T, best_f = float(T), float(f)

        # The refinement can land outside the coarse grid's own range, so the
        # reported reachable range must include it -- otherwise the caller is
        # told a value is unreachable that the calibration just reached.
        lo_f = float(min(np.nanmin(fracs), best_f))
        hi_f = float(max(np.nanmax(fracs), best_f))
        bracketed = bool(lo_f <= target_frac <= hi_f)
        return {"total_time": np.array([best_T]),
                "achieved_frac": np.array([best_f]),
                "bracketed": np.array([1.0 if bracketed else 0.0]),
                "reach_lo": np.array([lo_f]), "reach_hi": np.array([hi_f]),
                "grid_T": T_GRID, "grid_frac": fracs}

    d = cache.memo_arrays("entropy_cal", params, build)
    return {"total_time": float(d["total_time"][0]),
            "achieved_frac": float(d["achieved_frac"][0]),
            "bracketed": bool(d["bracketed"][0] > 0.5),
            "target_frac": float(target_frac),
            "reachable_range": [float(d["reach_lo"][0]), float(d["reach_hi"][0])]}
