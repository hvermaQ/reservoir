"""
runner.py — execute one unit and return one record.

Pure with respect to the filesystem: it takes a unit dict and returns a record
dict. The caller decides where that lands. That separation is what lets the same
function run under a local process pool, a SLURM array task, or a debugger.

Invariants this module exists to protect:

  1. Every arm of a unit sees the SAME split. Quantum, Haar-random control and
     the classical baselines are computed here, together, from one prepared
     task. Nothing downstream can pair numbers that were not produced together.

  2. The Haar control is passed in explicitly (`block_unitary=`), never by
     monkeypatching `ppe.build_block_unitary`. The old approach mutated a module
     global, which is invisible to the reader and actively unsafe as soon as two
     units share an interpreter -- exactly what a parallel sweep arranges.

  3. Every record can name the code that produced it. This repo has already had
     one generation of results invalidated by three silent Hamiltonian bugs; a
     result file that cannot identify its commit is a liability, not a record.

  4. Representation capacity is measured, not assumed. The quantum feature
     vector's nominal width overstates its usable dimension, so effective rank
     is recorded for every arm and the entanglement actually produced by the
     task inputs is recorded alongside the accuracy it supposedly explains.
"""
from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
import time
import traceback
from functools import lru_cache

import numpy as np
from scipy.stats import unitary_group

from qrc import streaming as stream
from qrc.calibrate import calibrate
from qrc.classify import (baselines_clf, evaluate_clf, quantum_features,
                          scale_for_encoding, separability)
from qrc.evaluate import baselines as ts_baselines
from qrc.evaluate import evaluate_features
from qrc.metrics import effective_rank, entropy_stats
from qrc.spec import canon
from qrc.tasks import prepare_classification, prepare_timeseries, load_series

SCHEMA_VERSION = 3


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def provenance() -> dict:
    """Git commit and library versions, resolved once per process."""
    out = {"python": sys.version.split()[0], "host": platform.node()}
    try:
        import importlib
        for mod in ("numpy", "scipy", "sklearn", "pandas", "qiskit"):
            try:
                out[mod] = importlib.import_module(mod).__version__
            except Exception:
                out[mod] = "unavailable"
    except Exception:
        pass
    try:
        root = __file__.rsplit("/", 2)[0]
        sha = subprocess.run(["git", "-C", root, "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5)
        dirty = subprocess.run(["git", "-C", root, "status", "--porcelain"],
                               capture_output=True, text=True, timeout=5)
        if sha.returncode == 0:
            out["git_sha"] = sha.stdout.strip()
            out["git_dirty"] = bool(dirty.stdout.strip())
    except Exception:
        out["git_sha"] = "unavailable"
    return out


# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

def derive_seed(master: int, *tags) -> int:
    """
    Independent sub-seed from the master seed and a role tag.

    Reusing one integer for the data split, the Haar draw and the disorder
    realisation correlates things that are supposed to be independent; deriving
    each from a hash keeps them independent while staying fully reproducible.
    """
    h = hashlib.sha1(f"{master}|{'|'.join(map(str, tags))}".encode()).hexdigest()
    return int(h[:8], 16)


def model_kwargs_for(model: str, seed: int) -> dict:
    """NNN_* models carry a quenched disorder realisation; tie it to the seed."""
    if model.startswith("NNN"):
        return {"use_random": True, "seed": derive_seed(seed, "disorder", model)}
    return {}


@lru_cache(maxsize=32)
def _haar(dim: int, seed: int) -> np.ndarray:
    return unitary_group.rvs(dim, random_state=seed)


# ---------------------------------------------------------------------------
# Interaction time: fixed, or resolved to hit an entropy target
# ---------------------------------------------------------------------------

def resolve_time(params: dict, model: str, num_memory: int, encode_len: int,
                 seed: int, encoding: str = "continuous") -> tuple[float, dict]:
    """
    Return (total_time, calibration_info).

    With `entropy_target` set, the time is chosen per model so the
    task-conditional entropy matches across models -- an entropy-matched
    comparison rather than a time-matched one, which is the only way to separate
    "entanglement does not matter" from "this particular T suited both models".
    """
    if "entropy_target" not in params:
        return float(params.get("total_time", 0.2)), {}
    cal = calibrate(model, num_memory, float(params["entropy_target"]),
                    n_steps=int(params.get("n_steps", 5)),
                    W=max(2, int(encode_len)), encoding=encoding,
                    initial_state=params.get("initial_state", "neel"),
                    model_kwargs=model_kwargs_for(model, seed),
                    seed=derive_seed(seed, "calib"))
    return cal["total_time"], cal


# ---------------------------------------------------------------------------
# Negative controls
# ---------------------------------------------------------------------------

def _permute_within(y, mask_a, mask_b, rng):
    """Permute inside each split half separately, never across the boundary."""
    out = np.array(y, copy=True)
    for m in (mask_a, mask_b):
        idx = np.flatnonzero(m)
        if len(idx) > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


def _shuffle_rows_within(F, mask_a, mask_b, rng):
    out = np.array(F, copy=True)
    for m in (mask_a, mask_b):
        idx = np.flatnonzero(m)
        if len(idx) > 1:
            out[idx] = out[rng.permutation(idx)]
    return out


# ---------------------------------------------------------------------------
# Memoised classical baselines
# ---------------------------------------------------------------------------
# Baselines depend on (task, seed, representation) only -- not on the
# Hamiltonian. Across a model sweep that is the same computation many times
# over, and a tuned RBF-SVM is not cheap, so it is memoised per worker.

@lru_cache(maxsize=16)
def _clf_baselines(task_json: str, seed: int, n_pca: int, n_rff: int, control: str):
    import json
    task = json.loads(task_json)
    Atr, Ate, ytr, yte = prepare_classification(task, seed, n_pca)
    if control == "shuffle_labels":
        rng = np.random.default_rng(derive_seed(seed, "control"))
        ytr = ytr[rng.permutation(len(ytr))]
        yte = yte[rng.permutation(len(yte))]
    return baselines_clf(Atr, ytr, Ate, yte, n_rff=n_rff, seed=derive_seed(seed, "rff"))


@lru_cache(maxsize=16)
def _ts_baselines(task_json: str, seed: int, W: int, embargo: int, control: str):
    import json
    Xs, Xr, y, sid, tr, te = prepare_timeseries(json.loads(task_json), seed, W, embargo)
    if control == "shuffle_labels":
        y = _permute_within(y, tr, te, np.random.default_rng(derive_seed(seed, "control")))
    return ts_baselines(Xs, Xr, y, tr, te)


@lru_cache(maxsize=8)
def _mem_capacity(model: str, nm: int, T: float, n_steps: int, encoding: str,
                  seed: int, haar_seed: int | None, n_tap: int, max_lag: int):
    U = None if haar_seed is None else _haar(1 << (1 + nm), haar_seed)
    return stream.memory_capacity(model, nm, T, n_steps=n_steps, encoding=encoding,
                                  model_kwargs=model_kwargs_for(model, seed),
                                  block_unitary=U, seed=derive_seed(seed, "mc"),
                                  n_tap=n_tap, max_lag=max_lag)


def clear_memo() -> None:
    for fn in (_clf_baselines, _ts_baselines, _mem_capacity, _haar, provenance):
        fn.cache_clear()


# ---------------------------------------------------------------------------
# Families
# ---------------------------------------------------------------------------

def _repr_metrics(F, L, prefix_entropy=True) -> dict:
    out = {"effective_rank": effective_rank(F)}
    if prefix_entropy:
        out["entropy"] = entropy_stats(F, L)
    return out


def _run_classification(unit: dict) -> dict:
    p, seed, task = unit["params"], unit["seed"], unit["task"]
    n_pca = int(p["n_pca"]); nm = int(p["num_memory"]); model = p["model"]
    rup = int(p.get("n_reupload", 1))
    shots = int(p.get("shots", 0)) or None
    n_steps = int(p.get("n_steps", 5)); init = p.get("initial_state", "neel")
    control = p.get("control", "none")
    L = 1 + nm

    T, cal = resolve_time(p, model, nm, n_pca * rup, seed)

    Atr, Ate, ytr, yte = prepare_classification(task, seed, n_pca)
    if control == "shuffle_labels":
        rng = np.random.default_rng(derive_seed(seed, "control"))
        ytr = ytr[rng.permutation(len(ytr))]
        yte = yte[rng.permutation(len(yte))]
    Utr, Ute = scale_for_encoding(Atr, Ate)

    n_feat = n_pca * rup * (2 * L - 1)
    base = _clf_baselines(canon(task), seed, n_pca, n_feat, control)

    common = dict(num_memory=nm, total_time=T, n_reupload=rup, n_steps=n_steps,
                  initial_state=init, shots=shots,
                  shot_seed=derive_seed(seed, "shots", model))

    kw = model_kwargs_for(model, seed)
    Ftr = quantum_features(Utr, model_key=model, model_kwargs=kw, **common)
    Fte = quantum_features(Ute, model_key=model, model_kwargs=kw, **common)

    U = _haar(1 << L, derive_seed(seed, "haar", nm))
    Rtr = quantum_features(Utr, model_key=model, block_unitary=U, **common)
    Rte = quantum_features(Ute, model_key=model, block_unitary=U, **common)

    if control == "shuffle_features":
        rng = np.random.default_rng(derive_seed(seed, "control"))
        for M in (Ftr, Rtr):
            M[:] = M[rng.permutation(len(M))]
        for M in (Fte, Rte):
            M[:] = M[rng.permutation(len(M))]

    return {
        "L": L, "n_features": int(Ftr.shape[1]), "resolved_total_time": T,
        "calibration": cal, "control": control,
        "baselines": base,
        "quantum": evaluate_clf(Ftr, ytr, Fte, yte),
        "random_unitary": evaluate_clf(Rtr, ytr, Rte, yte),
        "repr_quantum": _repr_metrics(Ftr, L),
        "repr_random_unitary": _repr_metrics(Rtr, L),
        "repr_pca": {"effective_rank": effective_rank(Atr)},
        "sep_pca": separability(Atr, ytr, seed=derive_seed(seed, "sep")),
        "sep_quantum": separability(Ftr, ytr, seed=derive_seed(seed, "sep")),
        "n_train": int(len(ytr)), "n_test": int(len(yte)),
    }


def _normalize_for_encoding(X_raw, tr):
    """Map raw window values into [-1,1] using TRAIN statistics only."""
    mu, sd = X_raw[tr].mean(), X_raw[tr].std()
    sd = sd if sd > 1e-12 else 1.0
    return np.clip((X_raw - mu) / (3.0 * sd), -1.0, 1.0)


def _run_timeseries(unit: dict) -> dict:
    from qrc.ppe import reservoir_features
    p, seed, task = unit["params"], unit["seed"], unit["task"]
    W = int(p.get("W", 10)); nm = int(p["num_memory"]); model = p["model"]
    enc = p.get("encoding", "continuous")
    n_steps = int(p.get("n_steps", 5)); wash = int(p.get("washout", 4))
    embargo = int(p.get("embargo", W))
    shots = int(p.get("shots", 0)) or None
    init = p.get("initial_state", "neel")
    control = p.get("control", "none")
    L = 1 + nm

    T, cal = resolve_time(p, model, nm, W, seed, encoding=enc)

    Xs, Xr, y, sid, tr, te = prepare_timeseries(task, seed, W, embargo)
    base = _ts_baselines(canon(task), seed, W, embargo, control)
    if control == "shuffle_labels":
        y = _permute_within(y, tr, te, np.random.default_rng(derive_seed(seed, "control")))
    Xin = Xs if enc == "symbolic" else _normalize_for_encoding(Xr, tr)

    common = dict(num_memory=nm, dt=T / n_steps, n_steps=n_steps, washout_length=wash,
                  encoding=enc, initial_state=init, shots=shots,
                  shot_seed=derive_seed(seed, "shots", model))

    kw = model_kwargs_for(model, seed)
    F = reservoir_features(Xin, model, model_kwargs=kw, **common)
    U = _haar(1 << L, derive_seed(seed, "haar", nm))
    R = reservoir_features(Xin, model, block_unitary=U, **common)

    if control == "shuffle_features":
        rng = np.random.default_rng(derive_seed(seed, "control"))
        F = _shuffle_rows_within(F, tr, te, rng)
        R = _shuffle_rows_within(R, tr, te, rng)

    # Dimension-matched control for the incremental-value question: concatenating
    # the reservoir adds features AND information, so raw+random-projection of
    # equal width separates "more columns" from "more signal".
    rng = np.random.default_rng(derive_seed(seed, "proj"))
    P = Xr @ rng.standard_normal((Xr.shape[1], F.shape[1])) / np.sqrt(Xr.shape[1])

    return {
        "L": L, "n_features": int(F.shape[1]), "resolved_total_time": T,
        "calibration": cal, "control": control,
        "baselines": base,
        "quantum": evaluate_features(F, y, tr, te),
        "random_unitary": evaluate_features(R, y, tr, te),
        "raw_only": evaluate_features(Xr, y, tr, te),
        "raw_plus_reservoir": evaluate_features(np.hstack([Xr, F]), y, tr, te),
        "raw_plus_random_proj": evaluate_features(np.hstack([Xr, P]), y, tr, te),
        "repr_quantum": _repr_metrics(F, L),
        "repr_random_unitary": _repr_metrics(R, L),
        "repr_raw": {"effective_rank": effective_rank(Xr)},
        "n_windows": int(len(y)), "n_series": int(len(np.unique(sid))),
        "n_train": int(tr.sum()), "n_test": int(te.sum()),
    }


def _run_streaming(unit: dict) -> dict:
    """
    The reservoir protocol proper: continuous drive, no per-window reset.

    Reports both the forecasting score and the linear memory capacity. MC is the
    load-bearing number: a unitary reservoir is measure-preserving and has no
    fading memory, so if MC is ~0 then the windowed results stop being a fact
    about these tasks and become a structural fact about unitary dynamics.
    """
    p, seed, task = unit["params"], unit["seed"], unit["task"]
    nm = int(p["num_memory"]); model = p["model"]
    enc = p.get("encoding", "continuous")
    n_steps = int(p.get("n_steps", 5)); wash = int(p.get("washout", 20))
    n_tap = int(p.get("n_tap", 1)); n_lags = int(p.get("n_lags", 10))
    max_lag = int(p.get("max_lag", 20)); embargo = int(p.get("embargo", 10))
    stream_len = p.get("stream_len")
    shots = int(p.get("shots", 0)) or None
    init = p.get("initial_state", "neel")
    control = p.get("control", "none")
    L = 1 + nm

    T, cal = resolve_time(p, model, nm, max(2, n_lags), seed, encoding=enc)
    series = load_series(task, seed)
    kw = model_kwargs_for(model, seed)
    U = _haar(1 << L, derive_seed(seed, "haar", nm))

    common = dict(num_memory=nm, total_time=T, n_steps=n_steps, washout=wash,
                  n_lags=n_lags, embargo=embargo, encoding=enc, initial_state=init,
                  shots=shots, shot_seed=derive_seed(seed, "shots", model),
                  max_len=int(stream_len) if stream_len else None, n_tap=n_tap)

    q = stream.forecast(series, model, model_kwargs=kw, **common)
    r = stream.forecast(series, model, block_unitary=U, **common)
    Xq = q.pop("state_matrix"); Xr_ = r.pop("state_matrix")

    mc_q = _mem_capacity(model, nm, T, n_steps, enc, seed, None, n_tap, max_lag)
    mc_r = _mem_capacity(model, nm, T, n_steps, enc, seed,
                         derive_seed(seed, "haar", nm), n_tap, max_lag)

    return {
        "L": L, "n_features": int(q["reservoir"]["n_features"]),
        "resolved_total_time": T, "calibration": cal, "control": control,
        "baselines": q["baselines"],
        "quantum": q["reservoir"],
        "random_unitary": r["reservoir"],
        "memory_capacity": mc_q,
        "memory_capacity_random": mc_r,
        "repr_quantum": {"effective_rank": effective_rank(Xq)},
        "repr_random_unitary": {"effective_rank": effective_rank(Xr_)},
        "n_series": q["n_series"], "n_steps_used": q["n_steps_used"],
        "n_tap": q["n_tap"], "n_train": q["n_train"], "n_test": q["n_test"],
    }


FAMILY_FN = {"classification": _run_classification,
             "timeseries": _run_timeseries,
             "streaming": _run_streaming}


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_unit(unit: dict) -> dict:
    """Execute one unit. Never raises: failures come back as status='failed'."""
    t0 = time.time()
    head = {"schema": SCHEMA_VERSION, "id": unit["id"], "config": unit["config"],
            "family": unit["family"], "task": unit["task"], "task_key": unit["task_key"],
            "params": unit["params"], "seed": unit["seed"],
            "provenance": provenance()}
    try:
        result = FAMILY_FN[unit["family"]](unit)
        head.update(status="ok", seconds=round(time.time() - t0, 3), **result)
    except Exception as exc:
        head.update(status="failed", seconds=round(time.time() - t0, 3),
                    error=f"{type(exc).__name__}: {exc}",
                    traceback=traceback.format_exc()[-4000:])
    return head
