"""
tasks.py — turn a task spec into split data, cached on disk and in-process.

Two families, one contract each:

  classification -> (Atr, Ate, ytr, yte)   PCA components, PCA fitted on train
  timeseries     -> (Xs, Xr, y, sid, tr, te)  pooled windows and a split

The seed reaches everything that should vary between replicates: which samples
are drawn, the initial condition of the chaotic generators, the PCA/train split,
and the series-level split. A sweep whose "seeds" only reshuffle a split reports
error bars that are far too narrow, because the dominant source of variability
-- the data itself -- was held fixed.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from qrc import cache
from qrc.classify import load_images, pca_split
from qrc.datasets import DATASETS, source_classification
from qrc.evaluate import windowize, make_split
from qrc.spec import canon


# ---------------------------------------------------------------------------
# Raw corpora (disk cached; seed-dependent where the corpus itself varies)
# ---------------------------------------------------------------------------

def _series_params(task: dict, seed: int) -> dict:
    """Loader kwargs for a timeseries task, with the seed folded in."""
    ds = task["dataset"]
    kw = {k: v for k, v in task.items() if k != "dataset"}
    if ds in ("narma10", "mackey_glass"):
        kw["seed"] = seed
    elif ds == "lorenz":
        kw["seed"] = seed                       # jitters the initial condition
    elif ds in ("stocks", "options"):
        kw["seed"] = seed                       # which series are subsampled
    elif ds == "stock_cohort":
        kw["seed"] = seed
    return kw


def load_series(task: dict, seed: int) -> list[np.ndarray]:
    ds = task["dataset"]
    kw = _series_params(task, seed)
    params = {"dataset": ds, **kw}

    def build():
        return cache.pack_series(DATASETS[ds](**kw))

    return cache.unpack_series(cache.memo_arrays("series", params, build))


def load_classification_corpus(task: dict, seed: int):
    """
    (X, y) for a classification task, disk cached.

    'source' is the four-generator window-classification corpus. It goes through
    the same PCA -> encode -> readout path as the image tasks on purpose: the
    width-matched RFF baseline is only a fair comparator if both arms consume
    the identical representation, and giving the quantum map raw windows while
    the baseline gets PCA components is precisely the asymmetry that manufactures
    an apparent advantage.
    """
    ds = task["dataset"]
    if ds == "source":
        params = {"dataset": ds, "W": task.get("W", 16),
                  "n_per_class": task.get("n_per_class", 2500),
                  "noise": task.get("noise", 0.0), "seed": seed}

        def build():
            X, y = source_classification(W=params["W"], n_per_class=params["n_per_class"],
                                         noise=params["noise"], seed=seed)
            return {"X": np.asarray(X, dtype=np.float32), "y": np.asarray(y, dtype=np.int64)}
    else:
        n = task.get("n_samples")
        params = {"dataset": ds, "n_samples": n, "seed": seed}

        def build():
            X, y = load_images(ds, n_samples=n, seed=seed)
            return {"X": np.asarray(X, dtype=np.float32), "y": np.asarray(y, dtype=np.int64)}

    d = cache.memo_arrays("clf_corpus", params, build)
    return d["X"].astype(np.float64), d["y"]


# backwards-compatible alias
load_image_corpus = load_classification_corpus


# ---------------------------------------------------------------------------
# Prepared tasks (in-process memo; units are ordered so this hits constantly)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=4)
def _classification_cached(task_json: str, seed: int, n_pca: int):
    import json
    task = json.loads(task_json)
    X, y = load_classification_corpus(task, seed)
    Atr, Ate, ytr, yte = pca_split(X, y, n_pca, seed=seed)
    return Atr, Ate, ytr, yte


@lru_cache(maxsize=4)
def _timeseries_cached(task_json: str, seed: int, W: int, embargo: int):
    import json
    task = json.loads(task_json)
    series = load_series(task, seed)
    Xs, Xr, y, sid = windowize(series, W)
    # seed reaches the split; embargo removes the overlap leak at a temporal cut
    tr, te = make_split(sid, seed=seed, embargo=embargo)
    if tr.sum() < 20 or te.sum() < 10:
        raise ValueError(f"degenerate split for {task['dataset']} at W={W}: "
                         f"train={int(tr.sum())} test={int(te.sum())}")
    return Xs, Xr, y, sid, tr, te


def prepare_classification(task: dict, seed: int, n_pca: int):
    return _classification_cached(canon(task), int(seed), int(n_pca))


def prepare_timeseries(task: dict, seed: int, W: int, embargo: int = 0):
    return _timeseries_cached(canon(task), int(seed), int(W), int(embargo))


def clear_memo() -> None:
    _classification_cached.cache_clear()
    _timeseries_cached.cache_clear()
