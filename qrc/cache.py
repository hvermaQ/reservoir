"""
cache.py — on-disk memoisation of everything expensive that is not the physics.

Why this exists: under a parallel sweep, dataset preparation is repeated once
per worker process, not once per run. `load_option_deviations` parses ~1.1 GB of
CSV and `load_images("mnist")` hits openml over the network -- doing either
inside N workers is both slow and, for the network case, a way to get rate
limited or to produce a sweep that cannot run offline. Everything cacheable is
therefore built ONCE by `qrc prepare` and read back as .npz afterwards.

Writes are atomic (tmp file + os.replace), so a worker that dies mid-write
cannot leave a truncated cache entry behind for the next one to read.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = Path(os.environ.get("QRC_CACHE_DIR", REPO_ROOT / "cache"))


def key_hash(kind: str, params: dict) -> str:
    blob = json.dumps(params, sort_keys=True, separators=(",", ":"), default=str)
    return f"{kind}-{hashlib.sha1(blob.encode()).hexdigest()[:16]}"


def _path(kind: str, params: dict) -> Path:
    return CACHE_DIR / f"{key_hash(kind, params)}.npz"


def save_npz(path: Path, **arrays) -> None:
    """
    Atomic .npz write.

    The temp file is written through an open handle rather than by path:
    `np.savez_compressed` APPENDS '.npz' to any path not already ending in it,
    so writing to 'x.npz.tmp123' silently produces 'x.npz.tmp123.npz' and the
    rename then fails on a missing file.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    try:
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **arrays)
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink()
        except OSError:
            pass
        raise


def memo_arrays(kind: str, params: dict, builder):
    """
    Return dict-of-arrays for (kind, params), building and caching on first call.

    `builder()` must return a dict of numpy arrays. Corrupt or unreadable cache
    entries are rebuilt rather than raising, so a partially written cache from an
    interrupted prepare cannot wedge a whole sweep.
    """
    path = _path(kind, params)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k] for k in z.files}
        except Exception:
            try:
                path.unlink()
            except OSError:
                pass
    out = builder()
    save_npz(path, **out)
    return out


# --- ragged series lists <-> flat arrays -----------------------------------

def pack_series(series_list) -> dict:
    """Ragged list of 1-D arrays -> {'flat', 'offsets'} for npz storage."""
    if not series_list:
        return {"flat": np.zeros(0), "offsets": np.zeros(1, dtype=np.int64)}
    lens = np.array([len(s) for s in series_list], dtype=np.int64)
    return {"flat": np.concatenate([np.asarray(s, float) for s in series_list]),
            "offsets": np.concatenate([[0], np.cumsum(lens)])}


def unpack_series(d: dict) -> list[np.ndarray]:
    off = d["offsets"].astype(int)
    flat = d["flat"]
    return [flat[off[i]:off[i + 1]] for i in range(len(off) - 1)]
