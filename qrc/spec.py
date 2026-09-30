"""
spec.py — experiment config -> a deterministic, content-addressed list of units.

A "unit" is the atom of work: one (task, reservoir configuration, seed) triple,
which is evaluated for EVERY arm (quantum, Haar-random control, classical
baselines) on one identical split. Keeping the arms together in a unit is not an
optimisation -- it is the methodological requirement. The whole point of these
experiments is the paired comparison, and pairing is only valid if the arms saw
the same split, so nothing may be able to schedule them apart.

Each unit's id is a hash of its own content, which buys three properties that a
sweep of this size needs:
  * resumable   -- a finished unit's file already exists, so it is skipped
  * idempotent  -- re-running never double-counts
  * shardable   -- any subset of units can run anywhere, in any order

Configs are JSON (not YAML) so that running a sweep needs no dependency that
isn't already required to compute one.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

FAMILIES = ("classification", "timeseries", "streaming")

# Grid keys each family understands. Anything else in a grid is a typo, and a
# typo that silently does nothing would quietly invalidate a whole sweep.
# `entropy_target` is an ALTERNATIVE to `total_time`: the interaction time is
# resolved per model so that the task-conditional entropy hits the target,
# giving an entropy-matched comparison instead of a time-matched one.
# `control` selects a negative control (shuffled labels / shuffled features),
# which must drive every arm to chance if the pipeline is free of leakage.
_COMMON = {"num_memory", "model", "total_time", "entropy_target", "shots",
           "initial_state", "n_steps", "control"}

GRID_KEYS = {
    "classification": _COMMON | {"n_pca", "n_reupload"},
    "timeseries": _COMMON | {"W", "encoding", "washout", "embargo"},
    "streaming": _COMMON | {"encoding", "washout", "n_tap", "n_lags", "max_lag",
                            "stream_len", "embargo"},
}

_TS_TASK = {"dataset", "n", "max_series", "scheme", "index", "n_cohorts",
            "min_len", "tau"}

TASK_KEYS = {
    "classification": {"dataset", "n_samples", "W", "n_per_class", "noise"},
    "timeseries": _TS_TASK,
    "streaming": _TS_TASK,
}

CONTROLS = ("none", "shuffle_labels", "shuffle_features")


def canon(obj) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str)


def unit_id(unit: dict) -> str:
    """Stable 16-hex id over everything that affects the numbers."""
    payload = {k: unit[k] for k in ("family", "task", "params", "seed")}
    return hashlib.sha1(canon(payload).encode()).hexdigest()[:16]


def task_key(family: str, task: dict) -> str:
    """Short human-readable task label, used in filenames and the tidy table."""
    ds = task["dataset"]
    if family in ("timeseries", "streaming") and ds == "stock_cohort":
        scheme = task.get("scheme", "vol")
        if scheme == "bootstrap":
            return f"stock:bootstrap{task.get('index', 0)}"
        return f"stock:{scheme}{task.get('index',0)}of{task.get('n_cohorts',3)}"
    if family == "classification":
        if ds == "source":
            return f"source:W{task.get('W', 16)}n{task.get('n_per_class', 2500)}"
        return f"{ds}:{task.get('n_samples') or 'all'}"
    return ds


def expand_grid(grid: dict):
    keys = sorted(grid)
    for combo in itertools.product(*(grid[k] for k in keys)):
        yield dict(zip(keys, combo))


def _validate(exp: dict, i: int) -> None:
    fam = exp.get("family")
    if fam not in FAMILIES:
        raise ValueError(f"experiment[{i}]: family must be one of {FAMILIES}, got {fam!r}")
    for t in exp.get("tasks", []):
        if "dataset" not in t:
            raise ValueError(f"experiment[{i}]: every task needs a 'dataset'")
        bad = set(t) - TASK_KEYS[fam]
        if bad:
            raise ValueError(f"experiment[{i}] task {t.get('dataset')}: unknown keys {sorted(bad)}")
    bad = set(exp.get("grid", {})) - GRID_KEYS[fam]
    if bad:
        raise ValueError(f"experiment[{i}]: unknown grid keys {sorted(bad)} "
                         f"(valid: {sorted(GRID_KEYS[fam])})")
    grid = exp.get("grid", {})
    for k, v in grid.items():
        if not isinstance(v, list) or not v:
            raise ValueError(f"experiment[{i}]: grid['{k}'] must be a non-empty list")
    if "total_time" in grid and "entropy_target" in grid:
        raise ValueError(f"experiment[{i}]: give total_time OR entropy_target, not both "
                         "-- entropy_target resolves the time itself")
    if "total_time" not in grid and "entropy_target" not in grid:
        raise ValueError(f"experiment[{i}]: needs total_time or entropy_target")
    for c in grid.get("control", []):
        if c not in CONTROLS:
            raise ValueError(f"experiment[{i}]: control must be one of {CONTROLS}, got {c!r}")
    if not exp.get("seeds"):
        raise ValueError(f"experiment[{i}]: 'seeds' must be a non-empty list")


PRIMARY_KEYS = {"family", "metric", "task_key", "arm_a", "arm_b", "margin", "params"}


def _validate_primary(pri: dict) -> None:
    """
    The pre-declared primary endpoint.

    Declaring one comparison in advance is what separates a negative result from
    a fishing expedition: with hundreds of cells, some will look significant by
    chance, and the reader has no way to know which were chosen after the fact.
    `margin` is the equivalence bound -- the smallest difference that would
    matter -- without which "no significant difference" is only a failure to
    reject, not evidence of absence.
    """
    bad = set(pri) - PRIMARY_KEYS
    if bad:
        raise ValueError(f"primary: unknown keys {sorted(bad)} (valid: {sorted(PRIMARY_KEYS)})")
    for k in ("family", "metric", "arm_a", "arm_b", "margin"):
        if k not in pri:
            raise ValueError(f"primary: missing required key '{k}'")
    if pri["family"] not in FAMILIES:
        raise ValueError(f"primary: family must be one of {FAMILIES}")
    if not isinstance(pri["margin"], (int, float)) or pri["margin"] <= 0:
        raise ValueError("primary: 'margin' must be a positive number "
                         "(the smallest effect size worth caring about)")


def load_config(path: str | Path) -> dict:
    cfg = json.loads(Path(path).read_text())
    cfg.setdefault("name", Path(path).stem)
    if "experiments" not in cfg:
        raise ValueError(f"{path}: config needs an 'experiments' list")
    for i, exp in enumerate(cfg["experiments"]):
        _validate(exp, i)
    if "primary" in cfg:
        _validate_primary(cfg["primary"])
    return cfg


def build_units(cfg: dict) -> list[dict]:
    """
    Expand a config into units, deduplicated and ordered for cache locality.

    Ordering matters at scale: units are sorted so that everything sharing a
    (task, seed) -- and therefore a prepared dataset and a set of baselines --
    runs consecutively in the same worker, which lets the in-process memo take
    the dataset build and the classical baselines out of the inner loop.
    """
    units, seen = [], set()
    for exp in cfg["experiments"]:
        fam = exp["family"]
        for task in exp["tasks"]:
            for seed in exp["seeds"]:
                for params in expand_grid(exp["grid"]):
                    u = {"config": cfg["name"], "family": fam, "task": dict(task),
                         "params": params, "seed": int(seed)}
                    u["task_key"] = task_key(fam, task)
                    uid = unit_id(u)
                    if uid in seen:
                        continue
                    seen.add(uid)
                    u["id"] = uid
                    units.append(u)
    units.sort(key=lambda u: (u["family"], u["task_key"], u["seed"], canon(u["params"])))
    return units


def shard(units: list[dict], index: int, total: int) -> list[dict]:
    """
    Contiguous, balanced shard, so each shard keeps its (task, seed) locality.

    Balanced rather than ceil-divided: with ceil division a large shard count
    leaves empty trailing shards (2200 units over 64 shards gives 63 shards of
    35 and one of 25, or worse), which on a SLURM array means paying for array
    tasks that immediately exit while the rest run long.
    """
    if total <= 1:
        return units
    if not 0 <= index < total:
        raise ValueError(f"shard index {index} out of range for {total} shards")
    n = len(units)
    base, extra = divmod(n, total)
    start = index * base + min(index, extra)
    return units[start:start + base + (1 if index < extra else 0)]
