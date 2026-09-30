"""
store.py — one JSON file per finished unit, written atomically.

A single aggregate JSON written at the end of a monolithic script (what the
current runners do) is the wrong shape for a sweep: it cannot resume, cannot be
written by more than one process, and loses everything if the job is killed at
90%. One small file per unit, named by the unit's content hash, is append-only,
lock-free and safe for any number of concurrent writers.

Failures are recorded, not swallowed and not fatal. A unit that raises writes a
record with status='failed' and its traceback, so one bad configuration cannot
take down a 10,000-unit sweep, and `qrc status` can list exactly what broke.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = Path(os.environ.get("QRC_RUNS_DIR", REPO_ROOT / "results" / "runs"))


def unit_path(unit: dict, runs_dir: Path | None = None) -> Path:
    root = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    return root / unit["config"] / unit["family"] / f"{unit['id']}.json"


def is_done(unit: dict, runs_dir: Path | None = None) -> bool:
    p = unit_path(unit, runs_dir)
    if not p.exists() or p.stat().st_size == 0:
        return False
    try:
        with p.open() as f:
            return json.load(f).get("status") == "ok"
    except Exception:
        return False          # unreadable/truncated -> treat as not done, rerun


def write_record(unit: dict, record: dict, runs_dir: Path | None = None) -> Path:
    p = unit_path(unit, runs_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(f".tmp{os.getpid()}")
    with tmp.open("w") as f:
        json.dump(record, f, default=float)
    os.replace(tmp, p)        # atomic on POSIX: readers see old or new, never partial
    return p


def iter_records(config: str | None = None, runs_dir: Path | None = None):
    root = Path(runs_dir) if runs_dir is not None else RUNS_DIR
    root = root / config if config else root
    if not root.exists():
        return
    for p in sorted(root.rglob("*.json")):
        try:
            with p.open() as f:
                yield json.load(f)
        except Exception:
            continue
