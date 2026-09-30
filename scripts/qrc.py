#!/usr/bin/env python3
"""
qrc — sweep driver: prepare | plan | run | status | aggregate | diagnostics

    python3 scripts/qrc.py prepare  -c configs/timeseries_seeds.json
    python3 scripts/qrc.py plan     -c configs/timeseries_seeds.json
    python3 scripts/qrc.py run      -c configs/timeseries_seeds.json -j 12
    python3 scripts/qrc.py run      -c configs/... --shard 3/16      # SLURM array
    python3 scripts/qrc.py status   -c configs/timeseries_seeds.json
    python3 scripts/qrc.py aggregate -c configs/timeseries_seeds.json

`run` is resumable and idempotent: finished units are skipped, so re-running
after an interruption picks up exactly where it stopped, and two shards racing
on the same unit can only ever write the same answer twice.
"""
import os
import sys

# BLAS thread pinning MUST happen before numpy is imported anywhere. Each unit is
# already a small dense linear-algebra job, so with N worker processes an
# unpinned BLAS spawns N x cores threads that thrash the machine and make the
# sweep slower than running it serially. One thread per worker, N workers.
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse           # noqa: E402
import json               # noqa: E402
import multiprocessing as mp  # noqa: E402
import numpy as np    # noqa: E402
import time               # noqa: E402
from collections import Counter  # noqa: E402
from pathlib import Path  # noqa: E402

from qrc import spec, store  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _load_units(paths, shard=None):
    units = []
    for p in paths:
        units.extend(spec.build_units(spec.load_config(p)))
    if shard:
        i, n = shard
        units = spec.shard(units, i, n)
    return units


def _parse_shard(s):
    if not s:
        return None
    i, n = s.split("/")
    return int(i), int(n)


def _fmt_hms(sec):
    sec = int(max(0, sec))
    return f"{sec // 3600:d}h{(sec % 3600) // 60:02d}m{sec % 60:02d}s"


def _start_method(requested):
    if requested != "auto":
        return requested
    # fork keeps worker startup and the page cache cheap, but on macOS forking a
    # process that has already loaded Accelerate/OpenBLAS can deadlock. The disk
    # cache makes spawn's re-import cheap enough that safety wins there.
    return "spawn" if sys.platform == "darwin" else "fork"


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------

def cmd_plan(a):
    units = _load_units(a.config, _parse_shard(a.shard))
    done = sum(store.is_done(u, a.runs_dir) for u in units)
    print(f"units: {len(units)}   done: {done}   todo: {len(units) - done}")
    for field in ("family", "task_key"):
        c = Counter(u[field] for u in units)
        print(f"\nby {field}:")
        for k, v in sorted(c.items()):
            print(f"  {k:34s} {v:6d}")
    c = Counter(u["params"].get("model", "-") for u in units)
    print("\nby model:")
    for k, v in sorted(c.items()):
        print(f"  {k:34s} {v:6d}")
    print(f"\nseeds: {sorted({u['seed'] for u in units})}")
    if a.show:
        print("\nfirst units:")
        for u in units[:a.show]:
            print(f"  {u['id']}  {u['family']:14s} {u['task_key']:22s} "
                  f"seed={u['seed']:<4d} {spec.canon(u['params'])}")
    if a.out:
        Path(a.out).write_text(json.dumps(units, indent=1))
        print(f"\nwrote plan to {a.out}")


def cmd_prepare(a):
    """Build every dataset cache serially, before any parallel run touches it."""
    from qrc import tasks
    units = _load_units(a.config)
    wanted = {}
    for u in units:
        if u["family"] == "classification":
            k = ("clf", spec.canon(u["task"]), u["seed"], int(u["params"]["n_pca"]), 0)
        elif u["family"] == "timeseries":
            W = int(u["params"].get("W", 10))
            k = ("ts", spec.canon(u["task"]), u["seed"], W,
                 int(u["params"].get("embargo", W)))
        else:   # streaming consumes raw series, no windowing
            k = ("stream", spec.canon(u["task"]), u["seed"], 0, 0)
        wanted.setdefault(k, u)
    print(f"{len(wanted)} distinct (task, seed, window) preparations")
    for i, (k, u) in enumerate(sorted(wanted.items()), 1):
        kind, task_json, seed, w, emb = k
        t0 = time.time()
        try:
            if kind == "clf":
                Atr, Ate, ytr, yte = tasks.prepare_classification(u["task"], seed, w)
                shape = f"train={Atr.shape} test={Ate.shape}"
            elif kind == "ts":
                Xs, Xr, y, sid, tr, te = tasks.prepare_timeseries(u["task"], seed, w, emb)
                shape = (f"{len(y)} windows from {len(set(sid.tolist()))} series "
                         f"(train {int(tr.sum())} / test {int(te.sum())})")
            else:
                series = tasks.load_series(u["task"], seed)
                shape = (f"{len(series)} series, lengths "
                         f"{min(len(s) for s in series)}..{max(len(s) for s in series)}")
            label = {"clf": "n_pca", "ts": "W", "stream": "-"}[kind]
            print(f"  [{i}/{len(wanted)}] {u['task_key']:24s} seed={seed:<4d} "
                  f"{label}={w:<3d} {shape}  [{time.time()-t0:.1f}s]")
        except Exception as exc:
            print(f"  [{i}/{len(wanted)}] {u['task_key']:24s} seed={seed:<4d} "
                  f"FAILED: {type(exc).__name__}: {exc}")
        tasks.clear_memo()      # keep prepare's memory flat; the cache is on disk

    _warm_calibrations(units)


def _warm_calibrations(units):
    """
    Resolve every entropy target serially, before any parallel run needs one.

    Each calibration is a ~24-point scan cached on disk. Left to the workers,
    N processes would run the same scan simultaneously and race on the same
    cache file the first time each target is seen.
    """
    from qrc.calibrate import calibrate
    from qrc.runner import model_kwargs_for
    combos = sorted({(u["params"]["model"], int(u["params"]["num_memory"]),
                      float(u["params"]["entropy_target"]),
                      int(u["params"].get("n_steps", 5)), u["seed"])
                     for u in units if "entropy_target" in u["params"]})
    if not combos:
        return
    print(f"\nwarming {len(combos)} entropy calibrations")
    for model, nm, tgt, n_steps, seed in combos:
        c = calibrate(model, nm, tgt, n_steps=n_steps,
                      model_kwargs=model_kwargs_for(model, seed))
        flag = "" if c["bracketed"] else "  <-- TARGET NOT REACHABLE"
        print(f"  {model:15s} L={1+nm:<3d} target={tgt:.2f} -> T*={c['total_time']:.4f} "
              f"achieved={c['achieved_frac']:.4f} range="
              f"[{c['reachable_range'][0]:.3f},{c['reachable_range'][1]:.3f}]{flag}")


def _worker(unit):
    from qrc.runner import run_unit
    rec = run_unit(unit)
    store.write_record(unit, rec, _WORKER_RUNS_DIR)
    return (unit["id"], rec.get("status"), rec.get("seconds"),
            rec.get("error"), unit["task_key"], unit["seed"])


_WORKER_RUNS_DIR = None


def _init_worker(runs_dir):
    global _WORKER_RUNS_DIR
    _WORKER_RUNS_DIR = runs_dir


def cmd_run(a):
    units = _load_units(a.config, _parse_shard(a.shard))
    if not a.force:
        units = [u for u in units if not store.is_done(u, a.runs_dir)]
    if a.limit:
        units = units[:a.limit]
    if not units:
        print("nothing to do (all units already complete)")
        return
    print(f"running {len(units)} units on {a.jobs} worker(s); "
          f"results -> {a.runs_dir or store.RUNS_DIR}")
    if a.dry_run:
        print("--dry-run: listing only, nothing executed")
        for u in units[:a.show or 20]:
            print(f"  {u['id']}  {u['family']:14s} {u['task_key']:22s} "
                  f"seed={u['seed']:<4d} {spec.canon(u['params'])}")
        if len(units) > (a.show or 20):
            print(f"  ... and {len(units) - (a.show or 20)} more")
        return

    t0 = time.time()
    ok = failed = 0
    failures = []

    def report(res):
        nonlocal ok, failed
        uid, status, secs, err, tkey, seed = res
        if status == "ok":
            ok += 1
        else:
            failed += 1
            failures.append((uid, tkey, seed, err))
        n = ok + failed
        if n % a.progress_every == 0 or n == len(units):
            rate = n / max(1e-9, time.time() - t0)
            eta = (len(units) - n) / max(1e-9, rate)
            print(f"  {n:6d}/{len(units)}  ok={ok} failed={failed}  "
                  f"{rate:.2f} units/s  elapsed {_fmt_hms(time.time()-t0)}  "
                  f"eta {_fmt_hms(eta)}", flush=True)

    if a.jobs == 1:
        _init_worker(a.runs_dir)
        for u in units:
            report(_worker(u))
    else:
        ctx = mp.get_context(_start_method(a.start_method))
        # Contiguous chunks keep each worker on one (task, seed) for a while, so
        # the in-process dataset and baseline memos actually hit.
        chunk = max(1, min(16, len(units) // (a.jobs * 4) or 1))
        with ctx.Pool(a.jobs, initializer=_init_worker, initargs=(a.runs_dir,),
                      maxtasksperchild=a.max_tasks_per_child) as pool:
            try:
                for res in pool.imap(_worker, units, chunksize=chunk):
                    report(res)
            except KeyboardInterrupt:
                pool.terminate(); pool.join()
                print("\ninterrupted; finished units are on disk, rerun to resume")
                return

    print(f"\ndone in {_fmt_hms(time.time()-t0)}: ok={ok} failed={failed}")
    for uid, tkey, seed, err in failures[:20]:
        print(f"  FAILED {uid} {tkey} seed={seed}: {err}")
    if len(failures) > 20:
        print(f"  ... and {len(failures)-20} more failures")


def cmd_status(a):
    units = _load_units(a.config, _parse_shard(a.shard))
    done = [u for u in units if store.is_done(u, a.runs_dir)]
    print(f"{len(done)}/{len(units)} complete ({100*len(done)/max(1,len(units)):.1f}%)")
    todo = Counter((u["family"], u["task_key"]) for u in units
                   if not store.is_done(u, a.runs_dir))
    if todo:
        print("\noutstanding:")
        for (fam, tk), n in sorted(todo.items()):
            print(f"  {fam:14s} {tk:26s} {n:6d}")
    bad = [r for r in store.iter_records(a.config_name, a.runs_dir)
           if r.get("status") == "failed"]
    if bad:
        print(f"\n{len(bad)} failed records:")
        seen = Counter(r.get("error", "?") for r in bad)
        for err, n in seen.most_common(10):
            print(f"  {n:5d}x  {err[:110]}")


def cmd_aggregate(a):
    from qrc import aggregate as agg
    cfg = spec.load_config(a.config[0]) if len(a.config) == 1 else {}
    primary = cfg.get("primary")
    margins = cfg.get("margins")
    recs = list(store.iter_records(a.config_name, a.runs_dir))
    ok = [r for r in recs if r.get("status") == "ok"]
    print(f"{len(ok)} ok / {len(recs)} records")
    if not ok:
        print("nothing to aggregate")
        return
    df = agg.tidy(ok)
    df_real = agg.drop_controls(df)
    out = Path(a.out or (REPO_ROOT / "results" / "aggregated" / (a.config_name or "all")))
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "tidy.csv", index=False)

    for family, sub in df_real.groupby("family", observed=True):
        metrics = agg.CLF_METRICS if family == "classification" else agg.TS_METRICS
        for m in metrics:
            if not (sub["metric"] == m).any():
                continue
            agg.summarize(sub, m).to_csv(out / f"summary_{family}_{m}.csv", index=False)
        pm = "acc" if family == "classification" else "NRMSE"
        marg = (margins or {}).get(pm, 0.01 if pm == "acc" else 0.02)
        for b in ("random_unitary", "rff", "rbf_svm", "linear_raw", "raw_only",
                  "raw_plus_random_proj", "persistence"):
            t = agg.paired(sub, pm, "quantum", b, margin=marg)
            if not t.empty:
                t.to_csv(out / f"paired_{family}_{pm}_vs_{b}.csv", index=False)
        lm = agg.learnable_mask(sub, pm)
        if not lm.empty:
            lm.to_csv(out / f"learnable_{family}.csv", index=False)

    head = agg.headline(df, margins=margins, primary=primary)
    (out / "headline.json").write_text(json.dumps(head, indent=2))
    print(json.dumps(head, indent=2))
    if primary is None:
        print("\nNOTE: this config declares no 'primary' endpoint. Every number above "
              "is exploratory and BH-corrected; none of it is confirmatory.")
    print(f"\nwrote {out}")


def cmd_calibrate(a):
    """Resolve and cache entropy targets without running any units."""
    _warm_calibrations(_load_units(a.config))


def cmd_power(a):
    """
    Paired standard deviation from whatever has already run -> seeds required.

    Run a small pilot first (`run --limit N`), then this. Choosing the seed
    count from a measured paired SD is the difference between a sweep that can
    detect the effect it cares about and one that produces wide CIs at full cost.
    """
    from qrc import aggregate as agg
    cfg = spec.load_config(a.config[0]) if len(a.config) == 1 else {}
    margins = {**{"acc": 0.01, "NRMSE": 0.02}, **(cfg.get("margins") or {})}
    recs = [r for r in store.iter_records(a.config_name, a.runs_dir)
            if r.get("status") == "ok"]
    if not recs:
        print("no completed records yet -- run a pilot first, e.g.:\n"
              f"  python3 scripts/qrc.py run -c {a.config[0]} --limit 40")
        return
    df = agg.tidy(recs)
    print(f"{len(recs)} records, {df['seed'].nunique()} distinct seeds\n")
    hdr = f"{'family':14s} {'comparator':22s} {'metric':7s} {'paired sd':>10} {'margin':>8} {'seeds@80%':>10}"
    print(hdr); print("-" * len(hdr))
    for family, sub in df.groupby("family", observed=True):
        metric = "acc" if family == "classification" else "NRMSE"
        margin = margins.get(metric, 0.01)
        for arm in ("random_unitary", "rff", "linear_raw"):
            t = agg.paired(sub, metric, "quantum", arm)
            if t.empty or t["n_seeds"].max() < 2:
                continue
            # typical within-configuration paired sd across the grid
            sd = float(np.nanmedian(t["sem_delta"] * np.sqrt(t["n_seeds"])))
            n = agg.required_seeds(sd, margin)
            print(f"{family:14s} {arm:22s} {metric:7s} {sd:10.5f} {margin:8.4f} {n:10d}")
    print("\nseeds@80% = replicates needed to detect a difference of `margin` "
          "at 80% power, alpha=0.05 (normal approximation on paired differences)")


def cmd_diagnostics(a):
    """
    PPE entropy moments per (model, size, time) -- no task, no seed dependence
    beyond the disorder realisation. Cheap, and it is the x-axis of the
    entanglement-vs-performance comparison, so it belongs in the same output.
    """
    from qrc.ppe import ppe_diagnostics
    from qrc.runner import model_kwargs_for
    units = _load_units(a.config)
    combos = sorted({(u["params"]["model"], int(u["params"]["num_memory"]),
                      float(u["params"].get("total_time", 0.2)), u["seed"])
                     for u in units})
    n_steps = 5
    out_rows = []
    print(f"{len(combos)} (model, size, time, seed) diagnostics")
    for model, nm, T, seed in combos:
        d = ppe_diagnostics(model, num_memory=nm, n_interventions=a.n_interventions,
                            dt=T / n_steps, n_steps=n_steps,
                            model_kwargs=model_kwargs_for(model, seed))
        d["seed"] = seed
        out_rows.append(d)
        print(f"  {model:15s} L={d['L']:<3d} T={T:<6g} seed={seed:<4d} "
              f"mean_S={d['mean_S']:.4f} std_S={d['std_S']:.4f} "
              f"frac_max={d['mean_S']/d['max_S']:.3f}")
    dest = Path(a.out or (REPO_ROOT / "results" / "aggregated" /
                          (a.config_name or "all") / "diagnostics.json"))
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out_rows, indent=2))
    print(f"wrote {dest}")


# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    # Common flags live on a parent parser so they may be given AFTER the
    # subcommand (`qrc.py plan -c cfg`), which is what everyone types.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", nargs="+", required=True)
    common.add_argument("--runs-dir", default=None, help="override results/runs")
    common.add_argument("--shard", default=None, metavar="i/N")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("plan", parents=[common]);  p.set_defaults(fn=cmd_plan)
    p.add_argument("--show", type=int, default=0)
    p.add_argument("--out", default=None)

    p = sub.add_parser("prepare", parents=[common]); p.set_defaults(fn=cmd_prepare)

    p = sub.add_parser("run", parents=[common]); p.set_defaults(fn=cmd_run)
    p.add_argument("-j", "--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--force", action="store_true", help="rerun units already done")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--show", type=int, default=0)
    p.add_argument("--progress-every", type=int, default=25)
    p.add_argument("--max-tasks-per-child", type=int, default=200)
    p.add_argument("--start-method", default="auto",
                   choices=["auto", "fork", "spawn", "forkserver"])

    p = sub.add_parser("status", parents=[common]); p.set_defaults(fn=cmd_status)
    p = sub.add_parser("aggregate", parents=[common]); p.set_defaults(fn=cmd_aggregate)
    p.add_argument("--out", default=None)
    p = sub.add_parser("diagnostics", parents=[common]); p.set_defaults(fn=cmd_diagnostics)
    p.add_argument("--out", default=None)
    p.add_argument("--n-interventions", type=int, default=5)
    p = sub.add_parser("calibrate", parents=[common]); p.set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("power", parents=[common]); p.set_defaults(fn=cmd_power)

    a = ap.parse_args()
    a.config_name = spec.load_config(a.config[0])["name"] if len(a.config) == 1 else None
    a.fn(a)


if __name__ == "__main__":
    main()
