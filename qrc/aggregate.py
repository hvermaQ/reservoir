"""
aggregate.py — shards -> tidy table -> paired statistics with error bars.

The comparisons this repo cares about are all PAIRED: quantum vs the Haar
control, and quantum vs a width-matched classical baseline, both measured on the
same split within the same seed. Aggregating each arm's mean separately and
differencing the means throws that pairing away and inflates the error bar,
which for gaps of ~0.01 accuracy is the difference between a claim and a
coincidence. So the difference is formed WITHIN each seed and only then averaged.

Reported spread is the standard error over seeds plus a percentile bootstrap CI.
With the ~5-20 seeds these sweeps run, the bootstrap is the more honest of the
two, since normality over so few replicates is an assumption nobody checked.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

CLF_METRICS = ("acc", "balanced_acc")
TS_METRICS = ("NRMSE", "RMSE", "MAE", "R2")

# Arms that live at the top level of a record, vs. those nested under baselines.
TOP_ARMS = ("quantum", "random_unitary", "raw_only", "raw_plus_reservoir")


def tidy(records) -> pd.DataFrame:
    """Long-form table: one row per (unit, arm, metric)."""
    rows = []
    for r in records:
        if r.get("status") != "ok":
            continue
        metrics = CLF_METRICS if r["family"] == "classification" else TS_METRICS
        base = {"config": r.get("config"), "family": r["family"],
                "task_key": r.get("task_key"), "seed": r["seed"],
                "L": r.get("L"), "n_features": r.get("n_features"),
                "seconds": r.get("seconds"), "unit_id": r.get("id")}
        for k, v in (r.get("params") or {}).items():
            base[k] = v

        def emit(arm, d):
            if not isinstance(d, dict):
                return
            for m in metrics:
                if m in d:
                    rows.append({**base, "arm": arm, "metric": m, "value": float(d[m])})

        for arm in TOP_ARMS:
            emit(arm, r.get(arm))
        for name, d in (r.get("baselines") or {}).items():
            # width-matched random Fourier features are stored as 'rff_<n>';
            # normalise the name so it groups across differing widths
            emit("rff" if name.startswith("rff_") else name, d)
    df = pd.DataFrame(rows)
    if not df.empty:
        df["arm"] = df["arm"].astype("category")
    return df


def drop_controls(df: pd.DataFrame, keep: str = "none") -> pd.DataFrame:
    """
    Remove negative-control units from a frame.

    Control units are deliberately broken (shuffled labels or shuffled feature
    rows), so pooling them with real runs drags every mean towards chance. They
    are diagnostics about the pipeline, never evidence about the model, and the
    default everywhere is to exclude them.
    """
    if "control" not in df.columns:
        return df
    return df[(df["control"] == keep) | df["control"].isna()]


def control_report(df: pd.DataFrame) -> dict:
    """
    Did the negative controls behave? Per family and control, the mean of each arm.

    shuffle_labels must drive EVERY arm to the chance level; shuffle_features
    must collapse only the quantum arms while the classical baselines are
    untouched. Anything else is leakage, and invalidates the rest of the sweep.
    """
    if "control" not in df.columns:
        return {}
    out = {}
    for (fam, ctrl), sub in df.groupby(["family", "control"], observed=True, dropna=False):
        if ctrl == "none" or pd.isna(ctrl):
            continue
        metric = "acc" if fam == "classification" else "NRMSE"
        d = sub[sub["metric"] == metric]
        if d.empty:
            continue
        out.setdefault(fam, {})[str(ctrl)] = {
            str(arm): round(float(v), 4) for arm, v in
            d.groupby("arm", observed=True)["value"].mean().items()}
    return out


def _group_cols(df: pd.DataFrame) -> list[str]:
    """
    Everything that identifies a configuration, excluding seed/arm/value.

    All-null columns are dropped. Families do not share a parameter set -- a
    classification row has n_pca and no W, a timeseries row the reverse -- so a
    mixed frame has all-NaN parameter columns, and both groupby and pivot_table
    silently DROP rows with NaN in a key. Left in, that turns a paired
    comparison into an empty table rather than an error.
    """
    drop = {"seed", "arm", "metric", "value", "seconds", "unit_id", "n_features", "L"}
    return [c for c in df.columns if c not in drop and df[c].notna().any()]


_NA = "__na__"


def _keyed(df: pd.DataFrame):
    """Frame plus grouping keys, with residual NaNs made groupable."""
    keys = _group_cols(df)
    d = df.copy()
    for c in keys:
        if d[c].isna().any():
            d[c] = d[c].astype(object).where(d[c].notna(), _NA)
    return d, keys


def summarize(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Per-configuration, per-arm mean and standard error across seeds."""
    d, keys = _keyed(df[df["metric"] == metric])
    keys = keys + ["arm"]
    g = d.groupby(keys, dropna=False, observed=True)["value"]
    out = g.agg(mean="mean", std="std", n="count").reset_index()
    out["sem"] = out["std"] / np.sqrt(out["n"].clip(lower=1))
    return out.sort_values(keys)


def _bootstrap_ci(x: np.ndarray, n_boot: int = 10000, alpha: float = 0.05, seed: int = 0):
    """Percentile CI and a two-sided p-value from the same bootstrap distribution."""
    x = np.asarray(x, dtype=float)
    if len(x) < 2:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = rng.choice(x, size=(n_boot, len(x)), replace=True).mean(axis=1)
    lo = float(np.quantile(means, alpha / 2))
    hi = float(np.quantile(means, 1 - alpha / 2))
    # inverting the percentile interval: the smallest alpha at which 0 leaves it
    frac_le = float(np.mean(means <= 0.0))
    pval = min(1.0, 2.0 * min(frac_le, 1.0 - frac_le))
    return lo, hi, max(pval, 1.0 / n_boot)


def benjamini_hochberg(pvals, alpha: float = 0.05):
    """
    BH step-up FDR control. Returns (rejected, qvalues).

    A sweep of this size runs hundreds of paired tests. At a nominal 5% roughly
    one cell in twenty will look significant with no effect present, and a reader
    scanning the table will find exactly those cells. Controlling the false
    discovery rate is what makes a per-cell claim survivable; it is not optional
    once the grid is larger than a handful.
    """
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.zeros(0, bool), np.zeros(0)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]        # enforce monotonicity
    q = np.clip(q, 0, 1)
    out_q = np.empty(n); out_q[order] = q
    return out_q <= alpha, out_q


HIGHER_IS_BETTER = ("acc", "balanced_acc", "R2", "MC")


def equivalence_verdict(effect_lo: float, effect_hi: float, margin: float) -> str:
    """
    Two one-sided-test style verdict on a signed effect (positive = arm A better).

    'equivalent' is the scientifically strong negative statement: the CI lies
    entirely inside +/-margin, so a difference large enough to matter has been
    RULED OUT. 'inconclusive' means the data cannot distinguish that from a real
    effect -- which is where an underpowered study lands, and reporting it as a
    negative result would be wrong.
    """
    if not np.isfinite(effect_lo) or not np.isfinite(effect_hi):
        return "undetermined"
    if effect_lo > margin:
        return "superior"
    if effect_hi < -margin:
        return "inferior"
    if effect_lo > -margin and effect_hi < margin:
        return "equivalent"
    return "inconclusive"


def paired(df: pd.DataFrame, metric: str, arm_a: str = "quantum",
           arm_b: str = "random_unitary", n_boot: int = 10000,
           margin: float | None = None, alpha: float = 0.05) -> pd.DataFrame:
    """
    Within-seed differences (arm_a - arm_b), averaged over seeds.

    `wins` counts seeds where arm_a came out ahead, which for a metric where
    lower is better (NRMSE) means a NEGATIVE difference -- handled below so the
    column always means "arm_a was better".
    """
    lower_better = metric in ("NRMSE", "RMSE", "MAE")
    d = df[(df["metric"] == metric) & (df["arm"].isin([arm_a, arm_b]))]
    if d.empty:
        return pd.DataFrame()
    d, keys = _keyed(d)
    wide = (d.pivot_table(index=keys + ["seed"], columns="arm", values="value",
                          observed=True, aggfunc="mean", dropna=False)
             .reset_index())
    if arm_a not in wide.columns or arm_b not in wide.columns:
        return pd.DataFrame()
    wide = wide.dropna(subset=[arm_a, arm_b])
    wide["delta"] = wide[arm_a] - wide[arm_b]
    wide["a_better"] = wide["delta"] < 0 if lower_better else wide["delta"] > 0

    sign = 1.0 if metric in HIGHER_IS_BETTER else -1.0
    out = []
    for key, grp in wide.groupby(keys, dropna=False, observed=True):
        vals = grp["delta"].to_numpy()
        lo, hi, pval = _bootstrap_ci(vals, n_boot=n_boot, alpha=alpha)
        rec = dict(zip(keys, key if isinstance(key, tuple) else (key,)))
        n = len(vals)
        sd = float(np.std(vals, ddof=1)) if n > 1 else float("nan")
        # signed effect: positive always means "arm_a is better", whichever
        # direction the metric runs, so verdicts read the same way everywhere
        e_lo, e_hi = sorted((sign * lo, sign * hi))
        rec.update(metric=metric, arm_a=arm_a, arm_b=arm_b, n_seeds=n,
                   mean_a=float(grp[arm_a].mean()), mean_b=float(grp[arm_b].mean()),
                   mean_delta=float(np.mean(vals)),
                   effect=float(sign * np.mean(vals)),
                   sem_delta=sd / np.sqrt(n) if n > 1 else float("nan"),
                   ci_lo=lo, ci_hi=hi, effect_lo=e_lo, effect_hi=e_hi,
                   p_value=pval,
                   wins=int(grp["a_better"].sum()),
                   significant=bool(n > 1 and (lo > 0 or hi < 0)))
        if margin is not None:
            rec["margin"] = float(margin)
            rec["verdict"] = equivalence_verdict(e_lo, e_hi, float(margin))
        out.append(rec)
    t = pd.DataFrame(out)
    if not t.empty:
        rej, q = benjamini_hochberg(t["p_value"].to_numpy(), alpha=alpha)
        t["q_value"] = q
        t["significant_bh"] = rej
    return t


def learnable_mask(df: pd.DataFrame, metric: str = "NRMSE",
                   floor_arm: str | None = None) -> pd.DataFrame:
    """
    Flag configurations where at least one arm actually beats the trivial floor.

    On the equity and option panels every arm sits at or above NRMSE 1, i.e. no
    model beats predicting the training mean. A "win" between two models that
    both fail is not a result, and reporting one invites a reader to believe the
    task was learned. This marks such cells so they can be excluded from any
    claim while remaining visible in the tables.
    """
    higher = metric in HIGHER_IS_BETTER
    if floor_arm is None:
        floor_arm = "majority" if higher else "mean"
    d, keys = _keyed(df[df["metric"] == metric])
    if d.empty:
        return pd.DataFrame()
    arms = ["quantum", "random_unitary", "linear_raw", "raw_only", "rff", "rbf_svm"]
    g = d[d["arm"].isin(arms)].groupby(keys, dropna=False, observed=True)["value"]
    best = (g.max() if higher else g.min()).rename("best_arm_value")
    floor = (d[d["arm"] == floor_arm]
             .groupby(keys, dropna=False, observed=True)["value"].mean()
             .rename("floor_value"))
    t = pd.concat([best, floor], axis=1).reset_index()
    ref = t["floor_value"].where(t["floor_value"].notna(), 0.0 if higher else 1.0)
    t["learnable"] = (t["best_arm_value"] > ref) if higher else (t["best_arm_value"] < ref)
    return t


def primary_result(df: pd.DataFrame, primary: dict, n_boot: int = 10000,
                   alpha: float = 0.05) -> dict:
    """
    Evaluate the single pre-declared comparison.

    Everything else in the sweep is exploratory and BH-corrected; this one was
    named before the data existed, so it needs no multiplicity adjustment and it
    is the only cell entitled to be read as confirmatory. Seeds are pooled across
    the remaining grid dimensions by first averaging within seed, which keeps the
    replicate structure (one number per seed) rather than treating correlated
    configurations as independent observations.
    """
    fam = primary["family"]
    metric = primary["metric"]
    a, b = primary.get("arm_a", "quantum"), primary.get("arm_b", "random_unitary")
    margin = float(primary["margin"])

    sub = df[(df["family"] == fam) & (df["metric"] == metric)]
    want_control = (primary.get("params") or {}).get("control")
    sub = sub if want_control else drop_controls(sub)
    if primary.get("task_key"):
        sub = sub[sub["task_key"] == primary["task_key"]]
    for k, v in (primary.get("params") or {}).items():
        if k in sub.columns:
            sub = sub[sub[k] == v]
    sub = sub[sub["arm"].isin([a, b])]
    if sub.empty:
        return {"status": "no_data", "primary": primary}

    wide = sub.pivot_table(index="seed", columns="arm", values="value",
                           observed=True, aggfunc="mean")
    if a not in wide.columns or b not in wide.columns:
        return {"status": "missing_arm", "primary": primary}
    wide = wide.dropna(subset=[a, b])
    delta = (wide[a] - wide[b]).to_numpy()
    lo, hi, pval = _bootstrap_ci(delta, n_boot=n_boot, alpha=alpha)
    sign = 1.0 if metric in HIGHER_IS_BETTER else -1.0
    e_lo, e_hi = sorted((sign * lo, sign * hi))
    return {
        "status": "ok", "primary": primary, "n_seeds": int(len(delta)),
        "mean_a": float(wide[a].mean()), "mean_b": float(wide[b].mean()),
        "mean_delta": float(delta.mean()), "effect": float(sign * delta.mean()),
        "ci_lo": lo, "ci_hi": hi, "effect_lo": e_lo, "effect_hi": e_hi,
        "p_value": pval, "margin": margin,
        "verdict": equivalence_verdict(e_lo, e_hi, margin),
    }


def required_seeds(paired_sd: float, margin: float, power: float = 0.8,
                   alpha: float = 0.05) -> int:
    """
    Seeds needed to detect a paired difference of `margin` at the given power.

    Normal approximation on the paired differences: n = ((z_a/2 + z_b) * sd / d)^2.
    Approximate, but the point is to discover that a planned sweep is
    underpowered BEFORE spending the compute, not to be exact at the margin.
    """
    from scipy.stats import norm as _norm
    if not np.isfinite(paired_sd) or paired_sd <= 0 or margin <= 0:
        return 0
    z = _norm.ppf(1 - alpha / 2) + _norm.ppf(power)
    return int(np.ceil((z * paired_sd / margin) ** 2))


def headline(df: pd.DataFrame, margins: dict | None = None,
             primary: dict | None = None) -> dict:
    """
    Sweep-level summary: per family, how often the quantum arm actually wins.

    Reports raw wins, BH-corrected significant wins, and -- where a margin is
    supplied -- how many configurations are positively EQUIVALENT, meaning a
    difference worth caring about has been ruled out rather than merely not
    detected.
    """
    comparators = {
        "classification": ["random_unitary", "rff", "rbf_svm"],
        "timeseries": ["random_unitary", "linear_raw", "raw_only", "raw_plus_random_proj"],
        "streaming": ["random_unitary", "linear_raw", "persistence"],
    }
    default_margin = {"acc": 0.01, "balanced_acc": 0.01, "NRMSE": 0.02, "R2": 0.02}
    margins = {**default_margin, **(margins or {})}

    controls = control_report(df)
    df = drop_controls(df)

    out = {}
    for family, sub in df.groupby("family", observed=True):
        metric = "acc" if family == "classification" else "NRMSE"
        entry = {"metric": metric, "margin": margins.get(metric),
                 "n_seeds": int(sub["seed"].nunique()),
                 "n_units": int(sub["unit_id"].nunique())}
        for arm in comparators.get(family, ["random_unitary"]):
            t = paired(sub, metric, "quantum", arm, margin=margins.get(metric))
            if t.empty:
                continue
            better = t["effect"] > 0
            cell = {"n_configs": int(len(t)), "wins": int(better.sum()),
                    "significant_wins": int((better & t["significant"]).sum()),
                    "significant_wins_bh": int((better & t.get("significant_bh", False)).sum())}
            if "verdict" in t:
                cell["verdicts"] = {k: int(v) for k, v in
                                    t["verdict"].value_counts().to_dict().items()}
            entry[f"vs_{arm}"] = cell
        if family in ("timeseries", "streaming"):
            lm = learnable_mask(sub, metric)
            if not lm.empty:
                entry["learnable_configs"] = f"{int(lm['learnable'].sum())}/{len(lm)}"
        out[family] = entry

    if controls:
        out["_negative_controls"] = controls
    if primary:
        out["_primary"] = primary_result(df, primary)
    return out
