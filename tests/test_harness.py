"""
Harness tests: the sweep infrastructure, not the physics.

These cover the failure modes that are silent at scale -- a duplicated unit id
would overwrite a result, an unbalanced shard would drop work, a NaN grouping
key would empty a comparison table, and a monkeypatched control would leak
between concurrent units. Every one of them produces plausible-looking output
rather than an error, which is exactly why they need tests.
"""
import json
import numpy as np
import pandas as pd
import pytest

from qrc import aggregate as agg
from qrc import spec, store
from qrc.runner import derive_seed, model_kwargs_for


CFG = {
    "name": "unit_test",
    "experiments": [
        {"family": "timeseries",
         "tasks": [{"dataset": "narma10", "n": 400},
                   {"dataset": "stock_cohort", "scheme": "vol", "index": 1, "n_cohorts": 3}],
         "seeds": [0, 1, 2],
         "grid": {"W": [10], "num_memory": [2, 4], "model": ["XXZ", "IAA_CHAOTIC"],
                  "total_time": [0.2, 8.75], "encoding": ["continuous"], "shots": [0]}},
        {"family": "classification",
         "tasks": [{"dataset": "digits"}],
         "seeds": [0, 1, 2],
         "grid": {"n_pca": [8], "num_memory": [2], "model": ["XXZ"],
                  "n_reupload": [1, 2], "total_time": [0.2], "shots": [0]}},
    ],
}


# ---------------------------------------------------------------------------
# spec
# ---------------------------------------------------------------------------

def test_unit_ids_unique_and_stable():
    a = spec.build_units(CFG)
    b = spec.build_units(json.loads(json.dumps(CFG)))
    ids = [u["id"] for u in a]
    assert len(set(ids)) == len(ids)
    assert ids == [u["id"] for u in b]


def test_unit_id_changes_with_every_field_that_changes_the_numbers():
    # units sort classification-first, so [0] is a classification unit; flip the
    # family to the other one so the mutation is actually a change
    base = spec.build_units(CFG)[0]
    assert base["family"] == "classification"
    for field, value in (("seed", 999), ("family", "timeseries")):
        other = dict(base, **{field: value})
        assert spec.unit_id(other) != spec.unit_id(base)
    other = dict(base, params=dict(base["params"], num_memory=7))
    assert spec.unit_id(other) != spec.unit_id(base)
    other = dict(base, task=dict(base["task"], n=99999))
    assert spec.unit_id(other) != spec.unit_id(base)


def test_unit_id_ignores_presentation_only_fields():
    base = spec.build_units(CFG)[0]
    assert spec.unit_id(dict(base, task_key="anything")) == spec.unit_id(base)


@pytest.mark.parametrize("n", [1, 2, 3, 7, 16, 97, 500])
def test_shards_are_a_balanced_exact_partition(n):
    units = spec.build_units(CFG)
    parts = [spec.shard(units, i, n) for i in range(n)]
    flat = [u["id"] for p in parts for u in p]
    assert sorted(flat) == sorted(u["id"] for u in units)
    sizes = [len(p) for p in parts]
    assert max(sizes) - min(sizes) <= 1


def test_unknown_grid_key_is_rejected():
    bad = json.loads(json.dumps(CFG))
    bad["experiments"][0]["grid"]["totl_time"] = [0.2]     # typo
    with pytest.raises(ValueError, match="unknown grid keys"):
        spec.build_units(spec._validate(bad["experiments"][0], 0) or bad)


def test_unknown_task_key_is_rejected():
    bad = json.loads(json.dumps(CFG))
    bad["experiments"][0]["tasks"][0]["n_sampels"] = 10
    with pytest.raises(ValueError, match="unknown keys"):
        spec._validate(bad["experiments"][0], 0)


def test_empty_seeds_rejected():
    bad = json.loads(json.dumps(CFG))
    bad["experiments"][0]["seeds"] = []
    with pytest.raises(ValueError, match="seeds"):
        spec._validate(bad["experiments"][0], 0)


# ---------------------------------------------------------------------------
# store
# ---------------------------------------------------------------------------

def test_store_roundtrip_and_resume(tmp_path):
    u = spec.build_units(CFG)[0]
    assert not store.is_done(u, tmp_path)
    store.write_record(u, {"status": "ok", "id": u["id"]}, tmp_path)
    assert store.is_done(u, tmp_path)
    assert len(list(store.iter_records(u["config"], tmp_path))) == 1


def test_failed_record_does_not_count_as_done(tmp_path):
    u = spec.build_units(CFG)[0]
    store.write_record(u, {"status": "failed", "error": "boom"}, tmp_path)
    assert not store.is_done(u, tmp_path)


def test_truncated_file_is_not_done(tmp_path):
    u = spec.build_units(CFG)[0]
    p = store.unit_path(u, tmp_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text('{"status": "ok"')          # truncated mid-write
    assert not store.is_done(u, tmp_path)


# ---------------------------------------------------------------------------
# seeds
# ---------------------------------------------------------------------------

def test_derived_seeds_are_independent_and_reproducible():
    assert derive_seed(0, "haar") == derive_seed(0, "haar")
    assert derive_seed(0, "haar") != derive_seed(0, "shots")
    assert derive_seed(0, "haar") != derive_seed(1, "haar")


def test_disorder_realisation_tracks_the_seed():
    assert model_kwargs_for("XXZ", 0) == {}
    a, b = model_kwargs_for("NNN_CHAOTIC", 0), model_kwargs_for("NNN_CHAOTIC", 1)
    assert a["use_random"] and a["seed"] != b["seed"]


# ---------------------------------------------------------------------------
# aggregation
# ---------------------------------------------------------------------------

def _fake_records(units, q_better: bool):
    rng = np.random.default_rng(0)
    out = []
    for u in units:
        if u["family"] == "timeseries":
            q = 0.75 + 0.01 * rng.standard_normal()
            r = q + (0.05 if q_better else -0.05)     # NRMSE: lower is better
            arms = {"quantum": {"NRMSE": q}, "random_unitary": {"NRMSE": r},
                    "raw_only": {"NRMSE": 0.7}, "raw_plus_reservoir": {"NRMSE": 0.71},
                    "baselines": {"linear_raw": {"NRMSE": 0.69}}}
        else:
            a = 0.88 + 0.01 * rng.standard_normal()
            b = a - (0.05 if q_better else -0.05)     # acc: higher is better
            arms = {"quantum": {"acc": a}, "random_unitary": {"acc": b},
                    "baselines": {"rff_40": {"acc": 0.9}, "rbf_svm": {"acc": 0.93}}}
        out.append(dict(status="ok", id=u["id"], config=u["config"], family=u["family"],
                        task=u["task"], task_key=u["task_key"], params=u["params"],
                        seed=u["seed"], L=3, n_features=40, seconds=0.1, **arms))
    return out


def test_tidy_normalises_rff_width_into_one_arm():
    units = spec.build_units(CFG)
    df = agg.tidy(_fake_records(units, q_better=True))
    assert "rff" in set(df["arm"])
    assert not any(str(a).startswith("rff_") for a in df["arm"].unique())


def test_failed_records_are_excluded():
    units = spec.build_units(CFG)[:3]
    recs = _fake_records(units, True)
    recs.append(dict(status="failed", id="x", config="c", family="timeseries",
                     task={}, task_key="t", params={}, seed=0, error="boom"))
    assert agg.tidy(recs)["unit_id"].nunique() == 3


def test_paired_survives_mixed_families():
    """Regression: all-NaN parameter columns used to empty every comparison."""
    units = spec.build_units(CFG)
    df = agg.tidy(_fake_records(units, q_better=True))
    assert df["family"].nunique() == 2
    for fam, metric in (("timeseries", "NRMSE"), ("classification", "acc")):
        t = agg.paired(df[df.family == fam], metric, "quantum", "random_unitary")
        assert not t.empty, f"{fam}: paired table empty on a mixed frame"
        assert (t["n_seeds"] == 3).all()


@pytest.mark.parametrize("fam,metric", [("timeseries", "NRMSE"), ("classification", "acc")])
def test_wins_respect_metric_direction(fam, metric):
    """'wins' must mean 'quantum was better', for both directions of metric."""
    units = spec.build_units(CFG)
    for q_better in (True, False):
        df = agg.tidy(_fake_records(units, q_better=q_better))
        t = agg.paired(df[df.family == fam], metric, "quantum", "random_unitary")
        expected = t["n_seeds"] if q_better else 0
        assert (t["wins"] == expected).all()


def test_headline_uses_family_appropriate_comparators():
    units = spec.build_units(CFG)
    h = agg.headline(agg.tidy(_fake_records(units, q_better=True)))
    assert "vs_rff" in h["classification"]
    assert "vs_linear_raw" in h["timeseries"]
    assert "vs_rff" not in h["timeseries"]        # RFF is not fitted for timeseries


def test_bootstrap_ci_brackets_the_mean():
    lo, hi, p = agg._bootstrap_ci(np.array([0.1, 0.11, 0.09, 0.105, 0.095]))
    assert lo < 0.1 < hi
    assert p < 0.05          # a mean far from zero must give a small p-value


# ---------------------------------------------------------------------------
# representation metrics
# ---------------------------------------------------------------------------

def test_effective_rank_recovers_a_known_dimension():
    """A matrix built from k independent directions must have PR close to k."""
    from qrc.metrics import effective_rank
    rng = np.random.default_rng(0)
    Z = rng.standard_normal((2000, 4))
    F = Z @ rng.standard_normal((4, 40))          # 40 columns, rank 4
    er = effective_rank(F)
    assert er["nominal"] == 40
    assert 3.5 < er["participation_ratio"] < 4.5


def test_effective_rank_of_independent_columns_is_the_width():
    from qrc.metrics import effective_rank
    rng = np.random.default_rng(0)
    er = effective_rank(rng.standard_normal((4000, 10)))
    assert er["participation_ratio"] > 8.5


def test_entropy_column_layout_matches_the_reservoir():
    from qrc.metrics import entropy_column_index
    idx, per_step = entropy_column_index(5)          # L=5: 4 cuts + 5 sigma_z
    assert per_step == 9 and idx.tolist() == [0, 1, 2, 3]
    idx, per_step = entropy_column_index(5, use_entropy=False)
    assert per_step == 5 and idx.tolist() == []


def test_entropy_stats_are_bounded_by_the_page_value():
    from qrc.metrics import entropy_stats
    from qrc.ppe import reservoir_features
    rng = np.random.default_rng(0)
    F = reservoir_features(rng.uniform(-1, 1, size=(64, 6)), "XXZ", num_memory=4,
                           dt=0.04, n_steps=5, washout_length=0, encoding="continuous")
    st = entropy_stats(F, 5)
    assert 0.0 <= st["S_mean"] <= st["S_max"]
    assert 0.0 <= st["S_frac_max"] <= 1.0


# ---------------------------------------------------------------------------
# streaming
# ---------------------------------------------------------------------------

def test_lag_matrix_row_order_is_series_major():
    """Row i of the baseline design must describe the same instant as row i of the state."""
    from qrc.streaming import _lag_matrix
    Y = np.arange(24, dtype=float).reshape(2, 12)
    Lg = _lag_matrix(Y, np.array([3, 4, 5]), 3)
    assert Lg.shape == (6, 3)
    assert np.allclose(Lg[0], [1, 2, 3])
    assert np.allclose(Lg[3], [13, 14, 15])


def test_streaming_state_is_not_reset_between_timesteps():
    """
    The defining property of the streaming protocol.

    Driving [a, b] must leave a different state than driving [c, b]: if the
    reservoir were reset each step, only the last input would matter.
    """
    from qrc.streaming import drive
    U = np.array([[0.9, -0.4], [-0.9, -0.4]])
    S = drive(U, "XXZ", num_memory=2, dt=0.06, n_steps=5)
    assert not np.allclose(S[0, -1], S[1, -1], atol=1e-8)


def test_time_multiplexing_widens_the_readout():
    from qrc.streaming import forecast
    from qrc.datasets import narma10
    s = narma10(n=400, seed=0)
    a = forecast(s, "XXZ", num_memory=2, total_time=0.2, washout=30, n_lags=5, n_tap=1)
    b = forecast(s, "XXZ", num_memory=2, total_time=0.2, washout=30, n_lags=5, n_tap=4)
    assert b["reservoir"]["n_features"] == 4 * a["reservoir"]["n_features"]


def test_memory_capacity_of_a_unitary_reservoir_is_near_zero():
    """
    The structural claim. A unitary map is measure preserving, so past inputs are
    scrambled into global correlations rather than retained locally, and a linear
    readout on local observables recovers almost nothing.
    """
    from qrc.streaming import memory_capacity
    mc = memory_capacity("XXZ", num_memory=2, total_time=0.2, T=500, washout=50, max_lag=8)
    assert len(mc["mc_curve"]) == 8
    assert mc["MC"] < 0.5


def test_streaming_rejects_a_drive_that_is_too_short():
    from qrc.streaming import forecast
    with pytest.raises(ValueError, match="too short"):
        forecast([np.arange(12, dtype=float)], "XXZ", num_memory=2,
                 total_time=0.2, washout=5, n_lags=3)


# ---------------------------------------------------------------------------
# entropy calibration
# ---------------------------------------------------------------------------

def test_calibration_reports_unreachable_targets_instead_of_pretending():
    from qrc.calibrate import calibrate
    c = calibrate("XXZ", 2, 0.99, n_probe=32, W=4)
    assert c["bracketed"] is False
    assert c["achieved_frac"] < 0.99
    assert c["reachable_range"][0] <= c["achieved_frac"] <= c["reachable_range"][1]


def test_calibration_hits_a_reachable_target():
    from qrc.calibrate import calibrate
    c = calibrate("XXZ", 2, 0.35, n_probe=64, W=4)
    assert c["bracketed"] is True
    assert abs(c["achieved_frac"] - 0.35) < 0.05


# ---------------------------------------------------------------------------
# statistics
# ---------------------------------------------------------------------------

def test_benjamini_hochberg_matches_a_worked_example():
    rej, q = agg.benjamini_hochberg([0.001, 0.02, 0.3, 0.5, 0.9], alpha=0.05)
    assert rej.tolist() == [True, True, False, False, False]
    assert np.isclose(q[0], 0.005) and np.isclose(q[1], 0.05)
    assert np.all(np.diff(q[np.argsort([0.001, 0.02, 0.3, 0.5, 0.9])]) >= -1e-12)


def test_benjamini_hochberg_controls_discoveries_under_the_global_null():
    """
    100 uniform p-values means nothing is real. Uncorrected testing at alpha=0.05
    flags ~5 of them; BH must flag essentially none. (100 identical p=0.04 values
    would legitimately all be rejected -- that pattern is not the global null.)
    """
    rng = np.random.default_rng(0)
    n_uncorrected, n_bh = 0, 0
    for _ in range(20):
        p = rng.uniform(size=100)
        n_uncorrected += int((p < 0.05).sum())
        n_bh += int(agg.benjamini_hochberg(p, alpha=0.05)[0].sum())
    assert n_uncorrected > 50          # ~5 per replicate, as expected by chance
    assert n_bh <= 5                   # BH suppresses nearly all of them


@pytest.mark.parametrize("lo,hi,expected", [
    (0.02, 0.05, "superior"),
    (-0.005, 0.005, "equivalent"),
    (-0.05, -0.02, "inferior"),
    (-0.5, 0.5, "inconclusive"),
])
def test_equivalence_verdicts(lo, hi, expected):
    assert agg.equivalence_verdict(lo, hi, 0.01) == expected


def test_equivalence_is_not_the_same_as_failing_to_reject():
    """A wide CI around zero is 'inconclusive', never 'equivalent'."""
    assert agg.equivalence_verdict(-0.4, 0.4, 0.01) == "inconclusive"
    assert agg.equivalence_verdict(-0.002, 0.002, 0.01) == "equivalent"


def test_required_seeds_grows_with_variance():
    a = agg.required_seeds(0.01, 0.01)
    b = agg.required_seeds(0.03, 0.01)
    assert 1 < a < b


def test_learnability_respects_metric_direction():
    df = pd.DataFrame([
        {"family": "timeseries", "task_key": "t", "seed": 0, "metric": "NRMSE",
         "arm": "quantum", "value": 0.5, "unit_id": "u"},
        {"family": "timeseries", "task_key": "t", "seed": 0, "metric": "NRMSE",
         "arm": "mean", "value": 1.0, "unit_id": "u"},
    ])
    assert bool(agg.learnable_mask(df, "NRMSE")["learnable"].iloc[0]) is True
    df.loc[df["arm"] == "quantum", "value"] = 1.5
    assert bool(agg.learnable_mask(df, "NRMSE")["learnable"].iloc[0]) is False

    dfa = pd.DataFrame([
        {"family": "classification", "task_key": "t", "seed": 0, "metric": "acc",
         "arm": "quantum", "value": 0.9, "unit_id": "u"},
        {"family": "classification", "task_key": "t", "seed": 0, "metric": "acc",
         "arm": "majority", "value": 0.1, "unit_id": "u"},
    ])
    assert bool(agg.learnable_mask(dfa, "acc")["learnable"].iloc[0]) is True


def _control_frame():
    rows = []
    for ctrl, val in (("none", 0.9), ("shuffle_labels", 0.1)):
        for seed in range(4):
            for arm in ("quantum", "random_unitary"):
                rows.append({"family": "classification", "task_key": "t", "seed": seed,
                             "metric": "acc", "arm": arm, "value": val,
                             "control": ctrl, "unit_id": f"{ctrl}{seed}{arm}"})
    return pd.DataFrame(rows)


def test_controls_are_excluded_from_summaries():
    """Regression: shuffled units used to be pooled into the primary endpoint."""
    df = _control_frame()
    assert agg.drop_controls(df)["control"].unique().tolist() == ["none"]
    res = agg.primary_result(df, {"family": "classification", "metric": "acc",
                                  "arm_a": "quantum", "arm_b": "random_unitary",
                                  "margin": 0.01})
    assert res["status"] == "ok"
    assert np.isclose(res["mean_a"], 0.9)      # not the 0.5 average with the controls


def test_control_report_surfaces_the_shuffled_arms():
    rep = agg.control_report(_control_frame())
    assert rep["classification"]["shuffle_labels"]["quantum"] == 0.1
    assert "none" not in rep.get("classification", {})


def test_paired_reports_pvalues_and_bh():
    units = spec.build_units(CFG)
    df = agg.tidy(_fake_records(units, q_better=True))
    t = agg.paired(df[df.family == "timeseries"], "NRMSE", "quantum",
                   "random_unitary", margin=0.02)
    for col in ("p_value", "q_value", "significant_bh", "effect", "verdict"):
        assert col in t.columns
    assert (t["q_value"] >= t["p_value"] - 1e-12).all()


def test_effect_sign_is_better_is_positive_for_both_directions():
    units = spec.build_units(CFG)
    df = agg.tidy(_fake_records(units, q_better=True))
    for fam, metric in (("timeseries", "NRMSE"), ("classification", "acc")):
        t = agg.paired(df[df.family == fam], metric, "quantum", "random_unitary")
        assert (t["effect"] > 0).all(), f"{fam}: 'better' must be a positive effect"


# ---------------------------------------------------------------------------
# spec: new families and validation
# ---------------------------------------------------------------------------

def test_streaming_family_expands():
    cfg = {"name": "s", "experiments": [
        {"family": "streaming", "tasks": [{"dataset": "narma10", "n": 500}],
         "seeds": [0, 1], "grid": {"num_memory": [2], "model": ["XXZ"],
                                   "total_time": [0.2], "n_tap": [1, 4]}}]}
    units = spec.build_units(cfg)
    assert len(units) == 4 and all(u["family"] == "streaming" for u in units)


def test_total_time_and_entropy_target_are_mutually_exclusive():
    exp = {"family": "classification", "tasks": [{"dataset": "digits"}], "seeds": [0],
           "grid": {"n_pca": [8], "num_memory": [2], "model": ["XXZ"],
                    "total_time": [0.2], "entropy_target": [0.4]}}
    with pytest.raises(ValueError, match="not both"):
        spec._validate(exp, 0)


def test_a_time_specification_is_required():
    exp = {"family": "classification", "tasks": [{"dataset": "digits"}], "seeds": [0],
           "grid": {"n_pca": [8], "num_memory": [2], "model": ["XXZ"]}}
    with pytest.raises(ValueError, match="total_time or entropy_target"):
        spec._validate(exp, 0)


def test_unknown_control_is_rejected():
    exp = {"family": "classification", "tasks": [{"dataset": "digits"}], "seeds": [0],
           "grid": {"n_pca": [8], "num_memory": [2], "model": ["XXZ"],
                    "total_time": [0.2], "control": ["shufle_labels"]}}
    with pytest.raises(ValueError, match="control must be one of"):
        spec._validate(exp, 0)


def test_primary_endpoint_requires_a_margin():
    with pytest.raises(ValueError, match="margin"):
        spec._validate_primary({"family": "classification", "metric": "acc",
                                "arm_a": "quantum", "arm_b": "rff"})
    with pytest.raises(ValueError, match="positive number"):
        spec._validate_primary({"family": "classification", "metric": "acc",
                                "arm_a": "quantum", "arm_b": "rff", "margin": 0})


def test_every_shipped_config_is_valid():
    """Every config in configs/ must load, validate and expand."""
    import glob
    seen = 0
    for f in sorted(glob.glob("configs/*.json")):
        cfg = spec.load_config(f)
        units = spec.build_units(cfg)
        assert units, f"{f} expanded to nothing"
        assert len({u["id"] for u in units}) == len(units), f"{f} has duplicate ids"
        seen += 1
    assert seen >= 5


# ---------------------------------------------------------------------------
# evaluate: split hygiene
# ---------------------------------------------------------------------------

def test_embargo_removes_windows_at_a_temporal_cut():
    from qrc.evaluate import make_split
    sid = np.zeros(100, dtype=int)
    tr0, te0 = make_split(sid, train_frac=0.8, embargo=0)
    tr1, te1 = make_split(sid, train_frac=0.8, embargo=10)
    assert tr0.sum() + te0.sum() == 100          # no gap
    assert tr1.sum() + te1.sum() == 90           # 10 windows dropped
    assert not (tr1 & te1).any()


def test_embargo_does_not_apply_to_a_by_series_split():
    from qrc.evaluate import make_split
    sid = np.repeat(np.arange(10), 5)
    tr, te = make_split(sid, embargo=3, seed=0)
    assert tr.sum() + te.sum() == len(sid)


def test_nrmse_uses_the_reference_sd_when_given():
    from qrc.evaluate import score
    y = np.array([0.0, 1.0, 2.0, 3.0])
    s = score(y, y + 0.5, sd_ref=1.0)
    assert np.isclose(s["NRMSE"], 0.5)
    assert not np.isclose(s["NRMSE"], s["NRMSE_test"])
