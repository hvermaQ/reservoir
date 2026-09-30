"""
datasets.py — evaluation corpora for the quantum reservoir.

Two families:

  Empirical (data/2013-06/): a single month of US equity data, so every series
  is capped at 20 daily observations. Breadth therefore comes from POOLING
  across many series (3895 tickers, ~414k option contracts) rather than from
  long histories. Window length is bounded by ~series_len - 2.

  Synthetic: NARMA-10, Mackey-Glass and Lorenz can be generated to arbitrary
  length, so they are the only sources here that can exercise long windows and
  long memory. NARMA-10 in particular has a known, exactly 10-step memory
  requirement, which makes it the right instrument for validating the memory
  measurements in the reservoir itself.

Every loader returns a list of 1-D float arrays (one per series) so that
downstream windowing can pool across series while still respecting boundaries.
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "2013-06"


# ---------------------------------------------------------------------------
# Empirical: equity close prices -> log returns
# ---------------------------------------------------------------------------

def load_stock_series(min_len: int = 20, folder: Path | None = None
                      ) -> tuple[list[np.ndarray], list[str]]:
    """
    Every usable per-ticker log-return series, with its ticker symbol.

    Unsubsampled and seed-independent, so it can be cached once and then carved
    into cohorts. Note the hard limit imposed by the corpus: data/2013-06 is a
    single month, so each ticker yields at most ~19 daily returns. Breadth must
    therefore come from the number of tickers, never from series length -- a
    per-ticker task is not viable at W=10 and cohorts are the only honest way to
    treat "multiple stocks" as multiple tasks.
    """
    src = Path(folder) if folder is not None else DATA_DIR
    files = sorted(glob.glob(str(src / "*stocks.csv")))
    if not files:
        raise FileNotFoundError(f"no *stocks.csv under {src}")
    frames = []
    for f in files:
        d = pd.read_csv(f, usecols=["symbol", "close"])
        d["date"] = pd.to_datetime(os.path.basename(f).split("stocks")[0])
        frames.append(d)
    st = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"])

    series, symbols = [], []
    for sym, grp in st.groupby("symbol", sort=True):
        px = grp["close"].to_numpy(dtype=float)
        if len(px) < min_len or not np.all(np.isfinite(px)) or np.any(px <= 0):
            continue
        r = np.diff(np.log(px))
        if np.std(r) < 1e-12:
            continue
        series.append(r)
        symbols.append(str(sym))
    return series, symbols


def load_stock_returns(min_len: int = 20, max_series: int | None = 1500,
                       folder: Path | None = None, seed: int = 0) -> list[np.ndarray]:
    """Per-ticker daily log-return series, pooled across tickers."""
    series, _ = load_stock_series(min_len=min_len, folder=folder)
    if max_series is not None and len(series) > max_series:
        idx = np.random.default_rng(seed).choice(len(series), max_series, replace=False)
        series = [series[i] for i in sorted(idx)]
    return series


# --- multiple stock tasks -------------------------------------------------

STOCK_COHORT_SCHEMES = ("vol", "disjoint", "bootstrap")


def stock_cohort(scheme: str = "vol", index: int = 0, n_cohorts: int = 3,
                 min_len: int = 20, max_series: int | None = 600,
                 folder: Path | None = None, seed: int = 0) -> list[np.ndarray]:
    """
    One cohort of tickers, treated as its own time-series task.

    scheme='vol'       tickers sorted by realised volatility, split into
                       `n_cohorts` equal buckets; index 0 is the calmest. This is
                       the scientifically interesting axis -- if the reservoir
                       ever helps, the prior is that it helps on the noisiest
                       cohort, and this is what tests that directly.
    scheme='disjoint'  deterministic non-overlapping partition by ticker (a
                       seed-independent hash), so cohorts share no ticker and
                       results across them are genuinely independent replicates.
    scheme='bootstrap' resample tickers WITH replacement using `seed`; a
                       different draw per seed, which turns seed variance into a
                       proper measure of ticker-sampling uncertainty.

    'vol' and 'disjoint' are fixed under `seed`, so with those the seed varies
    only the split and the model randomness -- which is what you want when the
    cohort itself is the experimental unit.
    """
    series, symbols = load_stock_series(min_len=min_len, folder=folder)
    n = len(series)
    if n == 0:
        raise ValueError("no usable stock series")
    if scheme == "vol":
        order = np.argsort([float(np.std(r)) for r in series], kind="stable")
        chunks = np.array_split(order, n_cohorts)
        sel = chunks[index % n_cohorts]
    elif scheme == "disjoint":
        # stable hash of the symbol -> cohort, independent of load order and seed
        import hashlib
        buck = np.array([int(hashlib.sha1(s.encode()).hexdigest(), 16) % n_cohorts
                         for s in symbols])
        sel = np.flatnonzero(buck == (index % n_cohorts))
    elif scheme == "bootstrap":
        rng = np.random.default_rng((seed, index))
        size = min(max_series or n, n)
        sel = rng.choice(n, size=size, replace=True)
    else:
        raise ValueError(f"unknown cohort scheme {scheme!r} (expected one of {STOCK_COHORT_SCHEMES})")

    sel = np.asarray(sel)
    if max_series is not None and len(sel) > max_series:
        # numpy seed sequences accept ints only -- a string in the tuple raises
        # "unrecognized seed string", and this path only runs when a cohort is
        # larger than max_series, so it is easy to miss in a small test.
        pick = np.random.default_rng([seed, index, 7717]).choice(
            len(sel), max_series, replace=False)
        sel = sel[np.sort(pick)]
    return [series[i] for i in sel]


# ---------------------------------------------------------------------------
# Empirical: option mid-price deviation from Black-Scholes
# ---------------------------------------------------------------------------

def _bs_call(S, K, T, r, sigma):
    """Vectorised Black-Scholes call price (NaN where inputs are invalid)."""
    S, K, T, sigma = map(np.asarray, (S, K, T, sigma))
    out = np.full(S.shape, np.nan)
    ok = (S > 0) & (K > 0) & (sigma > 0) & (T > 0) & np.isfinite(S) & np.isfinite(K) & np.isfinite(sigma)
    if not ok.any():
        return out
    s, k, t, v = S[ok], K[ok], T[ok], sigma[ok]
    d1 = (np.log(s / k) + (r + 0.5 * v ** 2) * t) / (v * np.sqrt(t))
    d2 = d1 - v * np.sqrt(t)
    out[ok] = s * norm.cdf(d1) - k * np.exp(-r * t) * norm.cdf(d2)
    return out


def load_option_deviations(min_len: int = 20, max_series: int | None = 800,
                           option_type: str = "call", r: float = 0.05,
                           folder: Path | None = None, seed: int = 0) -> list[np.ndarray]:
    """
    Per-contract (underlying, strike, expiry) deviation series: mid - Black-Scholes.

    Vectorised throughout; the original row-wise `.apply(axis=1)` in data_gen.py
    re-parsed timestamps per row and dominated load time.
    """
    src = Path(folder) if folder is not None else DATA_DIR
    ofiles = sorted(glob.glob(str(src / "*options.csv")))
    sfiles = sorted(glob.glob(str(src / "*stocks.csv")))
    if not ofiles or not sfiles:
        raise FileNotFoundError(f"missing *options.csv / *stocks.csv under {src}")

    sframes = []
    for f in sfiles:
        d = pd.read_csv(f, usecols=["symbol", "close"])
        d["date"] = pd.to_datetime(os.path.basename(f).split("stocks")[0])
        sframes.append(d)
    stocks = pd.concat(sframes, ignore_index=True).rename(columns={"close": "spot"})

    cols = ["underlying", "expiration", "type", "strike", "bid", "ask",
            "quote_date", "implied_volatility"]
    oframes = []
    for f in ofiles:
        d = pd.read_csv(f, usecols=cols)
        oframes.append(d[d["type"] == option_type])
    opts = pd.concat(oframes, ignore_index=True)
    opts["date"] = pd.to_datetime(opts["quote_date"])

    m = opts.merge(stocks, left_on=["underlying", "date"],
                   right_on=["symbol", "date"], how="inner")
    m["mid"] = (m["bid"] + m["ask"]) / 2.0
    ttm = (pd.to_datetime(m["expiration"]) - m["date"]).dt.days / 365.0
    m["T"] = np.maximum(ttm.to_numpy(dtype=float), 1e-6)
    m["bs"] = _bs_call(m["spot"], m["strike"], m["T"], r, m["implied_volatility"])
    m["dev"] = m["mid"] - m["bs"]
    m = m.dropna(subset=["dev"]).sort_values(["underlying", "strike", "expiration", "date"])

    series = []
    for _, grp in m.groupby(["underlying", "strike", "expiration"], sort=True):
        v = grp["dev"].to_numpy(dtype=float)
        if len(v) < min_len or np.std(v) < 1e-12:
            continue
        series.append(v)
        if max_series is not None and len(series) >= max_series * 4:
            break
    if max_series is not None and len(series) > max_series:
        idx = np.random.default_rng(seed).choice(len(series), max_series, replace=False)
        series = [series[i] for i in sorted(idx)]
    return series


# ---------------------------------------------------------------------------
# Synthetic reservoir-computing benchmarks
# ---------------------------------------------------------------------------

def narma10(n: int = 3000, seed: int = 0, burn: int = 200) -> list[np.ndarray]:
    """
    NARMA-10: the standard nonlinear-autoregressive RC benchmark.

        y[t+1] = 0.3 y[t] + 0.05 y[t] Σ_{i=0..9} y[t-i] + 1.5 u[t-9] u[t] + 0.1

    with u ~ U(0, 0.5). Its output depends explicitly on the last 10 inputs, so
    a reservoir that cannot retain 10 steps of history provably cannot fit it.
    """
    rng = np.random.default_rng(seed)
    N = n + burn + 10
    u = rng.uniform(0.0, 0.5, size=N)
    y = np.zeros(N)
    for t in range(9, N - 1):
        y[t + 1] = (0.3 * y[t] + 0.05 * y[t] * np.sum(y[t - 9:t + 1])
                    + 1.5 * u[t - 9] * u[t] + 0.1)
        if not np.isfinite(y[t + 1]) or abs(y[t + 1]) > 1e6:
            y[t + 1] = 0.0
    return [y[burn + 10:]]


def mackey_glass(n: int = 3000, tau: int = 17, seed: int = 0, burn: int = 500) -> list[np.ndarray]:
    """Mackey-Glass delay system (chaotic for tau=17), sampled at unit steps."""
    rng = np.random.default_rng(seed)
    beta, gamma, p, dt = 0.2, 0.1, 10, 1.0
    hist = 1.2 + 0.05 * rng.standard_normal(tau + 1)
    x = list(hist)
    for _ in range(n + burn):
        xt, xtau = x[-1], x[-tau - 1]
        x.append(xt + dt * (beta * xtau / (1.0 + xtau ** p) - gamma * xt))
    return [np.asarray(x[burn + tau + 1:], dtype=float)]


def lorenz_x(n: int = 3000, dt: float = 0.02, burn: int = 1000,
             seed: int | None = None) -> list[np.ndarray]:
    """
    x-component of the Lorenz attractor (sigma=10, rho=28, beta=8/3), RK4.

    `seed=None` reproduces the original fixed initial condition exactly. Passing
    an int jitters the initial condition, which is what makes a multi-seed sweep
    meaningful: with a fixed IC every seed re-simulates the identical trajectory
    and the only thing that varies is the split, so the error bars would be far
    too small and would not cover trajectory variability at all.
    """
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0

    def f(s):
        x, y, z = s
        return np.array([sigma * (y - x), x * (rho - z) - y, x * y - beta * z])

    s = np.array([1.0, 1.0, 1.0])
    if seed is not None:
        s = s + 0.05 * np.random.default_rng(seed).standard_normal(3)
    out = []
    for i in range(n + burn):
        k1 = f(s); k2 = f(s + dt * k1 / 2); k3 = f(s + dt * k2 / 2); k4 = f(s + dt * k3)
        s = s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if i >= burn:
            out.append(s[0])
    return [np.asarray(out, dtype=float)]


DATASETS = {
    "narma10":      lambda **kw: narma10(**kw),
    "mackey_glass": lambda **kw: mackey_glass(**kw),
    "lorenz":       lambda **kw: lorenz_x(**kw),
    "stocks":       lambda **kw: load_stock_returns(**kw),
    "stock_cohort": lambda **kw: stock_cohort(**kw),
    "options":      lambda **kw: load_option_deviations(**kw),
}


# ---------------------------------------------------------------------------
# Source classification: which dynamical system generated this window?
# ---------------------------------------------------------------------------

def rossler_x(n: int = 3000, dt: float = 0.08, burn: int = 1000,
              seed: int | None = None) -> np.ndarray:
    """x-component of the Rossler attractor (a=0.2, b=0.2, c=5.7), RK4.

    `seed=None` keeps the original fixed IC; an int jitters it (see `lorenz_x`).
    """
    a, b, c = 0.2, 0.2, 5.7

    def f(s):
        x, y, z = s
        return np.array([-y - z, x + a * y, b + z * (x - c)])

    s = np.array([1.0, 1.0, 1.0])
    if seed is not None:
        s = s + 0.05 * np.random.default_rng(seed).standard_normal(3)
    out = []
    for i in range(n + burn):
        k1 = f(s); k2 = f(s + dt * k1 / 2); k3 = f(s + dt * k2 / 2); k4 = f(s + dt * k3)
        s = s + dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        if i >= burn:
            out.append(s[0])
    return np.asarray(out, dtype=float)


def source_classification(W: int = 16, n_per_class: int = 2000, seed: int = 0,
                          noise: float = 0.0):
    """
    Windows drawn from four generators; the label is which one produced it.

    Each window is standardised individually, so amplitude and offset carry no
    information and the task can only be solved from temporal structure. That is
    what makes this a genuine reservoir benchmark, unlike static PCA features
    where the injection order is arbitrary.

    Classes: 0 Lorenz-x, 1 Rossler-x, 2 Mackey-Glass, 3 AR(1) noise.
    """
    rng = np.random.default_rng(seed)
    need = n_per_class * W + W
    series = [
        lorenz_x(n=need, seed=seed)[0],
        rossler_x(n=need, seed=seed),
        mackey_glass(n=need, seed=seed)[0],
        None,
    ]
    ar = np.zeros(need)
    for t in range(1, need):
        ar[t] = 0.85 * ar[t - 1] + rng.standard_normal()
    series[3] = ar

    Xs, ys = [], []
    for lab, s in enumerate(series):
        s = np.asarray(s, dtype=float)
        starts = rng.choice(len(s) - W - 1, n_per_class, replace=False)
        for st in starts:
            w = s[st:st + W].copy()
            if noise > 0:
                w = w + noise * np.std(w) * rng.standard_normal(W)
            sd = w.std()
            Xs.append((w - w.mean()) / (sd if sd > 1e-12 else 1.0))
            ys.append(lab)
    X, y = np.array(Xs), np.array(ys)
    p = rng.permutation(len(y))
    return X[p], y[p]
