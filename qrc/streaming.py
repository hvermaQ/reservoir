"""
streaming.py — the reservoir protocol proper: drive continuously, never reset.

Why this module exists. `reservoir_features` re-initialises the state for every
window, so the windowed experiments in `runner._run_timeseries` evaluate a
STATIC feature map on windows -- they contain no recurrence, no fading memory
and no echo-state property. Whatever they show, they cannot support a claim
about reservoir COMPUTING. This module supplies the missing protocol:

  * one continuous drive per series, state carried across every timestep
  * readout from the state at time t predicting the value at t+1
  * linear memory capacity MC_k (Jaeger), the standard RC memory measure

The distinction matters for the negative result in both directions. A unitary
reservoir is measure-preserving, so it has no fading memory at all: past inputs
are scrambled into global correlations rather than forgotten, and a linear
readout on local observables should recover very little. If that is what MC
shows, the windowed result stops looking like a property of these particular
tasks and starts looking like a structural property of unitary dynamics.

Panels are driven per series (each ticker gets its own trajectory from the same
initial state) and split by series. Within a series there is no reset, which is
the property that matters; resetting between two unrelated tickers is correct.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from qrc.evaluate import _fit_ridge, score
from qrc.ppe import reservoir_features


def _stack(series_list, max_len: int | None = None) -> np.ndarray:
    """Truncate a list of series to a common length and stack them."""
    if not series_list:
        raise ValueError("no series")
    T = min(len(s) for s in series_list)
    if max_len is not None:
        T = min(T, int(max_len))
    if T < 8:
        raise ValueError(f"series too short for a streaming drive (common length {T})")
    return np.stack([np.asarray(s, dtype=float)[:T] for s in series_list])


def drive(U: np.ndarray, model_key: str, num_memory: int, dt: float, n_steps: int,
          washout: int = 0, encoding: str = "continuous", initial_state: str = "neel",
          model_kwargs: dict | None = None, block_unitary: np.ndarray | None = None,
          shots: int | None = None, shot_seed: int = 0) -> np.ndarray:
    """
    Drive each row of U (n_series, T) continuously; return (n_series, T, per_step).

    `washout_length` is passed as 0 and the washout is applied afterwards by
    slicing: `reservoir_features` PREPENDS its washout with identity
    interventions, which would consume no input but would also discard the first
    recorded steps of the real drive. Recording everything and slicing keeps the
    time index aligned with the input, which the lag bookkeeping below depends on.
    """
    U = np.atleast_2d(np.asarray(U, dtype=float))
    n, T = U.shape
    L = 1 + num_memory
    F = reservoir_features(U, model_key, num_memory=num_memory, dt=dt, n_steps=n_steps,
                           washout_length=0, encoding=encoding,
                           initial_state=initial_state, model_kwargs=model_kwargs,
                           shots=shots, shot_seed=shot_seed, block_unitary=block_unitary)
    per_step = F.shape[1] // T
    S = F.reshape(n, T, per_step)
    return S[:, washout:, :] if washout else S


def _lag_matrix(U: np.ndarray, t_idx: np.ndarray, n_lags: int) -> np.ndarray:
    """
    Raw-lag design matrix: rows are [u[t-n_lags+1] .. u[t]] for each t in t_idx.

    Row order is series-major then time, matching how the state matrix is
    flattened, so baseline row i and reservoir row i describe the same instant.
    Getting this transpose wrong silently shuffles the baseline's rows against
    its targets and makes the classical comparator look far weaker than it is.
    """
    offs = np.arange(-n_lags + 1, 1)
    gathered = U[:, t_idx[None, :] + offs[:, None]]      # (n_series, n_lags, n_t)
    return gathered.transpose(0, 2, 1).reshape(-1, n_lags)


def _tapped(S: np.ndarray, t_idx: np.ndarray, n_tap: int) -> np.ndarray:
    """
    Time-multiplexed state readout: concatenate the last `n_tap` states.

    Standard practice in the QRC literature, and necessary for a fair width
    comparison: a single state carries only 2L-1 observables (9 at L=5), so
    against a 10-lag linear model the pure-state readout is outnumbered before
    the physics is considered. n_tap=1 is the pure, unmultiplexed protocol.
    """
    return np.concatenate([S[:, t_idx - j, :] for j in range(n_tap)], axis=2)


def forecast(series_list, model_key: str, num_memory: int, total_time: float,
             n_steps: int = 5, washout: int = 20, n_lags: int = 10,
             train_frac: float = 0.8, embargo: int = 0, encoding: str = "continuous",
             initial_state: str = "neel", model_kwargs: dict | None = None,
             block_unitary: np.ndarray | None = None, shots: int | None = None,
             shot_seed: int = 0, max_len: int | None = None, n_tap: int = 1,
             norm: tuple[float, float] | None = None) -> dict:
    """
    One-step-ahead forecasting from the streaming reservoir state.

    Returns the reservoir score, the classical baselines on identical indices,
    and the state matrix statistics. Splitting is temporal for a single series
    and by series for a panel, matching `evaluate.make_split`.
    """
    Y = _stack(series_list, max_len)
    n_series, T = Y.shape

    # usable prediction times: after washout, with enough history for the lag
    # baseline and the state taps, and a target at t+1
    t0 = max(washout, n_lags - 1, n_tap - 1)
    t_idx = np.arange(t0, T - 1)
    n_rows = n_series * len(t_idx)
    if len(t_idx) < 2 or n_rows < 40:
        raise ValueError(f"streaming drive too short: {n_series} series x "
                         f"{len(t_idx)} usable steps = {n_rows} rows")

    # split first, so encoding statistics are computed on training data only
    if n_series == 1:
        cut = int(train_frac * len(t_idx))
        tr_t = np.zeros(len(t_idx), bool); te_t = np.zeros(len(t_idx), bool)
        tr_t[:max(0, cut - embargo)] = True
        te_t[cut:] = True
        tr = tr_t; te = te_t
        train_vals = Y[0, t_idx[tr_t]]
    else:
        rng = np.random.default_rng(shot_seed)
        perm = rng.permutation(n_series)
        n_tr = max(1, int(train_frac * n_series))
        is_tr = np.zeros(n_series, bool); is_tr[perm[:n_tr]] = True
        tr = np.repeat(is_tr, len(t_idx))
        te = ~tr
        train_vals = Y[is_tr][:, t_idx].ravel()

    mu, sd = (float(np.mean(train_vals)), float(np.std(train_vals))) if norm is None else norm
    sd = sd if sd > 1e-12 else 1.0
    U = np.clip((Y - mu) / (3.0 * sd), -1.0, 1.0)

    S = drive(U, model_key, num_memory, total_time / n_steps, n_steps, washout=0,
              encoding=encoding, initial_state=initial_state, model_kwargs=model_kwargs,
              block_unitary=block_unitary, shots=shots, shot_seed=shot_seed)

    X = _tapped(S, t_idx, n_tap).reshape(n_series * len(t_idx), -1)
    y = Y[:, t_idx + 1].reshape(-1)                     # value at t+1
    sd_ref = float(np.std(y[tr]))

    mu_f, sd_f = X[tr].mean(0), X[tr].std(0)
    Xz = (X - mu_f) / np.where(sd_f < 1e-12, 1.0, sd_f)
    pred, alpha = _fit_ridge(Xz[tr], y[tr], Xz[te], y[te])
    out = score(y[te], pred, sd_ref)
    out.update(alpha=alpha, n_features=int(X.shape[1]))

    lags = _lag_matrix(Y, t_idx, n_lags)
    pl, _ = _fit_ridge(lags[tr], y[tr], lags[te], y[te])
    base = {
        "mean": score(y[te], np.full(int(te.sum()), float(np.mean(y[tr]))), sd_ref),
        "persistence": score(y[te], Y[:, t_idx].reshape(-1)[te], sd_ref),
        "linear_raw": score(y[te], pl, sd_ref),
    }
    return {"reservoir": out, "baselines": base, "n_series": int(n_series),
            "n_steps_used": int(len(t_idx)), "n_tap": int(n_tap),
            "n_train": int(tr.sum()), "n_test": int(te.sum()),
            "state_matrix": X}


def memory_capacity(model_key: str, num_memory: int, total_time: float,
                    n_steps: int = 5, max_lag: int = 20, T: int = 1500,
                    washout: int = 100, encoding: str = "continuous",
                    initial_state: str = "neel", model_kwargs: dict | None = None,
                    block_unitary: np.ndarray | None = None, seed: int = 0,
                    n_tap: int = 1) -> dict:
    """
    Jaeger linear memory capacity: MC_k = R^2 of reconstructing u(t-k), MC = sum_k.

    Driven with i.i.d. input so that anything recovered is memory rather than
    autocorrelation of the drive. Unlike a finite-difference sensitivity probe
    this is normalised and directly comparable across models and system sizes.
    """
    rng = np.random.default_rng(seed)
    u = (rng.integers(0, 4, size=T).astype(float) if encoding == "symbolic"
         else rng.uniform(-1.0, 1.0, size=T))
    S = drive(u[None, :], model_key, num_memory, total_time / n_steps, n_steps,
              washout=0, encoding=encoding, initial_state=initial_state,
              model_kwargs=model_kwargs, block_unitary=block_unitary)[0]
    if n_tap > 1:
        S = np.concatenate([np.roll(S, j, axis=0) for j in range(n_tap)], axis=1)
    mu, sd = S.mean(0), S.std(0)
    S = (S - mu) / np.where(sd < 1e-12, 1.0, sd)

    idx = np.arange(max(washout, n_tap), T)
    cut = int(0.7 * len(idx))
    tr_t, te_t = idx[:cut], idx[cut:]
    mcs = []
    for k in range(1, max_lag + 1):
        m = Ridge(alpha=1e-3).fit(S[tr_t], u[tr_t - k])
        mcs.append(max(0.0, float(r2_score(u[te_t - k], m.predict(S[te_t])))))
    return {"mc_curve": [float(v) for v in mcs], "MC": float(np.sum(mcs)),
            "max_lag": int(max_lag), "n_features": int(S.shape[1])}
