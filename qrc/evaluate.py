"""
evaluate.py — windowing, baselines and scoring for reservoir experiments.

Design decisions that matter for the validity of the numbers:

  * Symbolisation uses TRAILING statistics only (mean/std over the causal
    context up to and including t), so no future information leaks into the
    encoding. `create_lagged_quantile_features` in feature_engineering.py takes
    global quantiles over the whole series and does leak; it is not used here.

  * Panel datasets are split BY SERIES, not by window. Splitting pooled windows
    at random would put windows from the same ticker on both sides of the split.
    Synthetic single-series datasets are split temporally.

  * Every reservoir score is reported next to three baselines on identical
    splits. Without them an R^2 is uninterpretable: the symbolisation and the
    lag structure alone already carry predictive power.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

ALPHAS = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)


def symbolize_causal(values: np.ndarray, lag_window: int) -> np.ndarray:
    """Map a series to symbols {0,1,2,3} using trailing mean/std (no lookahead)."""
    v = np.asarray(values, dtype=float)
    T = len(v)
    out = np.zeros(T, dtype=int)
    for t in range(T):
        ctx = v[max(0, t - lag_window + 1): t + 1]
        mu, sd = ctx.mean(), ctx.std()
        x = v[t]
        if sd == 0:
            out[t] = 1 if x < mu else 2
            continue
        if x < mu - 2 * sd:
            out[t] = 0
        elif x < mu - sd:
            out[t] = 1
        elif x < mu + sd:
            out[t] = 2
        else:
            out[t] = 3
    return out


def windowize(series_list, W: int):
    """
    Build pooled windows over a list of series.

    Returns X_sym (n,W) int, X_raw (n,W) float, y (n,), sid (n,) series index.
    Window [t-W+1 .. t] predicts value at t+1.
    """
    Xs, Xr, ys, sid = [], [], [], []
    for i, s in enumerate(series_list):
        s = np.asarray(s, dtype=float)
        if len(s) < W + 2:
            continue
        sym = symbolize_causal(s, W)
        for t in range(W - 1, len(s) - 1):
            Xs.append(sym[t - W + 1: t + 1])
            Xr.append(s[t - W + 1: t + 1])
            ys.append(s[t + 1])
            sid.append(i)
    if not Xs:
        raise ValueError(f"no windows produced (W={W} too large for these series)")
    return (np.array(Xs), np.array(Xr, dtype=float),
            np.array(ys, dtype=float), np.array(sid))


def make_split(sid: np.ndarray, train_frac: float = 0.8, seed: int = 0,
               embargo: int = 0):
    """
    Split by series when there are many; temporally when there is one.

    `embargo` drops that many windows either side of a temporal cut, assigning
    them to NEITHER side. Windows of length W overlap, so a bare index cut puts
    roughly W-1 test windows that contain training timesteps on the test side.
    The leak is small but it is free to remove, and `embargo=W` removes it
    exactly. It has no meaning for a by-series split, where no window spans two
    series.
    """
    uniq = np.unique(sid)
    if len(uniq) == 1:
        n = len(sid); cut = int(train_frac * n)
        tr = np.zeros(n, bool); te = np.zeros(n, bool)
        tr[:max(0, cut - embargo)] = True
        te[cut:] = True
        return tr, te
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    n_tr = max(1, int(train_frac * len(uniq)))
    train_ids = set(perm[:n_tr].tolist())
    tr = np.array([s in train_ids for s in sid])
    return tr, ~tr


def _fit_ridge(Xtr, ytr, Xte, yte):
    """Ridge with alpha picked on a held-out tail of the training set."""
    n = len(ytr)
    if n < 20:
        best_a = 1.0
    else:
        cut = int(0.8 * n)
        best_a, best_s = None, np.inf
        for a in ALPHAS:
            m = Ridge(alpha=a).fit(Xtr[:cut], ytr[:cut])
            s = np.mean((m.predict(Xtr[cut:]) - ytr[cut:]) ** 2)
            if s < best_s:
                best_a, best_s = a, s
    m = Ridge(alpha=best_a).fit(Xtr, ytr)
    return m.predict(Xte), best_a


def score(y_true, y_pred, sd_ref: float | None = None) -> dict:
    """
    Error metrics. `sd_ref` is the NRMSE denominator.

    Pass the TRAINING standard deviation. Normalising by the test set's own
    spread makes NRMSE a ratio estimator with a random denominator: it adds
    variance to every comparison and shifts when test composition changes, which
    is exactly what happens across stock cohorts and across seeds. `NRMSE_test`
    is kept alongside for continuity with earlier results.
    """
    resid = y_true - y_pred
    rmse = float(np.sqrt(np.mean(resid ** 2)))
    sd_test = float(np.std(y_true))
    sd = float(sd_ref) if sd_ref is not None else sd_test
    return {
        "RMSE": rmse,
        "NRMSE": rmse / sd if sd > 1e-12 else float("nan"),
        "NRMSE_test": rmse / sd_test if sd_test > 1e-12 else float("nan"),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "R2": float(r2_score(y_true, y_pred)),
    }


def onehot(X_sym: np.ndarray, n_sym: int = 4) -> np.ndarray:
    n, W = X_sym.shape
    out = np.zeros((n, W * n_sym))
    out[np.arange(n)[:, None], np.arange(W)[None, :] * n_sym + X_sym] = 1.0
    return out


def baselines(X_sym, X_raw, y, tr, te) -> dict:
    """
    Persistence, Ridge on raw lags, Ridge on one-hot symbols, and the mean.

    `mean` (predict the training mean everywhere) is the learnability floor: a
    model that cannot beat it has found nothing, and comparing two arms that are
    both above NRMSE 1 is uninformative. Downstream gating uses it.
    """
    sd_ref = float(np.std(y[tr]))
    res = {}
    res["mean"] = score(y[te], np.full(int(te.sum()), float(np.mean(y[tr]))), sd_ref)
    res["persistence"] = score(y[te], X_raw[te][:, -1], sd_ref)
    p, _ = _fit_ridge(X_raw[tr], y[tr], X_raw[te], y[te])
    res["linear_raw"] = score(y[te], p, sd_ref)
    H = onehot(X_sym)
    p, _ = _fit_ridge(H[tr], y[tr], H[te], y[te])
    res["linear_symbolic"] = score(y[te], p, sd_ref)
    return res


def evaluate_features(F, y, tr, te) -> dict:
    """Score a reservoir feature matrix with a ridge readout."""
    mu, sd = F[tr].mean(0), F[tr].std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    Fz = (F - mu) / sd
    p, alpha = _fit_ridge(Fz[tr], y[tr], Fz[te], y[te])
    out = score(y[te], p, sd_ref=float(np.std(y[tr])))
    out["alpha"] = alpha
    out["n_features"] = int(F.shape[1])
    return out


# ---------------------------------------------------------------------------
# Linear memory capacity (Jaeger)
# ---------------------------------------------------------------------------

def memory_capacity(model_key: str, num_memory: int, dt: float, n_steps: int,
                    max_lag: int = 15, T: int = 1200, washout: int = 50,
                    initial_state: str = "neel", encoding: str = "symbolic",
                    seed: int = 0, model_kwargs: dict | None = None):
    """
    MC_k = R^2 of a linear readout reconstructing input u(t-k) from the reservoir
    state at time t; MC = sum_k MC_k. This is the standard reservoir-computing
    memory measure, and unlike a finite-difference sensitivity probe it is
    normalised, basis-independent and directly comparable across models.

    Driven with an i.i.d. random input so that any recovered structure is memory
    rather than input autocorrelation.
    """
    from qrc.ppe import reservoir_features
    rng = np.random.default_rng(seed)
    if encoding == "symbolic":
        u = rng.integers(0, 4, size=T)
    else:
        u = rng.uniform(-1.0, 1.0, size=T)

    F = reservoir_features(u[None, :], model_key, num_memory=num_memory, dt=dt,
                           n_steps=n_steps, washout_length=washout,
                           initial_state=initial_state, encoding=encoding,
                           model_kwargs=model_kwargs)
    # reservoir_features PREPENDS the washout, so one row is recorded per input
    # step; the washout does not consume elements of u.
    per = F.shape[1] // T
    S = F.reshape(T, per)
    u_post = u.astype(float)

    mu, sd = S.mean(0), S.std(0)
    S = (S - mu) / np.where(sd < 1e-12, 1.0, sd)

    n = len(u_post); cut = int(0.7 * n)
    mcs = []
    for k in range(1, max_lag + 1):
        if cut - k < 30:
            mcs.append(0.0); continue
        Xtr, ytr = S[k:cut], u_post[:cut - k]
        Xte, yte = S[cut:], u_post[cut - k:n - k]
        m = Ridge(alpha=1e-3).fit(Xtr, ytr)
        r2 = r2_score(yte, m.predict(Xte))
        mcs.append(max(0.0, float(r2)))
    return np.array(mcs), float(np.sum(mcs))


def window_memory_profile(model_key: str, num_memory: int, dt: float, n_steps: int,
                          W: int = 10, n_windows: int = 800, washout: int = 4,
                          initial_state: str = "neel", encoding: str = "continuous",
                          seed: int = 0, model_kwargs: dict | None = None):
    """
    Per-position input recoverability under the WINDOWED protocol.

    `memory_capacity` above measures fading memory in a streaming reservoir that
    is never reset. A purely unitary reservoir has no fading memory -- the map is
    measure-preserving, so past inputs are scrambled into global correlations
    rather than forgotten, and a linear readout on local observables recovers
    almost nothing. That is why the streaming MC is ~0 at every dt, and it is a
    property of unitary dynamics, not a defect of a particular operating point.

    The repo's actual protocol re-initialises the state for every window, so the
    operative question is different: from the features of one window, how well is
    the input at position k recoverable? k=0 is the oldest position in the window.
    """
    from qrc.ppe import reservoir_features
    rng = np.random.default_rng(seed)
    if encoding == "symbolic":
        U = rng.integers(0, 4, size=(n_windows, W))
        targets = U.astype(float)
    else:
        U = rng.uniform(-1.0, 1.0, size=(n_windows, W))
        targets = U

    F = reservoir_features(U, model_key, num_memory=num_memory, dt=dt, n_steps=n_steps,
                           washout_length=washout, initial_state=initial_state,
                           encoding=encoding, model_kwargs=model_kwargs)
    mu, sd = F.mean(0), F.std(0)
    F = (F - mu) / np.where(sd < 1e-12, 1.0, sd)
    cut = int(0.7 * n_windows)
    out = []
    for k in range(W):
        m = Ridge(alpha=1e-2).fit(F[:cut], targets[:cut, k])
        out.append(max(0.0, float(r2_score(targets[cut:, k], m.predict(F[cut:])))))
    return np.array(out), float(np.sum(out))
