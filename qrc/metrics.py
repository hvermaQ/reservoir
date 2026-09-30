"""
metrics.py — representation-level measurements shared by every arm.

These exist to answer objections that accuracy alone cannot:

  effective_rank  How many directions does a feature matrix actually use? The
                  quantum vector concatenates L-1 nested-cut entropies with L
                  single-site <sigma_z> values, which are functionally
                  dependent, so its NOMINAL width overstates its capacity.
                  Matching a random-feature baseline on nominal width therefore
                  hands the classical arm more usable directions, which biases
                  against the quantum arm. Reporting effective rank for both
                  makes that visible instead of arguable.

  entropy_stats   The mean and spread of the Renyi-2 entropy ACTUALLY produced
                  by the task inputs. `ppe_diagnostics` enumerates the symbolic
                  Pauli alphabet, which is a different input distribution from
                  the continuous encoding the experiments use -- so it cannot be
                  the x-axis of an entanglement-vs-performance claim about those
                  experiments. This can, and it is free: the entropy columns are
                  already in the feature matrix.
"""
from __future__ import annotations

import numpy as np


def effective_rank(F: np.ndarray, eps: float = 1e-12) -> dict:
    """
    Participation ratio and entropy-based effective rank of a feature matrix.

    PR = (sum lambda)^2 / sum(lambda^2) over the covariance eigenvalues: the
    number of directions carrying comparable variance. `erank` is exp(Shannon
    entropy of the normalised spectrum), which penalises long thin tails less.
    Both are reported because they disagree in informative ways.
    """
    F = np.asarray(F, dtype=float)
    if F.ndim != 2 or F.shape[0] < 2 or F.shape[1] == 0:
        return {"nominal": int(F.shape[1] if F.ndim == 2 else 0),
                "participation_ratio": float("nan"), "erank": float("nan")}
    Z = F - F.mean(0)
    sd = Z.std(0)
    Z = Z / np.where(sd < eps, 1.0, sd)
    # eigenvalues of the correlation matrix via SVD of the centred matrix
    s = np.linalg.svd(Z, compute_uv=False)
    lam = s ** 2
    tot = lam.sum()
    if tot <= eps:
        return {"nominal": int(F.shape[1]), "participation_ratio": float("nan"),
                "erank": float("nan")}
    pr = float(tot ** 2 / np.sum(lam ** 2))
    p = lam / tot
    p = p[p > eps]
    return {"nominal": int(F.shape[1]),
            "participation_ratio": pr,
            "erank": float(np.exp(-np.sum(p * np.log(p))))}


def entropy_column_index(L: int, cuts=None, use_entropy: bool = True,
                         use_sigmaz: bool = True) -> tuple[np.ndarray, int]:
    """
    Entropy column offsets within one recorded step, and the step width.

    `reservoir_features` emits, per recorded step, the entropy at each cut
    followed by <sigma_z> on each qubit; steps are concatenated along axis 1.
    """
    n_cuts = len(tuple(range(1, L)) if cuts is None else cuts) if use_entropy else 0
    per_step = n_cuts + (L if use_sigmaz else 0)
    return np.arange(n_cuts, dtype=int), per_step


def entropy_stats(F: np.ndarray, L: int, cuts=None, use_entropy: bool = True,
                  use_sigmaz: bool = True) -> dict:
    """
    Task-conditional entanglement statistics from the features themselves.

    Reports the half-chain cut specifically (the one `ppe_diagnostics` and the
    Page-value normalisation refer to) as well as the pooled statistics over all
    cuts, so the number is comparable to the paper's S_x.
    """
    F = np.asarray(F, dtype=float)
    if not use_entropy or F.ndim != 2 or F.shape[1] == 0:
        return {}
    idx, per_step = entropy_column_index(L, cuts, use_entropy, use_sigmaz)
    if per_step == 0 or F.shape[1] % per_step != 0:
        return {}
    n_steps = F.shape[1] // per_step
    cut_list = list(range(1, L)) if cuts is None else list(cuts)
    ent_cols = np.concatenate([idx + s * per_step for s in range(n_steps)]) if n_steps else idx
    E = F[:, ent_cols]
    S_max = float(np.log(2 ** (L // 2)))

    out = {"S_mean": float(E.mean()), "S_std": float(E.std()),
           "S_max": S_max, "S_frac_max": float(E.mean() / S_max) if S_max > 0 else float("nan")}

    # half-chain cut at the LAST recorded step: the closest analogue of S_x
    half = L // 2
    if half in cut_list:
        j = cut_list.index(half) + (n_steps - 1) * per_step
        col = F[:, j]
        out.update(S_half_mean=float(col.mean()), S_half_std=float(col.std()),
                   S_half_frac_max=float(col.mean() / S_max) if S_max > 0 else float("nan"))
    return out
