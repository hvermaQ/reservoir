"""
classify.py — static classification with a quantum feature map.

Scope note: with no time axis this is a quantum extreme learning machine, not
reservoir computing. Memory, washout and the echo-state property are irrelevant
here; the quantum system is a fixed nonlinear feature map. The relevant
comparison is therefore against classical kernels and random features, not
against RNNs.

What the encoding actually computes: PCA components are angle-encoded and
applied as sequential interventions with entangling evolution between them, so
each output feature is a fixed combination of products of sines and cosines of
the components -- a tensor-Fourier map, i.e. functionally an RBF-like kernel on
PCA space. RBF-SVM on the same components is thus the honest comparator, and it
is included in `baselines_clf` by default.
"""
from __future__ import annotations

import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.kernel_approximation import RBFSampler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score, silhouette_score
from sklearn.model_selection import train_test_split

from qrc.ppe import reservoir_features


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

_CACHE: dict = {}


def load_images(name: str = "digits", n_samples: int | None = None, seed: int = 0):
    """Return (X, y) raw pixels. 'digits' is 8x8 offline; 'mnist' is 28x28."""
    key = (name, n_samples, seed)
    if key in _CACHE:
        return _CACHE[key]
    if name == "digits":
        from sklearn.datasets import load_digits
        X, y = load_digits(return_X_y=True)
    elif name == "mnist":
        from sklearn.datasets import fetch_openml
        d = fetch_openml("mnist_784", version=1, as_frame=False, parser="liac-arff")
        X, y = d.data.astype(np.float64), d.target.astype(int)
    else:
        raise ValueError(f"unknown dataset {name!r}")
    if n_samples is not None and n_samples < len(y):
        idx = np.random.default_rng(seed).choice(len(y), n_samples, replace=False)
        X, y = X[idx], y[idx]
    _CACHE[key] = (X, y)
    return X, y


def pca_split(X, y, n_components: int, test_size: float = 0.3, seed: int = 0):
    """
    Stratified split, then PCA fitted on TRAIN ONLY.

    Fitting PCA on the full set before splitting leaks test information into the
    representation; it is a common and easily missed error in QML papers.
    """
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    pipe = make_pipeline(StandardScaler(), PCA(n_components, random_state=seed)).fit(Xtr)
    return pipe.transform(Xtr), pipe.transform(Xte), ytr, yte


def scale_for_encoding(Atr, Ate, n_sigma: float = 2.5):
    """Map PCA components into [-1,1] with TRAIN statistics, via tanh (no hard clip)."""
    mu, sd = Atr.mean(0), Atr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return np.tanh((Atr - mu) / (n_sigma * sd)), np.tanh((Ate - mu) / (n_sigma * sd))


# ---------------------------------------------------------------------------
# Quantum feature map
# ---------------------------------------------------------------------------

def quantum_features(U_in, model_key="XXZ", num_memory=4, total_time=0.2, n_steps=5,
                     n_reupload=1, cuts=None, use_entropy=True, use_sigmaz=True,
                     initial_state="neel", model_kwargs=None, batch_size=512,
                     shots=None, shot_seed=0, block_unitary=None):
    """
    Encode each row of U_in (values in [-1,1]) as a sequence of interventions.

    n_reupload > 1 tiles the component sequence. Repeated encoding provably
    enlarges the accessible frequency spectrum (Schuld/Sweke/Meyer 2021), which
    is the one lever that changes the FUNCTION CLASS rather than just which
    coefficients within it are reachable.
    """
    X = np.tile(np.asarray(U_in, dtype=float), (1, max(1, int(n_reupload))))
    return reservoir_features(
        X, model_key, num_memory=num_memory, dt=total_time / n_steps, n_steps=n_steps,
        washout_length=0, encoding="continuous", initial_state=initial_state,
        model_kwargs=model_kwargs, cuts=cuts, use_entropy=use_entropy,
        use_sigmaz=use_sigmaz, batch_size=batch_size, shots=shots, shot_seed=shot_seed,
        block_unitary=block_unitary)


# ---------------------------------------------------------------------------
# Diagnostics: is class structure preserved by the map?
# ---------------------------------------------------------------------------

def separability(F, y, sample: int = 3000, seed: int = 0) -> dict:
    """
    Training-free measures of how well classes are separated in a representation.

    Answers the question 'does the quantum map preserve the class structure PCA
    already found?' independently of any readout, so a poor result cannot be
    blamed on the classifier.
    """
    rng = np.random.default_rng(seed)
    if len(y) > sample:
        i = rng.choice(len(y), sample, replace=False)
        F, y = F[i], y[i]
    Z = (F - F.mean(0)) / np.where(F.std(0) < 1e-12, 1.0, F.std(0))
    knn = KNeighborsClassifier(5)
    n = len(y); cut = int(0.7 * n)
    knn.fit(Z[:cut], y[:cut])
    out = {"knn5_acc": float(knn.score(Z[cut:], y[cut:]))}
    try:
        out["silhouette"] = float(silhouette_score(Z, y))
    except Exception:
        out["silhouette"] = float("nan")
    # Fisher ratio: between-class scatter over within-class scatter
    gm = Z.mean(0); sb = sw = 0.0
    for c in np.unique(y):
        Zc = Z[y == c]
        sb += len(Zc) * np.sum((Zc.mean(0) - gm) ** 2)
        sw += np.sum((Zc - Zc.mean(0)) ** 2)
    out["fisher"] = float(sb / sw) if sw > 0 else float("nan")
    return out


# ---------------------------------------------------------------------------
# Readout + baselines
# ---------------------------------------------------------------------------

def evaluate_clf(Ftr, ytr, Fte, yte, max_iter: int = 2000) -> dict:
    mu, sd = Ftr.mean(0), Ftr.std(0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    m = LogisticRegression(max_iter=max_iter).fit((Ftr - mu) / sd, ytr)
    p = m.predict((Fte - mu) / sd)
    return {"acc": float(accuracy_score(yte, p)),
            "balanced_acc": float(balanced_accuracy_score(yte, p)),
            "n_features": int(Ftr.shape[1])}


def baselines_clf(Atr, ytr, Ate, yte, n_rff: int = 90, seed: int = 0,
                  svm_tune_subsample: int = 4000) -> dict:
    """
    Majority class, logistic, RBF-SVM, and random Fourier features.

    `n_rff` should be set to the quantum feature count so the random-feature
    baseline is matched in width. Every tunable classical hyperparameter -- RFF
    gamma, SVM C and gamma -- is selected on a held-out slice of the training
    set, never on test. Leaving a classical baseline untuned while tuning the
    quantum model is the specific methodological failure the 2026 QRC literature
    identifies, and it inflates any apparent quantum advantage.
    """
    res = {}
    maj = np.bincount(ytr).argmax()
    res["majority"] = {"acc": float((yte == maj).mean()),
                       "balanced_acc": float(balanced_accuracy_score(yte, np.full_like(yte, maj)))}
    res["logistic"] = evaluate_clf(Atr, ytr, Ate, yte)

    # RBF-SVM is the strongest classical comparator here, so leaving it at
    # library defaults while the quantum arm is swept is the same asymmetry the
    # tuned-RFF baseline exists to avoid -- just pointed the other way. C and
    # gamma are selected on a held-out slice of TRAIN, never on test.
    #
    # Selection runs on a subsample: SVC is O(n^2) and a 16-point grid on 14k
    # training rows would cost more than every quantum arm in the sweep
    # combined. The chosen pair is then refitted on the full training set.
    cut = int(0.8 * len(ytr))
    fit_idx = np.arange(cut)
    if cut > svm_tune_subsample:
        fit_idx = np.random.default_rng(seed).choice(cut, svm_tune_subsample,
                                                     replace=False)
    best_pair, best_score = ("scale", 10.0), -1.0
    for C in (1.0, 10.0, 100.0):
        for g in ("scale", 0.001, 0.01, 0.1):
            try:
                m = SVC(kernel="rbf", C=C, gamma=g).fit(Atr[fit_idx], ytr[fit_idx])
                sc = float(m.score(Atr[cut:], ytr[cut:]))
            except Exception:
                continue
            if sc > best_score:
                best_pair, best_score = (g, C), sc
    g_best, c_best = best_pair
    sv = SVC(kernel="rbf", C=c_best, gamma=g_best).fit(Atr, ytr)
    res["rbf_svm"] = {"acc": float(sv.score(Ate, yte)),
                      "balanced_acc": float(balanced_accuracy_score(yte, sv.predict(Ate))),
                      "C": float(c_best),
                      "gamma": (g_best if isinstance(g_best, str) else float(g_best))}
    cut = int(0.8 * len(ytr))
    best, best_s = None, -1.0
    for g in (0.001, 0.005, 0.02, 0.05, 0.2, 0.5, 1.0, "scale"):
        gv = 1.0 / (Atr.shape[1] * max(Atr.var(), 1e-12)) if g == "scale" else g
        rf = RBFSampler(n_components=n_rff, gamma=gv, random_state=seed).fit(Atr[:cut])
        sc = evaluate_clf(rf.transform(Atr[:cut]), ytr[:cut],
                          rf.transform(Atr[cut:]), ytr[cut:])["acc"]
        if sc > best_s:
            best, best_s = gv, sc
    rf = RBFSampler(n_components=n_rff, gamma=best, random_state=seed).fit(Atr)
    r = evaluate_clf(rf.transform(Atr), ytr, rf.transform(Ate), yte)
    r["gamma"] = float(best)
    res[f"rff_{n_rff}"] = r
    return res
