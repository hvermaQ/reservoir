"""
Systematic classification sweep: quantum feature map vs matched classical baselines.

Every configuration reports, on identical splits:
  - classical baselines (majority / logistic / RBF-SVM / RFF at matched width)
  - separability of the PCA representation BEFORE the quantum map
  - separability of the quantum representation AFTER it
  - classification accuracy from a logistic readout
  - a Haar-random unitary control of the same dimension

The random control is not optional. A physics-motivated Hamiltonian that cannot
beat a random unitary of equal dimension has contributed nothing, and reporting
one without the other is the failure mode identified in the 2026 QRC literature.

Usage:
  python3 scripts/run_classifier.py --quick
  python3 scripts/run_classifier.py --dataset mnist --n-samples 10000
"""
import sys, os, json, time, argparse, itertools
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from scipy.stats import unitary_group

from qrc.classify import (load_images, pca_split, scale_for_encoding, quantum_features,
                          separability, evaluate_clf, baselines_clf)

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "classifier")


def one_config(X, y, n_pca, num_memory, model, n_reupload, total_time, seed, n_rff):
    Atr, Ate, ytr, yte = pca_split(X, y, n_pca, seed=seed)
    Utr, Ute = scale_for_encoding(Atr, Ate)
    kw = {"use_random": True, "seed": seed} if model.startswith("NNN") else {}
    t0 = time.time()

    def feats(random_block):
        # The Haar control is injected as an explicit argument. It used to be a
        # monkeypatch of ppe.build_block_unitary, which mutates module state and
        # leaks across any concurrent caller sharing the interpreter.
        args = dict(model_key=model, num_memory=num_memory, total_time=total_time,
                    n_reupload=n_reupload, model_kwargs={} if random_block else kw)
        if random_block:
            args["block_unitary"] = unitary_group.rvs(2 ** (1 + num_memory),
                                                      random_state=seed)
        return quantum_features(Utr, **args), quantum_features(Ute, **args)

    Ftr, Fte = feats(False)
    Rtr, Rte = feats(True)
    rec = {
        "n_pca": n_pca, "L": 1 + num_memory, "model": model, "n_reupload": n_reupload,
        "total_time": total_time, "seed": seed,
        # RFF width matched to the quantum feature count, gamma tuned inside
        # baselines_clf. A fixed-width, fixed-gamma random-feature baseline is
        # not a baseline; it inflates the quantum result.
        "baselines": baselines_clf(Atr, ytr, Ate, yte,
                                   n_rff=n_pca * n_reupload * (2 * (1 + num_memory) - 1),
                                   seed=seed),
        "sep_pca": separability(Atr, ytr), "sep_quantum": separability(Ftr, ytr),
        "quantum": evaluate_clf(Ftr, ytr, Fte, yte),
        "random_unitary": evaluate_clf(Rtr, ytr, Rte, yte),
        "seconds": round(time.time() - t0, 2),
    }
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="digits", choices=["digits", "mnist"])
    ap.add_argument("--n-samples", type=int, default=None)
    ap.add_argument("--n-pca", type=int, nargs="+", default=[8, 16, 32])
    ap.add_argument("--num-memory", type=int, nargs="+", default=[2, 4, 6])
    ap.add_argument("--models", nargs="+", default=["XXZ", "IAA_CHAOTIC", "IAA_LOCALIZED"])
    ap.add_argument("--reupload", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--total-time", type=float, nargs="+", default=[0.2])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--quick", action="store_true", help="smallest grid, for a sanity run")
    ap.add_argument("--tag", default=None)
    a = ap.parse_args()

    if a.quick:
        a.n_pca, a.num_memory, a.models, a.reupload = [8, 16], [4], ["XXZ"], [1, 2]

    os.makedirs(OUT, exist_ok=True)
    X, y = load_images(a.dataset, n_samples=a.n_samples, seed=a.seed)
    grid = list(itertools.product(a.n_pca, a.num_memory, a.models, a.reupload, a.total_time))
    print(f"dataset={a.dataset}  X={X.shape}  configurations={len(grid)}")
    hdr = (f"{'nPCA':>5} {'L':>3} {'model':>14} {'rup':>4} {'T':>5} | "
           f"{'logistic':>8} {'rbfSVM':>7} {'RFF*':>6} | {'quantum':>8} {'random':>7} | "
           f"{'kNN_pca':>8} {'kNN_q':>7} | {'sec':>6}")
    print(hdr); print("-" * len(hdr))

    recs = []
    for n_pca, nm, model, rup, T in grid:
        r = one_config(X, y, n_pca, nm, model, rup, T, a.seed, n_rff=90)
        recs.append(r)
        b = r["baselines"]
        print(f"{n_pca:5d} {r['L']:3d} {model:>14} {rup:4d} {T:5.2f} | "
              f"{b['logistic']['acc']:8.4f} {b['rbf_svm']['acc']:7.4f} {list(b[k] for k in b if k.startswith('rff_'))[0]['acc']:6.4f} | "
              f"{r['quantum']['acc']:8.4f} {r['random_unitary']['acc']:7.4f} | "
              f"{r['sep_pca']['knn5_acc']:8.4f} {r['sep_quantum']['knn5_acc']:7.4f} | {r['seconds']:6.1f}")

    tag = a.tag or f"{a.dataset}_{a.n_samples or 'all'}"
    path = os.path.join(OUT, f"sweep_{tag}.json")
    with open(path, "w") as f:
        json.dump({"args": vars(a), "records": recs}, f, indent=2, default=float)
    print(f"\nWrote {path}")

    best = max(recs, key=lambda r: r["quantum"]["acc"])
    won = [r for r in recs if r["quantum"]["acc"] > max(r["baselines"]["logistic"]["acc"],
                                                        r["baselines"]["rbf_svm"]["acc"])]
    beat_rand = [r for r in recs if r["quantum"]["acc"] > r["random_unitary"]["acc"] + 0.005]
    print(f"best quantum acc = {best['quantum']['acc']:.4f} "
          f"(nPCA={best['n_pca']}, L={best['L']}, {best['model']}, reupload={best['n_reupload']})")
    print(f"configs beating BOTH logistic and RBF-SVM : {len(won)}/{len(recs)}")
    print(f"configs beating the random-unitary control: {len(beat_rand)}/{len(recs)}")


if __name__ == "__main__":
    main()
