"""Source classification: quantum feature map vs classical baselines."""
import sys, os, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from scipy.stats import unitary_group
from sklearn.model_selection import train_test_split
from qrc.datasets import source_classification
from qrc.classify import quantum_features, evaluate_clf, baselines_clf, separability

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "classifier")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--W", type=int, default=16)
    ap.add_argument("--n-per-class", type=int, default=2500)
    ap.add_argument("--noise", type=float, default=0.0)
    ap.add_argument("--num-memory", type=int, nargs="+", default=[4, 6])
    ap.add_argument("--models", nargs="+", default=["XXZ", "IAA_CHAOTIC", "IAA_LOCALIZED"])
    ap.add_argument("--total-time", type=float, nargs="+", default=[0.2])
    ap.add_argument("--reupload", type=int, nargs="+", default=[1])
    ap.add_argument("--shots", type=int, nargs="+", default=[0])
    a = ap.parse_args()

    os.makedirs(OUT, exist_ok=True)
    X, y = source_classification(W=a.W, n_per_class=a.n_per_class, noise=a.noise)
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
    Utr, Ute = np.tanh(Xtr / 2.5), np.tanh(Xte / 2.5)
    print(f"source classification: X={X.shape}, {len(np.unique(y))} classes, "
          f"train {len(ytr)} / test {len(yte)}, window noise={a.noise}")

    nfeat = a.W * max(a.reupload) * (2 * (1 + max(a.num_memory)) - 1)
    base = baselines_clf(Xtr, ytr, Xte, yte, n_rff=nfeat)
    for k, v in base.items():
        print(f"  baseline {k:>12}: acc={v['acc']:.4f}")
    print(f"  separability of raw windows: {separability(Xtr, ytr)}")

    recs = []
    hdr = (f"{'model':>14} {'L':>3} {'rup':>4} {'T':>5} {'shots':>6} | "
           f"{'quantum':>8} {'random':>7} {'nfeat':>6}")
    print(hdr); print("-" * len(hdr))
    for nm in a.num_memory:
        for model in a.models:
            for T in a.total_time:
                for rup in a.reupload:
                    for sh in a.shots:
                        kw = dict(model_key=model, num_memory=nm, total_time=T, n_reupload=rup)
                        ex = {} if sh == 0 else dict(shots=sh)
                        Ftr = quantum_features(Utr, **kw, **ex)
                        Fte = quantum_features(Ute, **kw, **ex)
                        q = evaluate_clf(Ftr, ytr, Fte, yte)
                        # Explicit injection, not a monkeypatch of the module
                        # global: the latter is unsafe under any concurrency.
                        U = unitary_group.rvs(2 ** (1 + nm), random_state=1)
                        rkw = {**kw, "block_unitary": U}
                        r = evaluate_clf(quantum_features(Utr, **rkw, **ex), ytr,
                                         quantum_features(Ute, **rkw, **ex), yte)
                        recs.append(dict(model=model, L=1 + nm, reupload=rup, total_time=T,
                                         shots=sh, quantum=q, random_unitary=r))
                        print(f"{model:>14} {1+nm:3d} {rup:4d} {T:5.2f} {(sh or 'exact'):>6} | "
                              f"{q['acc']:8.4f} {r['acc']:7.4f} {q['n_features']:6d}")

    with open(os.path.join(OUT, "source_clf.json"), "w") as f:
        json.dump({"args": vars(a), "baselines": base, "records": recs}, f, indent=2, default=float)
    print(f"\nWrote {os.path.join(OUT,'source_clf.json')}")
    beat = [r for r in recs if r["quantum"]["acc"] > max(base["logistic"]["acc"], base["rbf_svm"]["acc"])]
    br = [r for r in recs if r["quantum"]["acc"] > r["random_unitary"]["acc"] + 0.005]
    print(f"configs beating BOTH logistic and RBF-SVM : {len(beat)}/{len(recs)}")
    print(f"configs beating the random-unitary control: {len(br)}/{len(recs)}")


if __name__ == "__main__":
    main()
