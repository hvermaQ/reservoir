"""
Transducer characterisation: where does the reservoir map inputs onto the
widest spread of Renyi-2 entropy?

Needs no labelled data -- the reservoir is driven with random inputs and we
measure the response distribution P(S_x). The design point is the T at which
the width peaks: below it every input is crushed near S=0 (cutoff), above it
every input saturates at the Page value (the deep-thermalised regime the source
paper calls 'chaotic'). Both extremes are narrow and therefore uninformative.
"""
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from qrc.ppe import reservoir_features

MODELS = ["XXZ", "NNN_CHAOTIC", "NNN_LOCALIZED", "IAA_CHAOTIC", "IAA_LOCALIZED"]
TS = [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
NM, W, N, NSTEPS = 4, 16, 500, 5


def main():
    L = 1 + NM
    Smax = np.log(2 ** (L // 2))
    U = np.random.default_rng(0).uniform(-1, 1, size=(N, W))
    out = {}
    print(f"L={L}  max S = {Smax:.4f}   (n_steps={NSTEPS}, W={W})")
    print(f"{'model':>15} | " + " ".join(f"{t:>6g}" for t in TS))
    for tag, idx in (("mean S", 0), ("width", 1)):
        print(f"--- {tag} ---")
        for m in MODELS:
            kw = {"use_random": True, "seed": 0} if m.startswith("NNN") else {}
            vals = []
            for T in TS:
                key = (m, T)
                if key not in out:
                    F = reservoir_features(U, m, num_memory=NM, dt=T / NSTEPS, n_steps=NSTEPS,
                                           washout_length=0, encoding="continuous",
                                           initial_state="neel", model_kwargs=kw,
                                           cuts=(L // 2,), use_entropy=True, use_sigmaz=False)
                    S = F[:, -1]
                    out[key] = (float(S.mean()), float(S.std()))
                vals.append(out[key][idx])
            print(f"{m:>15} | " + " ".join(f"{v:6.3f}" for v in vals))

    print("\ndesign point = T at the width peak:")
    best = {}
    for m in MODELS:
        w = [out[(m, T)][1] for T in TS]
        i = int(np.argmax(w))
        best[m] = TS[i]
        frac = out[(m, TS[i])][0] / Smax
        print(f"  {m:>15}  T*={TS[i]:<5g} width={w[i]:.4f}  mean S = {frac*100:4.1f}% of max"
              f"   dt*={TS[i]/NSTEPS:.3f} ({'Trotter-faithful' if TS[i]/NSTEPS <= 0.06 else 'dt too large'})")
    with open(os.path.join("results", "classifier", "width_curve.json"), "w") as f:
        json.dump({"Smax": Smax, "TS": TS,
                   "curve": {f"{m}|{T}": out[(m, T)] for m in MODELS for T in TS},
                   "best_T": best}, f, indent=2)
    print("\nWrote results/classifier/width_curve.json")


if __name__ == "__main__":
    main()
