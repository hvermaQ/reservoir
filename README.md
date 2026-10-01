# Quantum Reservoir Computing: Benchmarks Against Classical Baselines

## Premise

A quantum reservoir computer uses a small, fixed quantum system as a feature
generator. Inputs are written into the system one step at a time and the
system evolves under its own Hamiltonian between inputs. Measured observables
become features for a simple trained readout, usually linear or ridge
regression. Only the readout is trained; the quantum system is not.

The usual argument for why this might help:

1. The state space grows exponentially with the number of qubits, so a few
   qubits might supply a very rich feature space.
2. The dynamics mixes inputs nonlinearly, which might replace a hand-designed
   nonlinear feature map.
3. Entanglement creates correlations that are costly to produce classically.
4. The system remembers past inputs, which is what time-series prediction
   needs.

This repository tests that argument directly. It builds a quantum reservoir
from the spin-chain models of O'Donovan et al. (see [Reference](#reference))
and applies it to two kinds of problem:

* **Classification**, where the reservoir is a fixed nonlinear feature map
  (a quantum extreme learning machine).
* **Time-series prediction**, where the reservoir processes a window of past
  values to predict the next one.

Every quantum result is reported next to two kinds of comparator, evaluated
on the same data split:

* **Classical baselines** whose hyperparameters are tuned and whose feature
  count matches the quantum model's.
* **A Haar-random unitary** of the same dimension, used in place of the
  physical Hamiltonian. If the random unitary does as well, the specific
  physics is not contributing.

**Result so far.** In every controlled comparison run to date, the quantum
feature map does not beat the best tuned classical baseline, and a
Haar-random unitary usually does as well as or better than the physical
Hamiltonian. The sections below describe how each pipeline works, step by
step, and what it found.

---

## The shared reservoir

Both arcs use the same core, `qrc/ppe.py::reservoir_features`.

**System.** A ring (periodic chain) of `L = 1 + num_memory` qubits (default `L = 5`). Qubit 0
is the *system* qubit that receives inputs; the rest form the memory.

**Initial state.** Néel state `|0101…⟩` at half filling, following the paper.
Other initial states (single excitation, ground state) make the dynamics
trivial or uninformative.

**Hamiltonians** (`qrc/ham_gen.py`, parameters from the paper's Table I):

| key | model | regime |
|---|---|---|
| `XXZ` | XXZ chain, J = 1, Δ = 0.55 | integrable, interacting |
| `NNN_CHAOTIC` | XXZ + next-nearest-neighbour, h = 0.6, g = 0.1 | chaotic |
| `NNN_LOCALIZED` | XXZ + next-nearest-neighbour, h = 1.5, g = 1.5 | localised |
| `IAA_CHAOTIC` | interacting Aubry–André, λ = 1 | chaotic |
| `IAA_LOCALIZED` | interacting Aubry–André, λ = 5 | localised |

Each model is compiled into a Trotterised circuit, then turned into a dense
unitary `U` for one inter-input interval of duration `total_time`. `U` is
built once and reused at every step, so any disorder is static. Every
two-qubit block is tested against `scipy.linalg.expm` in `tests/test_gates.py`.

**Overview** (`L = 5`, Néel start, `W` input slots):

```
 input slots     u_1               u_2                       u_W
                  │                 │                         │
 encode     θ_k = (π/2)(u_k + 1)    │                         │
                  ▼                 ▼                         ▼
 q0 sys |0⟩ ──[RY(θ_1)]──┬─────┬──[RY(θ_2)]──┬─────┬── ··· ──[RY(θ_W)]──┬─────┬──
 q1     |1⟩ ─────────────┤     ├─────────────┤     ├── ··· ─────────────┤     ├──
 q2     |0⟩ ─────────────┤  U  ├─────────────┤  U  ├── ··· ─────────────┤  U  ├──
 q3     |1⟩ ─────────────┤     ├─────────────┤     ├── ··· ─────────────┤     ├──
 q4     |0⟩ ─────────────┴─────┴─────────────┴─────┴── ··· ─────────────┴─────┴──
                               │                   │                          │
 record                       f_1                 f_2                        f_W
              f_k = [ S_1 … S_{L-1}  (Rényi-2 entropy at each cut) ,
                      ⟨σ_z⟩_0 … ⟨σ_z⟩_{L-1} ]                  2L−1 = 9 numbers
                               │                   │                          │
                               └─────────┬─────────┴────────── ··· ───────────┘
                                         ▼
                         x = [ f_1 | f_2 | … | f_W ]      W·(2L−1) features
                                         ▼
                 standardise ─► classical readout (trained; U is not)
                                ├─ classification: logistic regression
                                └─ time series:    ridge regression


 U = exp(−i H τ),  τ = total_time.  First-order Trotter, n_steps slices of dt = τ / n_steps:

     U ≈ [ for each ring bond (i, i+1), applied in turn:
             e^{−i J dt XᵢXᵢ₊₁},  then e^{−i J dt YᵢYᵢ₊₁},  then e^{−i Δ dt ZᵢZᵢ₊₁}
           then, NNN models: e^{−i hᵢ dt ZᵢZᵢ₊₂} and e^{−i gᵢ dt Zᵢ}  (hᵢ, gᵢ random per seed in the harness)
                 IAA models: J, Δ → −J, −Δ, plus e^{−i 2λ cos(2πq i) dt Zᵢ}  ]^n_steps

 Haar control: the same circuit with U replaced by a Haar-random 2^L × 2^L unitary.
```

The diagram shows the default continuous encoding. Symbolic inputs replace
`RY(θ_k)` with one of I, Z, X or Y.

The slots are filled differently in each arc:
* **Classification:** the 16 PCA components of one sample, repeated `k`
  times with re-uploading, so `W = 16k`.
* **Time series:** the `W = 10` values of one window, preceded by 4 washout
  slots that record nothing. The washout input is the value 0: the identity
  under symbolic encoding, but `RY(π/2)` under the default continuous
  encoding.

The state is re-initialised to Néel for every sample or window, except in
the streaming protocol.

**One step of the reservoir:**

1. **Encode.** Apply a single-qubit rotation to the system qubit.
   * *Continuous* inputs `u ∈ [-1, 1]` use `RY(θ)` with `θ = (π/2)(u + 1)`.
     The offset matters: without it `+u` and `−u` produce identical
     measurement statistics.
   * *Symbolic* inputs (4 letters) use one of the Pauli rotations
     {I, Z, X, Y}.
2. **Evolve.** Apply `U` to the whole chain.
3. **Record.** Read out features from the resulting pure state:
   * the Rényi-2 entanglement entropy `S = −log tr ρ_A²` across every cut
     of the chain (`L − 1` values), and
   * `⟨σ_z⟩` on every qubit (`L` values).

   That gives `2L − 1` features per step (9 at `L = 5`). Features from every
   step are concatenated.

There is no mid-circuit measurement. The state stays pure and the simulation
is an exact statevector calculation. Finite-shot noise can be switched on
with `shots=N`: `⟨σ_z⟩` is sampled binomially and the entropy is estimated
from a two-copy purity measurement, which is how it would be measured on
hardware.

**Haar control.** The same pipeline is run with `U` replaced by a
Haar-random unitary of the same size (`block_unitary=`). Everything else,
including encoding and readout, is identical.

---

## Arc 1: Classification

Here the quantum system acts as a fixed feature map, with no time axis. Code:
`qrc/classify.py`; runners `scripts/run_classifier.py`,
`scripts/run_source_clf.py`, and the `classification` family in the harness.

### Tasks

* **Digits / MNIST.** sklearn's 8×8 digits, or MNIST (20,000 samples).
* **Source classification** (`qrc/datasets.py::source_classification`).
  Windows of 16 points drawn from four generators: Lorenz-x, Rössler-x,
  Mackey–Glass, and AR(1) noise. The label is the generator. Each window is
  standardised individually, so amplitude and offset carry no information and
  only temporal structure can solve the task. A linear classifier on raw
  windows gets 0.265 (chance is 0.25), so the task needs a nonlinear model.

### Pipeline

1. **Split** into train and test (70/30, stratified).
2. **Reduce** (images only). Standardise, then PCA to 16 components. Both are
   fitted on the training set only, to avoid leaking test data.
3. **Scale to [−1, 1]** with `tanh((x − μ) / 2.5σ)`, using training
   statistics.
4. **Encode** each sample's 16 values as 16 consecutive reservoir inputs.
   With re-uploading (`n_reupload = k`) the sequence is repeated `k` times.
5. **Read out** features after every input: `16 × k × (2L − 1)` features in
   total (144 at `L = 5`, `k = 1`).
6. **Classify** with standardised logistic regression.
7. **Compare**, on the same split, against:
   * majority class and logistic regression on the PCA components;
   * RBF-SVM, with C and γ tuned on a held-out part of the training set;
   * random Fourier features (an RBF-kernel approximation) with the **same
     number of features** as the quantum map and γ tuned the same way;
   * the Haar-random-unitary control.
8. **Diagnose** whether the map preserves class structure, independently of
   the readout: 5-nearest-neighbour accuracy, silhouette score and Fisher
   ratio, before and after the quantum map.

### Results

Single-seed runs, `L ∈ {5, 7}`, models `XXZ`, `IAA_CHAOTIC`,
`IAA_LOCALIZED`, `total_time = 0.2`, re-uploading 1 or 2.

| task | best quantum | best classical | Haar ≥ quantum |
|---|---|---|---|
| MNIST, PCA-16, 20k (`results/classifier/sweep_mnist20k_fixed.json`) | 0.9075 | 0.9473 (RBF-SVM) | 9 / 12 configs |
| Source classification (`results/classifier/source_clf.json`) | 0.8677 | 0.9430 (RFF, 416 features, tuned) | 21 / 24 configs |

* **No configuration beats the best classical baseline** (0 / 12 on MNIST,
  0 / 24 on source classification). On MNIST, width-matched random Fourier
  features (0.924–0.938) also beat every quantum configuration.
* **The quantum map loses class structure.** kNN accuracy on MNIST falls from
  0.871 on the PCA components to 0.78–0.81 after the quantum map, in every
  configuration.
* **Re-uploading is the largest improvement found.** Source classification
  rises from 0.73 to 0.85 at `L = 5` and from 0.76 to 0.87 at `L = 7`
  (`XXZ`). Repeating the encoding enlarges the set of frequencies the model
  can represent (Schuld, Sweke & Meyer 2021), so it changes the function
  class, unlike changing the Hamiltonian.
* **Performance degrades sharply at 512 shots.** Source-classification
  accuracy falls to 0.48–0.56 for every model, and the differences between
  models are no longer resolvable.

---

## Arc 2: Time-series prediction

Here the reservoir processes a window of past values to predict the next
value. Code: `qrc/evaluate.py`, `qrc/streaming.py`; runner
`scripts/run_ppe_reservoir.py`, and the `timeseries` and `streaming`
families in the harness.

### Tasks

* **Synthetic:** NARMA-10 (depends explicitly on the last 10 inputs),
  Mackey–Glass (τ = 17, chaotic), Lorenz-x.
* **Equities:** daily log returns per ticker from `data/2013-06/`.
* **Options:** for each contract (underlying, strike, expiry), the deviation
  of the market mid price from the Black–Scholes call price.

`data/2013-06/` covers a single month, so each financial series has at most
about 20 points. Breadth comes from pooling many tickers or contracts, not
from series length. The harness groups tickers into *cohorts*:
* `vol`: grouped by realised volatility.
* `disjoint`: hash-partitioned, with no shared tickers.
* `bootstrap`: resampled per seed.

### Pipeline (windowed protocol)

1. **Window.** Slide a window of `W = 10` values over each series; the target
   is the value one step after the window.
2. **Split.**
   * A single series (synthetic tasks) is split in time, 80/20. An embargo of
     `W` windows is dropped at the cut so that no test window overlaps
     training data.
   * A panel of series (stocks, options) is split **by series**, so the same
     ticker never appears in both train and test.
3. **Encode.**
   * *Continuous* (default): scale values to [−1, 1] with training
     statistics, `clip((x − μ) / 3σ)`.
   * *Symbolic*: map each value to one of 4 letters using only trailing
     mean and standard deviation, so no future information leaks in.
4. **Run the reservoir.** Each window starts from a fresh Néel state.
   4 washout steps with input 0 come first, then the `W` inputs. Under the
   default continuous encoding, input 0 means `RY(π/2)`, not the identity.
   Features are recorded after each input: `W × (2L − 1)` features (90 at
   `L = 5`).
5. **Read out** with ridge regression; the regularisation strength is chosen
   on a held-out part of the training set.
6. **Score** with NRMSE, normalised by the *training* standard deviation
   (also RMSE, MAE, R²).
7. **Compare**, on the same split, against:
   * `mean`: predict the training mean. A model that can't beat this has
     learned nothing.
   * `persistence`: predict the last value.
   * `linear_raw`: ridge on the 10 raw lagged values.
   * `linear_symbolic`: ridge on one-hot symbols.
   * the Haar-random-unitary control.
   * `raw_plus_reservoir` versus `raw_plus_random_proj`: does adding
     reservoir features to the raw lags help more than adding the same number
     of random projections of the lags?

### Streaming protocol

The windowed protocol resets the state for every window, so it tests a
feature map applied to windows. It has no recurrence and cannot support a
claim about reservoir *computing*. `qrc/streaming.py` provides the
reservoir protocol proper:

1. Drive each series continuously from one initial state, **never
   resetting**.
2. Predict `x(t+1)` from the state at time `t`, optionally concatenating the
   last few states (`n_tap`).
3. Measure **linear memory capacity** (Jaeger): drive with i.i.d. random
   input and measure how well a linear readout recovers the input `k` steps
   back, summed over `k`.

A purely unitary reservoir conserves information, so it cannot forget:
past inputs are scrambled into global correlations that a linear readout on
local observables cannot recover. In exploratory runs, memory capacity was
close to zero at every evolution time tested. `configs/streaming.json`
measures this with error bars.

### Results

Single-seed run of `scripts/run_ppe_reservoir.py`, `L = 5`, five models,
`total_time ∈ {0.1, 0.2, 0.3, 0.5, 1.75, 8.75}`
(`results/ppe/results.json`). NRMSE, lower is better.

`linear_raw` is the baseline ridge model on the raw lags. "Raw lags only" is
the same model with the lags standardised first, exactly as reservoir features
are. It is the fair reference for the last column.

| task | best quantum alone | `linear_raw` | raw lags only | raw lags + reservoir |
|---|---|---|---|---|
| NARMA-10 | 0.737 | **0.697** | 0.697 | 0.706 |
| Mackey–Glass | 0.0135 | 0.0047 | 0.0026 | **0.0017** |
| Lorenz-x | 0.00085 | 0.0006 | 0.0014 | **0.0002** |
| Options | 0.237 | 0.104 | 0.106 | **0.098** |
| Stocks | 0.979 | 1.023 | 1.322 | 1.212 |

* **On its own, the reservoir never beats a linear model on the raw lags.**
* **Added to the raw lags, reservoir features help** on Mackey–Glass, Lorenz
  and options. This run did not include the matched comparison (raw lags plus
  the same number of random projections). Until that comparison is run, the
  gain can't be credited to the quantum features rather than to the added
  width. `configs/timeseries_seeds.json` includes it.
* **Stocks are not predictable** at one-day horizon from this data: every
  method sits at or above NRMSE ≈ 1, the level of predicting the mean.
* **Evolution time barely matters** in the windowed protocol, because nothing
  is lost between inputs. The best `total_time` was 8.75 on four tasks and
  0.3 on stocks, with NRMSE varying by a few percent across the range.

---

## Why there is no advantage here

Each link of the argument in [Premise](#premise) fails for a specific reason:

1. **State-space size is not feature count.** Features come from a fixed set
   of local observables. Reaching more of the state space needs exponentially
   many measurement settings, and the measured effective rank of the feature
   matrix is far below its nominal width.
2. **The nonlinearity comes from the encoding, not the dynamics.** Quantum
   evolution is linear. With single-qubit angle encoding, every feature is a
   fixed combination of products of sines and cosines of the inputs. The
   encoding fixes the function class; the Hamiltonian only selects
   coefficients within it. This is why a random unitary does as well, and
   why re-uploading, which changes the encoding, is the one lever that helps.
3. **Entanglement costs readout signal.** As entanglement spreads, local
   observables approach their maximally mixed values, so their spread shrinks
   and more shots are needed to resolve them.
4. **No fading memory.** Unitary dynamics cannot forget, so the streaming
   reservoir lacks the echo-state property a reservoir relies on.

The paper's chaos diagnostics measure how quickly local traces of past
inputs are scrambled. Prediction needs those traces preserved, so the two
goals pull in opposite directions.

One idea was tested and **did not hold**: using the width of the paper's
projected-ensemble entropy distribution to choose the best model. Its
correlation with accuracy was about zero at every shot budget.

Directions that remain open: tasks whose inputs are already quantum (no
classical encoding bottleneck), and hardware claims judged on speed or energy
rather than accuracy.

---

## Running it

### Install and test

```bash
pip install -e ".[test]"
python3 -m pytest tests/ -q        # 88 tests
```

`tests/test_gates.py` checks every two-qubit gate block against the exact
matrix exponential. `tests/test_harness.py` covers the harness failure modes
that are silent at scale (duplicate unit ids, unbalanced shards, control
units leaking into summaries, and so on).

### Single-seed runners

```bash
python3 scripts/run_ppe_reservoir.py   # time series  -> results/ppe/
python3 scripts/run_classifier.py      # MNIST/digits -> results/classifier/
python3 scripts/run_source_clf.py      # source clf   -> results/classifier/
```

These produced the results above. New work should use the sweep harness.

### Sweep harness

The harness splits an experiment into independent *units*, each one
(task, configuration, seed). Every unit evaluates all arms (quantum, Haar
control, classical baselines) on the same split, so comparisons are always
paired.

```bash
python3 scripts/qrc.py prepare   -c configs/smoke.json   # build dataset caches once
python3 scripts/qrc.py plan      -c configs/smoke.json   # list units without running
python3 scripts/qrc.py run       -c configs/smoke.json --jobs 8
python3 scripts/qrc.py status    -c configs/smoke.json
python3 scripts/qrc.py aggregate -c configs/smoke.json   # -> results/aggregated/<name>/
```

Other subcommands:
* `power`: after a pilot run, estimates how many seeds are needed to detect
  the configured margin.
* `calibrate`: finds the evolution time that produces a target entanglement.
* `diagnostics`: computes the paper's projected-ensemble diagnostics for a
  model.

`./scripts/run_local.sh configs/smoke.json 8` does `prepare`, `run` and
`aggregate` in one go.

On a cluster, run `prepare` once, then
`sbatch --array=0-15 scripts/slurm_array.sh <config>`. Each array task takes
one shard of the unit list; shards never overlap.

### Configs

| config | units | status | question |
|---|---|---|---|
| `smoke.json` | 14 | not run | does every code path work? (runs in seconds) |
| `negative_controls.json` | 54 | **done, passes** | do shuffled labels and shuffled features behave as expected? |
| `classification_seeds.json` | 2700 | not run | classification across tasks, models and evolution times |
| `entropy_matched.json` | 500 | not run | do models differ when compared at equal entanglement? |
| `streaming.json` | 960 | not run | reservoir computing proper, plus memory capacity |
| `timeseries_seeds.json` | 2200 | not run | 3 synthetic tasks, options, 7 stock cohorts |
| `shots.json` | 600 | not run | how fast does accuracy degrade with finite shots? |
| `stocks_bootstrap.json` | 120 | not run | uncertainty from which tickers are sampled |
| `size_scan.json` | 80 | not run | does a larger chain help? |

**Negative controls.** These check that nothing leaks between train and test
(`results/aggregated/negative_controls/`):
* `shuffle_labels` breaks the link between inputs and labels; every arm must
  fall to chance.
* `shuffle_features` shuffles the quantum features only; the quantum and Haar
  arms must fall to chance while the classical arms are unaffected.

Both behave as expected. On the unshuffled tasks in this config, the quantum
arm is `inferior` or `inconclusive` against every comparator.

### How comparisons are reported

* **Baselines are tuned and width-matched every time.** Random-feature width
  equals the quantum feature count; every classical hyperparameter is chosen
  on held-out training data. Earlier in this project, an untuned baseline
  reversed a conclusion.
* **Seeds vary the data**, not just the split: sample draw, initial
  conditions, disorder, Haar draw, shot noise and split each get an
  independent seed derived from the master seed.
* **Differences are paired.** Quantum minus comparator is computed within
  each seed, then averaged, with a bootstrap confidence interval over seeds.
* **Each comparison gets a verdict**, judged against a stated margin (the
  smallest difference that would matter): `superior`, `equivalent`,
  `inferior` or `inconclusive`. A confidence interval that includes zero is
  `inconclusive`, not `equivalent`.
* **One primary comparison per config** is declared in advance in its
  `primary` block. Everything else is exploratory and corrected for multiple
  comparisons (Benjamini–Hochberg).
* **Effective rank** of every arm's feature matrix is recorded, so
  differences in usable feature dimension are visible.
* **Every record stores its provenance:** git commit, dirty flag, Python
  and library versions.
* **BLAS is pinned to one thread per worker** (`scripts/qrc.py` does this),
  since each unit is already a small dense linear-algebra job.

---

## Repository layout

```
qrc/                    library
  ppe.py                  the reservoir: encoding, evolution, entropy and <sigma_z> readout
  ham_gen.py              XXZ / NNN / IAA Hamiltonians as Trotter circuits
  classify.py             classification arc: PCA, quantum features, classical baselines
  evaluate.py             time-series arc: windowing, splits, ridge readout, baselines
  streaming.py            streaming protocol and memory capacity
  datasets.py             synthetic series, stocks, options, source classification
  metrics.py              effective rank, entanglement statistics
  calibrate.py            evolution time for a target entanglement
  spec.py, tasks.py,      harness: config -> units -> records -> tables
  runner.py, store.py,
  aggregate.py, cache.py
  config.py, data_gen.py, earlier measured-reservoir pipeline (see History)
  reserve_mem.py, reservoir_gen.py, feature_engineering.py, swaptions_aid.py
configs/                harness experiment definitions
scripts/                runners, sweep driver, cluster scripts
results/                committed results (per-unit shards in results/runs/ are not committed)
data/2013-06/           one month of US equity and option quotes
data/*.xlsx             simulated swaption prices (swaption-imputation task)
tests/                  gate-level physics tests and harness tests
docs/                   early slide deck (outdated, see History)
archive/                superseded scripts
```

---

## History

1. **Nov 2025: financial prototype.** Option-price deviation from
   Black–Scholes and swaption-matrix imputation, using a reservoir with a
   projective measurement of the system qubit at every step and an MLP or
   ridge readout. Scripts: `scripts/reserve_end-to-end.py`,
   `scripts/swaptions_run.py`, `scripts/reserve_multip.py`. Settings live in
   `qrc/config.py`; there `dt` is the Trotter step, and the time between
   inputs is `n_steps · dt`.
2. **Dec 2025: comb and weak-measurement reservoirs.**
   `scripts/run_comb_reservoir.py`, `scripts/run_reservoir_comb_weak.py`.
3. **Sep 2026: audit and rebuild.** Three silent gate bugs were found and
   fixed:
   * `apply_yy` implemented ZZ instead of YY.
   * `heisenberg_pair` had no XX or YY term.
   * The weak-measurement probe carried no information about the system.

   The reservoir was then rebuilt as the pure-state model above, and the
   sweep harness was added.

**Results from phases 1 and 2 are invalid** because of the gate bugs. This
covers `results/heisen_disorder/`, `results/ising_tfim_disorder/` and the
slide deck in `docs/`. Do not cite them.

---

## Reference

O'Donovan, Dowling, Modi and Mitchison, *Diagnosing Chaos with
Projected Ensembles of Process Tensors*, PRX Quantum **7**, 020322 (2026),
doi:[10.1103/fgc4-hgk1](https://doi.org/10.1103/fgc4-hgk1). Data:
doi:[10.5281/zenodo.18256994](https://doi.org/10.5281/zenodo.18256994).

The Hamiltonians and their parameters come from Table I. The Néel initial
state follows Appendix F. The entanglement readout is the paper's Eq. (12).
