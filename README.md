# Quantum Reservoir Computing — benchmark harness

A quantum reservoir / quantum extreme-learning-machine implementation with the
controls needed to tell whether it actually does anything, and a sweep harness
for running that question over many seeds, tasks and configurations in parallel.

The headline finding so far is negative and the repository is built to state it
defensibly rather than to hide it: across every properly controlled comparison
run to date, the quantum feature map does not beat a tuned, width-matched
classical baseline, and usually does not beat a Haar-random unitary of the same
dimension. See [Results so far](#results-so-far) and `results/aggregated/`.

---

## Background

The reservoir design follows O'Donovan, Dowling, Modi & Mitchison, *Diagnosing
Chaos with Projected Ensembles of Process Tensors*, PRX Quantum **7**, 020322
(2026), doi:10.1103/fgc4-hgk1. Model Hamiltonians and parameters (Table I) are
copied into `qrc/ham_gen.py` as `HAM_PARAMS`. The paper's figures of merit
(QDE, spatiotemporal entanglement, PPE moments) measure how quickly local
traces of past interventions are scrambled; prediction needs them preserved.
Read the paper's regime (L = 12–14, Néel initial state, large δt) as a
diagnostic setting, not a recipe for a good reservoir.

The repo has gone through three phases:

1. **Financial QRC prototype (Nov 2025).** Option-price deviation from
   Black–Scholes and swaption-matrix imputation. Series are symbolised to 4
   letters and driven into a small spin chain, with mid-circuit measurement
   and an MLP/Ridge readout.
2. **Comb / process-tensor reservoirs (Dec 2025).** Comb-based and
   weak-measurement reservoirs. **Results from this phase are invalid**: see
   the gate bugs below.
3. **Audit and benchmark harness (Sep 2026).**
   - Found and fixed three silent gate bugs:
     - `apply_yy` was ZZ.
     - `heisenberg_pair` had no XX/YY term.
     - The weak probe carried no system information.
   - Rebuilt the reservoir as a pure-state process (`qrc/ppe.py`) with an
     entanglement readout.
   - Built the controlled, multi-seed sweep harness described below.

---

## Results so far

Single-seed runs from the original runners (`results/ppe/`,
`results/classifier/`) after the bugs were fixed. Each comparator is the best
tuned, width-matched classical method for that domain.

| domain | best quantum | best classical |
|---|---|---|
| financial regression (NRMSE, lower is better) | 0.2405 | 0.0879 (degree-2 features) |
| MNIST, PCA-16, 20k samples (acc) | 0.9075 | 0.9473 (RBF-SVM) |
| source classification (acc) | 0.8677 | 0.9430 (RFF, width-matched) |

* 0/12 MNIST and 0/24 source-classification configurations beat the best
  classical baseline.
* A Haar-random unitary matches or beats the physics Hamiltonian in 10/12 and
  21/24 of them. The choice of Hamiltonian contributes nothing measurable.
* **Re-uploading** the input is the one change that helped a lot: source
  classification went from 0.73 to 0.85 (L=5) and from 0.76 to 0.87 (L=7).
* **512 shots is catastrophic**: source classification drops from 0.85 to
  0.55 and the model ranking collapses into noise.

### Why

* With product encodings, every feature is a tensor-Fourier function of the
  inputs. The **encoding** fixes the function class; the Hamiltonian only
  picks coefficients within it (Schuld, Sweke & Meyer 2021). That is why a
  random unitary does as well.
* Nominal width overstates usable dimension. A 90-feature map had a
  participation ratio of about 13.
* Entanglement shrinks the readout signal. The spread of ⟨σ_z⟩ falls from
  0.28 to 0.03 as L goes from 3 to 11.
* A purely unitary reservoir has no fading memory (streaming memory capacity
  ≈ 0), so it needs either per-window reset or measurement-induced dissipation.

One hypothesis was tested and **falsified**: that PPE distribution width can
select the best model. Its correlation with accuracy is about 0 at every shot
budget.

### Harness runs

Only `negative_controls.json` has been run (54/54 units,
`results/aggregated/negative_controls/`). Both controls behave as designed:

* With `shuffle_labels`, every arm falls to chance.
* With `shuffle_features`, the quantum and Haar arms collapse to chance while
  the classical arms are unaffected.

On the unshuffled tasks the quantum arm is `inferior` or `inconclusive`
against every comparator. The one significant win (timeseries vs `raw_only`)
is still `inconclusive` against the 0.02 margin. Every other config is still to run.

---

## Layout

```
qrc/            library
  ppe.py            reservoir: unitary interventions, Renyi-2 + <sigma_z> readout
  ham_gen.py        XXZ / NNN / IAA Hamiltonian blocks
  classify.py       static feature-map path + classical baselines
  evaluate.py       windowing, splits, ridge readout, time-series baselines
  datasets.py       synthetic benchmarks, equities, options, stock cohorts
  streaming.py      RC proper: continuous drive, no reset, memory capacity
  metrics.py        effective rank, task-conditional entanglement
  calibrate.py      interaction time that hits a target entropy
  spec.py           config -> deterministic, content-addressed work units
  tasks.py          task spec -> split data (disk + in-process cached)
  runner.py         one unit -> one record (all arms, one split)
  store.py          atomic per-unit result files, resume detection
  aggregate.py      shards -> tidy table -> paired stats with error bars
  cache.py          on-disk memoisation of dataset preparation

configs/        experiment definitions (JSON)
scripts/qrc.py  sweep driver: prepare | plan | run | status | aggregate
                              | diagnostics | calibrate | power
results/runs/   one JSON per finished unit (gitignored; regenerate by rerunning)
results/aggregated/  tidy.csv, summary_*.csv, paired_*.csv, headline.json
tests/          gate-level physics tests + harness tests
```

---

## Running a sweep

```bash
# 1. build every dataset cache ONCE, serially
python3 scripts/qrc.py prepare -c configs/timeseries_seeds.json

# 2. see what would run, without running it
python3 scripts/qrc.py plan    -c configs/timeseries_seeds.json
python3 scripts/qrc.py run     -c configs/timeseries_seeds.json --dry-run

# 3. run it
python3 scripts/qrc.py run     -c configs/timeseries_seeds.json --jobs 12

# 4. pilot, then size the seed count from the MEASURED paired variance
python3 scripts/qrc.py run   -c configs/timeseries_seeds.json --limit 40
python3 scripts/qrc.py power -c configs/timeseries_seeds.json

# 5. progress / failures, then aggregate
python3 scripts/qrc.py status    -c configs/timeseries_seeds.json
python3 scripts/qrc.py aggregate -c configs/timeseries_seeds.json
```

`prepare` also warms the entropy calibrations. `power` reports the paired
standard deviation actually observed and the number of seeds needed to detect
the configured margin at 80% power -- run it after a pilot and before committing
to a seed count, because a sweep that cannot resolve its own effect size costs
the same as one that can.

Or end to end: `./scripts/run_local.sh configs/smoke.json 8`.

Start with `configs/smoke.json` (14 units). It exercises every code path and
costs seconds, so a broken config fails in seconds rather than at hour six.

### Many machines

Units are content-addressed and independent, so any subset can run anywhere:

```bash
python3 scripts/qrc.py prepare -c configs/timeseries_seeds.json   # once, first
sbatch --array=0-15 scripts/slurm_array.sh configs/timeseries_seeds.json
```

Each array task takes `--shard i/N` of the unit list. Shards are balanced and
contiguous, so no two tasks write the same file and no coordination is needed.

---

## Three families

| family | protocol | what it can support a claim about |
|---|---|---|
| `classification` | static feature map on PCA components | a quantum **feature map** / QELM |
| `timeseries` | windowed, state reset every window | a quantum **feature map** on windows |
| `streaming` | continuous drive, **no reset**, + memory capacity | **reservoir computing** |

This distinction is load-bearing. `reservoir_features` re-initialises the state
for every window, so the first two families contain no recurrence and no fading
memory — whatever they show, they cannot support a claim about *reservoirs*.
Only `streaming` can. It also reports Jaeger linear memory capacity, which is
the number that decides whether a null result is a fact about these tasks or a
structural fact about unitary dynamics.

---

## Why the harness is shaped this way

**One unit = one (task, configuration, seed), evaluated for every arm.** The
quantum arm, the Haar-random control and the classical baselines are computed
together from one prepared split. This is not an optimisation: every claim here
is a *paired* comparison, pairing is only valid if the arms saw the same split,
and keeping them in one unit makes it impossible to schedule them apart.

**One file per unit, named by a hash of its content.** A single aggregate JSON
written at the end of a monolithic script cannot resume, cannot be written by
more than one process, and loses everything if the job dies at 90%. Per-unit
files make the sweep resumable, idempotent and shardable, and a failed unit
records its traceback instead of taking down the run.

**Seeds reach the data, not just the split.** `seed` selects the sample draw,
the chaotic initial condition, the disorder realisation, the Haar draw, the
shot-noise stream and the train/test split, each derived independently by hash.
A sweep whose seeds only reshuffle one split reports error bars far narrower
than the real variability.

**Baselines are tuned and width-matched, every time.** `rff_<n>` is always
matched to the quantum feature count with gamma selected on held-out training
data. An untuned baseline is the specific failure that manufactures apparent
quantum advantage — in this repo it once flipped all 12 MNIST configurations
from "quantum wins" to "quantum loses" with the quantum numbers unchanged.

**Differences are formed within a seed, then averaged.** `aggregate.paired`
computes quantum − control inside each seed before averaging, and reports a
percentile bootstrap CI over seeds. Differencing separately-averaged arms throws
away the pairing and inflates the error bar, which for gaps of ~0.01 accuracy is
the difference between a claim and a coincidence.

**The Haar control is injected, never monkeypatched.** `reservoir_features(...,
block_unitary=U)` passes the control explicitly. The previous approach reassigned
the module-global `ppe.build_block_unitary`, which is invisible to the reader and
corrupts any concurrent caller sharing the interpreter — precisely what a
parallel sweep arranges.

**BLAS is pinned to one thread per worker.** Each unit is already a small dense
linear-algebra job. Unpinned, N workers spawn N×cores threads and the sweep runs
slower than serial. `scripts/qrc.py` sets this before numpy is imported.

**One comparison is declared before the data exists.** Each config carries a
`primary` block naming the family, task, arms and equivalence margin. Everything
else in the sweep is exploratory and BH-corrected; only the primary is
confirmatory. With hundreds of cells some will look significant by chance, and a
reader has no way to know which were chosen afterwards.

**The claim is bounded, not merely un-rejected.** "The CI includes zero" is a
failure to reject, which is not evidence of absence. Every comparison carries a
margin — the smallest difference that would matter — and reports one of
`superior` / `equivalent` / `inferior` / `inconclusive`. `equivalent` means a
difference worth caring about has been *ruled out*; `inconclusive` means the
study cannot tell, which is where an underpowered design lands.

**Representation capacity is measured, not assumed.** The quantum feature vector
concatenates functionally dependent quantities, so its nominal width overstates
its usable dimension — at L=5 the participation ratio is about 4 directions out
of 54 nominal features. Matching a random-feature baseline on nominal width
therefore hands the classical arm more effective capacity. Effective rank is
recorded for every arm so this is visible rather than arguable.

**Entanglement is measured on the inputs that were actually used.**
`ppe_diagnostics` enumerates the symbolic Pauli alphabet, which is a different
input distribution from the continuous encoding the experiments run on, so it
cannot be the x-axis of a claim about those experiments. The entropy columns of
the real feature matrix can, and they are free.

**Models can be compared at matched entanglement, not just matched time.**
Setting `entropy_target` instead of `total_time` resolves the interaction time
per model so every model sits at the same task-conditional entropy. Comparing a
chaotic and a localised model at a common `total_time` confounds "entanglement
does not matter" with "this particular time suited both". Calibration reports
the achieved fraction and whether the target was reachable at all — always read
`achieved_frac` rather than assuming the request was met.

**Negative controls are part of the sweep.** `control: shuffle_labels` must
drive every arm to chance; `control: shuffle_features` must collapse only the
quantum arms while the classical baselines are untouched. Control units are
excluded from every summary automatically and reported separately.

**Every record names the code that produced it.** Git SHA, dirty flag, Python
and library versions. This repo has already had one generation of results
invalidated by three silent Hamiltonian bugs.

---

## Configs

| config | units | status | what it answers |
|---|---|---|---|
| `smoke.json` | 14 | not run | harness validation, all three families + both controls |
| `negative_controls.json` | 54 | **done**, passes | **run this first** — leakage check; nothing else is trustworthy until it passes |
| `classification_seeds.json` | 2700 | not run | breadth grid + entropy dose-response via a `total_time` sweep within each model |
| `entropy_matched.json` | 500 | not run | models compared at equal task-conditional entanglement |
| `streaming.json` | 960 | not run | reservoir computing proper + memory capacity |
| `timeseries_seeds.json` | 2200 | not run | 3 synthetic + options + 7 stock cohorts |
| `shots.json` | 600 | not run | finite-shot degradation, the hardware-relevant ceiling |
| `stocks_bootstrap.json` | 120 | not run | ticker-resampling uncertainty on the equity universe |
| `size_scan.json` | 80 | not run | does more Hilbert space help (with error bars) |

Grid and task keys are validated on load, so a mistyped key fails immediately
instead of silently doing nothing to a 2000-unit sweep.

### A limit worth knowing

`data/2013-06` is a **single month**: each ticker yields at most ~19 daily
returns. "Multiple stocks" therefore cannot mean one task per ticker at `W=10` —
there would be ~9 windows each. It means multiple *cohorts*:

* `vol` — tickers bucketed by realised volatility (does the reservoir help more
  on noisier series?)
* `disjoint` — hash-partitioned, share no ticker, so cohorts are independent
  replicates
* `bootstrap` — resampled with replacement per seed, so seed spread measures
  ticker-sampling uncertainty

---

## Tests

```bash
python3 -m pytest tests/ -q
```

88 tests. `test_gates.py` checks every two-qubit block against
`scipy.linalg.expm` (three silent Hamiltonian bugs motivated these) and pins the
`block_unitary` injection. `test_harness.py` covers the failure modes that are
silent at scale: duplicate unit ids, unbalanced shards, NaN grouping keys
emptying a comparison table, `wins` and `effect` respecting metric direction,
control units leaking into summaries, effective rank recovering a known
dimension, the streaming state genuinely not resetting, and calibration
reporting unreachable targets instead of pretending.

### Known measurement caveats

* The achievable entropy range is model dependent and does not span [0, 1] — the
  pooled-over-cuts fraction saturates near 0.58 because outer cuts have a lower
  maximum than the half-chain cut. High targets are reported as unreachable
  rather than silently approximated.
* Single-series tasks measure interpolation within one trajectory, not
  out-of-distribution generalisation.
* At finite `shots` the quantum arm and its Haar control are both noisy while the
  classical baselines are exact. That is "hardware reality vs classical ceiling",
  not a like-for-like comparison, and should be framed as such.

---

## Legacy

`scripts/run_ppe_reservoir.py`, `run_classifier.py` and `run_source_clf.py` are
the original single-shot, single-seed runners that produced `results/ppe/` and
`results/classifier/`. They still work and are kept for provenance; new work
should go through `scripts/qrc.py`.

`results/heisen_disorder/` and `results/ising_tfim_disorder/` are **invalid** —
every entry has `R2: NaN` with `MAE == RMSE`, i.e. single-sample evaluation, and
they predate the gate-bug fixes. Do not cite them.

`docs/reservoir_explain.pptx` is the slide deck from phase 1. It predates
the gate-bug fixes; do not reuse its numbers.

`run_comb_reservoir.py`, `run_reservoir_comb_weak.py`, `reserve_end-to-end.py`,
`reserve_multip.py` and `swaptions_run.py` are the measured-reservoir pipelines
from phases 1–2. Their settings live in `qrc/config.py`, where `dt` is a Trotter
step: total evolution time is `n_steps · dt`.

`archive/` holds superseded myQLM-era scripts.
