"""
ppe.py — projected-process-ensemble reservoir with entanglement readout.

Differences from `reservoir_gen.py`, each tracing to a specific property of
O'Donovan et al., PRX Quantum 7, 020322 (2026):

  * No mid-circuit measurement. The paper's DETERMINISTIC interventions are
    unitary, so the conditional output state |Y_{R|x}> stays pure. Probing the
    system every step turns the process into the paper's monitored case, where
    entanglement is suppressed (Figs. 5d-f, 6b,d). Dropping the probe also makes
    the state exactly simulable, so features are shot-noise free.

  * Readout is the Renyi-2 entanglement entropy S_x = -log tr[rho_A^2], the
    paper's Eq. (12) -- a second-moment quantity. A single-site <sigma_z> is a
    first moment and is structurally blind to what the paper identifies as the
    discriminating signal.

  * Neel initial state at half filling (Eq. 15). Appendix F shows the probes
    fail for ground or typical states; |100...0> is a single-excitation state in
    which XXZ is a free magnon regardless of Delta.

  * Quenched disorder: the block unitary is built ONCE and reused at every
    timestep, so disorder is static rather than a per-step noise process.

Evolution is done by dense matrix application on a batch of statevectors, which
is far faster than per-shot circuit simulation at these system sizes.
"""
from __future__ import annotations

import numpy as np
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator

from qrc.ham_gen import MODEL_BLOCKS

# Qiskit convention: qubit 0 is the LEAST significant bit of the state index.
Q_SYS = 0


# ---------------------------------------------------------------------------
# Operators
# ---------------------------------------------------------------------------

def build_block_unitary(model_key: str, n_qubits: int, dt: float, n_steps: int,
                        model_kwargs: dict | None = None) -> np.ndarray:
    """Dense unitary for one inter-intervention evolution. Built once => quenched."""
    qc = QuantumCircuit(n_qubits)
    MODEL_BLOCKS[model_key](qc, list(range(n_qubits)), dt=dt, n_steps=n_steps,
                            **(model_kwargs or {}))
    return np.asarray(Operator(qc).data, dtype=complex)


def pauli_interventions() -> list[np.ndarray]:
    """The 4 deterministic single-qubit interventions {I, Z, X, Y} as pi-rotations."""
    I = np.eye(2, dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    return [I, -1j * Z, -1j * X, -1j * Y]      # exp(-i pi/2 P), phases are irrelevant


def ry(theta):
    """Batched RY(theta) as (n, 2, 2)."""
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    out = np.empty((theta.size, 2, 2), dtype=complex)
    out[:, 0, 0] = c; out[:, 0, 1] = -s
    out[:, 1, 0] = s; out[:, 1, 1] = c
    return out


def neel_state(n_qubits: int) -> np.ndarray:
    """|0101...> at half filling -- the paper's Eq. (15) initial condition."""
    idx = sum((q % 2) << q for q in range(n_qubits))
    psi = np.zeros(1 << n_qubits, dtype=complex)
    psi[idx] = 1.0
    return psi


def single_excitation_state(n_qubits: int) -> np.ndarray:
    """|100...0> -- what reservoir_gen.py uses; kept for A/B comparison."""
    psi = np.zeros(1 << n_qubits, dtype=complex)
    psi[1 << Q_SYS] = 1.0
    return psi


INITIAL_STATES = {"neel": neel_state, "single": single_excitation_state}


# ---------------------------------------------------------------------------
# Batched application
# ---------------------------------------------------------------------------

def _apply_1q(states: np.ndarray, ops: np.ndarray, q: int, n_qubits: int) -> np.ndarray:
    """Apply per-window 2x2 operators `ops` (n,2,2) to qubit q of `states` (n, 2^L)."""
    n = states.shape[0]
    lo, hi = 1 << q, 1 << (n_qubits - q - 1)
    v = states.reshape(n, hi, 2, lo)
    return np.einsum("nij,nhjl->nhil", ops, v, optimize=True).reshape(n, -1)


def renyi2(states: np.ndarray, k: int, n_qubits: int) -> np.ndarray:
    """
    Renyi-2 entanglement entropy across the cut {qubits 0..k-1} : {k..L-1}.

    S = -log tr[rho_A^2], with tr[rho_A^2] = ||M^dag M||_F^2 for M the state
    reshaped to (2^(L-k), 2^k). This is the paper's Eq. (12).
    """
    n = states.shape[0]
    M = states.reshape(n, 1 << (n_qubits - k), 1 << k)
    G = np.einsum("nba,nbc->nac", M.conj(), M, optimize=True)
    purity = np.einsum("nac,nac->n", G, G.conj(), optimize=True).real
    return -np.log(np.clip(purity, 1e-300, 1.0))


def sigma_z(states: np.ndarray, n_qubits: int) -> np.ndarray:
    """<sigma_z> on every qubit -> (n, L)."""
    p = np.abs(states) ** 2
    idx = np.arange(1 << n_qubits)
    signs = np.stack([1.0 - 2.0 * ((idx >> q) & 1) for q in range(n_qubits)], axis=1)
    return p @ signs


# ---------------------------------------------------------------------------
# Finite-shot estimation
# ---------------------------------------------------------------------------

def _sample_sigmaz(Z, shots, rng):
    """<sigma_z> from `shots` projective measurements: P(|1>) = (1 - <sz>)/2."""
    p1 = np.clip(0.5 * (1.0 - Z), 0.0, 1.0)
    return 1.0 - 2.0 * rng.binomial(shots, p1) / shots


def _sample_entropy(S, shots, rng, dim_A):
    """
    Renyi-2 entropy from a finite-sample purity estimate.

    Models the two-copy (SWAP-test / Bell-basis) measurement, which is how
    tr[rho^2] is actually obtained on hardware: the outcome is binary with
    P(+1) = (1 + tr[rho^2]) / 2, so the purity estimate is
    tr[rho^2]_hat = 2k/N - 1 with k ~ Binomial(N, P(+1)).

    This is the honest cost of a Renyi-2 readout. Exact-statevector simulation
    silently assumes infinite shots, which is exactly the assumption that makes
    a narrow P(S_x) look free.
    """
    P = np.exp(-S)
    p = np.clip(0.5 * (1.0 + P), 0.0, 1.0)
    Phat = 2.0 * rng.binomial(shots, p) / shots - 1.0
    return -np.log(np.clip(Phat, 1.0 / dim_A, 1.0))


# ---------------------------------------------------------------------------
# Reservoir
# ---------------------------------------------------------------------------

def reservoir_features(
    X_windows: np.ndarray,
    model_key: str,
    num_memory: int = 4,
    dt: float = 0.06,
    n_steps: int = 5,
    washout_length: int = 4,
    model_kwargs: dict | None = None,
    initial_state: str = "neel",
    encoding: str = "symbolic",
    cuts: tuple[int, ...] | None = None,
    use_entropy: bool = True,
    use_sigmaz: bool = True,
    continuous_scale: float = 1.0,
    batch_size: int = 512,
    shots: int | None = None,
    shot_seed: int = 0,
    block_unitary: np.ndarray | None = None,
) -> np.ndarray:
    """
    Run the reservoir over all windows and return a feature matrix.

    X_windows : (n_windows, W). Integer symbols 0-3 if encoding='symbolic',
                real values if encoding='continuous' (angle-encoded via RY).
    Returns   : (n_windows, n_post_washout_steps * n_features_per_step)

    Total physical time between interventions is n_steps * dt -- `dt` is the
    Trotter step, not the inter-intervention interval.

    `block_unitary` overrides the Hamiltonian-derived evolution with a supplied
    dense unitary. This is how the Haar-random control is built: passing it
    explicitly keeps the control local to the call, whereas the earlier approach
    of monkeypatching the module-level `build_block_unitary` mutates shared
    state and silently corrupts any concurrent caller in the same interpreter.
    """
    X_windows = np.asarray(X_windows)
    n_win, W = X_windows.shape
    L = 1 + num_memory
    dim = 1 << L
    if cuts is None:
        cuts = tuple(range(1, L))          # every contiguous bipartition

    _rng = np.random.default_rng(shot_seed)
    if block_unitary is None:
        U = build_block_unitary(model_key, L, dt, n_steps, model_kwargs)
    else:
        U = np.asarray(block_unitary, dtype=complex)
        if U.shape != (dim, dim):
            raise ValueError(f"block_unitary must be {(dim, dim)}, got {U.shape}")
    A = pauli_interventions()
    psi0 = INITIAL_STATES[initial_state](L)

    # Washout drives the reservoir with the identity intervention.
    wash = np.zeros((n_win, washout_length), dtype=X_windows.dtype)
    full = np.concatenate([wash, X_windows], axis=1)
    T = full.shape[1]

    feats = []
    for start in range(0, n_win, batch_size):
        blk = full[start:start + batch_size]
        nb = blk.shape[0]
        states = np.repeat(psi0[None, :], nb, axis=0)
        rows = []
        for t in range(T):
            # Order is intervene -> evolve -> record. A local intervention on the
            # system qubit cannot change the entanglement across any cut that
            # contains it, so recording immediately after the intervention would
            # make S_x lag the input by one step and be constant on the first
            # post-washout step. Evolving first lets the intervention spread.
            if encoding == "symbolic":
                lab = blk[:, t].astype(int)
                ops = np.stack([A[v] for v in lab])
            elif encoding == "continuous":
                # theta = (pi/2)(u+1) maps u in [-1,1] onto [0,pi], where
                # <sigma_z> = cos(theta) is MONOTONE. Encoding u directly as the
                # angle makes the response cos(u) -- an even function -- so +u and
                # -u give identical states and a linear readout recovers exactly
                # nothing by symmetry. The offset is what makes the encoding
                # injective over the input range.
                u = np.clip(continuous_scale * blk[:, t].astype(float), -1.0, 1.0)
                ops = ry(0.5 * np.pi * (u + 1.0))
            else:
                raise ValueError(f"unknown encoding {encoding!r}")
            states = _apply_1q(states, ops, Q_SYS, L)    # intervene
            states = states @ U.T                       # evolve
            if t >= washout_length:
                step = []
                if use_entropy:
                    E = np.stack([renyi2(states, k, L) for k in cuts], axis=1)
                    if shots:
                        for ci, k in enumerate(cuts):
                            E[:, ci] = _sample_entropy(E[:, ci], shots, _rng, 2 ** min(k, L - k))
                    step.append(E)
                if use_sigmaz:
                    Zf = sigma_z(states, L)
                    if shots:
                        Zf = _sample_sigmaz(Zf, shots, _rng)
                    step.append(Zf)
                rows.append(np.concatenate(step, axis=1))
        feats.append(np.concatenate(rows, axis=1))
    return np.concatenate(feats, axis=0)


# ---------------------------------------------------------------------------
# Paper diagnostics for THIS setup
# ---------------------------------------------------------------------------

def ppe_diagnostics(model_key: str, num_memory: int = 4, n_interventions: int = 5,
                    dt: float = 0.06, n_steps: int = 5, washout_length: int = 4,
                    initial_state: str = "neel", model_kwargs: dict | None = None) -> dict:
    """
    Enumerate the full intervention alphabet and report the paper's PPE moments.

    With 4 deterministic interventions and n_B steps this is 4^n_B sequences --
    1024 for n_B=5 -- so the entire ensemble is enumerable and the mean and
    variance of the bipartite entanglement (Eqs. 13-14) are exact, not sampled.
    """
    import itertools
    L = 1 + num_memory
    seqs = np.array(list(itertools.product(range(4), repeat=n_interventions)))
    F = reservoir_features(seqs, model_key, num_memory=num_memory, dt=dt, n_steps=n_steps,
                           washout_length=washout_length, initial_state=initial_state,
                           model_kwargs=model_kwargs, cuts=(max(1, L // 2),),
                           use_entropy=True, use_sigmaz=False)
    S_final = F[:, -1]                    # S_x at the last intervention
    mean, std = float(S_final.mean()), float(S_final.std())
    return {
        "model": model_key, "L": L, "n_B": n_interventions,
        "total_time": n_steps * dt, "n_sequences": len(seqs),
        "mean_S": mean, "std_S": std,
        "coeff_var": std / mean if mean > 1e-12 else float("nan"),
        "max_S": float(np.log(2 ** (L // 2))),
    }
