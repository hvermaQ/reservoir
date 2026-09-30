"""
reserve_mem.py — Qiskit implementation of the qubit-reuse quantum reservoir.

Implements Heisenberg and Ising-TFIM reservoirs with disorder fields,
qubit-reuse architecture, and mid-circuit sigma_z extraction.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

_backend = AerSimulator(method='statevector')


# ---------------------------------------------------------------------------
# Trotterized Heisenberg two-qubit block
# ---------------------------------------------------------------------------

def heisenberg_pair(qc: QuantumCircuit, q1: int, q2: int, Jdt: float):
    """
    Trotterized Heisenberg interaction on a qubit pair:

        exp(+i·Jdt·(XX + YY + ZZ))

    (The historical sign convention of this module is kept: angle = -2·Jdt,
    equivalent to H = -J(XX+YY+ZZ). It is uniform across all three terms, so it
    is a convention on the sign of J, not a bug.)

    Each term uses the CNOT sandwich with the rotation on the qubit whose Pauli
    propagates to both sites: RX on the CONTROL for XX, RZ on the TARGET for ZZ.
    Putting RX on the target instead makes the sandwich collapse to a bare
    single-qubit rotation, because CNOT·(I⊗X)·CNOT = I⊗X.
    """
    angle = -2 * Jdt
    # XX : CNOT · (Rx ⊗ I) · CNOT,  since CNOT·(X⊗I)·CNOT = X⊗X
    qc.cx(q1, q2)
    qc.rx(angle, q1)
    qc.cx(q1, q2)
    # YY : RX(π/2) basis change around ZZ,  since Rx(π/2)† Z Rx(π/2) = Y
    qc.rx(np.pi / 2, q1)
    qc.rx(np.pi / 2, q2)
    qc.cx(q1, q2)
    qc.rz(angle, q2)
    qc.cx(q1, q2)
    qc.rx(-np.pi / 2, q1)
    qc.rx(-np.pi / 2, q2)
    # ZZ : CNOT · (I ⊗ Rz) · CNOT,  since CNOT·(I⊗Z)·CNOT = Z⊗Z
    qc.cx(q1, q2)
    qc.rz(angle, q2)
    qc.cx(q1, q2)


def multi_qubit_heisenberg_block(
    qc: QuantumCircuit, block_qubits: list, J: float, dt: float, n_steps: int
):
    """n_steps of Heisenberg Trotterization on all pairs in block_qubits."""
    for _ in range(n_steps):
        for i in range(len(block_qubits)):
            for j in range(i + 1, len(block_qubits)):
                heisenberg_pair(qc, block_qubits[i], block_qubits[j], J * dt)


def multi_qubit_heisenberg_block_with_random(
    qc: QuantumCircuit,
    block_qubits: list,
    J: float,
    dt: float,
    n_steps: int,
    h_scale: float = 1.0,
    rng=None,
):
    """Heisenberg Trotterization with random on-site Z disorder fields."""
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(n_steps):
        for i in range(len(block_qubits)):
            for j in range(i + 1, len(block_qubits)):
                heisenberg_pair(qc, block_qubits[i], block_qubits[j], J * dt)
        h = h_scale * rng.uniform(-1.0, 1.0, size=len(block_qubits))
        for q_idx, h_i in zip(block_qubits, h):
            qc.rz(2.0 * h_i * dt, q_idx)


def multi_qubit_ising_block_with_random(
    qc: QuantumCircuit,
    block_qubits: list,
    J: float,
    dt: float,
    n_steps: int,
    h_scale: float = 1.0,
    g_scale: float = 1.0,
    rng=None,
):
    """
    Trotterized Ising-TFIM with random longitudinal (Z) and transverse (X) disorder.

    H = J ∑_{i<j} σ_z^i σ_z^j + ∑_i (h_i σ_z^i + g_i σ_x^i)
    """
    if rng is None:
        rng = np.random.default_rng()
    for _ in range(n_steps):
        # ZZ couplings
        for i in range(len(block_qubits)):
            for j in range(i + 1, len(block_qubits)):
                angle = 2.0 * J * dt
                qc.cx(block_qubits[i], block_qubits[j])
                qc.rz(angle, block_qubits[j])
                qc.cx(block_qubits[i], block_qubits[j])
        # Random longitudinal Z fields
        h = h_scale * rng.uniform(-1.0, 1.0, size=len(block_qubits))
        for q_idx, h_i in zip(block_qubits, h):
            qc.rz(2.0 * h_i * dt, q_idx)
        # Random transverse X fields
        g = g_scale * rng.uniform(-1.0, 1.0, size=len(block_qubits))
        for q_idx, g_i in zip(block_qubits, g):
            qc.rx(2.0 * g_i * dt, q_idx)


# ---------------------------------------------------------------------------
# Data-interactions reservoir (one dedicated qubit per timestep)
# ---------------------------------------------------------------------------

def reservoir_with_data_interactions(
    data_vec,
    num_memory: int = 2,
    shots: int = 1024,
    J: float = 1.0,
    dt: float = 0.1,
    n_steps: int = 1,
):
    """
    Non-reuse reservoir: each timestep t gets its own qubit.

    Layout: qubits 0..T-1 are data qubits; qubits T..T+num_memory-1 are memory.
    At each step t, x_t is encoded on qubit t, then all active data qubits
    (0..t) plus memory qubits evolve under a Heisenberg block.
    Final measurement: all T data qubits into classical bits 0..T-1.

    Use extract_sigmaz_reset(result, T) to get per-timestep ⟨σ_z⟩.
    """
    T = len(data_vec)
    total_qubits = T + num_memory
    qc = QuantumCircuit(total_qubits, T)

    mem_qubits = list(range(T, T + num_memory))

    for t, x_t in enumerate(data_vec):
        qc.ry((np.pi / 2) * (x_t + 1), t)
        block_qubits = list(range(t + 1)) + mem_qubits
        multi_qubit_heisenberg_block(qc, block_qubits, J, dt, n_steps)

    for t in range(T):
        qc.measure(t, t)

    qc_t = transpile(qc, _backend, optimization_level=0)
    return _backend.run(qc_t, shots=shots).result()


# ---------------------------------------------------------------------------
# Qubit-reuse reservoir
# ---------------------------------------------------------------------------

def reservoir_with_qubit_reuse(
    data_vec,
    num_memory: int = 2,
    shots: int = 1024,
    J: float = 1.0,
    dt: float = 1.0,
    n_steps: int = 1,
    disorder_scale: float = 1.0,
):
    """
    Qubit-reuse reservoir: 1 data qubit + num_memory memory qubits.

    At each timestep t:
      1. Reset data qubit.
      2. Angle-encode x_t via RY((π/2)(x_t + 1)).
      3. Apply Ising-TFIM Trotter block over [data, memory].
      4. Mid-circuit measure data qubit → classical bit t.

    Returns a Qiskit Result object.
    """
    T = len(data_vec)
    total_qubits = 1 + num_memory
    qc = QuantumCircuit(total_qubits, T)

    q_data = 0
    block_qubits = list(range(total_qubits))  # [0, 1, ..., num_memory]

    for t, x_t in enumerate(data_vec):
        qc.reset(q_data)
        qc.ry((np.pi / 2) * (x_t + 1), q_data)
        multi_qubit_ising_block_with_random(
            qc, block_qubits, J, dt, n_steps,
            h_scale=disorder_scale, g_scale=disorder_scale,
        )
        qc.measure(q_data, t)

    qc_t = transpile(qc, _backend, optimization_level=0)
    return _backend.run(qc_t, shots=shots).result()


# ---------------------------------------------------------------------------
# Feature extraction from Qiskit results
# ---------------------------------------------------------------------------

def _sigmaz_from_counts(counts: dict, n_bits: int) -> np.ndarray:
    """
    Compute ⟨σ_z⟩ for each of the first n_bits classical bits from Qiskit counts.

    Qiskit bitstrings are little-endian: rightmost character = classical bit 0.
    ⟨σ_z⟩_t = 1 - 2·P(bit_t = 1).
    """
    total_shots = sum(counts.values())
    bit1_count = np.zeros(n_bits)
    for bitstring, count in counts.items():
        bs = bitstring.replace(' ', '')
        for t in range(n_bits):
            idx = len(bs) - 1 - t  # bit t is at position -(t+1) from right
            if idx >= 0 and bs[idx] == '1':
                bit1_count[t] += count
    return 1 - 2 * (bit1_count / total_shots) if total_shots else np.ones(n_bits)


def extract_sigmaz_reset(result, n_steps: int) -> np.ndarray:
    """
    Compute ⟨σ_z⟩ for each of the n_steps timesteps from a reservoir result.
    """
    return _sigmaz_from_counts(result.get_counts(), n_steps)


def extract_sigmaz_reset_with_washout(
    result, n_steps: int, washout_length: int = 10
) -> np.ndarray:
    """
    Compute ⟨σ_z⟩ for n_steps timesteps after skipping the washout period.

    Classical bits 0..washout_length-1 are discarded; bits washout_length..washout_length+n_steps-1
    are used.
    """
    counts = result.get_counts()
    total_shots = sum(counts.values())
    bit1_count = np.zeros(n_steps)
    for bitstring, count in counts.items():
        bs = bitstring.replace(' ', '')
        for t in range(n_steps):
            actual_t = washout_length + t
            idx = len(bs) - 1 - actual_t
            if idx >= 0 and bs[idx] == '1':
                bit1_count[t] += count
    return 1 - 2 * (bit1_count / total_shots) if total_shots else np.ones(n_steps)


# ---------------------------------------------------------------------------
# Lagged feature construction (pure numpy, unchanged)
# ---------------------------------------------------------------------------

def make_lagged_features(features, targets, window: int):
    """Return X, y using past `window` features to predict next target."""
    X, y = [], []
    for i in range(window, len(features)):
        X.append(features[i - window:i])
        y.append(targets[i])
    return np.array(X), np.array(y)
