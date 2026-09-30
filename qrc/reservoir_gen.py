"""
reservoir_gen.py — Qiskit process-comb reservoir for XXZ / NNN / IAA models.

Responsibilities:
  - Take a PRE-BINARIZED intervention sequence x_seq (ints 0-3).
  - Apply deterministic local interventions A_{x_t} on the system qubit.
  - Between interventions, evolve {system + memory} under a Hamiltonian
    block chosen by model_key via ham_gen.MODEL_BLOCKS.
  - Supports standard (strong CNOT-measured) and ancilla weak-measurement variants.
"""

import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator

from qrc.ham_gen import MODEL_BLOCKS

_backend = AerSimulator(method='statevector')


# ---------------------------------------------------------------------------
# Deterministic intervention set on the system qubit
# ---------------------------------------------------------------------------

def det_I(qc: QuantumCircuit, q: int):
    pass  # Identity — no-op


def det_Z(qc: QuantumCircuit, q: int):
    qc.rz(np.pi, q)


def det_X(qc: QuantumCircuit, q: int):
    qc.rx(np.pi, q)


def det_Y(qc: QuantumCircuit, q: int):
    qc.ry(np.pi, q)


DEFAULT_DET_INTERVENTIONS = {
    0: det_I,
    1: det_Z,
    2: det_X,
    3: det_Y,
}


# ---------------------------------------------------------------------------
# Standard reservoir: CNOT-based ancilla measurement per timestep
# ---------------------------------------------------------------------------

def reservoir_results_per_window(
    X_windows,
    model_key: str,
    num_memory: int = 2,
    shots: int = 1024,
    dt: float = 1.0,
    n_steps: int = 1,
    det_basis: dict | None = None,
    model_kwargs: dict | None = None,
    washout_length: int = 5,
):
    """
    Run one circuit per input window with a washout prefix.

    Qubit layout: q_sys=0, q_mem=1..num_memory, q_anc=num_memory+1.
    Classical register: one bit per timestep (washout + window).

    Returns a list of Qiskit Result objects, one per window.
    """
    if det_basis is None:
        det_basis = DEFAULT_DET_INTERVENTIONS
    if model_kwargs is None:
        model_kwargs = {}

    num_windows, window_size = X_windows.shape
    ham_block_fn = MODEL_BLOCKS[model_key]
    washout_labels = np.zeros(washout_length, dtype=int)

    q_sys = 0
    q_mem = list(range(1, 1 + num_memory))
    q_anc = 1 + num_memory
    n_qubits = 2 + num_memory  # sys + memory + ancilla
    block_qubits = [q_sys] + q_mem

    all_results = []

    for i in range(num_windows):
        full_window = np.concatenate([washout_labels, X_windows[i]])
        n_total = len(full_window)

        qc = QuantumCircuit(n_qubits, n_total)
        qc.x(q_sys)  # initial state preparation

        for t, label in enumerate(full_window):
            ham_block_fn(qc, block_qubits, dt=dt, n_steps=n_steps, **model_kwargs)

            det_op = det_basis.get(int(label))
            if det_op is None:
                raise ValueError(f"No deterministic op for label {label}")
            det_op(qc, q_sys)

            # Measure ancilla via CNOT probe, reset ancilla for reuse
            qc.cx(q_sys, q_anc)
            qc.measure(q_anc, t)
            qc.reset(q_anc)

        qc_t = transpile(qc, _backend, optimization_level=0)
        all_results.append(_backend.run(qc_t, shots=shots).result())

    return all_results


# ---------------------------------------------------------------------------
# Weak measurement variant: ancilla-assisted weak probe
# ---------------------------------------------------------------------------

def _weak_probe(qc: QuantumCircuit, q_sys: int, q_anc: int, cbit: int, epsilon: float):
    """
    Ancilla-assisted weak measurement of σ_z on q_sys.

    Applies a genuine controlled-RY(epsilon) so the ancilla outcome depends on
    the system state -- P(anc=1) = 0 for |0> and sin^2(eps/2) for |1> -- then
    measures into classical bit `cbit` and resets the ancilla.

    A CNOT-RY-CNOT sandwich does NOT work here: it yields RY(+eps) for |0> and
    RY(-eps) for |1>, whose measurement statistics are identical, so the
    ancilla record would carry no information about the system.
    """
    qc.cry(epsilon, q_sys, q_anc)
    qc.measure(q_anc, cbit)
    qc.reset(q_anc)


def reservoir_results_per_window_ancilla(
    X_windows,
    model_key: str,
    num_memory: int = 2,
    shots: int = 1024,
    dt: float = 1.0,
    n_steps: int = 1,
    det_basis: dict | None = None,
    model_kwargs: dict | None = None,
    washout_length: int = 5,
    epsilon: float = 0.12,
    final_strong_measure: bool = False,
):
    """
    Quantum reservoir with ancilla-assisted weak measurement at each timestep.

    Classical register: one bit per timestep storing the ancilla outcome.
    Optionally appends a final strong measurement of the system qubit.

    Returns a list of Qiskit Result objects, one per window.
    """
    if det_basis is None:
        det_basis = DEFAULT_DET_INTERVENTIONS
    if model_kwargs is None:
        model_kwargs = {}

    num_windows, window_size = X_windows.shape
    ham_block_fn = MODEL_BLOCKS[model_key]
    washout_labels = np.zeros(washout_length, dtype=int)

    q_sys = 0
    q_mem = list(range(1, 1 + num_memory))
    q_anc = 1 + num_memory
    n_qubits = 2 + num_memory
    block_qubits = [q_sys] + q_mem

    all_results = []

    for i in range(num_windows):
        full_window = np.concatenate([washout_labels, X_windows[i]])
        n_total = len(full_window)
        n_cbits = n_total + (1 if final_strong_measure else 0)

        qc = QuantumCircuit(n_qubits, n_cbits)
        qc.x(q_sys)

        for t_idx, label in enumerate(full_window):
            ham_block_fn(qc, block_qubits, dt=dt, n_steps=n_steps, **model_kwargs)

            det_op = det_basis.get(int(label))
            if det_op is None:
                raise ValueError(f"No deterministic op for label {label}")
            det_op(qc, q_sys)

            _weak_probe(qc, q_sys, q_anc, t_idx, epsilon=epsilon)

        if final_strong_measure:
            qc.measure(q_sys, n_total)

        qc_t = transpile(qc, _backend, optimization_level=0)
        all_results.append(_backend.run(qc_t, shots=shots).result())

    return all_results
