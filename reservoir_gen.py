# reservoir_gen.py
"""
Reservoir construction with intermediate measurements at every timestep,
for time series prediction using spin-chain models.

Key features:
  - Apply intervention, Hamiltonian evolution, then measure data qubit for
    every timestep in one full circuit.
  - Reset data qubit between timesteps to encode next input.
  - Return a single qat.core.Result with all intermediate measurement data.
"""

import numpy as np
from qat.lang.AQASM import Program, RZ, RX, RY, X
from ham_gen import MODEL_BLOCKS


# ---------------------------------------------------------------------------
# Deterministic interventions on the system qubit
# ---------------------------------------------------------------------------

def det_I(pr, q):
    """Identity (no-op)."""
    return

def det_Z(pr, q):
    """Pi rotation around Z (σ_z)."""
    pr.apply(RZ(np.pi), q)

def det_X(pr, q):
    """Pi rotation around X (σ_x)."""
    pr.apply(RX(np.pi), q)

def det_Y(pr, q):
    """Pi rotation around Y (σ_y)."""
    pr.apply(RY(np.pi), q)


DEFAULT_DET_INTERVENTIONS = {
    0: det_I,   # small negative deviation
    1: det_Z,   # small positive deviation
    2: det_X,   # large negative deviation (future)
    3: det_Y,   # large positive deviation (future)
}


# ---------------------------------------------------------------------------
# Reservoir with intermediate measurements (one circuit, multiple timesteps)
# ---------------------------------------------------------------------------

def reservoir_from_binary_sequence(
    x_seq,
    model_key: str,
    num_memory: int = 2,
    shots: int = 1024,
    dt: float = 1.0,
    n_steps: int = 1,
    det_basis: dict | None = None,
    model_kwargs: dict | None = None,
):
    """
    Build and run a reservoir circuit applying interventions and Hamiltonian
    blocks with intermediate measurements at every timestep. Returns a single
    Result object with all intermediate measurement data.

    Parameters
    ----------
    x_seq : sequence of int
        Pre-binarized intervention sequence (labels) of length T.
    model_key : str
        Hamiltonian model string from ham_gen.MODEL_BLOCKS.
    num_memory : int
        Number of memory/environment qubits.
    shots : int
        Shots per circuit execution.
    dt : float
        Time step for Hamiltonian evolution.
    n_steps : int
        Number of Trotter steps per intervention.
    det_basis : dict or None
        Map from label → deterministic intervention function.
    model_kwargs : dict or None
        Additional parameters for the Hamiltonian block.

    Returns
    -------
    result : qat.core.Result
        Single circuit execution result with T intermediate measurements.
    """
    if det_basis is None:
        det_basis = DEFAULT_DET_INTERVENTIONS
    if model_kwargs is None:
        model_kwargs = {}

    x_seq = np.asarray(x_seq, dtype=int)
    T = len(x_seq)

    if model_key not in MODEL_BLOCKS:
        raise ValueError(
            f"Unknown model_key '{model_key}'. Valid keys: {list(MODEL_BLOCKS.keys())}"
        )
    ham_block_fn = MODEL_BLOCKS[model_key]

    pr = Program()
    q_sys = pr.qalloc(1)            # system qubit
    q_mem = pr.qalloc(num_memory)   # memory qubits
    cbit = pr.calloc(1)             # single classical bit for system measurement

    # Initial state: system in |1>, memory in |0...0>
    pr.apply(X, q_sys[0])

    block_qubits = [q_sys[0]] + [q_mem[m] for m in range(num_memory)]

    for t in range(T):
        label = int(x_seq[t])

        # Apply deterministic intervention on system qubit
        det_op = det_basis.get(label)
        if det_op is None:
            raise ValueError(
                f"No deterministic operation defined for label {label}. "
                f"Available: {list(det_basis.keys())}"
            )
        det_op(pr, q_sys[0])

        # Evolve under Hamiltonian block
        ham_block_fn(
            pr,
            block_qubits=block_qubits,
            dt=dt,
            n_steps=n_steps,
            **model_kwargs,
        )

        # Intermediate measurement of system qubit
        pr.measure(q_sys[0], cbit)

        # Reset data qubit for next timestep input
        pr.reset([q_sys[0]])

    circuit = pr.to_circ()
    job = circuit.to_job(nbshots=shots)
    result = job.submit()
    return result
