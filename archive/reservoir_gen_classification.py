# reservoir_gen.py
"""
Reservoir / process-comb construction for XXZ / NNN / IAA models.

Responsibilities:
  - Take a PRE-BINARIZED intervention sequence x_seq (ints).
  - Apply deterministic local interventions A_{x_t} on the system qubit
    according to a configurable intervention dictionary.
  - Between interventions, evolve {system + memory} under a Hamiltonian
    block chosen by model_key via ham_gen.MODEL_BLOCKS.
"""

import numpy as np
from qat.lang.AQASM import Program, RZ, X
from ham_gen import MODEL_BLOCKS

# reservoir_gen.py (relevant parts)

import numpy as np
from qat.lang.AQASM import Program, RZ, RX, RY, X, I
from ham_gen import MODEL_BLOCKS

# ---------------------------------------------------------------------------
# Default deterministic intervention set on the system qubit
# ---------------------------------------------------------------------------

def det_I(pr, q):
    """
    Deterministic intervention ~ I / sqrt(2).
    Implemented as identity (no-op).
    """
    pr.apply(I, q)
    return


def det_Z(pr, q):
    """
    Deterministic intervention ~ σ_z / sqrt(2).
    Implemented as a π rotation around Z (overall phase irrelevant).
    """
    pr.apply(RZ(np.pi), q)


def det_X(pr, q):
    """
    Deterministic intervention ~ σ_x / sqrt(2).
    Implemented as a π rotation around X.
    """
    pr.apply(RX(np.pi), q)


def det_Y(pr, q):
    """
    Deterministic intervention ~ σ_y / sqrt(2).
    Implemented as a π rotation around Y.
    """
    pr.apply(RY(np.pi), q)


# Map integer label → deterministic operation on system qubit.
# 0,1 used now; 2,3 ready for future use when binarization outputs 0–3
# based on ±2σ variance bands in the run script.[file:22][web:31]
DEFAULT_DET_INTERVENTIONS = {
    0: det_I,   # e.g. small negative deviation
    1: det_Z,   # e.g. small positive deviation
    2: det_X,   # e.g. large negative deviation (future use)
    3: det_Y,   # e.g. large positive deviation (future use)
}

# ---------------------------------------------------------------------------
# Comb-style reservoir construction (no preprocessing)
# ---------------------------------------------------------------------------

def reservoir_comb_from_binary_sequence(
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
    Build and run a reservoir / process-comb circuit for a given model and
    a pre-binarized intervention sequence.

    Parameters
    ----------
    x_seq : sequence of int
        Discrete intervention labels for each time step t. Typically
        0/1 for {I, σ_z}, but can be extended (e.g. 0..3 for
        {I, σ_x, σ_y, σ_z}) by extending det_basis.
        Length |x_seq| = n_B = number of interventions.
    model_key : str
        One of:
          "XXZ",
          "NNN_CHAOTIC", "NNN_LOCALIZED",
          "IAA_CHAOTIC", "IAA_LOCALIZED"
        as defined in ham_gen.MODEL_BLOCKS.
    num_memory : int
        Number of environment / memory qubits coupled to the system.
    shots : int
        Number of measurement shots for the final readout.
    dt : float
        Trotter time step passed to the Hamiltonian block.
    n_steps : int
        Number of Trotter steps between interventions.
    det_basis : dict or None
        Mapping from integer labels to deterministic interventions:
            det_basis[k] is a callable (pr, q_sys) → None.
        If None, DEFAULT_DET_INTERVENTIONS ({I, σ_z}) is used.
    model_kwargs : dict or None
        Extra keyword arguments forwarded to the chosen Hamiltonian block
        (e.g. rng, use_random for disordered NNN).

    Returns
    -------
    result : qat.core.Result
        Backend execution result containing bitstring statistics, etc.
    """
    if det_basis is None:
        det_basis = DEFAULT_DET_INTERVENTIONS
    if model_kwargs is None:
        model_kwargs = {}

    x_seq = np.asarray(x_seq, dtype=int)
    T = len(x_seq)

    # Get Hamiltonian block from ham_gen
    if model_key not in MODEL_BLOCKS:
        raise ValueError(
            f"Unknown model_key '{model_key}'. "
            f"Valid keys: {list(MODEL_BLOCKS.keys())}"
        )
    ham_block_fn = MODEL_BLOCKS[model_key]

    # Build circuit
    pr = Program()
    q_sys = pr.qalloc(1)            # system qubit where interventions act
    q_env = pr.qalloc(num_memory)   # memory / environment block
    cbits = pr.calloc(1)            # final readout bit

    # Initial state: system in |1>, env in |0...0>.
    pr.apply(X, q_sys[0])

    block_qubits = [q_sys[0]] + [q_env[m] for m in range(num_memory)]

    # Multi-time comb: interventions + Hamiltonian evolution
    for t in range(T):
        label = int(x_seq[t])

        # 1) Deterministic local intervention on system
        det_op = det_basis.get(label, None)
        if det_op is None:
            raise ValueError(
                f"No deterministic operator defined for label {label}. "
                f"Available labels: {list(det_basis.keys())}"
            )
        det_op(pr, q_sys[0])

        # 2) Many-body Hamiltonian block between interventions
        ham_block_fn(
            pr,
            block_qubits=block_qubits,
            dt=dt,
            n_steps=n_steps,
            **model_kwargs,
        )

    # Final readout of the system qubit
    pr.measure(q_sys[0], cbits[0])

    circuit = pr.to_circ()
    qpu = circuit.to_job(nbshots=shots)
    result = qpu.submit()
    return result