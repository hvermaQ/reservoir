import numpy as np
from qat.lang.AQASM import Program, RZ, RX, RY, X, CNOT, I
from ham_gen import MODEL_BLOCKS
from qat.qpus import get_default_qpu

def det_I(pr, q):
    pr.apply(I, q)

def det_Z(pr, q):
    pr.apply(RZ(np.pi), q)

def det_X(pr, q):
    pr.apply(RX(np.pi), q)

def det_Y(pr, q):
    pr.apply(RY(np.pi), q)

DEFAULT_DET_INTERVENTIONS = {
    0: det_I,
    1: det_Z,
    2: det_X,
    3: det_Y,
}

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
    Run separate circuits per window WITH washout prefix.
    Each window gets [washout_labels + window_labels].
    """
    if det_basis is None:
        det_basis = DEFAULT_DET_INTERVENTIONS
    if model_kwargs is None:
        model_kwargs = {}

    num_windows, window_size = X_windows.shape
    ham_block_fn = MODEL_BLOCKS[model_key]
    
    # Washout labels: all zeros (identity operations)
    washout_labels = np.zeros(washout_length, dtype=int)
    
    all_results = []

    for i in range(num_windows):
        qpu = get_default_qpu()
        # Full sequence for this window: washout + actual window
        full_window = np.concatenate([washout_labels, X_windows[i]])
        
        pr = Program()
        q_sys = pr.qalloc(1)
        q_mem = pr.qalloc(num_memory)
        q_anc = pr.qalloc(1)
        cbit = pr.calloc(1)

        pr.apply(X, q_sys[0])
        block_qubits = [q_sys[0]] + list(q_mem)

        for label in full_window:
            ham_block_fn(pr, block_qubits, dt=dt, n_steps=n_steps, **model_kwargs)

            det_op = det_basis.get(int(label))
            if det_op is None:
                raise ValueError(f"No deterministic op for label {label}")
            det_op(pr, q_sys[0])

            pr.apply(CNOT, [q_sys[0], q_anc[0]])
            pr.measure(q_anc[0], cbit)
            pr.reset([q_anc[0]])

        circuit = pr.to_circ()
        job = circuit.to_job(nbshots=shots)
        result = qpu.submit(job)
        all_results.append(result)

    return all_results  # List[num_windows] of Results
