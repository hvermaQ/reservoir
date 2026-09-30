#instantiate reservoir here based on memory size
from qat.lang.AQASM import Program, RY, RX, RZ, RY, CNOT
import numpy as np
from qat.qpus import get_default_qpu
import pandas as pd

#trotterisation of heisenber interaction between qubit pairs
def heisenberg_pair(pr, q1, q2, Jdt):
    # XX
    pr.apply(CNOT, q1, q2)
    pr.apply(RX(-2*Jdt), q2)
    pr.apply(CNOT, q1, q2)
    # YY
    pr.apply(RY(np.pi/2), q1)
    pr.apply(RY(np.pi/2), q2)
    pr.apply(CNOT, q1, q2)
    pr.apply(RX(-2*Jdt), q2)
    pr.apply(CNOT, q1, q2)
    pr.apply(RY(-np.pi/2), q1)
    pr.apply(RY(-np.pi/2), q2)
    # ZZ
    pr.apply(CNOT, q1, q2)
    pr.apply(RZ(-2*Jdt), q2)
    pr.apply(CNOT, q1, q2)

#nearest neighbour heisenberg between all block_qubits likely data + memory qubits
def multi_qubit_heisenberg_block(pr, block_qubits, J, dt, n_steps):
    """
    Applies n_steps of Heisenberg Trotterized evolution on ALL pairs in block_qubits
    """
    for _ in range(n_steps):
        for i in range(len(block_qubits)):
            for j in range(i+1, len(block_qubits)):
                heisenberg_pair(pr, block_qubits[i], block_qubits[j], J*dt)

# nearest neighbour Heisenberg between all block_qubits + random Z fields
def multi_qubit_heisenberg_block_with_random(pr, block_qubits, J, dt, n_steps, h_scale=1.0, rng=None):
    """
    Applies n_steps of Heisenberg Trotterized evolution on ALL pairs in block_qubits,
    plus random on-site Z fields (disorder) on each qubit.

    pr          : circuit / program object
    block_qubits: list of qubit indices
    J           : base Heisenberg coupling
    dt          : Trotter time step
    n_steps     : number of Trotter steps
    h_scale     : scale of random field strengths (dimensionless)
    rng         : np.random.Generator or None
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(n_steps):
        # 1) Two-qubit Heisenberg interactions
        for i in range(len(block_qubits)):
            for j in range(i + 1, len(block_qubits)):
                heisenberg_pair(pr, block_qubits[i], block_qubits[j], J * dt)

        # 2) Random on-site Z fields (disorder)
        #    H_field = sum_i h_i * sigma_z^i  ⇒  exp(-i h_i dt σ_z/2) ≈ RZ(2*h_i*dt)
        h = h_scale * rng.uniform(-1.0, 1.0, size=len(block_qubits))
        for q, h_i in zip(block_qubits, h):
            pr.apply(RZ(2.0 * h_i * dt), q)

def multi_qubit_ising_block_with_random(pr, block_qubits, J, dt, n_steps, h_scale=1.0, g_scale=1.0, rng=None):
    """
    Applies n_steps of Trotterized Ising evolution on ALL pairs in block_qubits,
    plus random on-site Z fields (disorder) and transverse X fields.

    Ising Hamiltonian: H = J ∑_{i<j} σ_z^i σ_z^j + ∑_i (h_i σ_z^i + g_i σ_x^i)

    pr          : circuit / program object
    block_qubits: list of qubit indices
    J           : base Ising ZZ coupling
    dt          : Trotter time step
    n_steps     : number of Trotter steps
    h_scale     : scale of random longitudinal Z field strengths
    g_scale     : scale of random transverse X field strengths
    rng         : np.random.Generator or None
    """
    if rng is None:
        rng = np.random.default_rng()

    for _ in range(n_steps):
        # Two-qubit Ising ZZ interactions implemented via CNOT-RZ-CNOT
        for i in range(len(block_qubits)):
            for j in range(i + 1, len(block_qubits)):
                angle = 2.0 * J * dt  # rotation angle
                pr.apply(CNOT, block_qubits[i], block_qubits[j])
                pr.apply(RZ(angle), block_qubits[j])
                pr.apply(CNOT, block_qubits[i], block_qubits[j])

        # Random on-site longitudinal Z fields: RZ(2 h_i dt)
        h = h_scale * rng.uniform(-1.0, 1.0, size=len(block_qubits))
        for q, h_i in zip(block_qubits, h):
            pr.apply(RZ(2.0 * h_i * dt), q)

        # Random transverse X fields: RX(2 g_i dt)
        g = g_scale * rng.uniform(-1.0, 1.0, size=len(block_qubits))
        for q, g_i in zip(block_qubits, g):
            pr.apply(RX(2.0 * g_i * dt), q)
"""
#defining the reservoir with data interactions
def reservoir_with_data_interactions(data_vec, num_memory=2, shots=1024, J=1.0, dt=0.1, n_steps=1):
    from qat.qpus import PyLinalg
    T = len(data_vec)
    total_qb = T + num_memory
    pr = Program()
    q = pr.qalloc(total_qb)
    mem_idxs = [T + i for i in range(num_memory)]

    for t, x_t in enumerate(data_vec):
        # Angle encoding
        pr.apply(RY((np.pi/2)*(x_t + 1)), q[t])

        # Form interaction block: all data qubits up to t, plus all memory qubits
        active_data = [q[i] for i in range(t+1)]
        mem_qubits = [q[m] for m in mem_idxs]
        block_qubits = active_data + mem_qubits

        # Apply Trotterized Heisenberg to all pairs in block
        multi_qubit_heisenberg_block(pr, block_qubits, J, dt, n_steps)

    # Measurement: all data qubits
    pr.measure([q[i] for i in range(T)])
    circuit = pr.to_circ()
    qpu = get_default_qpu()
    job = circuit.to_job(nbshots=shots)
    result = qpu.submit(job)
    return result
"""

def reservoir_with_qubit_reuse(data_vec, num_memory=2, shots=1024, J=1.0, dt=1, n_steps=1, disorder_scale=1.0):
    T = len(data_vec)
    pr = Program()
    q_data = pr.qalloc(1)
    q_mem = pr.qalloc(num_memory)
    cbits = pr.calloc(T)  # One classical bit per timestep

    for t, x_t in enumerate(data_vec):
        pr.reset([q_data[0]])
        pr.apply(RY((np.pi/2)*(x_t + 1)), q_data[0])
        block_qubits = [q_data[0]] + [q_mem[m] for m in range(num_memory)]
        #multi_qubit_heisenberg_block(pr, block_qubits, J, dt, n_steps)
        #multi_qubit_heisenberg_block_with_random(pr, block_qubits, J, dt, n_steps, h_scale=disorder_scale)
        multi_qubit_ising_block_with_random(pr, block_qubits, J, dt, n_steps, h_scale=disorder_scale, g_scale=disorder_scale)
        pr.measure(q_data[0], cbits[t])  # store this step's result in cbits[t]

    circuit = pr.to_circ()
    qpu = get_default_qpu()
    job = circuit.to_job(nbshots=shots)
    result = qpu.submit(job)
    return result

def extract_sigmaz_reset(result, n_steps):
    """
    Compute <sigma_z> per timestep from myQLM reservoir output.
    Args:
        result: myQLM Result object (as provided)
        n_steps: number of timesteps (number of intermediate_measurements per Sample)
    Returns:
        np.ndarray: shape (n_steps,), <sigma_z> for each time step
    """
    bit1_prob = np.zeros(n_steps)
    total_prob = 0.0

    # Loop over all shots/samples
    for sample in result.raw_data:
        prob = sample.probability if hasattr(sample, 'probability') else sample['probability']
        total_prob += prob
        # Each sample has a list intermediate_measurements, length n_steps
        # For each time step t, get measured classical bit value (cbits[0])
        for t in range(n_steps):
            int_meas = sample.intermediate_measurements[t]
            cbit_val = int_meas.cbits[0] if hasattr(int_meas, 'cbits') else int_meas['cbits'][0]
            # Accumulate probability for '1' outcome
            if cbit_val == 1:
                bit1_prob[t] += prob
    # Convert to <sigma_z> = 1 - 2*P(1) for each time step
    sigmaz = 1 - 2 * (bit1_prob / total_prob) if total_prob else np.ones(n_steps)
    return sigmaz

def extract_sigmaz_reset_with_washout(result, n_steps, washout_length=10):
    """
    Compute <sigma_z> per timestep, skipping first washout_length measurements.
    """
    bit1_prob = np.zeros(n_steps)
    total_prob = 0.0
    
    for sample in result.raw_data:
        prob = sample.probability if hasattr(sample, 'probability') else sample['probability']
        total_prob += prob
        
        # Skip first washout_length timesteps, extract next n_steps
        for t in range(n_steps):
            meas_idx = washout_length + t  # Start AFTER washout
            int_meas = sample.intermediate_measurements[meas_idx]
            cbit_val = int_meas.cbits[0] if hasattr(int_meas, 'cbits') else int_meas['cbits'][0]
            if cbit_val == 1:
                bit1_prob[t] += prob
    
    sigmaz = 1 - 2 * (bit1_prob / total_prob) if total_prob else np.ones(n_steps)
    return sigmaz


#lagged features for training over all time steps
def make_lagged_features(features, targets, window):
    """ Return X, y using past `window` features to predict next target. """
    X, y = [], []
    for i in range(window, len(features)):
        # X is a vector of window values: [x_{i-window}, ..., x_{i-1}]
        X.append(features[i-window:i])
        y.append(targets[i])  # true value at time step i
    return np.array(X), np.array(y)
