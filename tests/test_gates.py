"""
Gate-level correctness tests for the qrc reservoir primitives.

Every two-qubit block is checked against scipy's matrix exponential, up to a
global phase. These tests exist because all three bugs they cover were silent:
the circuits ran and produced plausible-looking metrics while implementing the
wrong Hamiltonian.
"""
import numpy as np
import scipy.linalg as la
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Operator, Statevector, partial_trace

from qrc.ham_gen import apply_xx, apply_yy, apply_zz
from qrc.reserve_mem import heisenberg_pair
from qrc.reservoir_gen import _weak_probe

X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]])
Z = np.diag([1, -1]).astype(complex)
kron = np.kron


def equal_up_to_phase(A, B, atol=1e-9):
    """True if A == e^{iφ}·B for some global phase φ."""
    idx = np.unravel_index(np.argmax(np.abs(A)), A.shape)
    if abs(A[idx]) < 1e-12:
        return np.allclose(A, B, atol=atol)
    return np.allclose(A * (B[idx] / A[idx]), B, atol=atol)


@pytest.mark.parametrize("theta", [0.0, 0.3, 0.7, 1.9, -1.1])
@pytest.mark.parametrize("fn,P,name", [(apply_xx, X, "xx"), (apply_yy, Y, "yy"), (apply_zz, Z, "zz")])
def test_two_qubit_pauli_blocks(fn, P, name, theta):
    """apply_PP(θ) must equal exp(-i·θ/2·P⊗P)."""
    qc = QuantumCircuit(2)
    fn(qc, 0, 1, theta)
    want = la.expm(-1j * theta / 2 * kron(P, P))
    assert equal_up_to_phase(Operator(qc).data, want), f"apply_{name} != exp(-i θ/2 {name.upper()})"


@pytest.mark.parametrize("theta", [0.4, 1.3])
def test_yy_is_not_zz(theta):
    """Regression: apply_yy used an RZ basis change, making it identical to apply_zz."""
    a = QuantumCircuit(2); apply_yy(a, 0, 1, theta)
    b = QuantumCircuit(2); apply_zz(b, 0, 1, theta)
    assert not equal_up_to_phase(Operator(a).data, Operator(b).data)


@pytest.mark.parametrize("Jdt", [0.0, 0.15, 0.3, -0.45])
def test_heisenberg_pair(Jdt):
    """heisenberg_pair(Jdt) must equal exp(+i·Jdt·(XX+YY+ZZ)) (module sign convention)."""
    qc = QuantumCircuit(2)
    heisenberg_pair(qc, 0, 1, Jdt)
    H = kron(X, X) + kron(Y, Y) + kron(Z, Z)
    # Trotter error is zero here: the three terms of the Heisenberg pair commute.
    want = la.expm(+1j * Jdt * H)
    assert equal_up_to_phase(Operator(qc).data, want, atol=1e-8)


@pytest.mark.parametrize("Jdt", [0.25])
def test_heisenberg_pair_is_entangling(Jdt):
    """Regression: XX and YY terms had collapsed to single-qubit rotations on q2."""
    qc = QuantumCircuit(2)
    heisenberg_pair(qc, 0, 1, Jdt)
    U = Operator(qc).data
    # A product of single-qubit gates cannot entangle |+0>.
    c = QuantumCircuit(2); c.h(0); c.append(Operator(U), [0, 1])
    rho = partial_trace(Statevector(c), [1]).data
    purity = np.trace(rho @ rho).real
    assert purity < 0.999, "heisenberg_pair is not entangling"


@pytest.mark.parametrize("eps", [0.12, 0.25, 0.6])
def test_weak_probe_is_informative(eps):
    """The ancilla outcome distribution must depend on the system state."""
    probs = []
    for bit in (0, 1):
        qc = QuantumCircuit(2, 1)
        if bit:
            qc.x(0)
        qc.cry(eps, 0, 1)          # same operation _weak_probe applies, pre-measurement
        rho = partial_trace(Statevector(qc), [0]).data
        probs.append(rho[1, 1].real)
    assert abs(probs[0] - probs[1]) > 1e-6, "weak probe carries no information about sigma_z"
    assert probs[0] == pytest.approx(0.0, abs=1e-9)
    assert probs[1] == pytest.approx(np.sin(eps / 2) ** 2, abs=1e-9)


def test_weak_probe_circuit_shape():
    """_weak_probe must emit a controlled rotation, measure, and reset."""
    qc = QuantumCircuit(2, 1)
    _weak_probe(qc, 0, 1, 0, epsilon=0.25)
    names = [inst.operation.name for inst in qc.data]
    assert "cry" in names and "measure" in names and "reset" in names


# ---------------------------------------------------------------------------
# block_unitary injection
# ---------------------------------------------------------------------------
# The Haar control used to be installed by monkeypatching the module-level
# build_block_unitary. It is now passed explicitly. These pin down that the
# refactor is behaviour-preserving on the default path and actually effective on
# the injected one -- a silently ignored argument would make every "random
# unitary control" column a duplicate of the quantum column.

def test_injected_block_unitary_reproduces_the_default_path():
    from qrc.ppe import build_block_unitary, reservoir_features
    rng = np.random.default_rng(0)
    U_in = rng.uniform(-1, 1, size=(6, 5))
    kw = dict(num_memory=2, dt=0.04, n_steps=5, washout_length=1,
              encoding="continuous", initial_state="neel")
    ref = reservoir_features(U_in, "XXZ", **kw)
    U = build_block_unitary("XXZ", 3, kw["dt"], kw["n_steps"], None)
    got = reservoir_features(U_in, "XXZ", block_unitary=U, **kw)
    assert np.allclose(ref, got, atol=1e-12)


def test_injected_block_unitary_actually_changes_the_features():
    from scipy.stats import unitary_group
    from qrc.ppe import reservoir_features
    rng = np.random.default_rng(0)
    U_in = rng.uniform(-1, 1, size=(6, 5))
    kw = dict(num_memory=2, dt=0.04, n_steps=5, washout_length=1,
              encoding="continuous", initial_state="neel")
    ref = reservoir_features(U_in, "XXZ", **kw)
    haar = reservoir_features(U_in, "XXZ", block_unitary=unitary_group.rvs(8, random_state=3), **kw)
    assert not np.allclose(ref, haar, atol=1e-6)


def test_injected_block_unitary_rejects_wrong_dimension():
    from qrc.ppe import reservoir_features
    with pytest.raises(ValueError, match="block_unitary must be"):
        reservoir_features(np.zeros((2, 4)), "XXZ", num_memory=2,
                           encoding="continuous", block_unitary=np.eye(4))


def test_reservoir_features_does_not_mutate_module_state():
    """A control arm must not leave the module changed for the next unit."""
    import qrc.ppe as ppe
    from scipy.stats import unitary_group
    before = ppe.build_block_unitary
    ppe.reservoir_features(np.zeros((2, 4)), "XXZ", num_memory=2, encoding="continuous",
                           block_unitary=unitary_group.rvs(8, random_state=1))
    assert ppe.build_block_unitary is before
