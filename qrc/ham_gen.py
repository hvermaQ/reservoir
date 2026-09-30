# ham_gen.py
"""
Hamiltonian block generation for XXZ, XXZ+NNN (chaotic / localized),
and interacting Aubry-Andre (IAA) models — Qiskit implementation.

This module defines:
  - HAM_PARAMS: global parameter sets for each model / regime.
  - Low-level Trotter blocks implementing the Hamiltonians.
  - Thin wrappers xxz_block / xxz_nnn_chaotic_block /
    xxz_nnn_localized_block / iaa_chaotic_block / iaa_localized_block
    with a common interface:
        block_fn(qc, block_qubits, dt, n_steps, ...)
  - MODEL_BLOCKS: dispatcher from string key -> block function.

All gates act on a Qiskit QuantumCircuit `qc` with integer qubit indices.
"""

import numpy as np
from qiskit import QuantumCircuit

# ---------------------------------------------------------------------------
# Global Hamiltonian parameters
# ---------------------------------------------------------------------------

HAM_PARAMS = {
    # Interacting integrable XXZ:
    #   H = J ∑ (σ_x^i σ_x^{i+1} + σ_y^i σ_y^{i+1}) + Δ ∑ σ_z^i σ_z^{i+1}
    "XXZ": {
        "J": 1.0,
        "Delta": 0.55,
    },

    # XXZ + NNN chaotic regime
    "NNN_CHAOTIC": {
        "J": 1.0,
        "Delta": 0.55,
        "h": 0.6,
        "g": 0.1,
    },

    # XXZ + NNN localized regime
    "NNN_LOCALIZED": {
        "J": 1.0,
        "Delta": 0.55,
        "h": 1.5,
        "g": 1.5,
    },

    # Interacting Aubry-Andre (IAA) chaotic regime
    "IAA_CHAOTIC": {
        "J": 1.0,
        "Delta": -1.0,
        "lam": 1.0,
        "q": 2.0 / (np.sqrt(5.0) + 1.0),
    },

    # IAA localized regime
    "IAA_LOCALIZED": {
        "J": 1.0,
        "Delta": -1.0,
        "lam": 5.0,
        "q": 2.0 / (np.sqrt(5.0) + 1.0),
    },
}


# ---------------------------------------------------------------------------
# Two-qubit interaction primitives
# ---------------------------------------------------------------------------

def apply_zz(qc: QuantumCircuit, q_control: int, q_target: int, angle: float):
    """exp(-i·angle/2·σ_z⊗σ_z) via CNOT-RZ-CNOT."""
    qc.cx(q_control, q_target)
    qc.rz(angle, q_target)
    qc.cx(q_control, q_target)


def apply_xx(qc: QuantumCircuit, q_control: int, q_target: int, angle: float):
    """exp(-i·angle/2·σ_x⊗σ_x) via basis change to Z and ZZ."""
    qc.h(q_control)
    qc.h(q_target)
    qc.cx(q_control, q_target)
    qc.rz(angle, q_target)
    qc.cx(q_control, q_target)
    qc.h(q_control)
    qc.h(q_target)


def apply_yy(qc: QuantumCircuit, q_control: int, q_target: int, angle: float):
    """
    exp(-i·angle/2·σ_y⊗σ_y) via RX basis change and ZZ.

    The basis change must be RX, not RZ: Rx(π/2)† Z Rx(π/2) = Y, whereas RZ
    commutes with Z⊗Z and would leave the ZZ evolution unchanged.
    """
    qc.rx(0.5 * np.pi, q_control)
    qc.rx(0.5 * np.pi, q_target)
    qc.cx(q_control, q_target)
    qc.rz(angle, q_target)
    qc.cx(q_control, q_target)
    qc.rx(-0.5 * np.pi, q_control)
    qc.rx(-0.5 * np.pi, q_target)


# ---------------------------------------------------------------------------
# XXZ nearest-neighbour block
# ---------------------------------------------------------------------------

def multi_qubit_xxz_block(
    qc: QuantumCircuit,
    block_qubits: list,
    J: float,
    Delta: float,
    dt: float,
    n_steps: int,
):
    """
    Trotterized XXZ evolution on a 1D chain with periodic boundaries:

        H_XXZ = J ∑_i (σ_x^i σ_x^{i+1} + σ_y^i σ_y^{i+1})
                + Δ ∑_i σ_z^i σ_z^{i+1}
    """
    L = len(block_qubits)
    for _ in range(n_steps):
        for i in range(L):
            j = (i + 1) % L
            qi, qj = block_qubits[i], block_qubits[j]
            angle_xy = 2.0 * J * dt
            apply_xx(qc, qi, qj, angle_xy)
            apply_yy(qc, qi, qj, angle_xy)
            angle_zz = 2.0 * Delta * dt
            apply_zz(qc, qi, qj, angle_zz)


# ---------------------------------------------------------------------------
# XXZ + NNN ZZ + on-site Z fields
# ---------------------------------------------------------------------------

def nnn_pairs(L: int):
    """
    Distinct next-nearest-neighbour bonds on a periodic chain of length L.

    Naively iterating (i, (i+2) % L) double-counts every bond when L == 4 and
    degenerates onto the nearest-neighbour bonds when L == 3, so the pairs are
    de-duplicated here and self-pairs (L <= 2) dropped.
    """
    return sorted({tuple(sorted((i, (i + 2) % L))) for i in range(L) if i != (i + 2) % L})


def multi_qubit_xxz_nnn_block(
    qc: QuantumCircuit,
    block_qubits: list,
    J: float,
    Delta: float,
    h: float,
    g: float,
    dt: float,
    n_steps: int,
    rng=None,
    use_random: bool = False,
    seed: int | None = None,
    h_field=None,
    g_field=None,
):
    """
    Trotterized XXZ + NNN model:

        H = H_XXZ(J, Δ) + ∑_i h_i σ_z^i σ_z^{i+2} + ∑_i g_i σ_z^i

    Disorder is QUENCHED: h_i and g_i are drawn once for the whole block and
    reused across all n_steps Trotter steps. Redrawing them per step would make
    the fields a time-dependent noise process rather than static disorder,
    destroying both reproducibility and the localized/chaotic distinction the
    NNN_LOCALIZED / NNN_CHAOTIC parameter sets exist to probe.

    Pass explicit h_field / g_field to quench across an entire reservoir run;
    otherwise they are drawn from `rng` (seeded by `seed` when given).
    """
    if rng is None:
        rng = np.random.default_rng(seed)
    L = len(block_qubits)

    if h_field is None:
        h_field = h * (rng.uniform(-1.0, 1.0, size=L) if use_random else np.ones(L))
    if g_field is None:
        g_field = g * (rng.uniform(-1.0, 1.0, size=L) if use_random else np.ones(L))

    pairs = nnn_pairs(L)
    for _ in range(n_steps):
        multi_qubit_xxz_block(qc, block_qubits, J, Delta, dt, 1)
        for i, j in pairs:
            apply_zz(qc, block_qubits[i], block_qubits[j], 2.0 * h_field[i] * dt)
        for idx, q in enumerate(block_qubits):
            qc.rz(2.0 * g_field[idx] * dt, q)


# ---------------------------------------------------------------------------
# Interacting Aubry-Andre (IAA) block
# ---------------------------------------------------------------------------

def multi_qubit_iaa_block(
    qc: QuantumCircuit,
    block_qubits: list,
    J: float,
    Delta: float,
    lam: float,
    q: float,
    dt: float,
    n_steps: int,
):
    """
    Trotterized IAA model:

        H_IAA = -H_XXZ(J, Δ) + 2λ ∑_i cos(2π q i) σ_z^i
    """
    for _ in range(n_steps):
        multi_qubit_xxz_block(qc, block_qubits, -J, -Delta, dt, 1)
        for i, qbit in enumerate(block_qubits):
            v_i = 2.0 * lam * np.cos(2.0 * np.pi * q * i)
            qc.rz(2.0 * v_i * dt, qbit)


# ---------------------------------------------------------------------------
# Thin wrappers using HAM_PARAMS (interface for reservoir_gen)
# ---------------------------------------------------------------------------

def xxz_block(qc: QuantumCircuit, block_qubits: list, dt: float, n_steps: int):
    p = HAM_PARAMS["XXZ"]
    multi_qubit_xxz_block(qc, block_qubits, J=p["J"], Delta=p["Delta"], dt=dt, n_steps=n_steps)


def xxz_nnn_chaotic_block(
    qc: QuantumCircuit, block_qubits: list, dt: float, n_steps: int,
    rng=None, use_random: bool = False, seed: int | None = None,
    h_field=None, g_field=None,
):
    p = HAM_PARAMS["NNN_CHAOTIC"]
    multi_qubit_xxz_nnn_block(
        qc, block_qubits, J=p["J"], Delta=p["Delta"], h=p["h"], g=p["g"],
        dt=dt, n_steps=n_steps, rng=rng, use_random=use_random, seed=seed,
        h_field=h_field, g_field=g_field,
    )


def xxz_nnn_localized_block(
    qc: QuantumCircuit, block_qubits: list, dt: float, n_steps: int,
    rng=None, use_random: bool = False, seed: int | None = None,
    h_field=None, g_field=None,
):
    p = HAM_PARAMS["NNN_LOCALIZED"]
    multi_qubit_xxz_nnn_block(
        qc, block_qubits, J=p["J"], Delta=p["Delta"], h=p["h"], g=p["g"],
        dt=dt, n_steps=n_steps, rng=rng, use_random=use_random, seed=seed,
        h_field=h_field, g_field=g_field,
    )


def iaa_chaotic_block(qc: QuantumCircuit, block_qubits: list, dt: float, n_steps: int):
    p = HAM_PARAMS["IAA_CHAOTIC"]
    multi_qubit_iaa_block(
        qc, block_qubits, J=p["J"], Delta=p["Delta"],
        lam=p["lam"], q=p["q"], dt=dt, n_steps=n_steps,
    )


def iaa_localized_block(qc: QuantumCircuit, block_qubits: list, dt: float, n_steps: int):
    p = HAM_PARAMS["IAA_LOCALIZED"]
    multi_qubit_iaa_block(
        qc, block_qubits, J=p["J"], Delta=p["Delta"],
        lam=p["lam"], q=p["q"], dt=dt, n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Dispatcher: string key -> block function
# ---------------------------------------------------------------------------

MODEL_BLOCKS = {
    "XXZ": xxz_block,
    "NNN_CHAOTIC": xxz_nnn_chaotic_block,
    "NNN_LOCALIZED": xxz_nnn_localized_block,
    "IAA_CHAOTIC": iaa_chaotic_block,
    "IAA_LOCALIZED": iaa_localized_block,
}
