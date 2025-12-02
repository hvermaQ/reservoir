# ham_gen.py
"""
Hamiltonian block generation for XXZ, XXZ+NNN (chaotic / localized),
and interacting Aubry–André (IAA) models.

This module defines:
  - HAM_PARAMS: global parameter sets for each model / regime.
  - Low-level Trotter blocks implementing the Hamiltonians.
  - Thin wrappers xxz_block / xxz_nnn_chaotic_block /
    xxz_nnn_localized_block / iaa_chaotic_block / iaa_localized_block
    with a common interface:
        block_fn(pr, block_qubits, dt, n_steps, ...)
  - MODEL_BLOCKS: dispatcher from string key -> block function.

All gates act on Atos QLM-style qubits via an existing Program `pr`.
"""

import numpy as np
from qat.lang.AQASM import CNOT, RX, RZ, RY, H

# ---------------------------------------------------------------------------
# Global Hamiltonian parameters (edit these to tune regimes)
# Values follow the PPE paper for XXZ, XXZ+NNN, and IAA.[file:22][web:2]
# ---------------------------------------------------------------------------

HAM_PARAMS = {
    # Interacting integrable XXZ:
    #   H = J ∑ (σ_x^i σ_x^{i+1} + σ_y^i σ_y^{i+1}) + Δ ∑ σ_z^i σ_z^{i+1}
    "XXZ": {
        "J": 1.0,
        "Delta": 0.55,
    },

    # XXZ + NNN chaotic regime:
    #   H = H_XXZ(J, Δ) + ∑ h σ_z^i σ_z^{i+2} + ∑ g σ_z^i
    # with J = 1, Δ = 0.55, h = 0.6, g = 0.1.[file:22][web:2]
    "NNN_CHAOTIC": {
        "J": 1.0,
        "Delta": 0.55,
        "h": 0.6,
        "g": 0.1,
    },

    # XXZ + NNN stronger-disorder / localized-like regime
    # (tune h, g as desired).
    "NNN_LOCALIZED": {
        "J": 1.0,
        "Delta": 0.55,
        "h": 1.5,
        "g": 1.5,
    },

    # Interacting Aubry–André (IAA) chaotic regime:
    #   H_IAA = -H_XXZ(J, Δ) + 2 λ ∑_i cos(2π q i) σ_z^i
    # with J = 1, Δ = -1, q = 2/(sqrt(5)+1), λ = 1.[file:22][web:2]
    "IAA_CHAOTIC": {
        "J": 1.0,
        "Delta": -1.0,
        "lam": 1.0,
        "q": 2.0 / (np.sqrt(5.0) + 1.0),
    },

    # IAA localized regime, λ = 5.[file:22][web:2]
    "IAA_LOCALIZED": {
        "J": 1.0,
        "Delta": -1.0,
        "lam": 5.0,
        "q": 2.0 / (np.sqrt(5.0) + 1.0),
    },
}

# ---------------------------------------------------------------------------
# Helper: two-qubit ZZ, XX, YY gates via native rotations
# ---------------------------------------------------------------------------
def apply_zz(pr, q_control, q_target, angle):
    """Implement exp(-i * angle/2 * σ_z ⊗ σ_z). Safe if q_control == q_target."""
    # CNOT-based construction
    pr.apply(CNOT, q_control, q_target)
    pr.apply(RZ(angle), q_target)
    pr.apply(CNOT, q_control, q_target)

def apply_xx(pr, q_control, q_target, angle):
    # Z→X
    pr.apply(H, q_control)
    pr.apply(H, q_target)
    # ZZ in X basis
    pr.apply(CNOT, q_control, q_target)
    pr.apply(RZ(angle), q_target)
    pr.apply(CNOT, q_control, q_target)
    # X→Z
    pr.apply(H, q_control)
    pr.apply(H, q_target)

def apply_yy(pr, q_control, q_target, angle):
    """
    Implement exp(-i * angle/2 * σ_y ⊗ σ_y).
    Safe if q_control == q_target: reduces to a single-qubit RY.
    """
    # Basis chanßge Z→Y on both qubits
    pr.apply(RZ(0.5 * np.pi), q_control)
    pr.apply(RZ(0.5 * np.pi), q_target)
    # ZZ in the rotated basis
    pr.apply(CNOT, q_control, q_target)
    pr.apply(RZ(angle), q_target)
    pr.apply(CNOT, q_control, q_target)
    # Rotate back
    pr.apply(RZ(-0.5 * np.pi), q_control)
    pr.apply(RZ(-0.5 * np.pi), q_target)


# ---------------------------------------------------------------------------
# XXZ nearest-neighbour block
# ---------------------------------------------------------------------------

def multi_qubit_xxz_block(pr, block_qubits, J, Delta, dt, n_steps):
    """
    Trotterized XXZ evolution on a 1D chain with periodic boundaries:

        H_XXZ = J ∑_i (σ_x^i σ_x^{i+1} + σ_y^i σ_y^{i+1})
                + Δ ∑_i σ_z^i σ_z^{i+1}.
    """
    L = len(block_qubits)
    for _ in range(n_steps):
        for i in range(L):
            j = (i + 1) % L
            qi, qj = block_qubits[i], block_qubits[j]

            # XX + YY terms
            angle_xy = 2.0 * J * dt
            apply_xx(pr, qi, qj, angle_xy)
            apply_yy(pr, qi, qj, angle_xy)

            # ZZ term
            angle_zz = 2.0 * Delta * dt
            apply_zz(pr, qi, qj, angle_zz)


# ---------------------------------------------------------------------------
# XXZ with NNN ZZ + on-site Z fields
# ---------------------------------------------------------------------------

def multi_qubit_xxz_nnn_block(
    pr,
    block_qubits,
    J,
    Delta,
    h,
    g,
    dt,
    n_steps,
    rng=None,
    use_random=False,
):
    """
    Trotterized XXZ + NNN model:

        H = H_XXZ(J, Δ)
            + ∑_i h σ_z^i σ_z^{i+2}
            + ∑_i g σ_z^i.

    If use_random=True, h and g are treated as *scales* and
    site-dependent values are drawn from uniform[-1,1] each Trotter step.
    """
    if rng is None:
        rng = np.random.default_rng()

    L = len(block_qubits)

    for _ in range(n_steps):
        # 1) Nearest-neighbour XXZ part
        multi_qubit_xxz_block(pr, block_qubits, J, Delta, dt, 1)

        # 2) NNN ZZ couplings
        if use_random:
            h_i = h * rng.uniform(-1.0, 1.0, size=L)
        else:
            h_i = h * np.ones(L)

        for i in range(L):
            j = (i + 2) % L
            qi, qj = block_qubits[i], block_qubits[j]
            angle_nnn = 2.0 * h_i[i] * dt
            apply_zz(pr, qi, qj, angle_nnn)

        # 3) On-site Z fields
        if use_random:
            g_i = g * rng.uniform(-1.0, 1.0, size=L)
        else:
            g_i = g * np.ones(L)

        for idx, q in enumerate(block_qubits):
            angle_z = 2.0 * g_i[idx] * dt
            pr.apply(RZ(angle_z), q)


# ---------------------------------------------------------------------------
# Interacting Aubry–André (IAA) block
# ---------------------------------------------------------------------------

def multi_qubit_iaa_block(
    pr,
    block_qubits,
    J,
    Delta,
    lam,
    q,
    dt,
    n_steps,
):
    """
    Trotterized IAA model:

        H_IAA = -H_XXZ(J, Δ) + 2 λ ∑_i cos(2π q i) σ_z^i,

    with site index i taken as 0..L-1 along block_qubits.[file:22]
    """
    L = len(block_qubits)

    for _ in range(n_steps):
        # -H_XXZ(J, Δ) implemented as XXZ with flipped couplings
        multi_qubit_xxz_block(pr, block_qubits, -J, -Delta, dt, 1)

        # Quasiperiodic on-site Z potential
        for i, qbit in enumerate(block_qubits):
            v_i = 2.0 * lam * np.cos(2.0 * np.pi * q * i)
            angle = 2.0 * v_i * dt
            pr.apply(RZ(angle), qbit)


# ---------------------------------------------------------------------------
# Thin wrappers using global HAM_PARAMS (for reservoir_gen)
# ---------------------------------------------------------------------------

def xxz_block(pr, block_qubits, dt, n_steps):
    """
    XXZ model using HAM_PARAMS['XXZ'].
    """
    p = HAM_PARAMS["XXZ"]
    multi_qubit_xxz_block(
        pr,
        block_qubits=block_qubits,
        J=p["J"],
        Delta=p["Delta"],
        dt=dt,
        n_steps=n_steps,
    )


def xxz_nnn_chaotic_block(pr, block_qubits, dt, n_steps, rng=None, use_random=False):
    """
    Chaotic XXZ+NNN regime using HAM_PARAMS['NNN_CHAOTIC'].
    """
    p = HAM_PARAMS["NNN_CHAOTIC"]
    multi_qubit_xxz_nnn_block(
        pr,
        block_qubits=block_qubits,
        J=p["J"],
        Delta=p["Delta"],
        h=p["h"],
        g=p["g"],
        dt=dt,
        n_steps=n_steps,
        rng=rng,
        use_random=use_random,
    )


def xxz_nnn_localized_block(pr, block_qubits, dt, n_steps, rng=None, use_random=False):
    """
    Localized / strong-disorder XXZ+NNN regime using HAM_PARAMS['NNN_LOCALIZED'].
    """
    p = HAM_PARAMS["NNN_LOCALIZED"]
    multi_qubit_xxz_nnn_block(
        pr,
        block_qubits=block_qubits,
        J=p["J"],
        Delta=p["Delta"],
        h=p["h"],
        g=p["g"],
        dt=dt,
        n_steps=n_steps,
        rng=rng,
        use_random=use_random,
    )


def iaa_chaotic_block(pr, block_qubits, dt, n_steps):
    """
    Interacting Aubry–André in chaotic regime (λ = 1).
    """
    p = HAM_PARAMS["IAA_CHAOTIC"]
    multi_qubit_iaa_block(
        pr,
        block_qubits=block_qubits,
        J=p["J"],
        Delta=p["Delta"],
        lam=p["lam"],
        q=p["q"],
        dt=dt,
        n_steps=n_steps,
    )


def iaa_localized_block(pr, block_qubits, dt, n_steps):
    """
    Interacting Aubry–André in localized regime (λ = 5).
    """
    p = HAM_PARAMS["IAA_LOCALIZED"]
    multi_qubit_iaa_block(
        pr,
        block_qubits=block_qubits,
        J=p["J"],
        Delta=p["Delta"],
        lam=p["lam"],
        q=p["q"],
        dt=dt,
        n_steps=n_steps,
    )


# ---------------------------------------------------------------------------
# Dispatcher from string key → block function
# ---------------------------------------------------------------------------

MODEL_BLOCKS = {
    "XXZ": xxz_block,
    "NNN_CHAOTIC": xxz_nnn_chaotic_block,
    "NNN_LOCALIZED": xxz_nnn_localized_block,
    "IAA_CHAOTIC": iaa_chaotic_block,
    "IAA_LOCALIZED": iaa_localized_block,
}
