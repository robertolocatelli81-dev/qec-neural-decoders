"""Physics of a cos(2φ) Cooper-pair-parity-protected transmon, from the public literature only, in JAX.

Hamiltonian of Roverc'h et al., "Experimental realization of a cos(2φ) transmon qubit", arXiv:2603.13114, Eq. (1)–(2),
signs as printed there:

    H = 4 E_C (N − N_g)^2 + E_J2 cos(2φ) − E_J1 cos(φ) − E_Jφ sin(φ) (φ_ext − π)

diagonalised in the charge basis |N⟩, N = −n_max … n_max (cos(2φ) couples N ↔ N ± 2, cos φ and sin φ couple N ↔ N ± 1).
Energies in GHz (h = 1). Table I of that paper (cos(2φ) transmon model): E_J2 = 1.14, E_J1 = 0.07, E_Jφ = 1.92 GHz,
E_C = 52 MHz. The relaxation rate from 1/f flux noise follows its Eq. (4). The output of this module is a single-qubit
Pauli channel (the Pauli twirl of the T1/Tφ channel) whose bias η = p_Z/(p_X + p_Y) feeds the decoder bench.

Standard quantum mechanics of a circuit; no new physics is claimed. The anharmonicity ratio, where quoted, is
r12 = (e2 − e0)/(e1 − e0) — the other convention (e2 − e1)/(e1 − e0) differs by one.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

H_PLANCK = 6.62607015e-34          # J s (exact, SI 2019)
HBAR = H_PLANCK / (2 * np.pi)
PHI0 = 2.067833848e-15             # Wb, h / 2e


@dataclass(frozen=True)
class Cos2PhiParams:
    """Energies in GHz. Defaults: Table I of arXiv:2603.13114 (cos(2φ) transmon model)."""
    ej2: float = 1.14
    ej1: float = 0.07
    ejphi: float = 1.92
    ec: float = 0.052

    def with_(self, **kw) -> "Cos2PhiParams":
        d = self.__dict__.copy()
        d.update(kw)
        return Cos2PhiParams(**d)


def _x64() -> None:
    """64-bit JAX arrays for the diagonalisation (kHz on top of hundreds of GHz). Enabled on first use, not on import:
    the decoder bench (float32 models) never imports this module."""
    if not jax.config.jax_enable_x64:
        jax.config.update("jax_enable_x64", True)


def charge_operators(n_max: int):
    """(N, cos φ, sin φ, cos 2φ) as dense matrices in the charge basis of dimension 2 n_max + 1."""
    _x64()
    dim = 2 * n_max + 1
    n = jnp.diag(jnp.arange(-n_max, n_max + 1, dtype=jnp.float64))
    up1 = jnp.eye(dim, k=1)                       # |N⟩⟨N+1|
    up2 = jnp.eye(dim, k=2)
    cos1 = 0.5 * (up1 + up1.T)                    # cos φ = (e^{iφ} + e^{−iφ})/2, e^{±iφ} shifts N by ∓1
    sin1 = (up1 - up1.T) / (2j)                   # sin φ = (e^{iφ} − e^{−iφ})/(2i), Hermitian
    cos2 = 0.5 * (up2 + up2.T)
    return n, cos1, sin1, cos2


def hamiltonian(p: Cos2PhiParams, ng: float = 0.0, phi_ext: float = np.pi, n_max: int = 40):
    n, cos1, sin1, cos2 = charge_operators(n_max)
    dim = 2 * n_max + 1
    h = (4 * p.ec * (n - ng * jnp.eye(dim)) @ (n - ng * jnp.eye(dim))
         + p.ej2 * cos2 - p.ej1 * cos1 - p.ejphi * (phi_ext - np.pi) * sin1)
    return h.astype(jnp.complex128)


@dataclass
class Spectrum:
    energies: np.ndarray           # GHz, ascending
    states: np.ndarray             # columns, charge basis
    parity: np.ndarray             # ⟨(−1)^N⟩ per state (±1 when Cooper-pair parity is exact)
    n_max: int


def spectrum(p: Cos2PhiParams, ng: float = 0.0, phi_ext: float = np.pi, n_max: int = 40, levels: int = 6) -> Spectrum:
    h = hamiltonian(p, ng, phi_ext, n_max)
    e, v = jnp.linalg.eigh(h)
    e, v = np.asarray(e[:levels]), np.asarray(v[:, :levels])
    par_op = (-1.0) ** np.arange(-n_max, n_max + 1)
    parity = np.real(np.einsum("ij,i,ij->j", np.conj(v), par_op, v))
    return Spectrum(e - e[0], v, parity, n_max)


def matrix_element(op: str, s: Spectrum, i: int, j: int) -> complex:
    """⟨i|op|j⟩ for op in {"N", "cos", "sin"} between eigenstates of `s`."""
    n, cos1, sin1, _ = charge_operators(s.n_max)
    m = {"N": n, "cos": cos1, "sin": sin1}[op]
    return complex(np.conj(s.states[:, i]) @ np.asarray(m) @ s.states[:, j])


def doublet_splitting_ghz(p: Cos2PhiParams, n_max: int = 40) -> float:
    return float(spectrum(p, n_max=n_max).energies[1])


def gamma1_flux_1f(p: Cos2PhiParams, s: Spectrum, a_phi_in_phi0: float = 5.6e-6, i: int = 0, j: int = 1) -> float:
    """Relaxation rate (1/s) between eigenstates i, j from 1/f flux noise, Eq. (4) of arXiv:2603.13114 as printed:
    Γ = 2 (2π E_Jφ/(Φ0 ħ))^2 S_ΦΦ(ω_q) |⟨i|sin φ|j⟩|^2, S_ΦΦ(ω) = A_Φ^2 (2π · 1 Hz)/ω, A_Φ in units of Φ0/√Hz."""
    omega = 2 * np.pi * abs(s.energies[j] - s.energies[i]) * 1e9           # rad/s
    ejphi_joule = H_PLANCK * p.ejphi * 1e9
    s_phi = (a_phi_in_phi0 * PHI0) ** 2 * (2 * np.pi * 1.0) / omega            # Wb^2/Hz
    m = abs(matrix_element("sin", s, i, j)) ** 2
    return float(2 * (2 * np.pi * ejphi_joule / (PHI0 * HBAR)) ** 2 * s_phi * m)


# ---------------------------------------------------------------- Pauli channel from (T1, Tφ): twirl of the Kraus map

_PAULIS = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]], [[0, -1j], [1j, 0]], [[1, 0], [0, -1]]], dtype=complex)


def kraus_t1_tphi(t: float, t1: float, tphi: float) -> list:
    """Kraus operators of amplitude damping (T1) followed by pure dephasing (Tφ) over a time t (same units)."""
    g = 1 - np.exp(-t / t1)
    k_damp = [np.array([[1, 0], [0, np.sqrt(1 - g)]]), np.array([[0, np.sqrt(g)], [0, 0]])]
    e = np.exp(-t / tphi)
    k_deph = [np.sqrt((1 + e) / 2) * _PAULIS[0], np.sqrt((1 - e) / 2) * _PAULIS[3]]
    return [d @ a for d in k_deph for a in k_damp]


def pauli_transfer_matrix(kraus: list) -> np.ndarray:
    r = np.zeros((4, 4))
    for i in range(4):
        for j in range(4):
            out = sum(k @ _PAULIS[j] @ k.conj().T for k in kraus)
            r[i, j] = np.real(np.trace(_PAULIS[i] @ out)) / 2
    return r


def pauli_channel(t: float, t1: float, tphi: float) -> dict:
    """Pauli twirl of the T1/Tφ channel: keep the diagonal of its Pauli transfer matrix (1, r_x, r_y, r_z) and read
    p_I, p_X, p_Y, p_Z off it. Returns the four probabilities and the bias η = p_Z/(p_X + p_Y)."""
    r = np.diag(pauli_transfer_matrix(kraus_t1_tphi(t, t1, tphi)))
    _, rx, ry, rz = r
    px = (1 + rx - ry - rz) / 4
    py = (1 - rx + ry - rz) / 4
    pz = (1 - rx - ry + rz) / 4
    return {"p_I": float(1 - px - py - pz), "p_X": float(px), "p_Y": float(py), "p_Z": float(pz),
            "eta": float(pz / (px + py)) if px + py > 0 else float("inf")}


def tphi_from_t2(t2: float, t1: float) -> float:
    """1/Tφ = 1/T2 − 1/(2 T1)."""
    return 1.0 / (1.0 / t2 - 1.0 / (2.0 * t1))


def pauli_probabilities_for_bias(p: float, eta: float) -> tuple:
    """(p_X, p_Y, p_Z) with p_X = p_Y, p_X + p_Y + p_Z = p and p_Z/(p_X + p_Y) = η (η = 1 is depolarising)."""
    a = p / (2 * (1 + eta))
    return a, a, p * eta / (1 + eta)
