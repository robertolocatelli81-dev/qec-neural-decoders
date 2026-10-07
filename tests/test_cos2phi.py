"""Tests of the cos(2φ) physics module and of the biased-noise circuits (seconds on CPU). Checks C1–C5 and F1–F2 of
prereg/PREREG_cos2phi_20261007.md. Run: python tests/test_cos2phi.py"""
import os
import sys

import numpy as np
import stim

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qecnd.biased import biased_circuit, data_qubits, deformed_face_pattern, deformed_qubits  # noqa: E402
from qecnd.cos2phi import (Cos2PhiParams, doublet_splitting_ghz, gamma1_flux_1f, hamiltonian, matrix_element,  # noqa: E402
                           pauli_channel, pauli_probabilities_for_bias, spectrum, tphi_from_t2)

P = Cos2PhiParams()


def test_c1_truncation_converges():
    e30 = spectrum(P, n_max=30, levels=4).energies
    e60 = spectrum(P, n_max=60, levels=4).energies
    assert np.max(np.abs(e30 - e60)) < 1e-6          # GHz: < 1 kHz


def test_c2_parity_blocks_and_charge_dispersion_positive_control():
    pure = P.with_(ej1=0.0)                            # φ_ext = π: the sin φ term vanishes too
    h = np.asarray(hamiltonian(pure, n_max=20))
    n = np.arange(-20, 21)
    odd_even = (n[:, None] + n[None, :]) % 2 == 1
    assert np.max(np.abs(h[odd_even])) == 0.0         # no coupling between parities
    s = spectrum(pure, n_max=40, levels=2)
    assert abs(abs(s.parity[0]) - 1) < 1e-9 and abs(abs(s.parity[1]) - 1) < 1e-9
    # positive control: the doublet is the charge dispersion between the parity sectors; it must change by orders of
    # magnitude when E_J2/E_C changes by 4 (transmon exponential suppression)
    light = doublet_splitting_ghz(pure.with_(ej2=P.ej2 / 4))
    heavy = doublet_splitting_ghz(pure.with_(ej2=P.ej2 * 4))
    mid = doublet_splitting_ghz(pure)
    assert light > 5 * mid > 50 * heavy             # measured 100.85 / 12.83 / 0.053 MHz (7 Oct 2026): a factor 16 in
    assert light > 1000 * heavy                     # E_J2 moves the doublet by 1900×


def test_c3_doublet_near_the_measured_13_6_mhz():
    f = doublet_splitting_ghz(P) * 1e3                  # MHz
    assert abs(f - 13.6) / 13.6 < 0.20, f
    # the sign of the cos(2φ) term matters: with −E_J2 the doublet is a different number (lesson of 29 September 2026)
    f_wrong = doublet_splitting_ghz(P.with_(ej2=-P.ej2)) * 1e3
    assert abs(f_wrong - f) / f > 0.05, (f, f_wrong)


def test_c4_charge_matrix_elements_doublet_vs_plasmon():
    s = spectrum(P, n_max=40, levels=4)
    assert s.parity[0] > 0.99 and s.parity[1] < -0.99 and s.parity[2] < -0.99 and s.parity[3] > 0.99   # |0+⟩|0−⟩|1−⟩|1+⟩
    protected = 2 * abs(matrix_element("N", s, 0, 1))     # in units of e (n = 2N): the paper's 1.3e-2 and 2.2 read so
    forbidden = 2 * abs(matrix_element("N", s, 0, 2))     # |0+⟩ → |1−⟩ changes parity: ~0 (E_J1 only)
    plasmon = 2 * abs(matrix_element("N", s, 0, 3))       # |0+⟩ → |1+⟩, the unprotected plasmon
    assert plasmon > 100 * protected                       # the paper: 2.2 / 1.3e-2 = 170 ("100-fold" in its abstract)
    assert 1.5 < plasmon < 3.0 and forbidden < 1e-6


def test_c5_t1_from_flux_noise_within_factor_2_of_70_us():
    s = spectrum(P, n_max=40, levels=4)
    t1 = 1.0 / gamma1_flux_1f(P, s)
    assert 35e-6 < t1 < 140e-6, t1


def test_twirl_matches_closed_form_and_bias_is_t1_over_tphi():
    t1, t2 = 70.0, 2.5
    tphi = tphi_from_t2(t2, t1)
    for t in (0.001, 0.1, 1.0):
        ch = pauli_channel(t, t1, tphi)
        px = (1 - np.exp(-t / t1)) / 4
        pz = (1 - 2 * np.exp(-t / t2) + np.exp(-t / t1)) / 4
        assert abs(ch["p_X"] - px) < 1e-12 and abs(ch["p_Y"] - px) < 1e-12 and abs(ch["p_Z"] - pz) < 1e-12
        assert abs(sum(ch[k] for k in ("p_I", "p_X", "p_Y", "p_Z")) - 1) < 1e-12
    assert abs(pauli_channel(1e-4, t1, tphi)["eta"] - t1 / tphi) / (t1 / tphi) < 1e-3
    px, py, pz = pauli_probabilities_for_bias(0.01, 27.5)
    assert abs(px + py + pz - 0.01) < 1e-15 and abs(pz / (px + py) - 27.5) < 1e-9


def test_f1_deformation_gives_uniform_xzzx():
    for d in (3, 5, 7):
        faces = deformed_face_pattern(d)
        assert len(faces) == d * d - 1
        for corners in faces.values():
            for (dx, dy), pauli in corners.items():
                assert pauli == ("X" if dx == dy else "Z")          # X on the main diagonal, Z on the other, every face


def test_f2_depolarising_noise_is_frame_invariant_and_biased_is_not():
    a = biased_circuit(3, 3, "z", 0.01, 0.01, 0.01, 0.01, xzzx=False)
    b = biased_circuit(3, 3, "z", 0.01, 0.01, 0.01, 0.01, xzzx=True)
    assert str(a) == str(b)
    c = biased_circuit(3, 3, "z", 0.001, 0.001, 0.03, 0.01, xzzx=True)
    assert str(a) != str(c)
    assert len(deformed_qubits(a)) == 5 and c.num_detectors == a.num_detectors == 24


def test_noise_placement_counts():
    c = biased_circuit(3, 3, "x", 0.001, 0.001, 0.03, 0.02, xzzx=False).flattened()
    names = [inst.name for inst in c]
    assert names.count("PAULI_CHANNEL_1") == 3 and names.count("X_ERROR") == 3 and "Z_ERROR" not in names
    assert "DEPOLARIZE1" not in names
    last_noise = max(i for i, n in enumerate(names) if n in ("X_ERROR", "PAULI_CHANNEL_1"))
    assert names.index("MX") > last_noise and names[last_noise + 1] != "MX"     # final data measurement ideal


def test_hamiltonian_hermitian_and_measured_numbers_pinned():
    """Regression pins of the numbers measured on 7 October 2026 (not the paper's): a wider tolerance let a non-Hermitian
    cos φ (12.83 MHz) and a missing factor 2 in Γ1 (86.7 µs) pass C3/C5."""
    h = np.asarray(hamiltonian(P, n_max=40))
    assert np.max(np.abs(h - h.conj().T)) == 0.0
    f = doublet_splitting_ghz(P) * 1e3
    assert abs(f - 12.975) < 0.01, f                       # MHz
    s = spectrum(P, n_max=40, levels=4)
    t1 = 1e6 / gamma1_flux_1f(P, s)
    assert abs(t1 - 43.3) < 0.1, t1                        # µs


def test_xzzx_frame_swaps_px_pz_on_the_deformed_qubits_only():
    """Without the swap the "XZZX" grid would be the CSS grid: every PAULI_CHANNEL_1 on a deformed data qubit carries
    (p_Z, p_Y, p_X), every other one (p_X, p_Y, p_Z), and the two circuits differ."""
    px, py, pz = 0.001, 0.001, 0.03
    c = biased_circuit(3, 3, "z", px, py, pz, 0.01, xzzx=True)
    u = biased_circuit(3, 3, "z", px, py, pz, 0.01, xzzx=False)
    assert str(c) != str(u)
    deformed = deformed_qubits(u)
    seen = {}
    for inst in c.flattened():
        if inst.name == "PAULI_CHANNEL_1":
            args = tuple(inst.gate_args_copy())
            for t in inst.targets_copy():
                seen.setdefault(t.value, []).append(args)
    assert set(seen) == set(deformed_qubits(u)) | (set(data_qubits(u)) - deformed)
    for q, argsets in seen.items():
        want = (pz, py, px) if q in deformed else (px, py, pz)
        assert all(np.allclose(a, want) for a in argsets), (q, argsets, want)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
