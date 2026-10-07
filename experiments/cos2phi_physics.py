"""Level 1 of the cos(2φ) study: spectrum, matrix elements, relaxation from flux noise, and the Pauli channel of the
qubit, from the published parameters (arXiv:2603.13114, Table I), compared with that paper's measurements.

Usage: python experiments/cos2phi_physics.py > results/cos2phi/physics.json
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from qecnd.cos2phi import (Cos2PhiParams, doublet_splitting_ghz, gamma1_flux_1f, matrix_element, pauli_channel,  # noqa: E402
                           spectrum, tphi_from_t2)

P = Cos2PhiParams()
PAPER = {"doublet_mhz_ramsey": 13.6, "t1_us": 70.0, "t2_echo_us": 2.5, "t2_ramsey_us": 1.4,
         "charge_me_protected_e": 1.3e-2, "charge_me_plasmon_e": 2.2, "a_phi_uphi0_sqrthz": 5.6,
         "readout_fidelity": 0.83, "readout_us": 5.0}

out = {"parameters_ghz": P.__dict__, "paper": PAPER}
conv = {n: spectrum(P, n_max=n, levels=4).energies.tolist() for n in (20, 30, 40, 60)}
out["convergence_levels_ghz_by_nmax"] = conv
out["convergence_max_abs_change_30_to_60_khz"] = float(np.max(np.abs(np.array(conv[30]) - np.array(conv[60]))) * 1e6)
s = spectrum(P, n_max=40, levels=6)
out["levels_ghz"] = s.energies.tolist()
out["parity"] = s.parity.tolist()
doublet = doublet_splitting_ghz(P) * 1e3
out["doublet_mhz"] = doublet
out["doublet_vs_paper_percent"] = (doublet - PAPER["doublet_mhz_ramsey"]) / PAPER["doublet_mhz_ramsey"] * 100
out["doublet_mhz_wrong_sign_minus_ej2"] = doublet_splitting_ghz(P.with_(ej2=-P.ej2)) * 1e3
out["plasmon_ghz_0plus_to_1plus"] = float(s.energies[3])
out["r12_(e2-e0)/(e1-e0)"] = float(s.energies[2] / s.energies[1])
out["r12_other_convention_(e2-e1)/(e1-e0)"] = float((s.energies[2] - s.energies[1]) / s.energies[1])
out["charge_matrix_element_e_protected_0plus_0minus"] = 2 * abs(matrix_element("N", s, 0, 1))
out["charge_matrix_element_e_plasmon_0plus_1plus"] = 2 * abs(matrix_element("N", s, 0, 3))
out["charge_matrix_element_units_note"] = ("in units of e, i.e. 2N with N the Cooper-pair number: our reading of the "
                                           "paper's 1.3e-2 and 2.2 (Fig. 5a); with N itself the numbers halve")
out["charge_suppression_ratio_plasmon_over_protected"] = (out["charge_matrix_element_e_plasmon_0plus_1plus"]
                                                          / out["charge_matrix_element_e_protected_0plus_0minus"])
out["sin_phi_matrix_element_0plus_0minus"] = abs(matrix_element("sin", s, 0, 1))
g1 = gamma1_flux_1f(P, s, a_phi_in_phi0=PAPER["a_phi_uphi0_sqrthz"] * 1e-6)
out["t1_flux_1f_us_eq4"] = 1e6 / g1
out["t1_vs_paper_ratio"] = (1e6 / g1) / PAPER["t1_us"]
# Pauli channel for a short idle t (bias is t-independent for t << T1, Tφ); T1 and T2 are the paper's measurements,
# declared inputs (the echo T2 is not derived here: second order in flux noise at the sweet spot)
t1 = PAPER["t1_us"]
chans = {}
for name, t2 in (("echo", PAPER["t2_echo_us"]), ("ramsey", PAPER["t2_ramsey_us"])):
    tphi = tphi_from_t2(t2, t1)
    c = {"t_phi_us": tphi, "channel_at_t_0.1us": pauli_channel(0.1, t1, tphi), "channel_at_t_1us": pauli_channel(1.0, t1, tphi)}
    c["eta_small_t"] = t1 / tphi
    chans[name] = c
# model-derived T1 with the measured dephasing: a second band edge
tphi_e = tphi_from_t2(PAPER["t2_echo_us"], t1)
chans["model_t1_with_echo_tphi"] = {"t1_us": 1e6 / g1, "t_phi_us": tphi_e, "eta_small_t": (1e6 / g1) / tphi_e}
out["pauli_channels"] = chans
out["eta_band"] = {"min": min(c["eta_small_t"] for c in chans.values()), "max": max(c["eta_small_t"] for c in chans.values()),
                   "pilot_value": 27.5, "note": "η = T1/Tφ; band from echo vs Ramsey T2 (same device) and model vs measured T1"}
# what a round would cost: p per data qubit per round for a round time t_round
out["p_per_round_for_t_round_us"] = {str(t): pauli_channel(t, t1, tphi_from_t2(PAPER["t2_echo_us"], t1))
                                     for t in (0.05, 0.1, 0.5, 1.0)}
print(json.dumps(out, indent=1))
