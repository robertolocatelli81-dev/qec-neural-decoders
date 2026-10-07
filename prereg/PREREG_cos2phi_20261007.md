# Pre-registration: learned decoders vs matching when the errors come from cos(2φ)-protected qubits

Written 2026-10-07 (UTC, see the sha256 file next to this one), BEFORE any decoder result and BEFORE the physics module
was run. Everything below is fixed now; anything measured afterwards that departs from it is reported as a departure.

## Question

When the physical errors have the structure of a cos(2φ) Cooper-pair-parity-protected transmon (strongly Z-biased, as
derived below from the published device parameters), do learned decoders beat minimum-weight perfect matching on the
surface code and on its bias-tailored variant (XZZX), and by how much?

## Level 1 — physics of the qubit (`qecnd/cos2phi.py`, JAX, written from the public literature only)

Hamiltonian (Roverc'h et al., arXiv:2603.13114, Eq. 1–2; signs as printed):

    H = 4 E_C (N − N_g)^2 + E_J2 cos(2φ) − E_J1 cos(φ) − E_Jφ sin(φ) (φ_ext − π)

Parameters, Table I of that paper (cos(2φ) transmon model): E_J2 = 1.14 GHz, E_J1 = 0.07 GHz, E_Jφ = 1.92 GHz,
E_C = 52 MHz. Sweet spot N_g = 0, φ_ext = π unless stated. Diagonalisation in the charge basis N ∈ [−N_max, N_max]:
cos(2φ) couples N ↔ N ± 2, cos(φ) and sin(φ) couple N ↔ N ± 1.

Checks fixed in advance (each one is a test in `tests/test_cos2phi.py`):

- C1 convergence: the lowest 4 levels change by < 1 kHz between N_max = 30 and N_max = 60.
- C2 parity: with E_J1 = E_Jφ(φ_ext − π) = 0 the Hamiltonian is block-diagonal in Cooper-pair parity; the doublet is then
  the even/odd ground-state splitting (positive control: the splitting must change by orders of magnitude when E_J2/E_C
  changes by a factor 4 — charge dispersion).
- C3 doublet: the |0+⟩–|0−⟩ splitting at the sweet spot is compared with the paper's measured 13.6 MHz (Ramsey).
  Acceptance for "reproduced": within 20 %; the number is reported whatever it is.
- C4 charge matrix elements: |⟨0+|N|0−⟩| (protected) and |⟨0+|N|1+⟩| (plasmon) are compared with the paper's 1.3 × 10^−2
  and 2.2 (Fig. 5a). Reported as ratios; a 100-fold suppression is the paper's claim.
- C5 relaxation from 1/f flux noise, Eq. (4) of the paper as printed: Γ1 = 2 (2π E_Jφ /(Φ0 ħ))^2 S_ΦΦ(ω_q) |⟨0+|sin φ|0−⟩|^2,
  S_ΦΦ(ω) = A_Φ^2 (2π · 1 Hz)/ω, A_Φ = 5.6 µΦ0/√Hz. Compared with the measured T1 = 70 µs. Acceptance: within a factor 2.
- C6 dephasing: the echo T2 is NOT derived from first principles here (at the sweet spot it is second order in flux noise
  and depends on the low-frequency cut-offs; the paper fits it). The measured T2^echo = 2.5 µs and T2^Ramsey = 1.4 µs
  are used as inputs, declared.
- r12 convention: whenever an anharmonicity ratio is quoted it is (e2 − e0)/(e1 − e0), stated next to the number.

Pauli channel (level-1 output): the T1/T2 channel of duration t (amplitude damping + pure dephasing) is Pauli-twirled.
Its Pauli transfer matrix is diagonal with (1, e^{−t/T2}, e^{−t/T2}, e^{−t/T1}), 1/T2 = 1/(2 T1) + 1/Tφ, so
p_X = p_Y = (1 − e^{−t/T1})/4, p_Z = (1 − 2 e^{−t/T2} + e^{−t/T1})/4 (computed numerically from the Kraus operators in the
module, checked against this closed form in a test). Bias η = p_Z/(p_X + p_Y) → T1/Tφ for t ≪ T1, Tφ.
From the paper's numbers (arithmetic, not a measurement of ours): Tφ^echo = 1/(1/2.5 − 1/140) µs = 2.545 µs →
η_echo = 27.5; Tφ^Ramsey = 1.414 µs → η_Ramsey = 49.5. The device is dephasing-dominated: the parity protection removes
charge-induced relaxation, the remaining noise is flux noise, and the resulting Pauli channel is Z-biased by a factor
27–50 (the band comes from echo vs Ramsey, same device). The pilot uses η = 27.5 (the conservative end).

Readout: the paper reports 83 % single-shot fidelity in 5 µs. A syndrome-measurement flip probability q = 0.17 is above
every known threshold; the pilot therefore uses q = p (the standard phenomenological choice) and states that the device's
readout is far from it. The physical cost of the ancilla measurement is kept in the model (no "free" ancilla).

## Level 2 — codes, noise placement, decoders (`qecnd/biased.py`, `experiments/cos2phi_pilot.py`)

Circuits: stim's rotated surface-code memory circuit (`surface_code:rotated_memory_x` / `_z`, distance d, rounds d)
with ALL built-in noise set to zero, rebuilt with: `PAULI_CHANNEL_1(p_X, p_Y, p_Z)` on every data qubit before every
round (where stim places `before_round_data_depolarization`), `X_ERROR(q)` before every stabiliser-ancilla measurement.
The final data-qubit measurement is ideal (phenomenological model). Gates ideal. This is NOT a circuit-level model:
no gate errors, no bias-preservation question for the two-qubit gates (Etxezarreta Martinez et al. 2505.17718; Ruiz
et al. 2607.20143); that is stated as a limit.

XZZX code: the Clifford-deformed surface code (Bonilla Ataides et al., Nat. Commun. 12, 2172, 2021, arXiv:2009.07851;
Dua et al. "Clifford-deformed surface codes", arXiv:2201.07802): Hadamard on the data qubits of one diagonal family.
In this repository it is implemented in the CSS FRAME: the same stim circuit, with (p_X, p_Z) swapped on the deformed
data qubits — those with (x − y)/2 even in stim's coordinates. Measurements and the observable are unchanged. This
equivalence is exact for the phenomenological model (ideal gates). Checks fixed in advance:
- F1 every face of the rotated lattice has exactly two deformed corners on the same diagonal (so the deformed code is
  XZZX, not a mixed pattern): asserted by a test on the coordinates for d = 3, 5, 7.
- F2 at η = 1 (p_X = p_Y = p_Z) the deformed and undeformed circuits are byte-identical (depolarising noise is
  Hadamard-invariant): positive control of the machinery.
- F3 code distance per circuit and basis with `stim.Circuit.search_for_undetectable_logical_errors`
  (`dont_explore_detection_event_sets_with_size_above = 6`, no edge-degree limit, ignore ungraphlike), reported for
  p_Y = 0 pure-Z noise and for η = 1; the expectation from the literature is that under pure Z the XZZX frame has a
  larger effective distance than CSS memory-X, which has distance d. Reported as measured, whatever it is.
- F4 the detector error model of every cell is scanned for parallel edges (same detector set) with different logical
  effect — the PyMatching merge rule (`Matching.from_detector_error_model` docstring, PyMatching#103) made a 3× "win"
  an artefact on 4 October 2026. The count is reported per cell; a cell with such edges is flagged and the MLP's win,
  if any, is compared in that cell with the near-optimal lookup rather than with matching.

Grid (fixed):
- distances d ∈ {3, 5}, rounds = d;
- bias η ∈ {1 (depolarising control: no advantage expected for either the XZZX frame or the MLP), 27.5 (device, echo)};
- strength p = p_X + p_Y + p_Z per data qubit per round ∈ {0.01, 0.03}, with p_X = p_Y = p/(2(1 + η)), p_Z = p η/(1 + η) (so that
  p_Z/(p_X + p_Y) = η); measurement flip q = p;
- codes/bases: CSS memory-X (the basis hurt by Z errors), CSS memory-Z, XZZX-frame memory-X, XZZX-frame memory-Z.
Total 32 cells. If time permits (stated now, not promised): η = 49.5 at d = 3 only.

Decoders on the SAME test shots (200,000, stim seed 2) in every cell:
- trivial ("no flip");
- PyMatching 2.4.0 from the circuit's detector error model (`decompose_errors=True`), plain;
- MLP (`qecnd/model.py`, hidden 256; 500,000 training shots, seed 1; 4,000 steps × 512; Adam 1e-3; val_frac 0.1 —
  parameters with the best validation loss kept, as in the correlated grid of 4 October);
- null: the same MLP on permuted labels (permutation seed 3, model seed 1);
- at d = 3 only: empirical near-optimal lookup (per 24-detector syndrome, majority outcome over 10,000,000 shots with
  stim seed 11; unseen syndromes fall back to PyMatching; same method as `experiments/lookup_independent.py`).
Logical error rate = fraction of test shots whose predicted observable differs from the true one.

Statistics (fixed): 95 % Wilson interval per rate; exact two-sided McNemar MLP vs PyMatching on the same shots;
Bonferroni over the 32 cells (α = 0.05/32 = 1.56e-3). A cell is a pre-registered WIN only if MLP < PyMatching with
McNemar p < 1.56e-3 AND the null of that cell is within 1.5 points of the trivial rate (the bench can fail there).

Predictions written now:
- P0 (null/control) at η = 1 the MLP does not beat PyMatching in any cell at d = 5, and at d = 3 any win is ≤ 0.3
  points (the near-optimal lookup bounds it; matching is near-optimal for independent noise).
- P1 at η = 27.5, matching on the XZZX frame has a lower logical error than matching on CSS memory-X at equal p
  (the literature's bias-tailoring result); if it does not, the frame implementation is suspect and is re-examined
  before any decoder claim.
- P2 (the actual question) at η = 27.5 the MLP beats PyMatching in at least one of the 16 biased cells by more than
  the P0 margin; the size of the gap is the result. If no biased cell is a WIN: "no learned-decoder advantage from
  cos(2φ) bias at this scale" is the result, reported as such.
- P3 the null stays at the trivial floor (within 1.5 points) in every cell.

Budget: CPU 8 cores / 6 GB; no single command over 30 minutes; the lookup table is the largest item (10 M shots).
All numbers in the report carry the command that produced them and the sha256 of this file.

— Roberto Locatelli, with Noûs (AI agent under his revocable mandate).
