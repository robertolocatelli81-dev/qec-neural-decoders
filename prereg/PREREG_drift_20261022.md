# PREREG — does per-qubit knowledge of flux-drift dephasing help the decoder? (cos(2φ), 2026-10-22)

Written before any measurement of this study. Its SHA-256 is recorded in `PREREG_drift_20261022.sha256` and in every
result file. Local work, not published.

## Question
A cos(2φ) qubit sits at the flux sweet spot (second-order flux sensitivity). Real flux drift moves each qubit off it;
the off-set opens a first-order slope to 1/f flux noise and the qubit's dephasing grows, differently per qubit.
Does a decoder that knows each qubit's current rate (oracle) beat one that only knows the current average rate
(global recalibration), by enough to justify building an online per-qubit estimator? If not, the candidate is closed.

## Real inputs (arXiv:2603.13114, read 2026-10-22 from arxiv.org/html)
- flux drift "0.1 mΦ0 per hour"; A_Φ = 5.6 µΦ0/√Hz with S_ΦΦ ∝ A_Φ²/f; T1 = 70 µs at φext = π and 300 µs at
  π(1 ± 0.002); T2echo = 2.5 µs, T2Ramsey = 1.44 µs; echo envelope exp(−Γν t − (Γφ t)²); doublet second-order in flux.
- Hamiltonian parameters of this repo (`qecnd/cos2phi.py`, fitted to the paper in the 2026-10-07 pilot).
- Model check done before this file (not a result): T1 from flux noise at δΦ = 1e-3 Φ0 = 330 µs (paper 300 µs);
  at δΦ = 0, 43 µs (paper 70 µs, known pilot discrepancy). Curvature d²f/dΦ² = 3.73e13 Hz/Φ0².

## Noise model (assumptions marked A)
- A1: each data qubit j drifts linearly since the last calibration: δ_j(t) = r_j t, r_j ~ N(0, r²), r = 0.1 mΦ0/h,
  independent across qubits and redrawn per episode. (Real: one drift number for one qubit; per-qubit spread assumed.)
- Dephasing rate (Gaussian composition): Γ_j² = Γ0² + Γ1(δ_j)², Γ0 = 1/(2.5 µs) (sweet-spot echo, upper bound of its
  Gaussian part), Γ1(δ) = κ A_Φ |2π f''δ|, two values of κ: √ln2 = 0.833 (echo-like) and 4.6 (Ramsey-like,
  √ln(1/(2π · 1e-4 Hz · 1 µs))). A2: syndrome rounds are not echoed; κ brackets the low-frequency cutoff.
- T1_j(δ) = 70 µs × T1_model(δ)/T1_model(0) (model shape, paper level).
- One round τ = 1 µs. Per round, per data qubit: p_Z = (1 − exp(−(Γ_j τ)²))/2, p_X = p_Y = τ/(4 T1_j). A3: Pauli
  twirl per round (the pilot's choice; coherent accumulation within a shot is not modelled).
- A4 (scale): the device is above threshold; all per-round probabilities are multiplied by one factor k chosen so that
  at t = 0 (all δ = 0) p_X + p_Y + p_Z = p_cal. The relative growth with δ is kept. Measurement flip q = p_cal, fixed.
- Code: rotated surface code, CSS and XZZX frames (`qecnd/biased.py` placement), distance d, d rounds, memory in Z and X.

## Decoders (PyMatching on the circuit DEM)
- M: calibrated at t = 0 (every qubit at its δ = 0 rates).
- G: global recalibration — every qubit at the CURRENT mean (over the episode's qubits) of p_X, p_Y, p_Z.
- O: oracle — each qubit at its own current rates.
- Null N1: O with the episode's per-qubit rates permuted among the data qubits (same multiset, wrong places).

## Grid
t ∈ {0, 0.5, 1, 2, 4} h × κ ∈ {0.833, 4.6} × d ∈ {3, 5} × frame ∈ {css, xzzx} × basis ∈ {z, x} × p_cal ∈ {0.005,
0.01} = 160 cells; 200 episodes × 1000 shots per cell; seeds fixed per cell.

## Controls (must pass before reading the grid)
- C0 (identity): at t = 0, O, G and M give identical predictions on every shot.
- C1 (positive): one data qubit per episode with p_Z = 0.25, the others at p_cal rates (d = 5, xzzx, z, p_cal = 0.005):
  R_OG lower 95 % bound > 0.10. If not, the bench cannot see a per-qubit gain and nothing below is read.
- C2 (null): in C1, N1 must not beat G (R_N1G upper bound ≤ 0.02).

## Metric and decision
L = logical error rate per shot. R_OG = (L_G − L_O)/L_G, R_OM = (L_M − L_O)/L_M; 95 % CI by cluster bootstrap over
episodes (2000 resamples). Cells read: t > 0 and L_G ≤ 0.2.
- OPEN (build the online per-qubit estimator next): R_OG lower bound ≥ 0.10 in at least one xzzx cell.
- CLOSED: R_OG upper bound < 0.10 in every read xzzx cell.
- INCONCLUSIVE otherwise (more episodes, same grid).
R_OM is reported, never used for the decision (it mixes per-qubit value with plain recalibration).
