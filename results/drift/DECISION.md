# Decision under PREREG_drift_20261022 (sha256 135f58eb…), read 2026-10-22 ~07:05 CEST

Grid: results/drift/grid_full.json (sha256 827fa2a276e7444d…), 160 cells, all seeds as preregistered; controls.json passed first
(C0 identical, C1 R_OG 0.50 [0.31, 0.66], C2 R_N1G upper 0.019). C0 also holds on the 32 grid cells at t = 0.

Rule: OPEN if R_OG lower 95 % bound >= 0.10 in at least one read xzzx cell (t > 0, L_G <= 0.2).
Outcome: **OPEN** — 6 of 64 read xzzx cells, all at kappa = 4.6 and d = 5 (best: t=2 h, x, p_cal 0.005:
R_OG 0.276 [0.147, 0.403]; t=1 h, z, p_cal 0.01: 0.186 [0.139, 0.230]).

By kappa and d (xzzx, read cells, 16 each): kappa 0.833 d3: 0 open (max lower 0.035); kappa 0.833 d5: 0 (0.029);
kappa 4.6 d3: 0 (0.098); kappa 4.6 d5: 6 (0.147).

Limits written with the result, not after it:
- The gain appears only with kappa = 4.6 (A2: syndrome rounds not echoed, Ramsey-like low-frequency cutoff). With the
  echo-like kappa = 0.833 no cell opens. Which kappa a real device has is not known here.
- A1 (per-qubit spread of the drift) is assumed, not measured: the paper gives one drift number for one qubit.
- The rule is "at least one of 64 cells"; the six open cells are all in one (kappa, d) block, and the N1 null does not
  beat G in any of them (point estimates of R_N1G negative), which argues against a chance pattern, but the rule is
  liberal and this is stated.
- R_OM (not used for the decision) has median 0.101 over the read xzzx cells: plain recalibration (G) already takes
  part of the gain.
Next under the prereg: the online per-qubit estimator, with its own prereg; it must be measured against G and O.

Erratum, 29 October 2026 (added at publication; nothing above is changed): the median of R_OM over the 64 read xzzx cells is
0.096, not 0.101 (0.101 is the upper of the two middle values, 0.091 and 0.101). The sentence it supports is unchanged.

Check outside the pre-registration, 30 October 2026 (nothing above is changed): with seeds 5000 + i instead of 1000 + i,
the 64 read xzzx cells give 5 open cells (still OPEN, none at kappa 0.833): 4 at kappa 4.6 d5 and 1 at kappa 4.6 d3
(t=2 h, x, p_cal 0.005: R_OG 0.236 [0.172, 0.295]). The six cells' grouping in one (kappa, d) block, used above as an
argument against a chance pattern, does not hold with other seeds.
