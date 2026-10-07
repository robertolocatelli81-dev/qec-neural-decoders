# qec-neural-decoders

Learned decoders for surface-code memory experiments, measured against minimum-weight perfect matching on the **same**
shots, with a null control that must fail. JAX/Flax models; data from [stim](https://github.com/quantumlib/Stim);
reference decoder [PyMatching](https://github.com/oscarhiggott/PyMatching).

## The question

Matching is near-optimal when errors are independent and syndrome measurements are reliable. A learned decoder has
something to add only where those assumptions fail: noisy syndrome extraction, errors correlated in time or space,
leakage. This project asks **where, and by how much, a detector-attention decoder beats matching — and where it does not** —
with every claim measured on shots the model never saw, against matching on those same shots, next to a model trained on
permuted labels.

## What is here

- `qecnd/data.py` — rotated surface-code memory circuits (stim, uniform circuit-level noise), detector records and
  logical labels.
- `qecnd/baselines.py` — PyMatching from the circuit's detector error model; logical error rate.
- `qecnd/model.py` — an MLP over the flattened record and a Transformer that reads each detector as a token; one training
  loop (Adam, sigmoid cross-entropy), compiled with `jax.jit`.
- `experiments/pilot.py` — trivial / matching / learned / null on one experiment.
- `experiments/scaling.py` — the cost of the Transformer on the machine it runs on, as the distance grows.
- `qecnd/cos2phi.py` — the cos(2φ) transmon of arXiv:2603.13114 (Eq. 1–2, Table I) in the charge basis, its 1/f-flux-noise
  T1 (its Eq. 4) and the Pauli twirl of the T1/Tφ channel; `qecnd/biased.py` — surface-code memory circuits with a Z-biased
  Pauli channel on the data qubits, in the CSS or the XZZX frame; `experiments/cos2phi_*.py` — that study (below).

## Measured on a laptop CPU (8 cores, 6 GB RAM, JAX 0.10.2 CPU), 4 October 2026

Logical error rate on held-out shots (different stim seed from training), p = 0.005, rounds = distance:

| distance | trivial | matching | learned | null (permuted labels) | training |
|---|---|---|---|---|---|
| 3 | 10.46% | 1.71% | MLP 1.90% | 10.46% | 3000 steps × 512 |
| 5 | 23.08% | 1.43% | MLP 7.03% | 24.37% | 4000 steps × 512 |
| 5 | 23.02% | 1.38% | Transformer 14.78% | 23.02% | 1500 steps × 64 (262 s) |

The learned decoders do **not** reach matching here. At distance 3 the MLP comes within 0.2 points; at distance 5 both are
far from converged within a CPU budget. The null stays at the floor in every row (equal to the trivial decoder at distance
3 and for the Transformer; 1.3 points above it for the distance-5 MLP, which overfits the permuted labels slightly): the
bench can fail. Error rates are deterministic for the seeds in the scripts (re-measured identical on 4 October 2026);
training times and throughputs vary from run to run on the same machine by up to about 25%. Three runs on 4 October 2026
(the tables and `results/` hold the third): distance-5 Transformer pilot 253 / 271 / 262 s; real-data Transformer
373 / 286 / 281 s; throughput 2218 / 2619 / 2640, 356 / 378 / 372, 65 / 59 / 62, 24 / 23 / 21, 8 / 7.5 / 7.8 shots/s.

Cost of the detector Transformer (width 64, depth 2, 4 heads) on the same CPU:

| distance | detectors per shot | training shots / s | batch |
|---|---|---|---|
| 3 | 24 | 2640 | 64 |
| 5 | 120 | 372 | 64 |
| 7 | 336 | 62 | 64 |
| 9 | 720 | 21 | 16 |
| 11 | 1320 | 7.8 | 4 |

At these rates, showing the model 10^8 shots takes about 75 hours at distance 5 and about 19 days at distance 7 (arithmetic
from the table; 73–78 hours and 18–20 days across the three runs). The cost is attention over all detectors of a shot: there are (d² − 1) × rounds of them (24, 120, 336,
720, 1320 above), and attention grows with the square of that number — a tensor workload, which is what accelerators are
built for.

**Can the Transformer learn at all?** Trained on one fixed batch of 64 distance-5 shots for 1000 steps and evaluated on
that same batch, it reaches 0.0% error (194.4 s), as the MLP (1.3 s) and the `GeoTransformer` (196.3 s) do — `experiments/overfit_check.py`,
`results/overfit_check_d5.json`. Its poor held-out numbers above are therefore not an inability to fit; whether they are
a budget or an architecture limit is what the larger runs would settle.

## On real hardware data (Google Sycamore, 4 October 2026)

Data: "Suppressing quantum errors by scaling a surface code logical qubit" (Nature 2023), [Zenodo 6804040](https://zenodo.org/records/6804040),
CC-BY 4.0, md5 verified. 50,000 shots per experiment, with Google's own predictions from four decoders.

**Positive control.** PyMatching here, configured with each experiment's `circuit_detector_error_model.dem` (the model
Google's README states was used for its PyMatching predictions), reproduces Google's published PyMatching predictions on
100.00% of the 50,000 shots at distance 3 with 1 and 5 rounds and at distance 5 with 5 rounds. At 25 rounds the agreement
is 99.89% (distance 3) and 96.87% (distance 5): the two implementations differ on a fraction of shots there, so the
25-round experiments are not used for the learned-decoder comparison. With the data-derived models
(`pij_from_even_for_odd.dem` on odd shots and the converse) agreement is 95.3% at distance 3, 5 rounds; that variant is
reported as its own decoder.

Distance 3, 5 rounds (`surface_code_bX_d3_r05_center_3_5`), logical error rate on the 25,000 **odd** real shots. The
learned decoders are trained only on shots sampled from `pij_from_even_for_odd.dem` (derived from the even shots); they
never see a real shot in training.

| decoder | logical error rate |
|---|---|
| tensor network contraction (Google) | 12.72% |
| belief matching (Google) | 13.06% |
| correlated matching (Google) | 14.43% |
| MLP, 500k simulated shots, 4000 steps × 512 (10 s of training) | 15.37% |
| PyMatching, data-derived models (here) | 15.51% |
| PyMatching (Google; reproduced here) | 15.74% |
| Transformer, 500k simulated shots, 3000 steps × 128 (281 s of training) | 20.39% |
| null (MLP and Transformer on permuted labels) | 27.84% |
| trivial | 27.84% |

The MLP is 0.37 points below PyMatching; with 25,000 shots the standard error of each rate is about 0.23 points, and a
paired test on the same shots (exact McNemar: 1,485 shots the MLP alone gets wrong against 1,578 PyMatching alone gets
wrong, p = 0.10) finds no significant difference. Over five training seeds the MLP averages 15.40% (s.d. 0.13) on the odd
shots and 15.65% (s.d. 0.35) on the mirror fold (trained on `pij_from_odd_for_even.dem`, evaluated on the even shots,
where PyMatching gets 15.56%): the sign of the difference flips between folds, so there is no evidence that the MLP beats
PyMatching here. Against correlated matching the MLP is worse in all ten fold × seed cases (McNemar p < 0.03 in each).
Neither learned decoder reaches correlated matching, belief matching or tensor-network contraction. The Transformer is
the furthest from converged within a CPU budget (dense attention, whose cost grows with the square of the number of
detectors: not the cheapest architecture at large distance, the one able to relate any two detectors). These learned
decoders are zero-shot on the hardware: trained only on simulated shots, never fine-tuned on real ones, so they are not
comparable with a decoder fine-tuned on experimental shots such as Bausch et al.'s.

## Errors from a cos(2φ)-protected qubit (pre-registered, 7 October 2026)

Pre-registration `prereg/PREREG_cos2phi_20261007.md` (sha256 `6cc3b0e5…505e1`, written before any run). The qubit of
Roverc'h et al. (arXiv:2603.13114) is diagonalised from its published Hamiltonian and Table I: doublet 12.98 MHz (measured
13.6), charge-matrix-element suppression 157× (their 2.2 / 1.3 × 10⁻²), T1 from their flux-noise formula and amplitude 43 µs
(measured 70); the wrong sign of the cos(2φ) term gives 128 MHz, so the check can fail. Pauli-twirled, the device's T1/T2 give a
Z-biased channel, η = p_Z/(p_X + p_Y) = 17–50 (27.5 with the measured T1 and echo T2). Its strength is above every known
threshold, so the grid keeps the structure (η = 27.5) and scales p to 0.01 and 0.03 per data qubit per round (phenomenological
model: ideal gates, measurement flips q = p). 32 pre-registered cells (d = 3, 5; η = 1 control and 27.5; CSS and XZZX frames,
both bases) plus 8 sensitivity cells at η = 49.5, each with PyMatching, an MLP, the null and, at d = 3, a 10-million-shot lookup
on the same 200,000 shots. **As measured** (`results/cos2phi/SUMMARY.md`): matching on the XZZX frame beats matching on the
vulnerable CSS basis by 8.7× / 7.3× at d = 3 and 21× / 14.5× at d = 5 (P1 holds); the null is at the trivial rate in 40/40 cells
(P3 holds); the MLP beats PyMatching in two biased cells at d = 3 (0.14 and 0.17 points) but also in four unbiased control
cells by up to 0.37 points (P0 fails), and no biased win exceeds that margin: **no learned-decoder advantage attributable to
the cos(2φ) bias at this scale** (P2, as pre-registered: no). At d = 5 the MLP does not reach matching in any cell (budget).

## Long-range correlated errors (pre-registered, 4 October 2026)

Pre-registration `prereg/PREREG_correlated_20261004.md` (sha256 `02d73e9e…14b02`, written before any run; one erratum on
the wording of the noise model, `prereg/ERRATUM_correlated_20261004.md`). Distance 3, 3 rounds, uniform circuit noise
p = 0.005, plus a joint X error on each of the two farthest data-qubit pairs after every single-qubit depolarising
instruction (three per round) with probability pc per insertion. Decoders on the same 200,000 held-out shots (stim seed 2):
PyMatching 2.4.0 on the circuit's own detector error model, plain and with `enable_correlations`; an MLP (hidden 256,
500,000 training shots, 4000 steps × 512, parameters with the best validation loss kept); the null (same MLP on permuted
labels); the trivial decoder. Primary test: exact two-sided McNemar, MLP against the better of the two matching modes,
Bonferroni over 5 cells (p < 0.01).

| pc | trivial | PyMatching | PyMatching, correlations | MLP | null | McNemar p | pre-registered win |
|---|---|---|---|---|---|---|---|
| 0 | 10.40% | 1.73% | 1.69% | 1.76% | 10.40% | 3.1e-4 (matching better) | no — as expected |
| 0.002 | 13.21% | 3.37% | 3.31% | 3.12% | 13.21% | 1.4e-7 | yes |
| 0.005 | 16.92% | 6.03% | 5.88% | 4.13% | 16.92% | 1.5e-236 | yes |
| 0.01 | 22.35% | 9.66% | 9.53% | 4.70% | 22.35% | < 1e-300 | yes |
| 0.02 | 30.91% | 16.62% | 16.43% | 5.38% | 30.91% | < 1e-300 | yes |

The MLP beats PyMatching in the four cells with correlated noise and, as pre-registered, not at pc = 0 (there correlated
matching is better, p = 3.1e-4). **What the win is, measured after the fact.** Each joint error lights exactly two
detectors plus the observable: an ordinary long edge of the matching graph, not a hyper-edge. For one of the two pairs
(the diagonal one, data qubits at (1,1) and (5,5)) those two detectors are at distance 3 also the syndrome of a single X
error on the central data qubit, which does not flip the observable (the other pair's syndrome is shared with no
single-qubit error). The two are parallel edges with different logical effect; PyMatching merges parallel edges keeping the logical effect of the first
one in the error model (the single-qubit one here), whatever the probabilities. This is documented PyMatching behaviour
(the `Matching.from_detector_error_model` docstring), discussed by its maintainer in
[PyMatching#103](https://github.com/oscarhiggott/PyMatching/issues/103): parallel edges with different logical effects
mean the code has distance at most 2 for those errors, which is exactly what this correlated noise does at distance 3.
A matching built from the same error model
that keeps, per merged edge, the most probable logical effect gets 1.73% / 3.37% / 4.33% / 4.70% / 5.11%: the MLP is
better than it by 0.26 and 0.20 points at pc = 0.002 and 0.005 (p = 2.7e-13 and 7.9e-10), equal at 0.01 (p = 0.93) and
worse by 0.27 points at 0.02 (p = 6.0e-15). So most of the gap is one library's rule for degenerate parallel edges, not a
limit of matching as a method; what remains for the learned decoder is real and small.

**Distance from the optimum.** An empirical near-optimal decoder (for each of the 2^24 syndromes, the more frequent
outcome over 20,000,000 shots sampled with a separate seed; unseen syndromes, under 1% of the test shots, fall back to
PyMatching) gets 1.54% / 2.81% / 3.78% / 4.21% / 4.80% on the same test shots. The MLP sits 0.23 / 0.31 / 0.35 / 0.49 /
0.57 points above it, significantly in every cell (McNemar p < 1e-32). `experiments/lookup_optimum.py` (post hoc, not
pre-registered) gives 1.56% / 2.78% / 3.76% / 4.21% with its own table seed. Scripts: `experiments/correlated.py` (the
pre-registered grid), `experiments/correlated_review.py` (the matching with the most probable logical effect per merged edge,
and the paired tests against it), `experiments/lookup_independent.py` (the near-optimal decoder above); raw outputs in
`results/correlated/`.

**Ablation of the Transformer (distance 5, 5 rounds, p = 0.005, 300,000 shots, 1500 steps × 64; negative).** Transformer
without validation 14.78%; with validation-selected checkpoint 18.72% (step 1200); a Transformer given stim's detector
coordinates and a [CLS] readout (`GeoTransformer`), with validation, 23.35% — the trivial rate is 23.02%. It memorises a
single batch of 64 shots at distance 3 and at distance 5 (0.0% error, like the MLP and the plain Transformer;
`results/overfit_check_d3.json`, `results/overfit_check_d5.json`), and at 300 steps both the plain Transformer and the
GeoTransformer still output a constant "no flip" on all 20,000 test shots (`fraction_predicted_1 = 0`,
`results/ablation/d_transformer_300steps.json`, `results/ablation/e_geo_300steps.json`): within this CPU budget the plain
Transformer leaves that plateau before 1500 steps and the coordinate/[CLS] one does not. Whether that is the readout, the
coordinates or only the budget was not separated (it needs ≥ 1500 steps per variant).

## Reproduce

The bench self-tests also run in CI on every push (`.github/workflows/tests.yml`: Python 3.11, the pinned
requirements; the job fails unless every test defined in `tests/test_bench.py` and `tests/test_cos2phi.py` runs and passes).

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt     # versions pinned to the ones measured
.venv/bin/python tests/test_bench.py                                    # bench self-tests (seconds)
.venv/bin/python tests/test_cos2phi.py                                  # cos(2φ) physics and biased-circuit checks (seconds)
.venv/bin/python experiments/pilot.py 3 3 0.005 200000 3000 mlp 512
.venv/bin/python experiments/pilot.py 5 5 0.005 300000 4000 mlp 512
.venv/bin/python experiments/pilot.py 5 5 0.005 300000 1500 transformer 64
.venv/bin/python experiments/overfit_check.py 5 64 1000 > results/overfit_check_d5.json
.venv/bin/python experiments/overfit_check.py 3 64 1000 > results/overfit_check_d3.json
.venv/bin/python experiments/scaling.py 64 20 3 5 7 && .venv/bin/python experiments/scaling.py 16 20 9 && .venv/bin/python experiments/scaling.py 4 20 11
# real data: download google_qec3v5_experiment_data.zip from Zenodo 6804040 (md5 a7fd8b481c3087090093106382dc217d) and unzip into data/qec3v5/
for e in surface_code_bX_d3_r01_center_3_5 surface_code_bX_d5_r05_center_5_5 surface_code_bX_d3_r25_center_3_5 surface_code_bX_d5_r25_center_5_5; do .venv/bin/python experiments/real_google.py data/qec3v5/$e; done   # positive controls
.venv/bin/python experiments/real_google.py data/qec3v5/surface_code_bX_d3_r05_center_3_5 --learned mlp --train-shots 500000 --steps 4000 --batch 512
.venv/bin/python experiments/real_google.py data/qec3v5/surface_code_bX_d3_r05_center_3_5 --learned transformer --train-shots 500000 --steps 3000 --batch 128
# correlated errors (pre-registered grid, its review, the two near-optimal references) and the Transformer ablation
for pc in 0 0.002 0.005 0.01 0.02; do .venv/bin/python experiments/correlated.py $pc > results/correlated/pc_$pc.json; done
for pc in 0 0.002 0.005 0.01 0.02; do .venv/bin/python experiments/correlated_review.py $pc results/correlated/review_pc_$pc.json; done
for pc in 0 0.002 0.005 0.01 0.02; do .venv/bin/python experiments/lookup_independent.py $pc 20000000 results/correlated/lookup_independent_pc_$pc.json results/correlated/review_pc_${pc}_mlp_pred.npy; done
for pc in 0 0.002 0.005 0.01; do .venv/bin/python experiments/lookup_optimum.py $pc > results/correlated/lookup_pc_$pc.json; done
.venv/bin/python experiments/pilot.py 5 5 0.005 300000 1500 transformer 64 > results/ablation/a_transformer_noval.json
.venv/bin/python experiments/pilot.py 5 5 0.005 300000 1500 transformer 64 0.1 > results/ablation/b_transformer_val.json
.venv/bin/python experiments/pilot.py 5 5 0.005 300000 1500 geo 64 0.1 > results/ablation/c_geo_val.json
.venv/bin/python experiments/pilot.py 5 5 0.005 300000 300 transformer 64 > results/ablation/d_transformer_300steps.json
.venv/bin/python experiments/pilot.py 5 5 0.005 300000 300 geo 64 > results/ablation/e_geo_300steps.json
# cos(2φ) qubit: physics, checks F3–F4, the pre-registered grid (+ η = 49.5 sensitivity cells), the summary and the hashes
.venv/bin/python experiments/cos2phi_physics.py > results/cos2phi/physics.json
.venv/bin/python experiments/cos2phi_checks.py > results/cos2phi/checks.json
experiments/cos2phi_grid.sh 3 1 27.5 && experiments/cos2phi_grid.sh 5 1 27.5 && experiments/cos2phi_grid.sh 3 49.5
.venv/bin/python experiments/cos2phi_summary.py > results/cos2phi/SUMMARY.md
(cd results/cos2phi && sha256sum -c SHA256SUMS)
```

Raw outputs are in `results/`, one file per command above, produced by the code in this commit
(`results/correlated/matching_merge_most_probable.json`, the edge-level comparison of the review, is the one file without a
command: its error rates are the `matching_merge_most_probable` values of `review_pc_*.json`). `fraction_predicted_1` in the
pilot output was added with the 300-step runs; the earlier `pilot_*.json` and `ablation/a_*`–`c_*` files predate it and do not carry it. Code: Apache License 2.0
(`LICENSE`); the Google data keep their own CC-BY 4.0 licence and are not redistributed here.

## Limits

Uniform circuit-level noise, plus one correlated-noise model at distance 3 only, where one of the two correlated errors shares
its syndrome with a single-qubit error, so the code has effective distance 2 for those errors (a degeneracy that does not carry
over as such to larger distances, not measured), plus one phenomenological Z-biased model (ideal gates, q = p, XZZX via the
Clifford frame, exact only for ideal gates) at distances 3 and 5;
one architecture size; no hyper-parameter search; CPU only. These are the conditions of a pilot, stated so the numbers are
not read as more than they are.

— Roberto Locatelli (individual developer). Written with Noûs, an AI agent operating under a revocable mandate from Roberto
Locatelli, who reviews and is accountable for what is published.
