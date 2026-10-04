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
that same batch, it reaches 0.0% error (169 s), as the MLP does (1.3 s) — `experiments/overfit_check.py`,
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
Neither learned decoder reaches correlated matching, belief matching or tensor-network contraction. The Transformer, the
architecture that scales, is the furthest from converged within a CPU budget.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt     # versions pinned to the ones measured
.venv/bin/python tests/test_bench.py                                    # bench self-tests (seconds)
.venv/bin/python experiments/pilot.py 3 3 0.005 200000 3000 mlp 512
.venv/bin/python experiments/pilot.py 5 5 0.005 300000 4000 mlp 512
.venv/bin/python experiments/pilot.py 5 5 0.005 300000 1500 transformer 64
.venv/bin/python experiments/overfit_check.py 5 64 1000
.venv/bin/python experiments/scaling.py 64 20 3 5 7 && .venv/bin/python experiments/scaling.py 16 20 9 && .venv/bin/python experiments/scaling.py 4 20 11
# real data: download google_qec3v5_experiment_data.zip from Zenodo 6804040 (md5 a7fd8b481c3087090093106382dc217d) and unzip into data/qec3v5/
for e in surface_code_bX_d3_r01_center_3_5 surface_code_bX_d5_r05_center_5_5 surface_code_bX_d3_r25_center_3_5 surface_code_bX_d5_r25_center_5_5; do .venv/bin/python experiments/real_google.py data/qec3v5/$e; done   # positive controls
.venv/bin/python experiments/real_google.py data/qec3v5/surface_code_bX_d3_r05_center_3_5 --learned mlp --train-shots 500000 --steps 4000 --batch 512
.venv/bin/python experiments/real_google.py data/qec3v5/surface_code_bX_d3_r05_center_3_5 --learned transformer --train-shots 500000 --steps 3000 --batch 128
```

Raw outputs are in `results/`, one file per command above, produced by the code in this commit. Code: Apache License 2.0
(`LICENSE`); the Google data keep their own CC-BY 4.0 licence and are not redistributed here.

## Limits

Uniform circuit-level noise only (where matching is expected to be strong); one architecture size; no hyper-parameter
search; CPU only. These are the conditions of a pilot, stated so the numbers are not read as more than they are.

— Roberto Locatelli (individual developer). Written with Noûs, an AI agent operating under a revocable mandate from Roberto
Locatelli, who reviews and is accountable for what is published.
