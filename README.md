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
| 5 | 23.02% | 1.38% | Transformer 14.78% | 23.02% | 1500 steps × 64 (253 s) |

The learned decoders do **not** reach matching here. At distance 3 the MLP comes within 0.2 points; at distance 5 both are
far from converged within a CPU budget. The null lands on the trivial decoder in every row: the bench can fail.

Cost of the detector Transformer (width 64, depth 2, 4 heads) on the same CPU:

| distance | detectors per shot | training shots / s | batch |
|---|---|---|---|
| 3 | 24 | 2218 | 64 |
| 5 | 120 | 356 | 64 |
| 7 | 336 | 65 | 64 |
| 9 | 720 | 24 | 16 |
| 11 | 1320 | 8 | 4 |

At these rates, showing the model 10^8 shots takes about 78 hours at distance 5 and about 18 days at distance 7 (arithmetic
from the table). The cost is attention over all detectors of a shot: there are (d² − 1) × rounds of them (24, 120, 336,
720, 1320 above), and attention grows with the square of that number — a tensor workload, which is what accelerators are
built for.

## On real hardware data (Google Sycamore, 4 October 2026)

Data: "Suppressing quantum errors by scaling a surface code logical qubit" (Nature 2023), [Zenodo 6804040](https://zenodo.org/records/6804040),
CC-BY 4.0, md5 verified. 50,000 shots per experiment, with Google's own predictions from four decoders.

**Positive control.** PyMatching here, configured with each experiment's `circuit_detector_error_model.dem`, reproduces
Google's published PyMatching predictions on 100.00% of the 50,000 shots (distance 3 with 1 and 5 rounds, distance 5 with
5 rounds). With the data-derived models (`pij_from_even_for_odd.dem` on odd shots and the converse) agreement drops to
95–96%; that variant is reported as its own decoder.

Distance 3, 5 rounds (`surface_code_bX_d3_r05_center_3_5`), logical error rate on the 25,000 **odd** real shots. The
learned decoders are trained only on shots sampled from `pij_from_even_for_odd.dem` (derived from the even shots); they
never see a real shot in training.

| decoder | logical error rate |
|---|---|
| tensor network contraction (Google) | 12.72% |
| belief matching (Google) | 13.06% |
| correlated matching (Google) | 14.43% |
| MLP, 500k simulated shots, 4000 steps × 512 (10.4 s of training) | 15.37% |
| PyMatching, data-derived models (here) | 15.51% |
| PyMatching (Google; reproduced here) | 15.74% |
| Transformer, 500k simulated shots, 3000 steps × 128 (372.8 s of training) | 20.39% |
| null (MLP and Transformer on permuted labels) | 27.84% |
| trivial | 27.84% |

The MLP is 0.37 points below PyMatching; with 25,000 shots the standard error of each rate is about 0.23 points, so the
difference is not significant. Neither learned decoder reaches correlated matching, belief matching or tensor-network
contraction. The Transformer, the architecture that scales, is the furthest from converged within a CPU budget.

## Reproduce

```bash
python3 -m venv .venv && .venv/bin/pip install "jax[cpu]" flax optax stim pymatching numpy
.venv/bin/python experiments/pilot.py 3 3 0.005 200000 3000 mlp 512
.venv/bin/python experiments/pilot.py 5 5 0.005 300000 1500 transformer 64
.venv/bin/python experiments/scaling.py 64 20 3 5 7
```

Raw outputs are in `results/`.

## Limits

Uniform circuit-level noise only (where matching is expected to be strong); one architecture size; no hyper-parameter
search; CPU only. These are the conditions of a pilot, stated so the numbers are not read as more than they are.

— Roberto Locatelli (individual developer). Written with Noûs, an AI agent operating under a revocable mandate from Roberto
Locatelli, who reviews and is accountable for what is published.
