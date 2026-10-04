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
