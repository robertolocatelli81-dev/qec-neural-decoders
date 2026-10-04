# TPU Research Cloud — application draft (not sent; to be reviewed and sent by Roberto Locatelli)

**Applicant.** Roberto Locatelli, independent developer (no institutional affiliation), Italy. Maintainer of open-source
verification tools on GitHub (robertolocatelli81-dev), released under open licences with independent re-implementations
and differential test oracles.

**Project.** Where do learned decoders beat matching on the surface code — measured, with controls.

**Research question.** Minimum-weight perfect matching is near-optimal when errors are independent and syndrome
measurements are reliable. A recurrent Transformer decoder (Bausch et al., "Learning high-accuracy error decoding for
quantum processors", Nature 2024) was reported to beat correlated matching on Sycamore data at distances 3 and 5 and on
simulated data with cross-talk and leakage up to distance 11. I want to map,
on public simulators and with every result reproducible from code, *which* noise features (measurement noise, errors
correlated in time and space, leakage) give a detector-attention decoder an advantage over matching, *how large* it is
as the code distance grows, and *where it vanishes*. Each claim is measured on held-out shots against matching on the
same shots, next to the same model trained on permuted labels; a positive result that does not survive that null is not
reported as one.

**What exists today (public code, measured on a laptop CPU, 4 October 2026).** A JAX/Flax pipeline: rotated
surface-code memory circuits generated with stim, PyMatching as the reference, an MLP and a Transformer that reads each
detector as a token, and the null control. Results under uniform circuit-level noise (p = 0.005, rounds = distance):

- distance 3: matching 1.71%, MLP 1.90%, null = trivial (10.46%);
- distance 5: matching 1.38–1.43%, MLP 7.03%, Transformer 14.78% after 1500 steps of 64 shots (253 s), null = trivial.

The learned decoders do not reach matching on CPU, and the reason is measured: the Transformer trains on 2218 shots/s at
distance 3, 356 at distance 5, 65 at distance 7, 24 at distance 9 and 8 at distance 11, because attention grows with the
square of the number of detectors ((d² − 1) × rounds: 24 → 1320). Showing a model 10^8 shots takes about 78 hours at
distance 5 and about 18 days at distance 7 on this machine.

**What the TPUs would be used for.**
1. Train detector-attention decoders to convergence at distances 3–11 under uniform noise, to establish where they
   match matching (the calibration: if they do not reach matching where matching is near-optimal, nothing else is
   claimed).
2. Repeat under noise models where matching's assumptions fail — correlated (two-qubit and time-correlated) errors,
   heavier measurement noise, leakage — all generated with stim, and measure the gap in each, with confidence intervals
   from independent seeds.
3. Ablations: model size, rounds seen, training-set size, so the advantage (or its absence) is attributed to a cause.

The workload is batched matrix multiplication over sequences of hundreds to thousands of detector tokens, written in JAX
(`jax.jit`); it moves to TPU without changes to the model code.

**Sharing.** All code, configurations, raw results and the null controls are published in the public repository as they
are produced; a write-up (arXiv preprint) at the end, including the negative results.

**Compute.** The CPU table above is the scale of the problem: the distance-7 and larger experiments are out of reach on
CPU. The first TPU runs will measure throughput per distance; the plan is sized from those measurements, and the
negative calibration result (no advantage where matching is near-optimal) is published whatever it is.
