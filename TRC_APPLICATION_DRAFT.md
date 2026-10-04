# TPU Research Cloud — application draft (not sent; to be reviewed and sent by Roberto Locatelli)

<!-- DO NOT SEND until the repository is public at the URL below: the text says the code is published. -->

<!-- Compute request, in case the form asks for one (the public TRC page lists only the conditions: share the research
through publications / open-source code / blog posts, give feedback, follow Google's AI principles; the form's fields
are not shown publicly): one TPU host, 60 days, order of 10^8 shots per (distance, noise model) × distances 3–11 × 4 noise
models × 3 seeds, re-sized after the first throughput runs. -->

**Applicant.** Roberto Locatelli, independent developer (no institutional affiliation), Italy. Maintainer of open-source
verification tools on GitHub (robertolocatelli81-dev), released under open licences with independent re-implementations
and differential test oracles.

**Project.** Where do learned decoders beat matching on the surface code — measured, with controls.

**Research question.** Minimum-weight perfect matching is near-optimal when errors are independent and syndrome
measurements are reliable. A recurrent Transformer decoder (Bausch et al., "Learning high-accuracy error decoding for
quantum processors", Nature 635, 834–840, 2024) was reported to beat the best published decoders — tensor-network
contraction and correlated matching — on Sycamore data at distances 3 and 5, after pre-training on 2 × 10^9 simulated
samples and fine-tuning an ensemble of 20 models on about 20,000 experimental shots, and to keep an advantage over
correlated matching on simulated data with cross-talk, leakage and soft readout up to distance 11 (on a simulator that
is not public). I found no public release of that decoder's code (searched 4 October 2026). I want to map, openly,
on public simulators and with every result reproducible from code, *which* noise features (measurement noise, errors
correlated in time and space, leakage) give a detector-attention decoder an advantage over matching, *how large* it is
as the code distance grows, and *where it vanishes*. Each claim is measured on held-out shots against matching on the
same shots, next to the same model trained on permuted labels; a positive result that does not survive that null is not
reported as one.

**What exists today (code at https://github.com/robertolocatelli81-dev/qec-neural-decoders, Apache License 2.0; measured on a
laptop CPU, 4 October 2026).** A JAX/Flax pipeline: rotated surface-code memory circuits generated with stim, PyMatching
as the reference, an MLP and a Transformer that reads each detector as a token, the null control, and a first run on
Google's public Sycamore data (Zenodo 6804040), where PyMatching configured here reproduces Google's published PyMatching
predictions on 100% of 50,000 shots at distance 3 and 5 (positive control), and an MLP trained only on simulated shots
lands within the noise of PyMatching on the real shots (15.37% vs 15.74% on 25,000 shots, paired test not significant)
but behind correlated matching (14.43%). Results under uniform circuit-level noise (p = 0.005, rounds = distance):

- distance 3: matching 1.71%, MLP 1.90%, null = trivial (10.46%);
- distance 5: matching 1.38–1.43%, MLP 7.03%, Transformer 14.78% after 1500 steps of 64 shots (262 s), null = trivial.
- distance 3 with long-range correlated errors (pre-registered grid, 5 cells, exact paired tests, Bonferroni): the MLP
  beats PyMatching in the four correlated cells (e.g. 4.70% vs 9.53% at pc = 0.01) and not in the control cell without
  correlated noise, as pre-registered. Measured afterwards: this noise gives the code effective distance 2 for those
  errors, and the gap is mostly how the library resolves the resulting degenerate parallel edges (documented PyMatching
  behaviour for that case); against a matching that keeps the most probable logical effect per edge the MLP wins by 0.2–0.3 points in
  two cells, ties in one and loses by 0.3 in one, and it stays 0.2–0.6 points above an empirical near-optimal decoder
  (20 million-shot lookup table). The Transformer variants at distance 5 did not leave the constant-output plateau
  within 1500 steps on CPU (negative ablation, reported).

The learned decoders do not reach matching on CPU, and the reason is measured: the Transformer trains on 2640 shots/s at
distance 3, 372 at distance 5, 62 at distance 7, 21 at distance 9 and 7.8 at distance 11 (±25% between runs), because attention grows with the
square of the number of detectors ((d² − 1) × rounds: 24 → 1320). Showing a model 10^8 shots takes about 75 hours at
distance 5 and about 19 days at distance 7 on this machine.

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

**Sharing.** All code, configurations, raw results and the null controls will be published in the repository above as
they are produced, under the Apache License 2.0; a write-up (arXiv preprint) at the end, including the negative results; TRC
support acknowledged in every output.

**Compute.** The CPU table above is the scale of the problem: the distance-7 and larger experiments are out of reach on
CPU. The first TPU runs will measure throughput per distance; the plan is sized from those measurements, and the
negative calibration result (no advantage where matching is near-optimal) is published whatever it is.
