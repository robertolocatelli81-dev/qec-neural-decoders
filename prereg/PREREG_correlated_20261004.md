# Pre-registration — learned decoder vs matching under long-range correlated errors (written 4 October 2026, BEFORE any run)

## Question
Under long-range correlated X errors (stim CORRELATED_ERROR on the 2 farthest data-qubit pairs, added at every per-round
data depolarisation of a rotated surface-code Z memory), does a learned decoder have a lower logical error rate than
matching given the TRUE detector error model of the same circuit?

## Fixed grid (every cell is reported, none dropped)
- distance 3, rounds 3, uniform circuit noise p = 0.005;
- correlated strength pc in {0, 0.002, 0.005, 0.01, 0.02} (pc = 0 is the control: no correlated noise).

## Decoders (same test shots for all)
- PyMatching 2.4.0 on the circuit's own DEM (decompose_errors=True): plain, and with enable_correlations=True;
- MLP (qecnd.model.MLP, hidden 256): 500,000 training shots (stim seed 1), 4000 steps × 512, Adam 1e-3, validation =
  last 10% of the training shots, parameters with the best validation loss kept;
- null: the same MLP on permuted labels;
- trivial: always "no flip".
Test: 200,000 shots, stim seed 2 (never in training).

## Primary outcome and test
For each pc: exact two-sided McNemar test on the 200,000 paired test shots, MLP vs the BETTER of the two matching modes
(the one with the lower error on those shots). 5 cells → Bonferroni: significance at p < 0.01 per cell.
A cell counts as "learned decoder beats matching" only if MLP error < better-matching error AND p < 0.01 AND the null
lands at the trivial rate (within 1 point). Any other outcome is reported as it is (no win, tie, or loss).

## Expectations stated before running (falsifiable)
- pc = 0: the MLP does NOT beat matching (matching is near-optimal there); a "win" at pc = 0 would point to a bench defect.
- pc > 0: unknown; that is the question.
