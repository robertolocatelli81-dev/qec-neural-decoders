Erratum to PREREG_correlated_20261004.md (sha256 02d73e9e…14b02), written after the runs.
The pre-registration says the joint X error is added "at every per-round data depolarisation". The code that ran
(qecnd/correlated.py) adds it after every single-qubit depolarising instruction, three times per round (once on the data
qubits, twice after the Hadamard layers on the measure qubits): the per-pair, per-round strength is 1 - (1 - pc)^3, about
3 pc, and the first round has two insertions, the final data measurement one. The detector error model given to matching
is computed from the same circuit, so the comparison is unaffected; the grid, seeds, decoders, test and decision rule
were run as pre-registered. Nothing else deviates.
