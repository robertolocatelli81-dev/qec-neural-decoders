"""POST-HOC reference (not part of prereg/PREREG_correlated_20261004.md): an empirical near-optimal decoder at distance 3.

With 24 detectors every syndrome is a 24-bit integer. From N shots sampled with a SEPARATE seed (5), the decoder predicts,
for each syndrome, the more frequent logical outcome seen with it; a syndrome never seen falls back to correlated matching.
On the same 200,000 test shots (seed 2) of experiments/correlated.py, this estimates how close any decoder can get
(unseen syndromes, under 1% of the test shots, fall back to correlated matching, so it is an upper bound on the optimum).

Usage: python experiments/lookup_optimum.py <pc> [table_shots]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pymatching  # noqa: E402

from qecnd.correlated import correlated_circuit  # noqa: E402


def keys(det: np.ndarray) -> np.ndarray:
    return det.astype(np.int64) @ (np.int64(1) << np.arange(det.shape[1], dtype=np.int64))


def main():
    pc = float(sys.argv[1])
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 20_000_000
    c = correlated_circuit(3, 3, 0.005, pc)
    sampler = c.compile_detector_sampler(seed=5)
    ones, total = {}, {}
    k_all, y_all = [], []
    for _ in range(n // 1_000_000):
        d, o = sampler.sample(1_000_000, separate_observables=True)
        k_all.append(keys(d)); y_all.append(o[:, 0].astype(np.int64))
    k, y = np.concatenate(k_all), np.concatenate(y_all)
    uk, inv = np.unique(k, return_inverse=True)
    cnt = np.bincount(inv)
    one = np.bincount(inv, weights=y)
    x_te, o_te = c.compile_detector_sampler(seed=2).sample(200_000, separate_observables=True)
    y_te = o_te[:, 0].astype(np.uint8)
    dem = c.detector_error_model(decompose_errors=True)
    fallback = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True).decode_batch(
        x_te, enable_correlations=True)[:, 0].astype(np.uint8)
    kt = keys(x_te)
    pos = np.searchsorted(uk, kt)
    pos[pos >= len(uk)] = 0
    seen = uk[pos] == kt
    pred = np.where(seen, (one[pos] * 2 > cnt[pos]).astype(np.uint8), fallback)
    out = {"pc": pc, "table_shots": int(len(k)), "distinct_syndromes": int(len(uk)), "test_shots": 200_000,
           "test_syndromes_seen_in_table": float(np.mean(seen)), "lookup_optimum": float(np.mean(pred != y_te))}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
