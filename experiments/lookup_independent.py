"""INDEPENDENT near-optimal reference at distance 3 (not derived from experiments/lookup_optimum.py).

For each pc: sample N shots with stim seed 11 (separate from training seed 1, test seed 2 and the author's table seed 5),
in chunks; accumulate, per 24-bit syndrome, the count of shots and the count of logical flips with a merge of sorted
unique keys (no dict). Decoder: per syndrome, predict flip iff flips > count/2 (strict: ties -> no flip); syndromes never
seen in the table -> DECLARED fallback = plain PyMatching from the circuit DEM (also reported with the 'no flip' fallback
and with the test-set-itself oracle, which is optimistic, to bracket). Evaluated on the SAME 200,000 test shots (seed 2).

Usage: python experiments/lookup_independent.py <pc> <table_shots> <out_json> [mlp_pred.npy]
"""
import json
import math
import sys

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pymatching

from qecnd.correlated import correlated_circuit

CHUNK = 2_000_000


def key_of(det: np.ndarray) -> np.ndarray:
    w = (np.uint64(1) << np.arange(det.shape[1], dtype=np.uint64))
    return (det.astype(np.uint64) * w).sum(axis=1)


def mcnemar(b, c):
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(logs)
    return min(1.0, 2 * math.exp(m) * sum(math.exp(v - m) for v in logs))


def main():
    pc, n_table, out_path = float(sys.argv[1]), int(sys.argv[2]), sys.argv[3]
    mlp_path = sys.argv[4] if len(sys.argv) > 4 else None
    c = correlated_circuit(3, 3, 0.005, pc)
    sampler = c.compile_detector_sampler(seed=11)
    keys = np.zeros(0, dtype=np.uint64)
    cnt = np.zeros(0, dtype=np.int64)
    flips = np.zeros(0, dtype=np.int64)
    done = 0
    while done < n_table:
        m = min(CHUNK, n_table - done)
        d, o = sampler.sample(m, separate_observables=True)
        k = key_of(d)
        y = o[:, 0].astype(np.int64)
        uk, inv = np.unique(k, return_inverse=True)
        ck = np.bincount(inv, minlength=len(uk))
        fk = np.bincount(inv, weights=y, minlength=len(uk)).astype(np.int64)
        allk = np.concatenate([keys, uk])
        allc = np.concatenate([cnt, ck])
        allf = np.concatenate([flips, fk])
        keys, inv2 = np.unique(allk, return_inverse=True)
        cnt = np.bincount(inv2, weights=allc, minlength=len(keys)).astype(np.int64)
        flips = np.bincount(inv2, weights=allf, minlength=len(keys)).astype(np.int64)
        done += m
    assert cnt.sum() == n_table
    x_te, o_te = c.compile_detector_sampler(seed=2).sample(200_000, separate_observables=True)
    y_te = o_te[:, 0].astype(np.uint8)
    kt = key_of(x_te)
    pos = np.searchsorted(keys, kt)
    pos_c = np.minimum(pos, len(keys) - 1)
    seen = keys[pos_c] == kt
    table_pred = (flips[pos_c] * 2 > cnt[pos_c]).astype(np.uint8)
    dem = c.detector_error_model(decompose_errors=True)
    fb_match = pymatching.Matching.from_detector_error_model(dem).decode_batch(x_te)[:, 0].astype(np.uint8)
    pred_match_fb = np.where(seen, table_pred, fb_match)
    pred_zero_fb = np.where(seen, table_pred, 0).astype(np.uint8)
    # optimistic oracle: majority vote computed on the test set itself
    ut, it = np.unique(kt, return_inverse=True)
    ct = np.bincount(it)
    ft = np.bincount(it, weights=y_te)
    oracle = (ft[it] * 2 > ct[it]).astype(np.uint8)
    # how much probability mass of the test set sits on syndromes with a clear majority in the table (|2f-c| > 2 sqrt(c))
    conf = np.abs(2 * flips[pos_c] - cnt[pos_c]) > 2 * np.sqrt(cnt[pos_c])
    out = {"pc": pc, "table_seed": 11, "table_shots": int(n_table), "distinct_syndromes": int(len(keys)),
           "test_seed": 2, "test_shots": 200_000, "test_shots_seen_in_table": float(seen.mean()),
           "test_shots_on_confident_syndromes": float((seen & conf).mean()),
           "lookup_fallback_matching": float(np.mean(pred_match_fb != y_te)),
           "lookup_fallback_noflip": float(np.mean(pred_zero_fb != y_te)),
           "oracle_on_test_itself_optimistic": float(np.mean(oracle != y_te)),
           "trivial": float(y_te.mean())}
    if mlp_path:
        mlp = np.load(mlp_path).astype(np.uint8)
        b = int(np.sum((mlp == y_te) & (pred_match_fb != y_te)))
        cc = int(np.sum((mlp != y_te) & (pred_match_fb == y_te)))
        out["mlp"] = float(np.mean(mlp != y_te))
        out["mlp_vs_lookup"] = {"mlp_right_lookup_wrong": b, "mlp_wrong_lookup_right": cc, "mcnemar_p": mcnemar(b, cc)}
    np.save(out_path.replace(".json", "_pred.npy"), pred_match_fb)
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
