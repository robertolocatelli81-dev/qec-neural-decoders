"""One cell of the cos(2φ) grid (prereg/PREREG_cos2phi_20261007.md): a rotated surface-code memory experiment under a
Z-biased Pauli channel on the data qubits (phenomenological model, measurement flips q = p), in the CSS or XZZX frame,
with the decoders of the pre-registration on the same 200,000 test shots:

  trivial, PyMatching (circuit DEM), MLP (validation-selected), null (permuted labels), and at distance 3 optionally
  the empirical near-optimal lookup (majority outcome per syndrome over N table shots, fallback PyMatching).

Example: python experiments/cos2phi_pilot.py --d 3 --eta 27.5 --p 0.01 --code xzzx --basis x --lookup 10000000 --out results/cos2phi/d3_eta27.5_p0.01_xzzx_x.json
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pymatching  # noqa: E402

from qecnd.biased import biased_circuit, parallel_edges_with_different_effect  # noqa: E402
from qecnd.cos2phi import pauli_probabilities_for_bias  # noqa: E402
from qecnd.model import MLP, train  # noqa: E402

PREREG = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "prereg", "PREREG_cos2phi_20261007.md")
TRAIN_SHOTS, TEST_SHOTS, STEPS, BATCH, VAL_FRAC = 500_000, 200_000, 4000, 512, 0.1
SEED_TRAIN, SEED_TEST, SEED_PERM, SEED_TABLE = 1, 2, 3, 11


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value for discordant counts b, c."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(logs)
    return min(1.0, 2 * math.exp(m) * sum(math.exp(v - m) for v in logs))


def wilson(k: int, n: int, z: float = 1.959964) -> tuple:
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return c - h, c + h


def paired(name_a, pred_a, name_b, pred_b, y):
    wa, wb = pred_a != y, pred_b != y
    b, c = int(np.sum(wa & ~wb)), int(np.sum(~wa & wb))
    return {"only_" + name_a + "_wrong": b, "only_" + name_b + "_wrong": c, "mcnemar_p": mcnemar_exact(b, c)}


def syndrome_keys(det: np.ndarray) -> np.ndarray:
    w = (np.uint64(1) << np.arange(det.shape[1], dtype=np.uint64))
    return (det.astype(np.uint64) * w).sum(axis=1)


def lookup_decoder(circuit, n_table: int, x_te: np.ndarray, fallback: np.ndarray, chunk: int = 2_000_000):
    """Majority outcome per syndrome over n_table shots (stim seed SEED_TABLE); unseen syndromes -> fallback."""
    sampler = circuit.compile_detector_sampler(seed=SEED_TABLE)
    keys = np.zeros(0, dtype=np.uint64)
    cnt = np.zeros(0, dtype=np.int64)
    flips = np.zeros(0, dtype=np.int64)
    done = 0
    while done < n_table:
        m = min(chunk, n_table - done)
        d, o = sampler.sample(m, separate_observables=True)
        uk, inv = np.unique(syndrome_keys(d), return_inverse=True)
        ck = np.bincount(inv, minlength=len(uk))
        fk = np.bincount(inv, weights=o[:, 0].astype(np.int64), minlength=len(uk)).astype(np.int64)
        keys, inv2 = np.unique(np.concatenate([keys, uk]), return_inverse=True)
        cnt = np.bincount(inv2, weights=np.concatenate([cnt, ck]), minlength=len(keys)).astype(np.int64)
        flips = np.bincount(inv2, weights=np.concatenate([flips, fk]), minlength=len(keys)).astype(np.int64)
        done += m
    kt = syndrome_keys(x_te)
    pos = np.minimum(np.searchsorted(keys, kt), len(keys) - 1)
    seen = keys[pos] == kt
    table_pred = (flips[pos] * 2 > cnt[pos]).astype(np.uint8)
    return np.where(seen, table_pred, fallback).astype(np.uint8), float(np.mean(seen)), int(len(keys))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--eta", type=float, required=True)
    ap.add_argument("--p", type=float, required=True)
    ap.add_argument("--code", choices=["css", "xzzx"], required=True)
    ap.add_argument("--basis", choices=["x", "z"], required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--lookup", type=int, default=0, help="table shots for the near-optimal lookup (0 = skip)")
    a = ap.parse_args()

    t0 = time.perf_counter()
    px, py, pz = pauli_probabilities_for_bias(a.p, a.eta)
    circuit = biased_circuit(a.d, a.d, a.basis, px, py, pz, q=a.p, xzzx=(a.code == "xzzx"))
    out = {"distance": a.d, "rounds": a.d, "eta": a.eta, "p": a.p, "q": a.p, "p_x": px, "p_y": py, "p_z": pz,
           "code": a.code, "basis": a.basis, "detectors": circuit.num_detectors,
           "prereg_sha256": hashlib.sha256(open(PREREG, "rb").read()).hexdigest(),
           "train_shots": TRAIN_SHOTS, "test_shots": TEST_SHOTS, "steps": STEPS, "batch": BATCH, "val_frac": VAL_FRAC}
    x_tr, o_tr = circuit.compile_detector_sampler(seed=SEED_TRAIN).sample(TRAIN_SHOTS, separate_observables=True)
    x_te, o_te = circuit.compile_detector_sampler(seed=SEED_TEST).sample(TEST_SHOTS, separate_observables=True)
    x_tr, y_tr = x_tr.astype(np.uint8), o_tr[:, 0].astype(np.uint8)
    x_te, y_te = x_te.astype(np.uint8), o_te[:, 0].astype(np.uint8)
    out["flip_rate_test"] = float(y_te.mean())
    out["trivial"] = float(np.mean(y_te != 0))

    dem = circuit.detector_error_model(decompose_errors=True)
    out["dem_parallel_edges"] = parallel_edges_with_different_effect(dem)
    pm = pymatching.Matching.from_detector_error_model(dem).decode_batch(x_te)[:, 0].astype(np.uint8)
    out["matching"] = float(np.mean(pm != y_te))

    res = train(MLP(), x_tr, y_tr, x_te, y_te, steps=STEPS, batch=BATCH, val_frac=VAL_FRAC)
    out["mlp"] = res.test_error
    out["mlp_best_step"] = res.best_step
    out["mlp_train_seconds"] = round(res.train_seconds, 1)
    out["mlp_fraction_predicted_1"] = float(np.mean(res.predictions))
    y_perm = np.random.default_rng(SEED_PERM).permutation(y_tr)
    out["null_permuted_labels"] = train(MLP(), x_tr, y_perm, x_te, y_te, steps=STEPS, batch=BATCH, seed=1,
                                        val_frac=VAL_FRAC).test_error
    out["mlp_vs_matching"] = paired("mlp", res.predictions, "matching", pm, y_te)
    for k in ("trivial", "matching", "mlp", "null_permuted_labels"):
        out[k + "_ci95"] = wilson(int(round(out[k] * TEST_SHOTS)), TEST_SHOTS)
    if a.lookup:
        lk, seen, n_syn = lookup_decoder(circuit, a.lookup, x_te, pm)
        out["lookup"] = float(np.mean(lk != y_te))
        out["lookup_ci95"] = wilson(int(np.sum(lk != y_te)), TEST_SHOTS)
        out["lookup_table_shots"], out["lookup_test_fraction_seen"], out["lookup_syndromes"] = a.lookup, seen, n_syn
        out["mlp_vs_lookup"] = paired("mlp", res.predictions, "lookup", lk, y_te)
        out["matching_vs_lookup"] = paired("matching", pm, "lookup", lk, y_te)
    out["wall_seconds"] = round(time.perf_counter() - t0, 1)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps({k: out[k] for k in ("distance", "eta", "p", "code", "basis", "trivial", "matching", "mlp",
                                           "null_permuted_labels", "wall_seconds")}))


if __name__ == "__main__":
    main()
