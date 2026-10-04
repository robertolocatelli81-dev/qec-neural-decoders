"""Pre-registered experiment prereg/PREREG_correlated_20261004.md (sha256 in the .sha256 file next to it): a learned decoder
against matching given the TRUE detector error model, under long-range correlated X errors.

Usage: python experiments/correlated.py <pc> [train_shots] [steps] [test_shots]
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pymatching  # noqa: E402

from qecnd.correlated import correlated_circuit  # noqa: E402
from qecnd.model import MLP, train  # noqa: E402


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value: b, c = discordant counts; binomial(b + c, 1/2), computed in log space."""
    n, k = b + c, min(b, c)
    if n == 0:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2) for i in range(k + 1)]
    m = max(logs)
    tail = math.exp(m) * sum(math.exp(v - m) for v in logs)
    return min(1.0, 2 * tail)


def main():
    pc = float(sys.argv[1])
    n_tr = int(sys.argv[2]) if len(sys.argv) > 2 else 500_000
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 4000
    n_te = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000
    c = correlated_circuit(3, 3, 0.005, pc)
    x_tr, o_tr = c.compile_detector_sampler(seed=1).sample(n_tr, separate_observables=True)
    x_te, o_te = c.compile_detector_sampler(seed=2).sample(n_te, separate_observables=True)
    x_tr, y_tr, x_te, y_te = x_tr.astype(np.uint8), o_tr[:, 0].astype(np.uint8), x_te.astype(np.uint8), o_te[:, 0].astype(np.uint8)
    dem = c.detector_error_model(decompose_errors=True)
    plain = pymatching.Matching.from_detector_error_model(dem).decode_batch(x_te)[:, 0].astype(np.uint8)
    corr = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True).decode_batch(
        x_te, enable_correlations=True)[:, 0].astype(np.uint8)
    res = train(MLP(), x_tr, y_tr, x_te, y_te, steps=steps, batch=512, val_frac=0.1)
    mlp = res.predictions
    null = train(MLP(), x_tr, np.random.default_rng(3).permutation(y_tr), x_te, y_te, steps=steps, batch=512, seed=1,
                 val_frac=0.1).test_error
    out = {"prereg_sha256": open(os.path.join(os.path.dirname(__file__), "..", "prereg",
                                              "PREREG_correlated_20261004.sha256")).read().split()[0],
           "pc": pc, "train_shots": n_tr, "test_shots": n_te, "steps": steps, "best_step": res.best_step,
           "trivial": float(np.mean(y_te)), "matching": float(np.mean(plain != y_te)),
           "matching_correlated": float(np.mean(corr != y_te)), "mlp": res.test_error, "null_permuted_labels": null,
           "mlp_train_seconds": round(res.train_seconds, 1)}
    better_name, better = min((("matching", plain), ("matching_correlated", corr)), key=lambda kv: np.mean(kv[1] != y_te))
    mlp_right_m_wrong = int(np.sum((mlp == y_te) & (better != y_te)))
    mlp_wrong_m_right = int(np.sum((mlp != y_te) & (better == y_te)))
    out["paired_vs"] = better_name
    out["discordant_mlp_right_matching_wrong"] = mlp_right_m_wrong
    out["discordant_mlp_wrong_matching_right"] = mlp_wrong_m_right
    out["mcnemar_p_two_sided"] = mcnemar_exact(mlp_right_m_wrong, mlp_wrong_m_right)
    out["prereg_win"] = bool(out["mlp"] < out[better_name] and out["mcnemar_p_two_sided"] < 0.01
                             and abs(null - out["trivial"]) <= 0.01)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
