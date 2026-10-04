"""Pilot: one surface-code memory experiment, four decoders on the same test shots.

  trivial   always "no flip"                                   (the floor)
  matching  PyMatching on the circuit's detector error model     (the reference)
  learned   MLP or detector-Transformer trained on detector records
  null      the same model trained on PERMUTED labels               (must land on the floor, or the bench is broken)

Usage: python experiments/pilot.py [distance] [rounds] [p] [train_shots] [steps] [mlp|transformer] [batch]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402

from qecnd.baselines import logical_error_rate, matching_predictions  # noqa: E402
from qecnd.data import Experiment, sample  # noqa: E402
from qecnd.model import MLP, DetectorTransformer, train  # noqa: E402

d = int(sys.argv[1]) if len(sys.argv) > 1 else 3
r = int(sys.argv[2]) if len(sys.argv) > 2 else 3
p = float(sys.argv[3]) if len(sys.argv) > 3 else 0.005
n_tr = int(sys.argv[4]) if len(sys.argv) > 4 else 200_000
steps = int(sys.argv[5]) if len(sys.argv) > 5 else 3000
arch = sys.argv[6] if len(sys.argv) > 6 else "mlp"
batch = int(sys.argv[7]) if len(sys.argv) > 7 else 512

exp = Experiment(d, r, p)
x_tr, y_tr = sample(exp, n_tr, seed=1)
x_te, y_te = sample(exp, 20_000 if arch == "transformer" else 100_000, seed=2)          # different seed: test shots never seen in training
out = {"distance": d, "rounds": r, "p": p, "train_shots": n_tr, "test_shots": len(y_te), "detectors": int(x_tr.shape[1]),
       "flip_rate_test": float(y_te.mean())}
out["trivial"] = logical_error_rate(np.zeros_like(y_te), y_te)
out["matching"] = logical_error_rate(matching_predictions(exp, x_te), y_te)
mk = (lambda: MLP()) if arch == "mlp" else (lambda: DetectorTransformer(n_det=x_tr.shape[1]))
out["arch"], out["steps"], out["batch"] = arch, steps, batch
res = train(mk(), x_tr, y_tr, x_te, y_te, steps=steps, batch=batch)
out["learned"] = res.test_error
out["train_seconds"] = round(res.train_seconds, 1)
y_perm = np.random.default_rng(3).permutation(y_tr)
out["null_permuted_labels"] = train(mk(), x_tr, y_perm, x_te, y_te, steps=steps, batch=batch, seed=1).test_error
print(json.dumps(out, indent=1))
