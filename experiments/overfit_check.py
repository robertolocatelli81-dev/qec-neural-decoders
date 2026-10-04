"""Can the model learn at all? Train on ONE fixed batch and measure the error on that same batch.

A model without a bug must memorise a single batch (training error -> 0). If the detector-Transformer cannot, its poor
test numbers are a defect, not a compute budget. The MLP is run as the comparison.

Usage: python experiments/overfit_check.py [distance] [batch] [steps]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qecnd.data import Experiment, sample  # noqa: E402
from qecnd.model import MLP, DetectorTransformer, train  # noqa: E402


def main():
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    b = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    steps = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
    x, y = sample(Experiment(d, d, 0.005), b, seed=7)
    out = {"distance": d, "batch": b, "steps": steps, "ones_in_batch": int(y.sum())}
    for name, model in (("mlp", MLP()), ("transformer", DetectorTransformer(n_det=x.shape[1]))):
        res = train(model, x, y, x, y, steps=steps, batch=b)        # train and evaluate on the SAME fixed batch
        out[name] = {"error_on_the_memorised_batch": res.test_error, "train_seconds": round(res.train_seconds, 1)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
