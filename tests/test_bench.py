"""Minimal tests of the bench itself (seconds on CPU): the test set is disjoint from training, every decoder sees the
same shots, the null control can fail, matching beats the trivial decoder, and the loop is deterministic for a seed.

Run: python -m pytest -q tests   (or: python tests/test_bench.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qecnd.baselines import logical_error_rate, matching_predictions  # noqa: E402
from qecnd.data import Experiment, sample  # noqa: E402
from qecnd.model import MLP, DetectorTransformer, train  # noqa: E402

EXP = Experiment(3, 3, 0.005)


def test_sample_shapes_and_values():
    x, y = sample(EXP, 1000, seed=1)
    assert x.shape == (1000, 24) and y.shape == (1000,)
    assert set(np.unique(x)) <= {0, 1} and set(np.unique(y)) <= {0, 1}


def test_different_seeds_give_different_shots_and_same_seed_is_deterministic():
    x1, _ = sample(EXP, 2000, seed=1)
    x2, _ = sample(EXP, 2000, seed=2)
    x1b, _ = sample(EXP, 2000, seed=1)
    assert np.array_equal(x1, x1b)
    assert not np.array_equal(x1, x2)


def test_matching_beats_trivial_on_same_shots():
    x, y = sample(EXP, 20_000, seed=2)
    trivial = logical_error_rate(np.zeros_like(y), y)
    matching = logical_error_rate(matching_predictions(EXP, x), y)
    assert matching < trivial / 2


def test_null_control_lands_at_chance_and_learned_beats_it():
    x_tr, y_tr = sample(EXP, 20_000, seed=1)
    x_te, y_te = sample(EXP, 10_000, seed=2)
    trivial = float(y_te.mean())
    learned = train(MLP(hidden=64), x_tr, y_tr, x_te, y_te, steps=1000, batch=256).test_error
    y_perm = np.random.default_rng(3).permutation(y_tr)
    null = train(MLP(hidden=64), x_tr, y_perm, x_te, y_te, steps=1000, batch=256, seed=1).test_error
    assert learned < trivial / 2                 # measured 3.6% vs 10.4% (4 Oct 2026): far from the floor
    assert abs(null - trivial) < 0.03            # measured 10.4% = trivial: the null stays at the floor, the bench can fail


def test_train_is_deterministic_for_a_seed():
    x_tr, y_tr = sample(EXP, 5_000, seed=1)
    x_te, y_te = sample(EXP, 2_000, seed=2)
    a = train(MLP(hidden=32), x_tr, y_tr, x_te, y_te, steps=50, batch=128, seed=7).test_error
    b = train(MLP(hidden=32), x_tr, y_tr, x_te, y_te, steps=50, batch=128, seed=7).test_error
    assert a == b


def test_transformer_runs_and_evaluates_in_chunks():
    x_tr, y_tr = sample(EXP, 2_000, seed=1)
    x_te, y_te = sample(EXP, 1_000, seed=2)      # 1000 is not a multiple of the batch: the last chunk is shorter
    res = train(DetectorTransformer(n_det=24, width=16, depth=1, heads=2), x_tr, y_tr, x_te, y_te, steps=5, batch=64)
    assert 0.0 <= res.test_error <= 1.0 and res.steps == 5


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn(); print("ok", name)
