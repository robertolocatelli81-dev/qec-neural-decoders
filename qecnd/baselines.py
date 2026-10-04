"""Reference decoders on the same samples: minimum-weight perfect matching (PyMatching, from the circuit's detector error
model) and the trivial decoder (always predict "no flip"), which fixes the floor any learned decoder must beat."""
from __future__ import annotations

import numpy as np
import pymatching

from .data import Experiment


def matching_predictions(exp: Experiment, detectors: np.ndarray) -> np.ndarray:
    dem = exp.circuit().detector_error_model(decompose_errors=True)
    matcher = pymatching.Matching.from_detector_error_model(dem)
    return matcher.decode_batch(detectors)[:, 0].astype(np.uint8)


def logical_error_rate(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(pred != truth))
