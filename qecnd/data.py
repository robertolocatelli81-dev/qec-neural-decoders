"""Syndrome data for a rotated surface-code memory experiment, generated with stim.

Each sample is the full detector record of one shot (rounds x stabilisers, flattened by stim) and one label: whether the
logical observable flipped. The same samples feed every decoder, so the comparison is on identical inputs.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim


@dataclass(frozen=True)
class Experiment:
    distance: int
    rounds: int
    p: float                      # physical error rate of the uniform circuit-level noise model
    basis: str = "z"

    def circuit(self) -> stim.Circuit:
        return stim.Circuit.generated(
            f"surface_code:rotated_memory_{self.basis}",
            distance=self.distance,
            rounds=self.rounds,
            after_clifford_depolarization=self.p,
            before_round_data_depolarization=self.p,
            before_measure_flip_probability=self.p,
            after_reset_flip_probability=self.p,
        )


def sample(exp: Experiment, shots: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(detectors uint8 [shots, n_det], logical flip uint8 [shots])."""
    sampler = exp.circuit().compile_detector_sampler(seed=seed)
    det, obs = sampler.sample(shots, separate_observables=True)
    return det.astype(np.uint8), obs[:, 0].astype(np.uint8)
