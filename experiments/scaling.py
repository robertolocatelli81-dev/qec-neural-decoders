"""Cost of the detector-Transformer on THIS machine as the code distance grows (rounds = distance), measured, not assumed.

For each distance: detectors per shot, stim sampling time for 100k shots, seconds per training step at a fixed batch,
and peak resident memory. The point is the curve, not one number.

Usage: python experiments/scaling.py [batch] [timed_steps] [distances...]
"""
import json
import os
import resource
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qecnd.data import Experiment, sample  # noqa: E402
from qecnd.model import DetectorTransformer, train  # noqa: E402


def main():
    batch = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    steps = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    dists = [int(a) for a in sys.argv[3:]] or [3, 5, 7, 9]
    rows = []
    for d in dists:
        exp = Experiment(d, d, 0.005)
        t0 = time.perf_counter(); x, y = sample(exp, 100_000, seed=1); t_sample = time.perf_counter() - t0
        res = train(DetectorTransformer(n_det=x.shape[1]), x[:20_000], y[:20_000], x[:4096], y[:4096], steps=steps, batch=batch)
        rows.append({"distance": d, "rounds": d, "detectors": int(x.shape[1]), "stim_seconds_per_100k_shots": round(t_sample, 3),
                     "batch": batch, "seconds_per_step": round(res.seconds_per_step, 4),
                     "shots_per_second_training": round(batch / res.seconds_per_step, 1),
                     "peak_rss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024})
        print(json.dumps(rows[-1]), flush=True)


if __name__ == "__main__":
    main()
