"""Checks F3 and F4 of prereg/PREREG_cos2phi_20261007.md, before the decoder grid:

  F3  code distance (stim's search for undetectable logical errors) of every (code, basis) at d = 3, 5 under pure-Z
      noise (p_X = p_Y = 0) and under depolarising noise (η = 1);
  F4  parallel edges with different logical effect in the detector error model of every grid cell.

Usage: python experiments/cos2phi_checks.py > results/cos2phi/checks.json
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from qecnd.biased import biased_circuit, code_distance, parallel_edges_with_different_effect  # noqa: E402
from qecnd.cos2phi import pauli_probabilities_for_bias  # noqa: E402

out = {"f3_distance": {}, "f4_dem_parallel_edges": {}}
for d in (3, 5):
    for code in ("css", "xzzx"):
        for basis in ("x", "z"):
            key = f"d{d}_{code}_{basis}"
            pure_z = biased_circuit(d, d, basis, 0.0, 0.0, 0.01, 0.01, xzzx=(code == "xzzx"))
            px, py, pz = pauli_probabilities_for_bias(0.01, 1.0)
            depol = biased_circuit(d, d, basis, px, py, pz, 0.01, xzzx=(code == "xzzx"))
            t0 = time.perf_counter()
            out["f3_distance"][key] = {}
            for name, c in (("pure_z", pure_z), ("eta_1", depol)):
                try:
                    out["f3_distance"][key][name] = code_distance(c)
                except ValueError as e:                         # stim: "Failed to find any logical errors" within the limits
                    out["f3_distance"][key][name] = f"not found within limits: {e}"
                print(key, name, out["f3_distance"][key][name], f"{time.perf_counter() - t0:.1f}s", file=sys.stderr)
            for eta in (1.0, 27.5):
                for p in (0.01, 0.03):
                    px, py, pz = pauli_probabilities_for_bias(p, eta)
                    c = biased_circuit(d, d, basis, px, py, pz, p, xzzx=(code == "xzzx"))
                    dem = c.detector_error_model(decompose_errors=True)
                    out["f4_dem_parallel_edges"][f"{key}_eta{eta}_p{p}"] = parallel_edges_with_different_effect(dem)
print(json.dumps(out, indent=1))
