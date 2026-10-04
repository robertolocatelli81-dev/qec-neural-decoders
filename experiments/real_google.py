"""Real hardware data: Google's "Suppressing quantum errors by scaling a surface code logical qubit" (Nature 2023),
Zenodo record 6804040 (CC-BY 4.0), unzipped under data/qec3v5/.

For one experiment folder:
  1. positive control — PyMatching here, configured with the experiment's circuit_detector_error_model.dem, must
     reproduce Google's own obs_flips_predicted_by_pymatching (measured: 100.00% of shots); PyMatching with the
     data-derived models (pij_from_even_for_odd.dem on odd shots, the other on even shots) is reported as a variant;
  2. the logical error rate of every decoder Google published (pymatching, correlated matching, belief matching, tensor
     network contraction) and of ours, computed here from the raw files on the same 50,000 shots;
  3. optionally a learned decoder: pre-trained on shots SAMPLED from the even-shot model, evaluated on the ODD real shots
     only (it never sees a real shot in training), against the decoders above on those same odd shots.

Usage: python experiments/real_google.py <experiment dir> [--learned mlp|transformer] [--train-shots N] [--steps N] [--batch N]
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np  # noqa: E402
import pymatching  # noqa: E402
import stim  # noqa: E402

DECODERS = ("pymatching", "correlated_matching", "belief_matching", "tensor_network_contraction")


def read_01(path):
    with open(path) as fh:
        return np.array([int(line.strip()) for line in fh if line.strip()], dtype=np.uint8)


def props(d):
    out = {}
    with open(os.path.join(d, "properties.yml")) as fh:
        for line in fh:
            if ":" in line:
                k, v = line.split(":", 1)
                out[k.strip()] = v.strip()
    return out


def load(d):
    p = props(d)
    n_det = int(p["circuit_detectors"])
    det = stim.read_shot_data_file(path=os.path.join(d, "detection_events.b8"), format="b8", num_detectors=n_det)
    actual = read_01(os.path.join(d, "obs_flips_actual.01"))
    preds = {k: read_01(os.path.join(d, f"obs_flips_predicted_by_{k}.01")) for k in DECODERS
             if os.path.exists(os.path.join(d, f"obs_flips_predicted_by_{k}.01"))}
    return p, det.astype(np.uint8), actual, preds


def matching_by_parity(d, det):
    pred = np.zeros(len(det), dtype=np.uint8)
    for model, parity in (("pij_from_even_for_odd.dem", 1), ("pij_from_odd_for_even.dem", 0)):
        m = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel.from_file(os.path.join(d, model)))
        idx = np.arange(parity, len(det), 2)
        pred[idx] = m.decode_batch(det[idx])[:, 0]
    return pred


def main():
    d = sys.argv[1]
    arch = sys.argv[sys.argv.index("--learned") + 1] if "--learned" in sys.argv else None
    n_tr = int(sys.argv[sys.argv.index("--train-shots") + 1]) if "--train-shots" in sys.argv else 500_000
    steps = int(sys.argv[sys.argv.index("--steps") + 1]) if "--steps" in sys.argv else 3000
    batch = int(sys.argv[sys.argv.index("--batch") + 1]) if "--batch" in sys.argv else 256
    p, det, actual, preds = load(d)
    m = pymatching.Matching.from_detector_error_model(stim.DetectorErrorModel.from_file(os.path.join(d, "circuit_detector_error_model.dem")))
    ours = m.decode_batch(det)[:, 0].astype(np.uint8)
    ours_pij = matching_by_parity(d, det)
    out = {"experiment": os.path.basename(d.rstrip("/")), "shots": int(len(actual)), "detectors": int(det.shape[1]),
           "positive_control_agreement_with_google_pymatching": float(np.mean(ours == preds["pymatching"])) if "pymatching" in preds else None,
           "ler_all_shots": {k: float(np.mean(v != actual)) for k, v in preds.items()}}
    out["ler_all_shots"]["pymatching_here"] = float(np.mean(ours != actual))
    out["ler_all_shots"]["pymatching_here_pij"] = float(np.mean(ours_pij != actual))
    odd = np.arange(1, len(actual), 2)
    out["ler_odd_shots"] = {k: float(np.mean(v[odd] != actual[odd])) for k, v in preds.items()}
    out["ler_odd_shots"]["pymatching_here"] = float(np.mean(ours[odd] != actual[odd]))
    out["ler_odd_shots"]["pymatching_here_pij"] = float(np.mean(ours_pij[odd] != actual[odd]))
    out["ler_odd_shots"]["trivial"] = float(np.mean(actual[odd]))
    if arch:
        from qecnd.model import MLP, DetectorTransformer, train
        dem = stim.DetectorErrorModel.from_file(os.path.join(d, "pij_from_even_for_odd.dem"))
        x_tr, y_tr, _ = dem.compile_sampler(seed=1).sample(n_tr)
        x_tr, y_tr = x_tr.astype(np.uint8), y_tr[:, 0].astype(np.uint8)
        mk = (lambda: MLP()) if arch == "mlp" else (lambda: DetectorTransformer(n_det=det.shape[1]))
        res = train(mk(), x_tr, y_tr, det[odd], actual[odd], steps=steps, batch=batch)
        out["learned"] = {"arch": arch, "train_shots_simulated": n_tr, "steps": steps, "batch": batch,
                          "ler_odd_real_shots": res.test_error, "train_seconds": round(res.train_seconds, 1)}
        perm = np.random.default_rng(3).permutation(y_tr)
        out["learned"]["null_permuted_labels_ler_odd"] = train(mk(), x_tr, perm, det[odd], actual[odd], steps=steps,
                                                               batch=batch, seed=1).test_error
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
