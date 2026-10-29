"""Flux-drift study of prereg/PREREG_drift_20261022.md: each data qubit of a rotated surface code (CSS or XZZX frame)
gets its own Pauli channel from its own flux offset; decoders M (calibrated at t = 0), G (current mean rates),
O (per-qubit current rates) and N1 (O with the rates permuted among the data qubits) decode the same shots.

  python experiments/drift_gap.py --controls --out results/drift/rerun/controls.json
  python experiments/drift_gap.py --grid --out results/drift/rerun/grid_full.json
--out must name a file that does not exist yet (results are never overwritten); with --grid its folder must exist.
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
import stim  # noqa: E402

from qecnd.biased import data_qubits, deformed_qubits  # noqa: E402
from qecnd.cos2phi import Cos2PhiParams, gamma1_flux_1f, spectrum  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PREREG = os.path.join(ROOT, "prereg", "PREREG_drift_20261022.md")
A_PHI = 5.6e-6            # Φ0/√Hz
DRIFT = 1e-4              # Φ0 per hour (0.1 mΦ0/h)
T1_PAPER_US, T2E_US, TAU_US = 70.0, 2.5, 1.0
EPISODES, SHOTS, BOOT = 200, 1000, 2000


# ------------------------------------------------------------------ physics: rates of one qubit at flux offset δ (Φ0)
def _tables():
    p = Cos2PhiParams()
    grid = np.linspace(0.0, 2.5e-3, 201)
    f, t1 = [], []
    for d in grid:
        s = spectrum(p, phi_ext=np.pi + 2 * np.pi * d, n_max=40, levels=2)
        f.append(s.energies[1] * 1e9)
        t1.append(1.0 / gamma1_flux_1f(p, s, a_phi_in_phi0=A_PHI))
    f, t1 = np.array(f), np.array(t1)
    slope = np.gradient(f, grid)                      # Hz per Φ0
    slope[0] = 0.0                                    # sweet spot: f is even in δ (one-sided edge difference is not)
    return grid, slope, t1 / t1[0]


GRID, SLOPE, T1_RATIO = _tables()


def raw_rates(delta: float, kappa: float) -> tuple:
    a = min(abs(delta), GRID[-1])
    slope = float(np.interp(a, GRID, SLOPE))
    g1 = kappa * A_PHI * 2 * math.pi * slope           # 1/s
    g0 = 1.0 / (T2E_US * 1e-6)
    g = math.sqrt(g0 * g0 + g1 * g1)
    tau = TAU_US * 1e-6
    pz = (1 - math.exp(-(g * tau) ** 2)) / 2
    t1 = T1_PAPER_US * 1e-6 * float(np.interp(a, GRID, T1_RATIO))
    px = tau / (4 * t1)
    return px, px, pz


# ------------------------------------------------------------------ circuits with one channel per data qubit
_BASE = {}


def base_circuit(d: int, basis: str) -> stim.Circuit:
    key = (d, basis)
    if key not in _BASE:
        _BASE[key] = stim.Circuit.generated(f"surface_code:rotated_memory_{basis}", distance=d, rounds=d,
                                            before_round_data_depolarization=0.5, before_measure_flip_probability=0.5)
    return _BASE[key]


def circuit(d: int, basis: str, xzzx: bool, rates: dict, q: float) -> stim.Circuit:
    """rates: data qubit -> (p_X, p_Y, p_Z) in the CSS frame; on deformed qubits X and Z are swapped (qecnd/biased.py)."""
    base = base_circuit(d, basis)
    deformed = deformed_qubits(base) if xzzx else set()
    data = set(data_qubits(base))

    def rebuild(block: stim.Circuit) -> stim.Circuit:
        out = stim.Circuit()
        for inst in block:
            if isinstance(inst, stim.CircuitRepeatBlock):
                out.append(stim.CircuitRepeatBlock(inst.repeat_count, rebuild(inst.body_copy())))
                continue
            if inst.name == "DEPOLARIZE1":
                for t in inst.targets_copy():
                    px, py, pz = rates[t.value]
                    out.append("PAULI_CHANNEL_1", [t.value], [pz, py, px] if t.value in deformed else [px, py, pz])
                continue
            if inst.name in ("X_ERROR", "Z_ERROR"):
                targets = [t.value for t in inst.targets_copy()]
                if set(targets) <= data:
                    continue
                if q > 0:
                    out.append("X_ERROR", targets, q)
                continue
            out.append(inst)
        return out

    return rebuild(base)


def matcher(c: stim.Circuit) -> pymatching.Matching:
    return pymatching.Matching.from_detector_error_model(c.detector_error_model(decompose_errors=True))


# ------------------------------------------------------------------ one cell
def episode_rates(rng, qubits, t_h, kappa, k):
    deltas = rng.normal(0.0, DRIFT, size=len(qubits)) * t_h
    true = {qb: tuple(min(0.5, k * x) for x in raw_rates(dl, kappa)) for qb, dl in zip(qubits, deltas)}
    return true


def run_cell(d, basis, frame, p_cal, t_h, kappa, seed, episodes=EPISODES, shots=SHOTS, override=None):
    rng = np.random.default_rng(seed)
    base = base_circuit(d, basis)
    qubits = data_qubits(base)
    px0, py0, pz0 = raw_rates(0.0, kappa)
    k = p_cal / (px0 + py0 + pz0)
    cal = {qb: (k * px0, k * py0, k * pz0) for qb in qubits}
    xzzx = frame == "xzzx"
    m_dec = matcher(circuit(d, basis, xzzx, cal, p_cal))
    errs = {n: [] for n in ("M", "G", "O", "N1")}
    identical_t0 = True
    for e in range(episodes):
        true = override(rng, qubits, cal) if override else episode_rates(rng, qubits, t_h, kappa, k)
        mean = tuple(float(np.mean([true[qb][i] for qb in qubits])) for i in range(3))
        perm = rng.permutation(len(qubits))
        shuffled = {qb: true[qubits[perm[i]]] for i, qb in enumerate(qubits)}
        c_true = circuit(d, basis, xzzx, true, p_cal)
        det, obs = c_true.compile_detector_sampler(seed=int(rng.integers(2**63))).sample(shots, separate_observables=True)
        obs = obs[:, 0]
        preds = {"M": m_dec.decode_batch(det)[:, 0],
                 "G": matcher(circuit(d, basis, xzzx, {qb: mean for qb in qubits}, p_cal)).decode_batch(det)[:, 0],
                 "O": matcher(c_true).decode_batch(det)[:, 0],
                 "N1": matcher(circuit(d, basis, xzzx, shuffled, p_cal)).decode_batch(det)[:, 0]}
        if t_h == 0 and override is None:
            identical_t0 &= bool(np.array_equal(preds["M"], preds["O"]) and np.array_equal(preds["G"], preds["O"]))
        for n, pr in preds.items():
            errs[n].append(int(np.sum(pr != obs)))
    return {n: np.array(v) for n, v in errs.items()}, identical_t0


def ratio_ci(err_a, err_b, rng):
    """R = (L_a − L_b)/L_a with cluster bootstrap over episodes; returns (R, lo, hi)."""
    def r(a, b):
        return (a.sum() - b.sum()) / a.sum() if a.sum() > 0 else float("nan")
    n = len(err_a)
    bs = []
    for _ in range(BOOT):
        i = rng.integers(0, n, n)
        bs.append(r(err_a[i], err_b[i]))
    lo, hi = np.nanpercentile(bs, [2.5, 97.5])
    return float(r(err_a, err_b)), float(lo), float(hi)


def summarise(errs, shots):
    rng = np.random.default_rng(12345)
    tot = len(errs["O"]) * shots
    out = {"L_" + n: float(v.sum() / tot) for n, v in errs.items()}
    for a, b in (("G", "O"), ("M", "O"), ("G", "N1")):
        out["R_%s%s" % (b, a)] = ratio_ci(errs[a], errs[b], rng)
    return out


def prereg_sha():
    with open(PREREG, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controls", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--out", required=True)
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    res = {"prereg_sha256": prereg_sha(), "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "stim": stim.__version__, "pymatching": pymatching.__version__, "episodes": EPISODES, "shots": SHOTS}
    if a.controls:
        def one_hot(rng, qubits, cal):
            hot = qubits[int(rng.integers(len(qubits)))]
            return {qb: ((cal[qb][0], cal[qb][1], 0.25) if qb == hot else cal[qb]) for qb in qubits}
        errs, _ = run_cell(5, "z", "xzzx", 0.005, None, 0.833, seed=101, override=one_hot)
        res["C1_C2"] = summarise(errs, SHOTS)
        errs, ident = run_cell(3, "z", "xzzx", 0.005, 0, 4.6, seed=102, episodes=20)
        res["C0_identical_at_t0"] = ident
    if a.grid:
        cells = []
        i = 0
        for t_h in (0, 0.5, 1, 2, 4):
            for kappa in (0.833, 4.6):
                for d in (3, 5):
                    for frame in ("css", "xzzx"):
                        for basis in ("z", "x"):
                            for p_cal in (0.005, 0.01):
                                i += 1
                                name = f"t{t_h}_k{kappa}_d{d}_{frame}_{basis}_p{p_cal}"
                                if a.only and a.only not in name:
                                    continue
                                t0 = time.time()
                                errs, ident = run_cell(d, basis, frame, p_cal, t_h, kappa, seed=1000 + i)
                                s = summarise(errs, SHOTS)
                                s.update(name=name, t_h=t_h, kappa=kappa, d=d, frame=frame, basis=basis, p_cal=p_cal,
                                         identical_t0=ident if t_h == 0 else None, seconds=round(time.time() - t0, 1))
                                cells.append(s)
                                print(name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in s.items()
                                             if k.startswith("L_")}, "R_OG", [round(x, 3) for x in s["R_OG"]], flush=True)
                                with open(a.out + ".partial", "w") as fh:
                                    json.dump(cells, fh)
        res["cells"] = cells
    res["finished_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "x") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
