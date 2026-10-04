"""Review driver: the SAME calls as experiments/correlated.py (same circuit, seeds, decoders, training) but it saves the
MLP's per-shot predictions (<out_json> with .json replaced by _mlp_pred.npy) and pairs the MLP also against the matching
whose merged edges carry the MOST PROBABLE fault set (merge_most_probable below). Output: <out_json>
(results/correlated/review_pc_<pc>.json). results/correlated/matching_merge_most_probable.json is the edge-level
comparison between PyMatching's merged edges and this rule, written during the review; its error rates are the ones this
script reproduces. Usage: python experiments/correlated_review.py <pc> <out_json>"""
import json
import sys

import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pymatching

from experiments.correlated import mcnemar_exact
from qecnd.correlated import correlated_circuit
from qecnd.model import MLP, train


def merge_most_probable(dem):
    groups = {}
    for inst in dem.flattened():
        if inst.type != "error":
            continue
        p = inst.args_copy()[0]
        comps = [[]]
        for t in inst.targets_copy():
            if t.is_separator():
                comps.append([])
            else:
                comps[-1].append(t)
        for comp in comps:
            dets = tuple(sorted(t.val for t in comp if t.is_relative_detector_id()))
            obs = frozenset(t.val for t in comp if t.is_logical_observable_id())
            key = (dets[0], None) if len(dets) == 1 else dets
            groups.setdefault(key, {}).setdefault(obs, []).append(p)
    m = pymatching.Matching()
    for (u, v), byobs in groups.items():
        podd = 0.0
        for p in [p for ps in byobs.values() for p in ps]:
            podd = podd * (1 - p) + (1 - podd) * p
        best = set(max(byobs.items(), key=lambda kv: sum(kv[1]))[0])
        w = float(np.log((1 - podd) / podd))
        if v is None:
            m.add_boundary_edge(u, fault_ids=best, weight=w, error_probability=podd)
        else:
            m.add_edge(u, v, fault_ids=best, weight=w, error_probability=podd)
    return m


def main():
    pc, out_path = float(sys.argv[1]), sys.argv[2]
    c = correlated_circuit(3, 3, 0.005, pc)
    x_tr, o_tr = c.compile_detector_sampler(seed=1).sample(500_000, separate_observables=True)
    x_te, o_te = c.compile_detector_sampler(seed=2).sample(200_000, separate_observables=True)
    x_tr, y_tr = x_tr.astype(np.uint8), o_tr[:, 0].astype(np.uint8)
    x_te, y_te = x_te.astype(np.uint8), o_te[:, 0].astype(np.uint8)
    dem = c.detector_error_model(decompose_errors=True)
    corr = pymatching.Matching.from_detector_error_model(dem, enable_correlations=True).decode_batch(
        x_te, enable_correlations=True)[:, 0].astype(np.uint8)
    plain = pymatching.Matching.from_detector_error_model(dem).decode_batch(x_te)[:, 0].astype(np.uint8)
    mine = merge_most_probable(dem).decode_batch(x_te)[:, 0].astype(np.uint8)
    res = train(MLP(), x_tr, y_tr, x_te, y_te, steps=4000, batch=512, val_frac=0.1)
    mlp = res.predictions
    np.save(out_path.replace(".json", "_mlp_pred.npy"), mlp)

    def pair(a, b):
        r, w = int(np.sum((a == y_te) & (b != y_te))), int(np.sum((a != y_te) & (b == y_te)))
        return {"first_right_second_wrong": r, "first_wrong_second_right": w, "mcnemar_p": mcnemar_exact(r, w)}

    out = {"pc": pc, "trivial": float(y_te.mean()), "matching": float(np.mean(plain != y_te)),
           "matching_correlated": float(np.mean(corr != y_te)), "matching_merge_most_probable": float(np.mean(mine != y_te)),
           "mlp": res.test_error, "best_step": res.best_step, "mlp_train_seconds": round(res.train_seconds, 1),
           "mlp_vs_matching_correlated": pair(mlp, corr), "mlp_vs_matching_merge_most_probable": pair(mlp, mine),
           "matching_merge_most_probable_vs_matching_correlated": pair(mine, corr)}
    json.dump(out, open(out_path, "w"), indent=1)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
