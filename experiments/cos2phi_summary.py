"""Summary of the cos(2φ) grid: one table from results/cos2phi/d*_eta*_p*_*.json, the pre-registered tests
(McNemar MLP vs PyMatching, Bonferroni over the cells present, null within 1.5 points of trivial) and the verdicts on
P0–P3 of prereg/PREREG_cos2phi_20261007.md, written as measured.

Usage: python experiments/cos2phi_summary.py > results/cos2phi/SUMMARY.md
"""
import glob
import json
import os

ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "cos2phi")
ALPHA = 0.05

cells = []
for f in sorted(glob.glob(os.path.join(ROOT, "d*_eta*_p*_*_*.json"))):
    j = json.load(open(f))
    j["_file"] = os.path.basename(f)
    cells.append(j)
cells.sort(key=lambda j: (j["distance"], j["eta"], j["p"], j["code"], j["basis"]))
PREREG_ETAS = (1.0, 27.5)                      # the 32 pre-registered cells; other η values are sensitivity cells
prereg_cells = [j for j in cells if j["eta"] in PREREG_ETAS]
n = len(cells)
n_b = len(prereg_cells)
alpha_b = ALPHA / n_b if n_b else ALPHA


def pct(x):
    return f"{100 * x:.3f}%"


def ci(j, k):
    lo, hi = j[k + "_ci95"]
    return f"{pct(j[k])} [{100 * lo:.3f}, {100 * hi:.3f}]"


print(f"# cos(2φ) grid — {n} cells ({n_b} pre-registered, η ∈ {PREREG_ETAS}; the rest are sensitivity cells), "
      f"Bonferroni α = {ALPHA}/{n_b} = {alpha_b:.2e}\n")
print("Logical error rate on the same 200,000 held-out shots per cell (95 % Wilson interval); MLP vs PyMatching: exact two-sided "
      "McNemar on the discordant shots. WIN = MLP < PyMatching with p < α_B and the null within 1.5 points of trivial.\n")
print("| d | η | p | code | basis | trivial | PyMatching | MLP | null | lookup | MLP−PM (pts) | McNemar p | WIN | gap recovered | parallel edges |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
wins, verdict = [], {}
for j in cells:
    pm, mlp, null, triv = j["matching"], j["mlp"], j["null_permuted_labels"], j["trivial"]
    p_mc = j["mlp_vs_matching"]["mcnemar_p"]
    null_ok = abs(null - triv) <= 0.015
    win = mlp < pm and p_mc < alpha_b and null_ok
    if win:
        wins.append(j)
    lk = ci(j, "lookup") if "lookup" in j else "—"
    rec = "—"
    if "lookup" in j and pm - j["lookup"] >= 0.001:          # fraction of the matching→lookup gap the MLP recovers
        j["_recovered"] = (pm - mlp) / (pm - j["lookup"])
        rec = f"{100 * j['_recovered']:.0f}%"
    print(f"| {j['distance']} | {j['eta']} | {j['p']} | {j['code'].upper()} | {j['basis'].upper()} | {pct(triv)} | {ci(j, 'matching')} | "
          f"{ci(j, 'mlp')} | {pct(null)} | {lk} | {100 * (mlp - pm):+.3f} | {p_mc:.1e} | {'yes' if win else 'no'} | {rec} | "
          f"{j['dem_parallel_edges']['parallel_edge_sets_with_different_effect']} |")

print("\n## Pre-registered predictions, as measured\n")
# P3 null
worst_null = max(abs(j["null_permuted_labels"] - j["trivial"]) for j in cells) if cells else 0
print(f"- P3 (null at the floor in every cell): largest |null − trivial| = {100 * worst_null:.3f} points → "
      f"{'holds' if worst_null <= 0.015 else 'FAILS'}.")
# P0 control, as pre-registered: no MLP win at η = 1 at d = 5; at d = 3 any win ≤ 0.3 points
ctrl = [j for j in cells if j["eta"] == 1.0]
ctrl_wins = [j for j in wins if j["eta"] == 1.0]
p0_d5 = [j for j in ctrl_wins if j["distance"] == 5]
p0_d3_over = [j for j in ctrl_wins if j["distance"] == 3 and j["matching"] - j["mlp"] > 0.003]
p0_margin = max((j["matching"] - j["mlp"] for j in ctrl_wins), default=0.0)      # the largest MLP win without bias
if ctrl:
    wins_d3 = ", ".join(f"{100 * (j['matching'] - j['mlp']):.3f}" for j in ctrl_wins if j["distance"] == 3) or "none"
    over = ", ".join(f"{100 * (j['matching'] - j['mlp']):.3f} ({j['p']}, {j['code'].upper()}-{j['basis'].upper()})" for j in p0_d3_over)
    print(f"- P0 (control, as pre-registered: no MLP win at η = 1 at d = 5, and at d = 3 any win ≤ 0.3 points): "
          f"{len(ctrl_wins)} WIN cells of {len(ctrl)} control cells, {len(p0_d5)} at d = 5; MLP wins at d = 3: {wins_d3} points → "
          + (f"FAILS: {over} exceed 0.3 points. The P0 margin (largest MLP win without bias) is {100 * p0_margin:.3f} points."
             if p0_d5 or p0_d3_over else "holds."))
else:
    print("- P0: no control cells yet.")
# P1 XZZX vs CSS-X for matching at eta 27.5
print("- P1 (matching on the XZZX frame below matching on CSS memory-X at η = 27.5, equal d and p):")
for d in sorted({j["distance"] for j in cells}):
    for p in sorted({j["p"] for j in cells}):
        c = {(j["code"], j["basis"]): j for j in cells if j["distance"] == d and j["p"] == p and j["eta"] == 27.5}
        if ("css", "x") in c and ("xzzx", "x") in c and ("xzzx", "z") in c:
            cx, xx, xz = c[("css", "x")]["matching"], c[("xzzx", "x")]["matching"], c[("xzzx", "z")]["matching"]
            worst_xzzx = max(xx, xz)
            print(f"  d = {d}, p = {p}: CSS-X {pct(cx)}, XZZX-X {pct(xx)}, XZZX-Z {pct(xz)} → worst XZZX basis "
                  f"{'below' if worst_xzzx < cx else 'NOT below'} CSS-X")
# P2, as pre-registered: at η = 27.5 the MLP beats PyMatching in at least one of the 16 biased cells BY MORE THAN the P0 margin
biased_pre = [j for j in prereg_cells if j["eta"] != 1.0]
bw_pre = [j for j in wins if j["eta"] in PREREG_ETAS and j["eta"] != 1.0]
bw_over = [j for j in bw_pre if j["matching"] - j["mlp"] > p0_margin]
print(f"- P2 (the question, as pre-registered: at η = 27.5 the MLP beats PyMatching in at least one of the {len(biased_pre)} biased "
      f"cells by more than the P0 margin, {100 * p0_margin:.3f} points): {len(bw_pre)} WIN cells, {len(bw_over)} of them above the "
      f"P0 margin → " + ("YES" if bw_over else "NO: no learned-decoder advantage attributable to the cos(2φ) bias at this scale") + ".")
for j in bw_pre:
    print(f"  WIN below the P0 margin: d = {j['distance']}, η = {j['eta']}, p = {j['p']}, {j['code'].upper()}-{j['basis'].upper()}: "
          f"MLP {pct(j['mlp'])} vs PyMatching {pct(j['matching'])} ({100 * (j['matching'] - j['mlp']):.3f} points, "
          f"p = {j['mlp_vs_matching']['mcnemar_p']:.1e})" + (f", lookup {pct(j['lookup'])}" if "lookup" in j else ""))
sens = [j for j in cells if j["eta"] not in PREREG_ETAS]
if sens:
    sw = [j for j in wins if j["eta"] not in PREREG_ETAS]
    print(f"- Sensitivity cells (η ∉ {PREREG_ETAS}, outside the pre-registered set and its Bonferroni count): {len(sens)} cells, "
          f"{len(sw)} WIN" + ("".join(f"; d = {j['distance']}, η = {j['eta']}, p = {j['p']}, {j['code'].upper()}-{j['basis'].upper()}: "
          f"{100 * (j['matching'] - j['mlp']):.3f} points" + (" (below the P0 margin)" if j["matching"] - j["mlp"] <= p0_margin else " (above the P0 margin)") for j in sw) if sw else "") + ".")
bw = [j for j in wins if j["eta"] != 1.0]
# where the MLP is worse, by how much
worse = [(j["mlp"] - j["matching"], j) for j in cells if j["mlp"] > j["matching"]]
if worse:
    g, j = max(worse, key=lambda t: t[0])
    print(f"- MLP worse than PyMatching in {len(worse)} of {n} cells; largest gap {100 * g:+.3f} points "
          f"(d = {j['distance']}, η = {j['eta']}, p = {j['p']}, {j['code'].upper()}-{j['basis'].upper()}).")
lk = [j for j in cells if "lookup" in j]
if lk:
    gaps_pm = [j["matching"] - j["lookup"] for j in lk]
    gaps_mlp = [j["mlp"] - j["lookup"] for j in lk]
    rw = [j["_recovered"] for j in wins if "_recovered" in j]
    if rw:
        r_ctrl = ", ".join(f"{100 * j['_recovered']:.0f}%" for j in wins if "_recovered" in j and j["eta"] == 1.0)
        r_bias = ", ".join(f"{100 * j['_recovered']:.0f}%" for j in wins if "_recovered" in j and j["eta"] != 1.0)
        print(f"- In the WIN cells the MLP recovers {100 * min(rw):.0f}%–{100 * max(rw):.0f}% of the matching→lookup gap "
              f"(η = 1: {r_ctrl}; η ≠ 1: {r_bias}).")
    print(f"- Distance from the near-optimal lookup (d = 3 cells, {len(lk)}): PyMatching − lookup from {100 * min(gaps_pm):+.3f} to "
          f"{100 * max(gaps_pm):+.3f} points; MLP − lookup from {100 * min(gaps_mlp):+.3f} to {100 * max(gaps_mlp):+.3f} points.")
secs = sum(j["wall_seconds"] for j in cells)
tr = sum(j["mlp_train_seconds"] for j in cells)
print(f"\nWall time of the {n} cells: {secs:.0f} s (MLP training {tr:.0f} s of it, both the model and its null), "
      f"prereg sha256 {cells[0]['prereg_sha256'][:12]}… in every file." if cells else "")
