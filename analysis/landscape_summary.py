#!/usr/bin/env python3
"""Summarize the DHFR single-mutant landscape and place the RL results in it.

Run from the repo root:  python3 analysis/landscape_summary.py [--out results/landscape_summary.json]
Writes numbers only (no figures); the figures in the report are drawn from this file.
"""
import argparse, collections, glob, json, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, ".")
from net.rew import DHFROracle  # noqa: E402

A = "ACDEFGHIKLMNPQRSTVWY"
IX = {a: i for i, a in enumerate(A)}
MIN_F, MAX_F = -5.215, 4.437
norm = lambda x: float(np.clip((x - MIN_F) / (MAX_F - MIN_F), 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/landscape_summary.json")
    a = ap.parse_args()

    d = pd.read_csv("data/DHFR/medium.csv")
    D = np.array([[IX[c] for c in s] for s in d.sequence])
    y = d.target.values
    L = D.shape[1]
    ref = np.array([collections.Counter(D[:, i]).most_common(1)[0][0] for i in range(L)])
    pos = np.array([int(np.where(r != ref)[0][0]) for r in D])
    aa = D[np.arange(len(D)), pos]
    ent = pd.read_csv("data/DHFR/conservation_scores.csv").set_index("position").shannon_entropy

    m = DHFROracle(n_tokens=20, make_one_hot=True)
    m.load_state_dict(torch.load("ckpt/DHFR/oracle.ckpt", map_location="cpu", weights_only=False)["model_state_dict"])
    m.eval()
    with torch.no_grad():
        p = torch.cat([m(torch.as_tensor(D[i:i + 1024])) for i in range(0, len(D), 1024)]).numpy()
    slope, icpt = np.polyfit(p, y, 1)

    ny = (y - MIN_F) / (MAX_F - MIN_F)
    out = {"n_variants": int(len(y)), "positions": int(L),
           "label_quantiles_norm": {q: float(np.quantile(ny, q)) for q in (0.05, 0.25, 0.5, 0.75, 0.9, 0.99)},
           "n_labels_ge": {str(t): int((ny >= t).sum()) for t in (0.738, 0.8, 0.866, 0.9, 0.95, 0.999)}}

    # top singles
    top = np.argsort(-y)[:15]
    out["top_singles"] = [{"mut": f"{A[ref[pos[k]]]}{pos[k] + 1}{A[aa[k]]}", "label_norm": float(ny[k]), "entropy": float(ent[pos[k] + 1])} for k in top]

    # per-position summary (mean/min/max normalized label, 219 rows) for the heatmap
    out["per_position"] = [{"pos": int(i + 1), "ref": A[ref[i]], "mean": float(ny[pos == i].mean()),
                            "min": float(ny[pos == i].min()), "max": float(ny[pos == i].max()), "entropy": float(ent[i + 1])} for i in range(L)]
    # full 219 x 20 matrix of normalized labels (NaN -> null at the reference residue)
    M = np.full((L, 20), np.nan)
    M[pos, aa] = ny
    out["matrix_norm"] = [[None if np.isnan(v) else round(float(v), 4) for v in row] for row in M]

    # oracle vs label (every 4th point keeps the file small)
    out["oracle_vs_label"] = [[round(float(ny[i]), 4), round(float(norm(slope * p[i] + icpt)), 4)] for i in range(0, len(y), 4)]
    out["oracle_corr"] = float(np.corrcoef(p, y)[0, 1])

    # where the RL runs landed: mutations vs reference, and how good each is as a single mutant
    single = {(int(pp), int(c)): float(v) for pp, c, v in zip(pos, aa, ny)}
    rl = []
    for f in sorted(glob.glob("results/runs/*/final_research_metrics.json")):
        j = json.load(open(f))
        if j.get("total_rounds", 0) < 2:
            continue
        s = np.array([IX[c] for c in j["global_best_sequence"]])
        muts = [(int(i), int(s[i])) for i in range(L) if s[i] != ref[i]]
        rl.append({"run": f.split("/")[-2], "rounds": j["total_rounds"], "best": j.get("global_max_fitness_normalized", j.get("global_max_fitness")),
                   "n_mut_vs_ref": len(muts), "muts": [f"{A[ref[i]]}{i + 1}{A[c]}" for i, c in muts],
                   "single_labels": [round(single[(i, c)], 3) for i, c in muts],
                   "single_pctile": [round(float((ny <= single[(i, c)]).mean()), 3) for i, c in muts]})
    out["rl_runs"] = rl
    json.dump(out, open(a.out, "w"))
    print(json.dumps({k: out[k] for k in ("label_quantiles_norm", "n_labels_ge", "top_singles")}, indent=1)[:2500])
    print("\nRL final mutations vs reference, with single-mutant label percentile:")
    for r in rl:
        print(f"  {r['best']:.3f}  {r['run'][:46]:46s} {' '.join(r['muts']):28s} singles pct={r['single_pctile']}")
    pm = pd.DataFrame(out["per_position"])
    print("\npositions with highest mean label:", pm.nlargest(8, "mean")[["pos", "ref", "mean"]].values.tolist())
    print("n-terminal (pos 1-10) mean label:", round(pm[pm.pos <= 10]["mean"].mean(), 3), "| rest:", round(pm[pm.pos > 10]["mean"].mean(), 3))


if __name__ == "__main__":
    main()
