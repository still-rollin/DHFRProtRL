#!/usr/bin/env python3
"""No-RL baselines scored by the SAME oracle the RL agent optimizes.

Purpose: show how much of the RL result can be reproduced without reinforcement learning.
Run from the repo root:  python3 analysis/baseline_no_rl.py [--out results/baseline_no_rl.json]

Everything starts from the reference sequence (the residue present at each position in all but
19 rows of data/DHFR/medium.csv). The dataset is a complete single-substitution scan of that
reference (219 positions x 19 substitutions); the reference itself is not in the file.
"""
import argparse, collections, itertools, json, sys
import numpy as np, pandas as pd, torch

sys.path.insert(0, ".")
from net.rew import DHFROracle  # noqa: E402

A = "ACDEFGHIKLMNPQRSTVWY"
IX = {a: i for i, a in enumerate(A)}
MIN_F, MAX_F = -5.215, 4.437            # normalization used by config.py
RL_BUDGETS = (15360, 38400)             # oracle calls of the 4-round and 10-round RL runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/baseline_no_rl.json")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)

    d = pd.read_csv("data/DHFR/medium.csv")
    D = np.array([[IX[c] for c in s] for s in d.sequence], dtype=np.int64)
    y = d.target.values
    L = D.shape[1]
    ref = np.array([collections.Counter(D[:, i]).most_common(1)[0][0] for i in range(L)])
    pos = np.array([int(np.where(r != ref)[0][0]) for r in D])
    aa = D[np.arange(len(D)), pos]
    assert ((D != ref).sum(1) == 1).all(), "dataset is not a pure single-mutant scan"

    m = DHFROracle(n_tokens=20, make_one_hot=True)
    m.load_state_dict(torch.load("ckpt/DHFR/oracle.ckpt", map_location="cpu", weights_only=False)["model_state_dict"])
    m.eval()

    def raw(T):
        with torch.no_grad():
            return torch.cat([m(torch.as_tensor(T[i:i + 4096])) for i in range(0, len(T), 4096)]).numpy()

    # linear calibration fitted on the dataset, as net/envr.py does at run time
    slope, icpt = np.polyfit(raw(D), y, 1)
    norm = lambda r: np.clip((slope * r + icpt - MIN_F) / (MAX_F - MIN_F), 0, 1)
    score = lambda T: norm(raw(T))
    def names(muts):
        return " ".join(f"{A[ref[p]]}{p + 1}{A[c]}" for p, c in muts)

    def apply(muts):
        s = ref.copy()
        for p, c in muts:
            s[p] = c
        return s

    # label-additive expectation: ref_label + sum of single-mutant effects. The reference label is not
    # in the data, so the oracle's estimate of it is used.
    ref_oracle_raw = float(slope * raw(ref[None])[0] + icpt)
    eff = {(int(p), int(c)): float(t) for p, c, t in zip(pos, aa, y)}
    additive = lambda muts: ref_oracle_raw + sum(eff[(p, c)] - ref_oracle_raw for p, c in muts)
    to_norm = lambda r: float(np.clip((r - MIN_F) / (MAX_F - MIN_F), 0, 1))

    res = {"reference_oracle_norm": to_norm(ref_oracle_raw),
           "best_single_label_norm": to_norm(y.max()),
           "best_single_oracle_norm": float(score(D).max()),
           "calibration": {"slope": float(slope), "intercept": float(icpt)}, "baselines": {}}

    # 1) exhaustive combinations of the top-20 single mutants (distinct positions), up to 5 mutations
    cand, seen = [], set()
    for k in np.argsort(-y):
        if pos[k] not in seen:
            cand.append((int(pos[k]), int(aa[k]))); seen.add(pos[k])
        if len(cand) == 20:
            break
    best, calls = None, 0
    for K in range(1, 6):
        combos = list(itertools.combinations(cand, K))
        T = np.stack([apply(c) for c in combos])
        v = score(T); calls += len(T)
        b = int(v.argmax())
        row = {"k": K, "combos": len(combos), "best_oracle_norm": float(v[b]), "muts": names(combos[b]),
               "additive_label_norm": to_norm(additive(combos[b]))}
        res["baselines"].setdefault("top20_singles_combined", []).append(row)
        if best is None or v[b] > best[0]:
            best = (float(v[b]), combos[b])
    res["baselines"]["top20_singles_combined_total_calls"] = calls

    # 2) random search with the same oracle budgets as the RL runs (1-5 random substitutions)
    for budget in RL_BUDGETS:
        ks = rng.integers(1, 6, size=budget)
        T = np.stack([apply([(int(pos[i]), int(aa[i])) for i in rng.choice(len(D), k, replace=False)]) for k in ks])
        v = score(T)
        res["baselines"][f"random_search_{budget}"] = {"best_oracle_norm": float(v.max()), "mean_oracle_norm": float(v.mean())}

    # 3) greedy hill-climb with the oracle: add the single substitution that most improves the oracle score
    cur, muts, calls, path = ref.copy(), [], 0, []
    singles = [(int(p), int(c)) for p, c in zip(pos, aa)]
    for step in range(1, 7):
        T = np.stack([np.where(np.arange(L) == p, c, cur) for p, c in singles])
        v = score(T); calls += len(T)
        b = int(v.argmax())
        muts.append(singles[b]); cur = T[b]
        path.append({"step": step, "oracle_norm": float(v[b]), "muts": names(muts), "additive_label_norm": to_norm(additive(muts)),
                     "calls_so_far": calls})
        singles = [s for s in singles if s[0] != singles[b][0]]
    res["baselines"]["greedy_hill_climb"] = path

    json.dump(res, open(a.out, "w"), indent=2)
    print(f"reference (oracle est.) = {res['reference_oracle_norm']:.3f}   best single mutant (label) = {res['best_single_label_norm']:.3f}")
    print("\ntop-20 singles combined (exhaustive):")
    for r in res["baselines"]["top20_singles_combined"]:
        print(f"  k={r['k']}  best={r['best_oracle_norm']:.3f}  additive-from-labels={r['additive_label_norm']:.3f}  {r['muts']}")
    for b in RL_BUDGETS:
        r = res["baselines"][f"random_search_{b}"]
        print(f"random search, {b:>6} calls: best={r['best_oracle_norm']:.3f}  mean={r['mean_oracle_norm']:.3f}")
    print("greedy hill-climb:")
    for r in path:
        print(f"  step {r['step']}  oracle={r['oracle_norm']:.3f}  additive={r['additive_label_norm']:.3f}  calls={r['calls_so_far']}  {r['muts']}")


if __name__ == "__main__":
    main()
