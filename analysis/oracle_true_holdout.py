#!/usr/bin/env python3
"""Score the shipped DHFR oracle on its real train / validation / test split.

The oracle was trained by oracle/train_dhfr_oracle.py on data/DHFR/medium.csv (same 4,161 rows, same
order as the oracle's own data/dhfr.csv) with
    train_test_split(df, test_size=0.15, random_state=42)  then  train_test_split(train, test_size=0.15, random_state=42)
so the held-out test set is reproducible. Run from the repo root:
    python3 analysis/oracle_true_holdout.py [--out results/oracle_true_holdout.json]
Requires scikit-learn. The split follows scikit-learn's ShuffleSplit; checked with scikit-learn 1.8.0.
"""
import argparse, json, sys
import numpy as np, pandas as pd, torch
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split

sys.path.insert(0, ".")
from net.rew import DHFROracle  # noqa: E402

A = "ACDEFGHIKLMNPQRSTVWY"
IX = {a: i for i, a in enumerate(A)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/oracle_true_holdout.json")
    a = ap.parse_args()

    df = pd.read_csv("data/DHFR/medium.csv")
    train, test = train_test_split(df, test_size=0.15, random_state=42)
    train, val = train_test_split(train, test_size=0.15, random_state=42)

    m = DHFROracle(n_tokens=20, make_one_hot=True)
    m.load_state_dict(torch.load("ckpt/DHFR/oracle.ckpt", map_location="cpu", weights_only=False)["model_state_dict"])
    m.eval()

    def metrics(x, name):
        t = torch.tensor([[IX[c] for c in s] for s in x.sequence])
        with torch.no_grad():
            p = torch.cat([m(t[i:i + 512]) for i in range(0, len(t), 512)]).numpy()
        y = x.target.values
        r = p - y
        return {"set": name, "n": int(len(x)), "pearson": float(pearsonr(p, y)[0]), "spearman": float(spearmanr(p, y)[0]),
                "rmse_raw": float(np.sqrt((r ** 2).mean())), "mae_raw": float(np.abs(r).mean()),
                "r2": float(1 - (r ** 2).sum() / ((y - y.mean()) ** 2).sum())}

    res = [metrics(train, "train"), metrics(val, "validation"), metrics(test, "test (held out)"), metrics(df, "all 4,161 (as previously reported)")]

    # the test set used by oracle/evaluate_dhfr_oracle.py is a different shuffle; how much of it was training data?
    d2 = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    ev = d2[int(0.85 * len(d2)):]
    leak = float(ev.sequence.isin(set(train.sequence)).mean())
    res.append(metrics(ev, "evaluate_dhfr_oracle.py 'test'"))

    out = {"splits": res, "evaluate_script_test_fraction_in_training": leak, "raw_to_normalized_divisor": 9.652}
    json.dump(out, open(a.out, "w"), indent=2)
    for r in res:
        print(f"{r['set']:40s} n={r['n']:5d} pearson={r['pearson']:.3f} spearman={r['spearman']:.3f} r2={r['r2']:.3f} rmse={r['rmse_raw']:.3f}")
    print(f"fraction of the evaluate script's 'test' rows that were training data: {leak:.3f}")


if __name__ == "__main__":
    main()
