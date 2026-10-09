#!/usr/bin/env python3
"""Extract per-evaluation results from run logs and AAV evaluation files into CSV.

    python3 analysis/extract_experiments_data.py --src <LatProtRL cluster copy> --out results/experiments

Reads, read-only:
  <src>/*.log                         evaluation blocks printed after each round ("EVALUATION METRICS - Round N")
  <src>/results/AAV_*/N.txt           per-round evaluator output of AAV runs (JSON lines)
Writes:
  log_evaluations.csv   one row per evaluation block
  log_summary.csv       one row per log (blocks found, last new_best printed, md5 of the log)
  aav_evaluations.csv   one row per AAV evaluation file
"""
import argparse, csv, glob, hashlib, json, os, re

FIELDS = {
    "avg_fitness_top128": r"Average Fitness \(top-128\):\s*(-?[0-9.]+)",
    "diversity": r"Diversity:\s*(-?[0-9.]+)",
    "novelty": r"Novelty:\s*(-?[0-9.]+)",
    "highest_in_batch": r"Highest in batch:\s*(-?[0-9.]+)",
    "global_max": r"Global Maximum Fitness \(normalized\):\s*(-?[0-9.]+)",
    "initial_max": r"Initial Maximum Fitness:\s*(-?[0-9.]+)",
    "pct_improvement": r"Percentage Improvement:\s*(-?[0-9.]+)%",
    "unique_sequences": r"Total Unique Sequences Discovered:\s*([0-9]+)",
    "oracle_calls": r"Total Oracle Calls:\s*([0-9,]+)",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    evals, summary = [], []
    for path in sorted(glob.glob(os.path.join(a.src, "*.log"))):
        name = os.path.basename(path)
        text = open(path, errors="ignore").read()
        parts = re.split(r"EVALUATION METRICS - Round (\d+)", text)
        n = 0
        for i in range(1, len(parts), 2):
            block = parts[i + 1][:4000]
            row = {"log": name, "round": int(parts[i])}
            for k, pat in FIELDS.items():
                m = re.search(pat, block)
                row[k] = m.group(1).replace(",", "") if m else ""
            if row["global_max"] != "":
                evals.append(row)
                n += 1
        nb = re.findall(r"new_best=([0-9.]+)", text)
        summary.append({"log": name, "evaluation_blocks": n, "last_new_best_printed": nb[-1] if nb else "",
                        "bytes": len(text.encode()), "md5": hashlib.md5(open(path, "rb").read()).hexdigest()})

    cols = ["log", "round"] + list(FIELDS)
    with open(os.path.join(a.out, "log_evaluations.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols); w.writeheader(); w.writerows(evals)
    with open(os.path.join(a.out, "log_summary.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, list(summary[0])); w.writeheader(); w.writerows(summary)

    aav = []
    for d in sorted(glob.glob(os.path.join(a.src, "results", "AAV_*"))):
        files = glob.glob(os.path.join(d, "*.txt"))
        for p in sorted(files, key=lambda p: int(os.path.basename(p)[:-4])):
            j = json.load(open(p))
            aav.append({"run": os.path.basename(d), "file": os.path.basename(p), "round": int(j["eval/round"]),
                        "fitness": j["eval/fitness"], "diversity": j["eval/diversity"],
                        "novelty": j["eval/novelty"], "high": j["eval/high"]})
        if not files:
            aav.append({"run": os.path.basename(d), "file": "", "round": "", "fitness": "", "diversity": "", "novelty": "", "high": ""})
    with open(os.path.join(a.out, "aav_evaluations.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, ["run", "file", "round", "fitness", "diversity", "novelty", "high"]); w.writeheader(); w.writerows(aav)
    print(f"{len(evals)} evaluation blocks from {len(summary)} logs; {len(aav)} AAV rows -> {a.out}")


if __name__ == "__main__":
    main()
