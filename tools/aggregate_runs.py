#!/usr/bin/env python3
"""Scan LatProtRL/DHFR result folders and build one master table.

Read-only on the source tree. Safe to re-run while rsync is still pushing files.

    python3 tools/aggregate_runs.py --src ../basework/LatProtRL --out ../organized/tables
"""
import argparse, csv, glob, json, os, re
from datetime import datetime

WT = 0.7381          # wild-type, normalized (PROJECT_SUMMARY.md)
START = 0.3124       # starting seed, normalized


def classify(name):
    """Map a run-folder name to (algorithm, experiment_group)."""
    n = name
    if n.startswith("SAC_") or "_sac_" in n:
        algo = "SAC"
    elif "_ppo_" in n or re.search(r"_(10\d|11\d)_", n):
        algo = "PPO"
    else:
        algo = "PPO"  # default LatProtRL policy
    if "ablation" in n:
        grp = "ablation"
    elif "unconstrained" in n:
        grp = "unconstrained"
    elif "dense" in n:
        grp = "dense-reward"
    elif "cold_start" in n:
        grp = "cold-start"
    elif "hard" in n:
        grp = "hard-task"
    elif "blosum" in n:
        grp = "blosum"
    elif "fixed_config" in n:
        grp = "fixed-config-debug"
    elif "final" in n or "production" in n or "baseline" in n or "confirmation" in n:
        grp = "main"
    else:
        grp = "other/debug"
    if n.startswith("AAV_") or n.startswith("GFP_"):
        grp = "original-protein"
    return algo, grp


def read_run(path):
    name = os.path.basename(path)
    algo, grp = classify(name)
    row = dict(run=name, algorithm=algo, group=grp, protein=name.split("_")[0] if not name.startswith("SAC_") else name.split("_")[1],
               status="", rounds="", best_fitness="", initial_fitness="", improvement_pct="",
               beats_wt="", oracle_calls="", unique_seqs="", best_sequence="", source="")
    fj = os.path.join(path, "final_research_metrics.json")
    if os.path.exists(fj):
        try:
            d = json.load(open(fj))
        except Exception as e:  # file may be mid-transfer
            row["status"] = f"unreadable json ({e.__class__.__name__})"
            return row
        best = d.get("global_max_fitness_normalized", d.get("global_max_fitness"))
        row.update(status="complete", rounds=d.get("total_rounds", ""), best_fitness=best,
                   initial_fitness=d.get("initial_max_fitness", ""), improvement_pct=d.get("percentage_improvement", ""),
                   oracle_calls=d.get("total_oracle_calls", ""), unique_seqs=d.get("total_unique_sequences", ""),
                   best_sequence=d.get("global_best_sequence", ""), source="final_research_metrics.json")
        # Pool-best (buffer) is reported separately in the docs; keep it visible
        if d.get("buffer_best_in_pool") is not None:
            row["source"] += f"; buffer_best_in_pool={d['buffer_best_in_pool']:.4f}"
    else:
        txts = glob.glob(os.path.join(path, "*.txt"))
        fits = []
        for t in txts:
            try:
                fits.append(json.load(open(t)).get("eval/fitness"))
            except Exception:
                pass
        fits = [f for f in fits if f is not None]
        if fits:
            row.update(status="partial (round logs only)", rounds=len(fits), best_fitness=max(fits), source="N.txt eval/fitness")
        else:
            row["status"] = "empty/failed"
    try:
        if row["best_fitness"] != "":
            row["beats_wt"] = "yes" if float(row["best_fitness"]) > WT else "no"
    except ValueError:
        pass
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--results", help="results dir (default: <src>/results)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    runs = sorted(p for p in glob.glob(os.path.join(a.results or os.path.join(a.src, "results"), "*")) if os.path.isdir(p))
    rows = [read_run(p) for p in runs]
    cols = list(rows[0].keys())
    with open(os.path.join(a.out, "master_runs.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, cols)
        w.writeheader()
        w.writerows(rows)

    complete = [r for r in rows if r["status"] == "complete" and isinstance(r["best_fitness"], (int, float))]
    # Bucket by (protein, status) for the header summary
    from collections import Counter
    st = Counter((r["protein"], r["status"].split(" ")[0]) for r in rows)
    lines = [f"# Run inventory (auto-generated {datetime.now():%Y-%m-%d %H:%M})", "",
             f"Source: `{os.path.basename(os.path.abspath(a.src))}/results`  \nTotal run folders: **{len(rows)}**", "",
             "| protein | status | count |", "|--|--|--|"]
    lines += [f"| {p} | {s} | {c} |" for (p, s), c in sorted(st.items())]
    lines += ["", f"Reference points (normalized): start seed = {START}, wild-type = {WT}.", "",
              "## Completed DHFR runs, best fitness first", "",
              "| run | algo | group | rounds | best | +% vs start | > WT | oracle calls |", "|--|--|--|--|--|--|--|--|"]
    for r in sorted([r for r in complete if r["protein"] == "DHFR"], key=lambda r: -r["best_fitness"]):
        imp = r["improvement_pct"]
        imp = f"{imp:.0f}" if isinstance(imp, (int, float)) else imp
        lines.append(f"| {r['run']} | {r['algorithm']} | {r['group']} | {r['rounds']} | {r['best_fitness']:.4f} | {imp} | {r['beats_wt']} | {r['oracle_calls']} |")
    open(os.path.join(a.out, "run_inventory.md"), "w").write("\n".join(lines) + "\n")
    print(f"{len(rows)} runs, {len(complete)} complete -> {a.out}")


if __name__ == "__main__":
    main()
