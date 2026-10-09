#!/usr/bin/env python3
"""Generate docs/experiments.md from results/experiments/*.csv, results/runs/*, results/baseline_no_rl.json.

    python3 tools/build_experiments_md.py            # run from the repo root
"""
import glob, json, os
import pandas as pd

D = "results/experiments"
ev = pd.read_csv(f"{D}/log_evaluations.csv")
summ = pd.read_csv(f"{D}/log_summary.csv")
aav = pd.read_csv(f"{D}/aav_evaluations.csv")
base = json.load(open("results/baseline_no_rl.json"))


def series(log):
    return ev[ev.log == log].sort_values("round")


def saved_best(prefix):
    for p in glob.glob(f"results/runs/{prefix}*/final_research_metrics.json"):
        j = json.load(open(p))
        return j.get("global_max_fitness_normalized", j.get("global_max_fitness"))
    return None


def fmt(v, n=3):
    return f"{v:.{n}f}"


out = []
w = out.append
w("# Experiments\n")
w("Per-evaluation results of the runs that have a log or result file, with the source of each number. "
  "Values come from `results/experiments/log_evaluations.csv` (extracted from the run logs by `analysis/extract_experiments_data.py`), "
  "`results/experiments/aav_evaluations.csv`, `results/runs/*/final_research_metrics.json` and `results/baseline_no_rl.json`. "
  "\"Evaluation k\" is the block printed after round k (\"EVALUATION METRICS - Round k\"). \"Global best\" is the printed "
  "\"Global Maximum Fitness (normalized)\", an oracle prediction on the 0 to 1 scale. Run settings are given only where a launch script "
  "or the file name states them.\n")

# 1 ablation
w("## 1. Constraint ablation\n")
w("Launched by `scripts/run_ablations.sh` (PPO, `--protein DHFR --level medium --use_oracle`, run indices 10 to 14), one run per setting, 4 rounds.\n")
w("| Setting | Flags | Log | Global best at evaluations 1, 2, 3, 4 | Unique sequences | Oracle calls |")
w("|--|--|--|--|--|--|")
abl = [("No constraints", "`--no_entropy --no_size --no_blosum`", "ablation_none.log"),
       ("Entropy only", "`--no_size --no_blosum`", "ablation_entropy.log"),
       ("Size only", "`--no_entropy --no_blosum`", "ablation_size.log"),
       ("BLOSUM only", "`--no_entropy --no_size`", "ablation_blosum.log"),
       ("All three", "none", "ablation_all.log")]
finals = []
for name, flags, log in abl:
    s = series(log)
    finals.append(s.global_max.iloc[-1])
    w(f"| {name} | {flags} | `{log}` | {' / '.join(fmt(v) for v in s.global_max)} | {int(s.unique_sequences.iloc[-1])} | {int(s.oracle_calls.iloc[-1]):,} |")
w(f"\nFinal global best across the five settings: minimum {fmt(min(finals))}, maximum {fmt(max(finals))}.\n")
w("![constraint ablation](../figures/experiments/ablation_final.png)\n")
w("Other logs with constraint settings given by their file names:\n")
w("| Log | Evaluations | Global best at each evaluation | Oracle calls |")
w("|--|--|--|--|")
for log in ["final_run.log", "dhfr_blosum_run.log"]:
    s = series(log)
    w(f"| `{log}` | {len(s)} | {' / '.join(fmt(v) for v in s.global_max)} | {int(s.oracle_calls.iloc[-1]):,} |")
w("")

# 2 PPO / SAC
w("## 2. PPO and SAC runs\n")
w("| Log | Algorithm | Evaluations | Global best at each evaluation | Oracle calls | Saved metrics (`results/runs`) |")
w("|--|--|--|--|--|--|")
rows = [("ppo_baseline_106", "PPO", "DHFR_medium_106_"), ("ppo_confirmation_107", "PPO", "DHFR_medium_107_"),
        ("ppo_production_108", "PPO", None), ("ppo_dense_113", "PPO", "DHFR_medium_113_"),
        ("ppo_unconstrained_111", "PPO", "DHFR_medium_111_"), ("sac_unconstrained_211", "SAC", "SAC_DHFR_medium_211_"),
        ("sac_high_delta_210", "SAC", "SAC_DHFR_medium_210_"), ("sac_high_entropy_212", "SAC", "SAC_DHFR_medium_212_"),
        ("sac_dense_215", "SAC", "SAC_DHFR_medium_215_"), ("sac_cold_start_213_v2", "SAC", "SAC_DHFR_medium_213_sac_cold_start_v2")]
for log, algo, pref in rows:
    s = series(log + ".log")
    sv = saved_best(pref) if pref else None
    if sv is not None:
        assert abs(sv - s.global_max.iloc[-1]) < 1e-4, (log, sv, s.global_max.iloc[-1])
    ser = " / ".join(fmt(v) for v in s.global_max)
    w(f"| `{log}.log` | {algo} | {len(s)} | {ser} | {int(s.oracle_calls.iloc[-1]):,} | {fmt(sv, 4) if sv is not None else 'none'} |")
s = series("dhfr_training.log")
w(f"| `dhfr_training.log` | not stated | {len(s)} | {' / '.join(fmt(v) for v in s.global_max)} | {int(s.oracle_calls.iloc[-1]):,} | none |")
w("\nIn all nine rows with saved metrics, the final value in the log equals the saved global best to within 0.0001.\n")
w("![best fitness by evaluation](../figures/experiments/ppo_sac_curves.png)\n")

# 3 no evaluations
w("## 3. Logs with no completed evaluation\n")
w("| Log | Evaluation blocks | Last `new_best` printed |")
w("|--|--|--|")
for log in ["ppo_cold_start_109.log", "ppo_cold_start_109_v2.log", "ppo_unconstrained_110.log", "ppo_hard_112.log",
            "sac_cold_start_213.log", "sac_hard_214.log"]:
    r = summ[summ.log == log].iloc[0]
    assert r.evaluation_blocks == 0
    nb = r.last_new_best_printed
    w(f"| `{log}` | 0 | {fmt(float(nb), 6) if pd.notna(nb) else 'none'} |")
w("")

# 4 early SAC
w("## 4. Early SAC development logs\n")
w("| Log | Evaluations | Global best at every evaluation | Last `Total Oracle Calls` printed |")
w("|--|--|--|--|")
for log in ["sac_run.log", "sac_fresh.log", "sac_final_v2.log", "sac_isolated.log"]:
    s = series(log)
    vals = sorted(set(round(v, 6) for v in s.global_max))
    assert vals == [0.311339], (log, vals)
    w(f"| `{log}` | {len(s)} | 0.311339 (equal to the initial maximum) | {int(s.oracle_calls.iloc[-1])} |")
w("")

# 5 AAV
w("## 5. AAV (the original paper's protein)\n")
run = aav[aav.run == "AAV_medium_0_13_55_57"].sort_values("round")
w("Run `AAV_medium_0_13_55_57`: 16 evaluation files (rounds 0 to 15). Evaluator definitions (`metric.py`): fitness = median normalized oracle fitness of the 128 evaluated sequences; diversity = median pairwise distance; novelty = median distance to the nearest initial-dataset sequence (d_init); high = median distance to the nearest of the top-10% sequences (d_high). The flags of this run are not recorded.\n")
w("| Round | Fitness | Diversity | Novelty (d_init) | d_high |")
w("|--|--|--|--|--|")
for _, r in run.iterrows():
    w(f"| {int(r['round'])} | {fmt(r.fitness)} | {int(r.diversity)} | {int(r.novelty)} | {int(r.high)} |")
last = run.iloc[-1]
w("\nCompared with the paper (Lee et al., ICML 2024, arXiv:2405.18986, Table 2, LatProtRL on AAV medium, oracle setting; mean of 5 runs, standard deviation in parentheses; 15 rounds, 256 oracle calls per round):\n")
w("| | Fitness | Diversity | d_init | d_high |")
w("|--|--|--|--|--|")
w("| Paper | 0.71 (0.0) | 5.4 (0.9) | 6.1 (0.2) | 2.4 (0.5) |")
w(f"| This run, round 15 | {fmt(last.fitness, 2)} | {int(last.diversity)} | {int(last.novelty)} | {int(last.high)} |")
w("\n![AAV run against the paper](../figures/experiments/aav_run.png)\n")
others = aav.groupby("run").agg(n=("file", lambda x: int((x.fillna("") != "").sum())), fit_min=("fitness", "min"), fit_max=("fitness", "max"),
                                 divs=("diversity", lambda x: sorted(set(x.dropna()))), novs=("novelty", lambda x: sorted(set(x.dropna()))),
                                 highs=("high", lambda x: sorted(set(x.dropna()))))
w("All AAV result folders:\n")
w("| Run | Evaluation files | Fitness (min to max) | Diversity values | Novelty values | d_high values |")
w("|--|--|--|--|--|--|")
for name, r in others.iterrows():
    if r.n == 0:
        w(f"| `{name}` | 0 | | | | |")
    else:
        w(f"| `{name}` | {r.n} | {fmt(r.fit_min)} to {fmt(r.fit_max)} | {', '.join(str(int(x)) for x in r.divs)} | {', '.join(str(int(x)) for x in r.novs)} | {', '.join(str(int(x)) for x in r.highs)} |")
w("")

# 6 baselines
w("## 6. Search without RL, same oracle (`analysis/baseline_no_rl.py`)\n")
g = base["baselines"]["greedy_hill_climb"]
c = base["baselines"]["top20_singles_combined"]
w("| Method | Best oracle score | Oracle calls |")
w("|--|--|--|")
w(f"| Score every single mutant, take the best (D1M) | {fmt(g[0]['oracle_norm'])} | {g[0]['calls_so_far']:,} |")
w(f"| Greedy hill-climb, 2 steps | {fmt(g[1]['oracle_norm'])} (at the normalization ceiling) | {g[1]['calls_so_far']:,} |")
w(f"| All pairs among the 20 best-labelled single mutants | {fmt(c[1]['best_oracle_norm'])} (at the ceiling) | {c[1]['combos']} |")
for b in (15360, 38400):
    r = base["baselines"][f"random_search_{b}"]
    w(f"| Random combinations of 1 to 5 single mutants, {b:,} draws | {fmt(r['best_oracle_norm'])} (at the ceiling); mean {fmt(r['mean_oracle_norm'])} | {b:,} |")
w(f"\nFor comparison, the best RL run in `results/runs` scores 0.8657 with 38,400 oracle calls. The baselines start from the reference sequence; the RL runs start from a pool of 128 sequences around the 10th percentile of fitness (normalized 0.28 to 0.31).\n")

# coverage
w("## 7. Coverage\n")
nlogs, nblocks = summ.shape[0], ev.shape[0]
w(f"- {nlogs} run logs read, containing {nblocks} evaluation blocks (`results/experiments/log_summary.csv` lists each log with its MD5).")
w("- 19 DHFR runs have saved final metrics (`results/runs/`); 265 result folders exist on the cluster, the rest partial or empty.")
w("- The 222 offline Weights & Biases runs are not parsed here.")
open("docs/experiments.md", "w").write("\n".join(out) + "\n")
print("wrote docs/experiments.md", len("\n".join(out)), "chars")
