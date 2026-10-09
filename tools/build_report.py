#!/usr/bin/env python3
"""Fill report_template.html from the repo's result files.  python3 tools/build_report.py <repo> <out.html>"""
import collections, json, sys
repo, out = sys.argv[1], sys.argv[2]
L = json.load(open(f"{repo}/results/landscape_summary.json")); B = json.load(open(f"{repo}/results/baseline_no_rl.json"))
names = {
    "DHFR_medium_111": ("PPO · unconstrained (run 111)", "10 rounds"), "DHFR_medium_106": ("PPO · baseline (run 106)", "4 rounds"),
    "DHFR_medium_113": ("PPO · dense reward (run 113)", "10 rounds"), "SAC_DHFR_medium_213": ("SAC · cold start v2 (run 213)", "4 rounds"),
    "DHFR_medium_107": ("PPO · confirmation of 106 (run 107)", "4 rounds"), "DHFR_medium_0_fixed_config_07": ("PPO · fixed-config debug run", "3 rounds"),
    "SAC_DHFR_medium_211": ("SAC · unconstrained (run 211)", "4 rounds"), "SAC_DHFR_medium_203": ("SAC · 4-round check (run 203)", "4 rounds"),
    "DHFR_medium_102": ("PPO · all constraints (run 102)", "4 rounds"), "SAC_DHFR_medium_205": ("SAC · production (run 205)", "4 rounds"),
    "SAC_DHFR_medium_215": ("SAC · dense reward (run 215)", "4 rounds"), "SAC_DHFR_medium_212": ("SAC · high entropy (run 212)", "4 rounds"),
    "SAC_DHFR_medium_210": ("SAC · high delta (run 210)", "4 rounds"),
}
rl = []
for r in L["rl_runs"]:
    key = next(k for k in sorted(names, key=len, reverse=True) if r["run"].startswith(k))
    j = json.load(open(f"{repo}/results/runs/{r['run']}/final_research_metrics.json"))
    rl.append({"label": names[key][0], "note": names[key][1], "algo": "SAC" if r["run"].startswith("SAC") else "PPO",
               "value": r["best"], "calls": f"{j['total_oracle_calls']:,}"})
rl.sort(key=lambda r: -r["value"])
g = B["baselines"]["greedy_hill_climb"]
data = {
    "refs": [{"label": "Best measured single mutant (D1M)", "value": B["best_single_label_norm"], "note": "highest label in the data"},
             {"label": "D1A mutant, called “wild-type” in the notes", "value": 0.7381, "note": "first row of the data file"},
             {"label": "Reference sequence, model estimate", "value": B["reference_oracle_norm"], "note": "no measurement exists"}],
    "baselines": [{"label": "Greedy, 2 steps", "value": g[1]["oracle_norm"], "clipped": True, "calls": f"{g[1]['calls_so_far']:,}", "note": "add the best substitution, twice"},
                  {"label": "Oracle scan, 1 step", "value": g[0]["oracle_norm"], "calls": f"{g[0]['calls_so_far']:,}", "note": "score every single mutant, take the best (D1M)"}],
    "rl": rl,
    "labels": [round(v, 3) for row in L["matrix_norm"] for v in row if v is not None],
    "marks": [{"v": B["reference_oracle_norm"], "color": "var(--ref)", "label": "reference sequence, model estimate"},
              {"v": 0.7381, "color": "var(--ref)", "label": "D1A mutant, the notes’ “wild-type”"},
              {"v": max(r["value"] for r in rl), "color": "var(--ppo)", "label": "best RL run (111, 38,400 calls)"},
              {"v": g[0]["oracle_norm"], "color": "var(--base)", "label": "oracle scan, one step (4,161 calls)"}],
}
cnt = collections.Counter(m for r in L["rl_runs"] for m in r["muts"]); pct, lab = {}, {}
for r in L["rl_runs"]:
    for m, p, v in zip(r["muts"], r["single_pctile"], r["single_labels"]): pct[m], lab[m] = p, v
data["muts"] = [{"mut": m, "count": c, "label": lab[m], "pct": pct[m]} for m, c in sorted(cnt.items(), key=lambda kv: (-kv[1], kv[0]))]
t = open(__file__.replace("build_report.py", "report_template.html")).read()
open(out, "w").write(t.replace("/*DATA*/null", json.dumps(data, separators=(",", ":"))))
print("wrote", out, len(t) // 1024, "KB template;", len(data["labels"]), "labels;", len(rl), "runs")
