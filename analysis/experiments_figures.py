#!/usr/bin/env python3
"""Draw the experiment figures from results/experiments/*.csv.

    python3 analysis/experiments_figures.py [--data results/experiments] [--out figures/experiments]
"""
import argparse, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

SURFACE, INK, INK2, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
PPO, SAC, BAR = "#2a78d6", "#eb6834", "#2a78d6"

# Paper values: Lee et al., ICML 2024, Table 2, LatProtRL on AAV medium (oracle setting), mean of 5 runs
PAPER = {"fitness": 0.71, "diversity": 5.4, "novelty": 6.1, "high": 2.4}

ABLATION = [("ablation_none.log", "No constraints"), ("ablation_entropy.log", "Entropy only"),
            ("ablation_size.log", "Size only"), ("ablation_blosum.log", "BLOSUM only"), ("ablation_all.log", "All three")]
CURVES = ["ppo_baseline_106", "ppo_confirmation_107", "ppo_dense_113", "ppo_unconstrained_111",
          "sac_unconstrained_211", "sac_high_delta_210", "sac_high_entropy_212", "sac_dense_215", "sac_cold_start_213_v2"]


def style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(AXIS)
    ax.tick_params(colors=INK2, labelsize=9, length=3)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="results/experiments")
    ap.add_argument("--out", default="figures/experiments")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    ev = pd.read_csv(os.path.join(a.data, "log_evaluations.csv"))
    aav = pd.read_csv(os.path.join(a.data, "aav_evaluations.csv"))

    # 1. ablation: final global best per setting
    fig, ax = plt.subplots(figsize=(7, 3.2), facecolor=SURFACE)
    vals = [ev[ev.log == f].sort_values("round").global_max.iloc[-1] for f, _ in ABLATION]
    labels = [l for _, l in ABLATION]
    y = range(len(vals))[::-1]
    ax.barh(list(y), vals, height=0.55, color=BAR)
    for yi, v in zip(y, vals):
        ax.text(v + 0.01, yi, f"{v:.3f}", va="center", fontsize=9, color=INK)
    ax.set_yticks(list(y)); ax.set_yticklabels(labels, color=INK, fontsize=10)
    ax.set_xlim(0, 1); ax.set_xlabel("best fitness after 4 rounds (normalized, oracle prediction)", color=INK2, fontsize=9)
    style(ax); ax.grid(axis="y", visible=False); ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_title("Constraint ablation, one PPO run per setting", loc="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "ablation_final.png"), dpi=170, facecolor=SURFACE); plt.close(fig)

    # 2. PPO and SAC curves
    fig, ax = plt.subplots(figsize=(7, 3.8), facecolor=SURFACE)
    for name in CURVES:
        d = ev[ev.log == name + ".log"].sort_values("round")
        ax.plot(d["round"], d.global_max, color=PPO if name.startswith("ppo") else SAC, linewidth=2, solid_capstyle="round")
        ax.plot(d["round"].iloc[-1], d.global_max.iloc[-1], "o", color=PPO if name.startswith("ppo") else SAC, markersize=5,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
    ax.plot([], [], color=PPO, linewidth=2, label="PPO"); ax.plot([], [], color=SAC, linewidth=2, label="SAC")
    ax.set_xlabel("evaluation (after round k)", color=INK2, fontsize=9); ax.set_ylabel("best fitness so far", color=INK2, fontsize=9)
    ax.set_xticks(range(1, 11)); ax.set_ylim(0.7, 0.9)
    style(ax); ax.legend(frameon=False, fontsize=9, labelcolor=INK2, loc="lower right")
    ax.set_title("Best fitness so far, by evaluation (9 runs)", loc="left", color=INK, fontsize=11)
    fig.tight_layout(); fig.savefig(os.path.join(a.out, "ppo_sac_curves.png"), dpi=170, facecolor=SURFACE); plt.close(fig)

    # 3. AAV run against the paper's reported means
    run = aav[aav.run == "AAV_medium_0_13_55_57"].sort_values("round")
    fig, axs = plt.subplots(2, 2, figsize=(8.2, 5), facecolor=SURFACE)
    panels = [("fitness", "median fitness"), ("diversity", "diversity"), ("novelty", "d_init (novelty)"), ("high", "d_high")]
    for ax, (col, title) in zip(axs.flat, panels):
        ax.plot(run["round"], run[col], color=PPO, linewidth=2, solid_capstyle="round")
        ax.plot(run["round"].iloc[-1], run[col].iloc[-1], "o", color=PPO, markersize=5, markeredgecolor=SURFACE, markeredgewidth=1.5)
        ax.axhline(PAPER[col], color=INK2, linewidth=1)
        ax.text(15.3, PAPER[col], f"paper {PAPER[col]}", fontsize=8, color=INK2, va="center", ha="left", clip_on=False)
        lo, hi = min(run[col].min(), PAPER[col]), max(run[col].max(), PAPER[col])
        pad = (hi - lo) * 0.18
        ax.set_ylim(lo - pad, hi + pad); ax.set_xticks(range(0, 16, 3))
        ax.set_title(title, loc="left", color=INK, fontsize=10); ax.set_xlabel("round", color=INK2, fontsize=8)
        style(ax)
    fig.suptitle("AAV medium: the repository's 15-round run against the paper's reported mean", x=0.02, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0, 0.98, 0.95)); fig.savefig(os.path.join(a.out, "aav_run.png"), dpi=170, facecolor=SURFACE); plt.close(fig)
    print("wrote", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
