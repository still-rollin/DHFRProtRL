# Figures

Produced by the scripts in `analysis/` and the earlier oracle validation. Read them with these limits in mind:

- `dhfr_oracle_*.png` and `medium_ckpt_scatter.png` plot the oracle against all 4,161 variants. About 72% of those were training data, so they show fit, not generalization (held-out Pearson 0.742; see `oracle/README.md`).
- `ablation_*.png`, `gravy_*.png` and `instability_*.png` come from single runs per setting and use sequence-level proxies (GRAVY, instability index). They do not measure stability or activity.
- `dhfr_medium_*.png` summarize one earlier checkpoint-testing session; the model they used is not recorded.
- `calibration_debug.png` is from the oracle calibration debugging.
- `experiments/` holds the three figures used in `docs/experiments.md`, drawn by `analysis/experiments_figures.py` from `results/experiments/*.csv`.
