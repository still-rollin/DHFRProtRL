# Earlier findings (from the "New Findings" folder)

Early-stage outputs and the first write-up. Kept for the record; read ../findings.md for the checked results.

| File | What it is | Checked |
|--|--|--|
| `Findings.docx` (not included; the author keeps the original) | First write-up: dataset list, oracle architecture and metrics, reward function, one PPO run's tables and top sequences | Metrics (Pearson 0.935, RMSE 0.522) match the shipped oracle checkpoint. See the notes below for what does not line up. |
| `early_run_mislabelled_AAV_medium_0.csv` | Called "baseline results on the AAV-medium dataset" in the write-up. It actually holds **DHFR** sequences (219 residues, all in the DHFR data): 6 rounds × 128 rows | Per-round count, mean (0.576) and max (1.075) are identical in every round, which suggests the optimizer produced no change between rounds. A max above 1 suggests this run used an unnormalized oracle. |
| `DHFROnLatProtRL.csv`, `final_DHFR_scores.csv` | The same 128 sequences from that early run. `base_score` is each sequence's true normalized label (matches the dataset exactly, 128 of 128); `predicted_score` is what the early pipeline assigned | Spearman between the two is 0.03: the early pipeline's scores were unrelated to the true fitness (e.g. D1M, true label 1.0, scored 0.517). This is the "zero-shot port fails" stage, caused at least in part by the wrongly loaded oracle. |
| `dhfr_predicted.csv` | Per-variant oracle predictions for all 4,161 variants | Pearson with the labels is 0.844 (RMSE 0.79), not 0.935. It comes from an earlier model than the shipped checkpoint (agreement with the checkpoint's output: r = 0.885). |
| `residuals_of_dhfr_predicted.png` | Residual histogram from the write-up | Its range (−3.8 to 4.9) matches `dhfr_predicted.csv`, i.e. the older model, not the shipped checkpoint (−4.7 to 5.2). |

## Things in `Findings.docx` that should be corrected before it is shared
- It describes the oracle metrics as measured on "a held-out test set of 4,161 DHFR variants". 4,161 is the entire dataset, of which the oracle trained on 85%. On the real held-out test set (625 variants) the checkpoint scores Pearson 0.742, R² 0.525 (see `oracle/README.md`).
- "Relative accuracy 73.66%" reproduces, but only because of a metric bug: the formula `|error| / actual < 0.1` counts every variant with a negative target (48.3% of them) as correct. With `|actual|` in the denominator it is 45.3%.
- Charts and text come from different model versions (see the residual chart above).
- Mutation positions are given as "~57, ~104, ~150, ~190". In the dataset's numbering the recurring changes are K106, V116, V160 and Y196, and position 57 is the reference residue (the "I→N" reversion is relative to a start sequence that carries N57I).
- The round table mixes scales: "Avg Fitness (Top-128)" is about −4.4 while the global best is on a 0–1 scale; the diversity and novelty columns are not meaningful for DHFR.
- The reward section is accurate to the code: `1.0 + 100·Δ` for a new episode best, an entropy factor `1 + α·mean(entropy)`, and a −1 penalty with termination for too many mutations. The sentence "highly conserved positions are rewarded less for mutation" matches the code only in the weak sense that the factor is closer to 1 there.
