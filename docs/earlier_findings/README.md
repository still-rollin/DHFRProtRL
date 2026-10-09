# Earlier findings

Early-stage outputs delivered with the first write-up (`Findings.docx`, not included). The facts from that write-up are in [../findings.md](../findings.md).

| File | Contents |
|--|--|
| `early_run_mislabelled_AAV_medium_0.csv` | Named `AAV_medium_0.csv` in the original folder. It holds DHFR sequences (219 residues, all in the DHFR data): 6 rounds × 128 rows. Per-round count, mean (0.576) and max (1.075) are identical in every round. |
| `DHFROnLatProtRL.csv`, `final_DHFR_scores.csv` | The same 128 sequences. `base_score` is each sequence's normalized dataset label (matches the dataset for all 128); `predicted_score` is the early pipeline's score. Spearman correlation between the two: 0.03 (for example D1M, label 1.0, scored 0.517). |
| `dhfr_predicted.csv` | Per-variant predictions for all 4,161 variants. Pearson with the labels 0.844, RMSE 0.79; agreement with the shipped checkpoint's output r = 0.885. |
| `residuals_of_dhfr_predicted.png` | Residual histogram. Its range (−3.8 to 4.9) matches `dhfr_predicted.csv`; the shipped checkpoint's residual range is −4.7 to 5.2. |

## Notes on the write-up
- The oracle metrics (Pearson 0.935, RMSE 0.522) are computed over all 4,161 variants. On the 625-variant test split the checkpoint scores Pearson 0.742, R² 0.525 (see `../../oracle/README.md`).
- "Relative accuracy 73.66%" is `|error| / actual < 0.1`; it is 45.3% with `|actual|` in the denominator.
- Mutation positions in the write-up (57, 91, 96, 104, 150, 190) differ from the dataset numbering, where the recurring changes are K106, V116, V160 and Y196. The write-up's "I to N at 57" is a reversion relative to its initial sequence, which carries N57I.
- The round table prints "Avg Fitness (Top-128)" near −4.4 beside a global best on the 0 to 1 scale.
