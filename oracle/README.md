# Oracle training code

Copied from the separate `dhfr_Oracle` project. This is the code that produced `ckpt/DHFR/oracle.ckpt` (the checkpoint is byte-identical to that project's `model/dhfr_oracle.pt`). The only edit is the data path, which pointed to a cluster home directory and now reads `data/DHFR/medium.csv` (same 4,161 rows, same order as that project's `data/dhfr.csv`).

| File | Notes |
|--|--|
| `train_dhfr_oracle.py` | Trains the 1-D CNN. Random 85/15 train/test split (`random_state=42`), then 15% of the training part as validation. Adam, MSE, up to 300 epochs. |
| `evaluate_dhfr_oracle.py` | Evaluation script for a different model class (`DHFRBaseCNN`). |
| `dhfr_basecnn.py` | The alternative model class used by the script above. |
| `oracle_validation_results.json` | The 0.935 / 0.874 figures. `sample_size` is 4161: all variants, not the test set. |

## What the real held-out score is
Reproduced by `analysis/oracle_true_holdout.py`:

| Set | n | Pearson | R² | RMSE (raw) |
|--|--|--|--|--|
| Train | 3,005 | 0.996 | 0.991 | 0.14 |
| Validation | 531 | 0.791 | 0.615 | 0.92 |
| **Test (held out)** | **625** | **0.742** | **0.525** | **1.01** |
| All 4,161 (the reported figure) | 4,161 | 0.935 | 0.874 | 0.52 |

The reported 0.935 is mostly the fit to training data. On unseen variants the oracle is considerably less accurate: an RMSE of 1.0 raw units is about 0.10 on the 0–1 scale used in the optimization.

## Problems in the code as written
1. **No early stopping takes effect.** `patience_counter` is incremented but nothing breaks out of the loop, so all 300 epochs run.
2. **"Restore the best model" restores nothing.** `best_state = model.state_dict()` returns references to the live weights, so it changes as training continues; `load_state_dict(best_state)` then loads the final-epoch weights into themselves. (Checked in PyTorch: the saved dict changes after an optimizer step.) The saved model is therefore the last epoch, which fits the near-perfect training score above.
3. **`evaluate_dhfr_oracle.py` tests on training data.** It builds its "test" set from a different shuffle than the one used in training; 87% of those rows were in the training set, which is why it reports Pearson 0.971.
4. **The test split was never reported.** No script or result file in the project records metrics on the 625 held-out rows.
5. **The "relative accuracy" metric is wrong.** In `test_oracle_validation.py` (in the original `dhfr_Oracle` project) the formula is `np.abs(actual - pred) / actual < 0.1`, with no absolute value on the denominator. Any variant with a negative target gives a negative ratio and counts as correct. That is 48.3% of the data; the reported 73.66% reproduces exactly under this formula and drops to 45.3% with `|actual|`.
