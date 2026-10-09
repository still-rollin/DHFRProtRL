# Oracle training code

Copied from the separate `dhfr_Oracle` project. This is the code that produced `ckpt/DHFR/oracle.ckpt` (byte-identical to that project's `model/dhfr_oracle.pt`). The data path was changed to `data/DHFR/medium.csv`, which has the same 4,161 rows in the same order as that project's `data/dhfr.csv`.

| File | Notes |
|--|--|
| `train_dhfr_oracle.py` | Trains the 1-D CNN. Random 85/15 train/test split (`random_state=42`), then 15% of the training part as validation. Adam, MSE, 300 epochs. |
| `evaluate_dhfr_oracle.py` | Evaluation script for the `DHFRBaseCNN` model class. |
| `dhfr_basecnn.py` | The alternative model class used by the script above. |
| `oracle_validation_results.json` | Validation figures computed over all 4,161 variants (`sample_size: 4161`). |

## Accuracy by split
Reproduced by `analysis/oracle_true_holdout.py`:

| Set | n | Pearson | R² | RMSE (raw) |
|--|--|--|--|--|
| Train | 3,005 | 0.996 | 0.991 | 0.14 |
| Validation | 531 | 0.791 | 0.615 | 0.92 |
| Test | 625 | 0.742 | 0.525 | 1.01 |
| All 4,161 | 4,161 | 0.935 | 0.874 | 0.52 |

An RMSE of 1.0 raw units is about 0.10 on the 0 to 1 scale used in the optimization.

## Training details
- The validation loss is computed each epoch and a patience counter is incremented, but training always runs the full 300 epochs.
- The saved model is the final-epoch model: `best_state = model.state_dict()` holds references to the live weights.
- `evaluate_dhfr_oracle.py` builds its test set with a different shuffle (`df.sample(frac=1.0, random_state=42)`); 87% of those 625 rows were in the training set, and it reports Pearson 0.971.
- "Relative accuracy" in `test_oracle_validation.py` is `np.abs(actual - pred) / actual < 0.1`. With the target in the denominator unsigned, the 48.3% of variants with a negative target count as within tolerance: 73.66%. With `|actual|` in the denominator it is 45.30%.
