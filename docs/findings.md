# Findings

Facts from the original findings document (`Findings.docx`) and the files delivered with it. Numbers were recomputed from the files in this repository unless the source column says "recorded", which marks a value printed in `Findings.docx` that was not recomputed. Fitness is on the 0 to 1 scale `(target + 5.215) / 9.652` unless marked raw.

## 1. Dataset

| Item | Value |
|--|--|
| File | `data/DHFR/medium.csv` (identical to `pfdhfr_fitness_data_single_mut.csv`) |
| Rows | 4,161 |
| Sequence length | 219 |
| Columns | `sequence`, `augmented` (0 in every row), `target` |
| Raw target range | −5.2148 to 4.4368 |
| Median target | 0.0648 raw, 0.547 normalized |
| Structure | 219 positions × 19 substitutions; each sequence differs at exactly one position from a common reference sequence; the reference is not in the file |
| Highest values | D1M 1.000 (raw 4.4368), D1Y 0.982, D1F 0.955, D1W 0.943 |
| First row | D1A, raw 1.9093, normalized 0.7381 |
| Variants at or above 0.738 / 0.866 / 0.9 / 0.95 | 150 / 11 / 6 / 3 |
| `easy.csv`, `medium.csv`, `hard.csv` | same sequences and targets |
| "Initial sequence" printed in `Findings.docx` | the reference with N57I (a dataset row; raw 0.3125, normalized 0.5727) |

## 2. Files delivered with the document

| File | Contents |
|--|--|
| `AAV_medium_0.csv` | Columns `round`, `sequence`, `target`. 768 rows: 6 rounds × the same 128 DHFR sequences. Target 0.2292 to 1.0745, mean 0.5764. Rows are identical in all six rounds. |
| `DHFROnLatProtRL.csv` | The 128 rows of round 1 of `AAV_medium_0.csv` (sequence and target identical). |
| `final_DHFR_scores.csv` | The same 128 sequences. `base_score` is the sequence's normalized dataset label (0.745 to 1.0; matches the dataset for all 128). `predicted_score` is 0.229 to 1.075. Spearman correlation between the two: 0.029. |
| `dhfr_predicted.csv` | 4,161 rows: `sequence`, `augmented`, `target`, `predicted`. Pearson between `predicted` and `target`: 0.844; RMSE 0.793 (raw units). |
| `chart (3).png` | Residual histogram (predicted minus target). Range −3.77 to 4.88, which equals the residual range of `dhfr_predicted.csv`. |

## 3. DHFR fitness oracle

### 3.1 Architecture
Input: one-hot sequence, 20 channels × 219. Layers: Conv1d 20→128 (kernel 3, padding 1), ReLU, Conv1d 128→256 (kernel 3, padding 1), ReLU, adaptive average pooling to 1, Linear 256→64, ReLU, Dropout 0.2, Linear 64→1. Amino-acid order `ACDEFGHIKLMNPQRSTVWY`.

### 3.2 Training (`oracle/train_dhfr_oracle.py`)
| Item | Value |
|--|--|
| Data | the 4,161 rows of the dataset |
| Split | `train_test_split(test_size=0.15, random_state=42)`, then 15% of the remainder for validation: train 3,005, validation 531, test 625 |
| Optimizer / loss | Adam, learning rate 1e-3, weight decay 1e-5, MSE, batch size 32 |
| Epochs | 300; validation loss is computed each epoch |
| Saved model | final-epoch weights (`best_state = model.state_dict()` references the live weights) |
| Checkpoint | `ckpt/DHFR/oracle.ckpt` equals `dhfr_Oracle/model/dhfr_oracle.pt` (same file) |

### 3.3 Accuracy of the checkpoint (`analysis/oracle_true_holdout.py`)
| Set | n | Pearson | Spearman | R² | RMSE (raw) | MAE (raw) |
|--|--|--|--|--|--|--|
| Train | 3,005 | 0.996 | 0.997 | 0.991 | 0.137 | 0.097 |
| Validation | 531 | 0.791 | 0.770 | 0.615 | 0.915 | 0.680 |
| Test | 625 | 0.742 | 0.760 | 0.525 | 1.008 | 0.722 |
| All 4,161 | 4,161 | 0.935 | 0.934 | 0.874 | 0.522 | 0.265 |

The values reported in `Findings.docx` (Pearson 0.935, Spearman 0.934, R² 0.874, RMSE 0.522, MAE 0.265) equal the "All 4,161" row.

### 3.4 Other reported oracle figures
| Figure | Value | Source |
|--|--|--|
| Predicted range / actual range (raw) | −5.544 to 4.361 / −5.215 to 4.437 | recomputed |
| Concordance index (all 4,161) | 0.915 | `oracle/oracle_validation_results.json`, recorded |
| Effect size (Cohen's d of predictions, top 10% vs bottom 10% by actual) | 5.59 | recomputed (5.589) |
| "Relative accuracy" (share of predictions with `|error| / actual < 0.1`) | 73.66% | recomputed: 73.66% with the script's formula, 45.30% with `|actual|` in the denominator (48.3% of variants have a negative target) |

## 4. Reward function (`net/envr.py`)
For a step that produces sequence `s` with normalized oracle fitness `f` and episode best `f_best`, with `Δ = f − f_best`:

| Condition | Reward |
|--|--|
| `Δ > 0` | `(1 + 100·Δ) × entropy_factor × size_factor × blosum_factor`; `f_best` is updated |
| `−0.001 < Δ ≤ 0` | `0.05 × entropy_factor × size_factor × blosum_factor` |
| `Δ ≤ −0.001` | `−0.5 × (1 + 50·|Δ|)` |
| Too many mutations in a step or in total | `−1`, episode ends |

| Factor | Definition |
|--|--|
| `entropy_factor` | `1 + w · mean(entropy value for each mutated residue)`, `w = 1.0` by default; 1 if no mutations. The conservation table is keyed by 1-based position; the code looks each value up with the residue's 0-based index. |
| `size_factor` | `max(0.1, 1 − Σ size penalties)`; penalty per substitution 0 (no cost), 0.1 (mild), 0.3 (moderate), 0.5 (severe), from `size_rules.yaml` |
| `blosum_factor` | `max(0.1, 1 + Σ adjustments)`; per-substitution adjustment from the position-specific score in `blosum_substitution_scores.csv`: score ≥ 1 → +0.05; ≥ 0 → 0; ≥ −1 → −0.1; ≥ −2 → −0.3; below → −0.5 (thresholds in `blosum_rules.yaml`) |

The entropy values are in `data/DHFR/conservation_scores.csv`: 219 positions, Shannon entropy 0.0 to 2.2018 (median 1.19). Zero-entropy positions: 32 (G), 38 (P), 39 (W), 95 (M), 104 (P). Each constraint can be disabled with `--no_entropy`, `--no_size`, `--no_blosum`.

## 5. Optimization runs

### 5.1 Setup
| Item | Value |
|--|--|
| Start pool | 128 sequences around the 10th percentile of fitness; normalized fitness 0.2798 to 0.3113 (recorded in the run logs) |
| Initial maximum fitness | 0.311339 |
| PPO | n_steps 4,096; `optimize.py` sets `total_timesteps=40,960` (10 rounds) |
| SAC | `train_freq` 4,096; `gradient_steps` 4,096; `ent_coef` auto; `optimize_sac.py` sets `total_timesteps=16,384` (4 rounds) |
| Oracle calls per round | 3,840 (15,360 for 4 rounds) |
| Hardware | NVIDIA RTX A6000 (recorded) |

### 5.2 Zero-shot port (recorded)
Values printed in `Findings.docx`: round 0 start 1.0; median fitness 0.6047; top fitness 0.6966; diversity 4.0; novelty 2.0; training time 5.3 s. The document states that initial baseline evaluations were below 0.40.

### 5.3 Per-round results of the 3-round run (start percentile 10)
Run `DHFR_medium_0_fixed_config_07_55_39`. Its saved metrics match round 3 of the document's table (global max 0.812024, 328 unique sequences, 11,520 oracle calls, initial 0.311339); rounds 1 and 2 are recorded.

| Round | Global max fitness | Improvement | Unique sequences | Oracle calls | Avg fitness top-128 (as printed) | Diversity | Novelty |
|--|--|--|--|--|--|--|--|
| 1 | 0.311339 | 0 | 128 | 3,840 | −4.480017 | 2 | 0 |
| 2 | 0.765216 | +0.453877 (145.78%) | 254 | 7,680 | −4.414618 | 3 | 1 |
| 3 | 0.812024 | +0.500685 (160.82%) | 328 | 11,520 | −4.372625 | 3 | 1 |

Percentage improvement is `(global max − 0.311339) / 0.311339`.

### 5.4 Completed multi-round runs (`results/runs/`)
Best fitness is the tracked global best from each run's `final_research_metrics.json`; re-scoring the best sequence with the oracle agrees to within 0.005 in all 13 runs. Reward scheme and guardrail settings are those listed in `Findings.docx` (Table 2) where given.

| Run | Algorithm | Rounds | Oracle calls | Best fitness | Mutations vs reference | Listed setup |
|--|--|--|--|--|--|--|
| 111 ppo_unconstrained_final | PPO | 10 | 38,400 | 0.8657 | K106N V116I T121S I134V V160I | sparse reward, guardrails off |
| 106 ppo_baseline_fixed | PPO | 4 | 15,360 | 0.8430 | K106N V116I V160I | sparse, guardrails on |
| 113 ppo_dense_reward | PPO | 10 | 38,400 | 0.8423 | K106N V160I Y196W | dense, guardrails on |
| 213 sac_cold_start_v2 | SAC | 4 | 15,360 | 0.8364 | V116I V160I Y196W | |
| 107 ppo_confirmation_verify | PPO | 4 | 15,360 | 0.8328 | V116I V160I Y196F | |
| 0_fixed_config_07_55_39 | PPO | 3 | 11,520 | 0.8120 | K106N V116I Y196W | |
| 211 sac_unconstrained | SAC | 4 | 15,360 | 0.8099 | K106N V116I Y196F | sparse, guardrails off |
| 203 sac_4round_check | SAC | 4 | 15,360 | 0.7738 | K106D V116I | |
| 102 final_full_model_run | PPO | 4 | 15,360 | 0.7703 | K106D Y196W | |
| 205 sac_production_final | SAC | 4 | 15,360 | 0.7685 | F107C V116I Y196W | |
| 215 sac_dense_reward | SAC | 4 | 15,360 | 0.7680 | K106N V116I | dense, guardrails on |
| 212 sac_high_entropy | SAC | 4 | 15,360 | 0.7665 | K106N V116I | |
| 210 sac_high_delta | SAC | 4 | 15,360 | 0.7640 | K106N V116I | |

Substitution counts across the 13 sequences: V116I 11, K106N 8, Y196W 5, V160I 5, K106D 2, Y196F 2, T121S 1, I134V 1, F107C 1. Each of these substitutions is also a variant in the dataset; their normalized labels are V116I 0.716, K106N 0.721, K106D 0.723, V160I 0.754, Y196W 0.748, Y196F 0.719, T121S 0.663, I134V 0.702, F107C 0.685.

### 5.5 The two runs shown in `Findings.docx`
Both runs: initial maximum fitness 0.311339, 4 rounds, 15,360 oracle calls (recorded).

| | Without entropy constraint | With entropy constraint |
|--|--|--|
| Best fitness | 0.834945 | 0.763875 |
| Absolute improvement | +0.523606 | +0.452536 |
| Percentage improvement | 168.18% | 145.35% |
| Unique sequences | 365 | 346 |
| Round-4 avg fitness top-128 (as printed) | −4.399999 | −4.367587 |
| Round-4 diversity / novelty | 3.0 / 1.0 | 3.0 / 1.0 |

Top sequences. "Re-scored" is the shipped oracle plus the dataset calibration. Mutations are given against the reference sequence; the start sequence carries N57I, so each sequence also differs from the start sequence by I57N (a reversion). The document lists positions 57, 91, 96, 104, 150 and 190 for these changes; the positions below are in the dataset's numbering.

| Run | Rank | Fitness (recorded) | Re-scored | Mutations vs reference |
|--|--|--|--|--|
| Without entropy | 1 | 0.834945 | 0.8362 | V116I V160I Y196W |
| Without entropy | 2 | 0.788986 | 0.7901 | K106N V160I |
| Without entropy | 3 | 0.787934 | 0.7890 | V116I V160I |
| With entropy | 1 | 0.763875 | 0.7619 | V116I Y196W |
| With entropy | 2 | 0.759656 | 0.7578 | V116I Y196F |
| With entropy | 3 | 0.740149 | 0.7385 | V94I V116I Y196F |

### 5.6 Algorithm comparison (`Findings.docx`, section "ppo vs sac")
| Item | Value |
|--|--|
| Environment changes (recorded) | oracle outputs normalized to [0, 1] with the dataset bounds; `buffer.update()` called inside the step so episodes restart from new maxima |
| SAC pipeline (recorded) | off-policy replay buffer of (start sequence, Δz, reward, mutated sequence) tuples; Gaussian policy with tanh squashing |
| Best fitness, PPO runs (table 5.4) | 0.8120 to 0.8657 |
| Best fitness, SAC runs (table 5.4) | 0.7640 to 0.8364 |

Values in `Findings.docx` Table 2 against the saved results:

| Run | Algorithm | Listed setup | Value in document | Saved result |
|--|--|--|--|--|
| 113 | PPO | dense, guardrails on | 0.8405 | 0.8423 |
| 106 | PPO | sparse, guardrails on | 0.8430 | 0.8430 |
| 111 | PPO | sparse, guardrails off | 0.8400 | 0.8657 |
| 211 | SAC | sparse, guardrails off | 0.8099 | 0.8099 |
| 206 | SAC | sparse, guardrails on | 0.7812 | no saved result |
| 215 | SAC | dense, guardrails on | 0.7647 | 0.7680 |
