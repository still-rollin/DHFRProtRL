# Findings

Full results, checks and caveats for DHFRProtRL. Short overview: [../README.md](../README.md).

[LatProtRL](https://github.com/haewonc/LatProtRL) (Lee et al., ICML 2024) ported to a DHFR single-mutant fitness dataset, with a relative-improvement reward, optional heuristic reward-shaping constraints, and a PPO vs SAC comparison.

> **Status: research prototype with negative baseline results.** All fitness values are predictions of a surrogate model trained on a random 85% of the same 4,161 variants that define the optimum; its accuracy on unseen variants (Pearson 0.74) is much lower than the 0.935 reported. Under that oracle, simple no-RL baselines match or beat every RL run (see [Results](#results)). Read [What is and is not established](#what-is-and-is-not-established) before citing a number.

## Summary
- **Built:** DHFR support for LatProtRL (config, oracle loader, data), a dense reward, three optional soft constraints, a SAC variant, and ablation/diagnostic tooling.
- **Found:** across 13 multi-round runs, PPO and SAC repeatedly converge on the same 2–5 substitutions (K106N/D, V116I, V160I, Y196W/F, sometimes T121S, I134V). The four most frequent (V116I, K106N, Y196W, V160I) are each already in the top 6% of single mutants in the data. The RL agent combines known-good single mutants.
- **Did not find:** evidence that RL beats simple search. A scan of the oracle over all single mutants finds a sequence scoring 0.979 with 4,161 oracle calls; the best RL run reaches 0.866 with 38,400.

## The data (verified)
`data/DHFR/medium.csv` has 4,161 sequences of 219 residues. **It is a complete single-substitution scan**: 219 positions × 19 substitutions, every row exactly one substitution away from one reference sequence. The reference itself is not in the file. `easy.csv`, `medium.csv` and `hard.csv` contain the same sequences and targets; "level" does not change the data here.

Fitness is min–max normalized (raw −5.215 → 0, raw 4.437 → 1). In normalized units the median variant is 0.547, 150 variants (3.6%) score ≥ 0.738, 11 score ≥ 0.866, and the best single mutant (D1M) scores 1.000. The top four single mutants are all at position 1 (D1M, D1Y, D1F, D1W); whether that reflects activity or an assay effect is not known from the files here.

**There is no measured wild-type value.** The code's `REFSEQ` for DHFR is the first row of the file (D1A, raw 1.909, normalized 0.738), which is a single mutant, not the reference. The oracle estimates the true reference at about 0.665.

## Results
### RL runs (oracle-predicted normalized fitness)
From each run's `final_research_metrics.json` (`results/runs/`). Independently re-scored with the oracle: all 13 agree to within ±0.005.

| Run | Algo | Rounds | Best | Mutations vs reference |
|--|--|--|--|--|
| 111 `ppo_unconstrained_final` | PPO | 10 | 0.866 | K106N V116I T121S I134V V160I |
| 106 `ppo_baseline_fixed` | PPO | 4 | 0.843 | K106N V116I V160I |
| 113 `ppo_dense_reward` | PPO | 10 | 0.842 | K106N V160I Y196W |
| 213 `sac_cold_start_v2` | SAC | 4 | 0.836 | V116I V160I Y196W |
| 107 `ppo_confirmation_verify` | PPO | 4 | 0.833 | V116I V160I Y196F |
| 211 `sac_unconstrained` | SAC | 4 | 0.810 | K106N V116I Y196F |
| 102 `final_full_model_run` | PPO | 4 | 0.770 | K106D Y196W |
| 205 `production_sac_final` | SAC | 4 | 0.769 | F107C V116I Y196W |

Full list: `results/tables/run_inventory.md`. 19 DHFR runs finished (6 are 1-round debug runs); the other 246 result folders on the cluster are partial or empty.

### No-RL baselines, same oracle (`analysis/baseline_no_rl.py`)
| Baseline | Best oracle score | Oracle calls |
|--|--|--|
| Oracle scan of all single mutants (1 greedy step) | **0.979** (D1M) | 4,161 |
| Greedy hill-climb, 2 steps | 1.000 (clipped at the normalization ceiling) | 8,303 |
| Best 2 of the top-20 single mutants combined | 1.000 (clipped) | 190 |
| Random combinations of 1–5 known single mutants | 1.000 (clipped; mean 0.340) | 15,360 and 38,400 |
| **Best RL run (111)** | **0.866** | 38,400 |

Caveats on the comparison: baselines start from the reference, RL starts from a pool of poor variants (10th percentile, normalized 0.28–0.31); the top-20 and random baselines use dataset labels or sequences, which the RL agent does not see directly. The oracle-scan and greedy baselines use only oracle queries. The oracle was trained on a random 85% of these 4,161 variants, so the best-measured variant (D1M) was in its training data, and none of these numbers measures discovery beyond the data.

### Where the RL picks sit in the landscape (`analysis/landscape_summary.py`)
Each recurring RL mutation is already a strong single mutant: K106N/D at the 95th percentile of single-mutant labels, V116I at the 94th, V160I at the 97th, Y196W/F at the 94th–97th. The agent's contribution is combining them.

## What is and is not established
Established (checked against files or reproduced):
- The saved metrics are internally consistent (re-scored sequences match).
- Oracle Pearson 0.935 reproduces on all 4,161 variants (on its real held-out test set it is 0.742).
- PPO and SAC runs converge on the same small mutation set.
- Simple baselines score higher than the RL runs under this oracle.

Not established:
1. **No independent validation.** Fitness is the output of the CNN the agent optimizes. No second oracle, structural or stability prediction, or wet-lab data.
2. **The oracle's reported accuracy is mostly training fit.** The oracle (`oracle/train_dhfr_oracle.py`) was trained on a random 85% of the 4,161 variants. The reported Pearson 0.935 / R² 0.874 were computed on all 4,161 (`sample_size: 4161`), though the earlier write-up (`Findings.docx`, not included here) calls them a "held-out test set". On the real held-out test set (625 variants, reproduced by `analysis/oracle_true_holdout.py`) the shipped checkpoint scores **Pearson 0.742, R² 0.525, RMSE 1.0 raw units (about 0.10 normalized)**; on its training part, 0.996. Its own `evaluate_dhfr_oracle.py` reports 0.971 only because 87% of its "test" rows were training data. A model of the same design retrained for this review agreed (random 80/20: 0.99 train, 0.755 unseen; holding out whole positions: 0.14 unseen; `analysis/oracle_generalization.py`). Two code problems explain the overfit (`oracle/README.md`): early stopping never fires and the "best model" is never restored.
3. **Multi-mutant predictions have no ground truth.** The oracle saw only single mutants, but the RL sequences carry 2–5 substitutions. For the 13 best sequences the oracle scores them 0.00–0.52 raw units *below* a simple additive estimate built from the single-mutant labels (median −0.26), so the oracle behaves roughly additively there. That estimate needs the reference's own fitness, which is not in the data, so it uses the oracle's value (1.205 raw). Whether real epistasis would help or hurt these combinations is unknown.
4. **The reward-function claim is untested.** The oracle was loaded with the wrong architecture for part of the project (from the project's debugging notes, which are not included). No completed run compares the old absolute reward with the relative reward on the corrected oracle.
5. **Constraints.** They are soft reward multipliers. The entropy table is `data/DHFR/conservation_scores.csv` (source alignment not included) and the five blocked positions are exactly its zero-entropy positions. The recurring RL positions are variable (entropy 0.66–1.66). The active-site rules in `docs/drafts/` were never applied. A stability benefit (instability index, p = 0.041) comes from a single-seed comparison.
6. **Single seed, 4–10 rounds** (upstream uses 15 rounds, 5 runs). Gaps of ~0.01 between runs are not interpretable.
7. **Most run settings are not recorded** (constraint flags, delta, seed). Run names are descriptive only.

## Corrections to the earlier notes
The project's earlier working notes and write-up are not included here. Where they disagree with the checks above:

| Note says | Check shows |
|--|--|
| Wild-type fitness 0.738 | That is the label of the D1A mutant; no wild-type measurement exists; oracle estimate 0.665 |
| "Exceeded wild-type" | 150 single mutants (3.6%) already exceed 0.738 |
| +147% to +178% improvement | Measured against the best member of a deliberately poor start pool (0.31); not informative |
| "I57N is present in every top sequence" | Relative to a mutated start sequence (N57I); relative to the true reference it is the reference residue, not a mutation |
| Oracle evaluated on "a held-out test set of 4,161 variants" (`Findings.docx`) | 4,161 is the whole dataset. On the real 625-variant test set: Pearson 0.742, R² 0.525 |
| Oracle "relative accuracy 73.66%" (share of predictions within 10% error) | Reproduces exactly, but the script divides by the raw target without taking its absolute value, so every variant with a negative target (48.3% of the data) counts as correct regardless of error. With the absolute value it is 45.3% |
| Top mutations at positions "~57, ~104, ~150, ~190" (`Findings.docx`) | Those are approximate. In the data's numbering the recurring changes are K106, V116, V160 and Y196; position 57 is the reference residue |
| Round tables report "Avg Fitness (Top-128) ≈ −4.4" next to a 0–1 global best | Two different scales in one table; the evaluator's average, diversity and novelty columns are not usable for DHFR |
| Fitness scores 1.50 / 1.94 / 2.24, "~0.3 = wild type" | A different, undocumented scale; not comparable with this README |
| Constraints are "biologically grounded safeguards" | Soft heuristics; stability benefit unverified |

## What would make this a result
Remove the best-labelled variants from the oracle's training data and test whether the agent recovers them; retrain the oracle with working early stopping and report its held-out accuracy (an ensemble would also give uncertainty estimates); obtain multi-mutant measurements; run the old-vs-new reward comparison on the corrected oracle; add seeds and the upstream baselines (PEX, AdaLead, CMA-ES) on DHFR.

## Reproducing
```bash
conda env create -f env.yml
# Weights are not in this repo (~5 GB each). Download ESM-2 650M to ckpt/ and train the VED:
python train_seq.py --protein DHFR --level medium --device cuda:0 --batch 16 -r 16
python optimize.py  --protein DHFR --level medium --device cuda:0 --run 0 --use_oracle
python optimize_sac.py --protein DHFR --level medium --device cuda:0 --run 0 --use_oracle
# analyses (CPU, no weights needed beyond ckpt/DHFR/oracle.ckpt):
python analysis/baseline_no_rl.py
python analysis/landscape_summary.py
python tools/aggregate_runs.py --src . --results results/runs --out results/tables
```
The training commands are adapted from `scripts/` and were not re-run when this repo was assembled; the analysis scripts were run.

## Layout
```
optimize*.py config.py metric.py train_seq.py   entry points and config
net/ utils/ baseline/ *.yaml                    model, environment, buffer, baselines, rules
oracle/                                         oracle training code and notes on its problems
data/DHFR/  ckpt/DHFR/                          dataset, conservation table, small oracle checkpoint
analysis/                                       baselines, landscape summary, result analysis
scripts/  tools/                                cluster launchers; run aggregation
results/                                        per-run metrics, tables, analysis outputs
figures/ logs/ docs/                            figures, PPO/SAC logs, earlier outputs, unused drafts, review page
```

## Credits and license
Built on LatProtRL (Lee et al., ICML 2024; [arXiv:2405.18986](https://arxiv.org/abs/2405.18986)). The upstream repository shows no license file, so none is granted here and this repository is not licensed for redistribution; keep it private pending permission and a license decision.
