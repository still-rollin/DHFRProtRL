# DHFRProtRL

[LatProtRL](https://github.com/haewonc/LatProtRL) (Lee et al., ICML 2024) applied to a DHFR single-mutant fitness dataset (4,161 variants), with a relative-improvement reward, optional soft constraints, and PPO and SAC runs.

**Result.** The RL runs reach a predicted fitness of 0.77 to 0.87 (0–1 scale). A plain scan of the same surrogate model over all single mutants reaches 0.98 with fewer model calls, so RL does not beat simple search here. The surrogate is also less accurate on unseen variants (Pearson 0.74) than the 0.935 first reported, and nothing is validated beyond the model.

Details, checks and limitations: [docs/findings.md](docs/findings.md).

## Layout
```
optimize*.py config.py metric.py train_seq.py   entry points and config
net/ utils/ baseline/                           model, environment, buffer, baselines
oracle/                                         surrogate model training code
data/DHFR/  ckpt/DHFR/                          dataset and small oracle checkpoint
analysis/                                       baselines and held-out checks
results/  logs/  figures/                       run metrics, tables, logs, plots
docs/                                           findings, earlier outputs, review page
```

## Run
```bash
conda env create -f env.yml
python optimize.py --protein DHFR --level medium --device cuda:0 --use_oracle
python analysis/baseline_no_rl.py    # CPU
```
The VED weights (about 5 GB) are not included; train them with `train_seq.py`.

Built on upstream LatProtRL, which has no license file; none is granted here.
