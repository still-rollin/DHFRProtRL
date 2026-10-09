# DHFRProtRL

[LatProtRL](https://github.com/haewonc/LatProtRL) (Lee et al., ICML 2024) applied to a DHFR single-mutant fitness dataset (4,161 variants), with a relative-improvement reward, optional reward-shaping constraints, and PPO and SAC runs.

## Results
- PPO and SAC runs reach a predicted fitness of 0.76 to 0.87 on the 0 to 1 scale (best 0.866, run 111).
- The substitutions that recur across runs are V116I, K106N or K106D, V160I and Y196W or Y196F.
- Scoring every single mutant with the same surrogate model gives 0.979 with 4,161 model calls.
- The AAV run in this repository reaches fitness 0.68 after 15 rounds; the paper reports 0.71.

All fitness values are predictions of the surrogate model in `oracle/`.

- Facts from the original findings document: [docs/findings.md](docs/findings.md)
- Per-run results from the logs: [docs/experiments.md](docs/experiments.md)

## Layout
```
optimize*.py config.py metric.py train_seq.py   entry points and config
net/ utils/ baseline/                           model, environment, buffer, baselines
oracle/                                         surrogate model training code
data/DHFR/  ckpt/DHFR/                          dataset and oracle checkpoint
analysis/                                       baselines, evaluation scripts, figures
results/  logs/  figures/                       run metrics, tables, logs, plots
docs/                                           findings, experiments, earlier outputs
```

## Run
```bash
conda env create -f env.yml
python optimize.py --protein DHFR --level medium --device cuda:0 --use_oracle
python analysis/baseline_no_rl.py
python tools/build_experiments_md.py   # rebuilds docs/experiments.md from results/experiments/
```
The VED weights (about 5 GB) are not included; train them with `train_seq.py`. Training commands are adapted from `scripts/`.

Built on LatProtRL, which has no license file; none is granted here.
