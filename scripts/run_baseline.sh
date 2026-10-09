#!/bin/bash
# Run DHFR optimization WITHOUT biological constraints (baseline)

echo "=========================================="
echo "Running DHFR Optimization WITHOUT Biological Constraints (Baseline)"
echo "=========================================="

# Need to temporarily disable entropy constraint
# This requires modifying config to set use_entropy_constraint=False
python optimize_baseline.py \
    --protein DHFR \
    --level medium \
    --device cuda:0 \
    --run 1 \
    --step_mut 15 \
    --tag "baseline_no_entropy" \
    --ent_coef 0.0

echo "=========================================="
echo "Training completed!"
echo "Results saved with tag: baseline_no_entropy"
echo "=========================================="
