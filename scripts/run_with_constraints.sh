#!/bin/bash
# Run DHFR optimization WITH biological constraints (entropy-based)

echo "=========================================="
echo "Running DHFR Optimization WITH Biological Constraints"
echo "=========================================="

python optimize.py \
    --protein DHFR \
    --level medium \
    --device cuda:0 \
    --run 0 \
    --step_mut 15 \
    --tag "with_entropy_constraints" \
    --ent_coef 0.0

echo "=========================================="
echo "Training completed!"
echo "Results saved with tag: with_entropy_constraints"
echo "=========================================="
