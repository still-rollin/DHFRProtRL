#!/bin/bash
# Ablation Study for DHFRProtRL Constraints
# Run each configuration for 10,000 steps to compare trajectory

DEVICE="cuda:2"
PROTEIN="DHFR"
LEVEL="medium"
RUN_ID="ablation_$(date +%s)"

echo "Starting Ablation Study Suite on $DEVICE..."

# 1. No Constraints (Baseline - Modified Reward only)
echo "Running: No Constraints..."
bash -c "source "$HOME"/miniconda3/bin/activate latprotrl && python optimize.py --protein $PROTEIN --level $LEVEL --device $DEVICE --run 10 --tag ablation_none --no_entropy --no_size --no_blosum --use_oracle" > ablation_none.log 2>&1

# 2. Entropy Only
echo "Running: Entropy Only..."
bash -c "source "$HOME"/miniconda3/bin/activate latprotrl && python optimize.py --protein $PROTEIN --level $LEVEL --device $DEVICE --run 11 --tag ablation_entropy --no_size --no_blosum --use_oracle" > ablation_entropy.log 2>&1

# 3. Size Only
echo "Running: Size Only..."
bash -c "source "$HOME"/miniconda3/bin/activate latprotrl && python optimize.py --protein $PROTEIN --level $LEVEL --device $DEVICE --run 12 --tag ablation_size --no_entropy --no_blosum --use_oracle" > ablation_size.log 2>&1

# 4. BLOSUM Only
echo "Running: BLOSUM Only..."
bash -c "source "$HOME"/miniconda3/bin/activate latprotrl && python optimize.py --protein $PROTEIN --level $LEVEL --device $DEVICE --run 13 --tag ablation_blosum --no_entropy --no_size --use_oracle" > ablation_blosum.log 2>&1

# 5. Full Constraints (Proposed)
echo "Running: All Constraints..."
bash -c "source "$HOME"/miniconda3/bin/activate latprotrl && python optimize.py --protein $PROTEIN --level $LEVEL --device $DEVICE --run 14 --tag ablation_all --use_oracle" > ablation_all.log 2>&1

echo "Ablation Study Suite Complete."
