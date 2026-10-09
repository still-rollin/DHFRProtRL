#!/bin/bash

# Script for running DHFR optimization overnight with extended training
# Results will be saved to timestamped log files

echo "========================================="
echo "DHFR Optimization - Extended Training"
echo "========================================="
echo ""
echo "Configuration:"
echo "  - Total timesteps: 500,000 (increased from 20,000)"
echo "  - Episode length: 50 steps"
echo "  - Entropy coefficient: 0.12"
echo "  - Start percentile: 10th"
echo "  - Expected runtime: ~1 hour"
echo ""

# Create results directory if it doesn't exist
mkdir -p overnight_results

# Generate timestamp for unique filenames
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="overnight_results/dhfr_training_${TIMESTAMP}.log"
SUMMARY_FILE="overnight_results/dhfr_summary_${TIMESTAMP}.txt"

echo "Results will be saved to:"
echo "  - Full log: $LOG_FILE"
echo "  - Summary: $SUMMARY_FILE"
echo ""
echo "Starting training at $(date)..."
echo ""

# Run optimization with all output captured
WANDB_MODE=offline python optimize.py \
    --protein DHFR \
    --level medium \
    --device cuda \
    --ent_coef 0.12 \
    --not_sparse \
    2>&1 | tee "$LOG_FILE"

# Extract summary statistics
echo ""
echo "========================================="
echo "Training Complete at $(date)"
echo "========================================="
echo ""

# Generate summary
cat > "$SUMMARY_FILE" <<EOF
DHFR Optimization Training Summary
Generated: $(date)
========================================

CONFIGURATION:
- Total timesteps: 500,000
- Episode length: 50 steps
- Entropy coefficient: 0.12 (actual: 0.6)
- Start percentile: 10th
- Protein: DHFR medium

RESULTS:
EOF

# Count successful improvements
SUCCESS_COUNT=$(grep -c "SUCCESS" "$LOG_FILE" || echo "0")
echo "- Total improvements found: $SUCCESS_COUNT" >> "$SUMMARY_FILE"

# Extract episode statistics
EPISODES=$(grep -c "Episode " "$LOG_FILE" || echo "0")
echo "- Total episodes completed: $EPISODES" >> "$SUMMARY_FILE"

# Find best reward
BEST_REWARD=$(grep "SUCCESS" "$LOG_FILE" | grep -oP "reward=\K[0-9.]+" | sort -rn | head -1 || echo "N/A")
echo "- Best reward achieved: $BEST_REWARD" >> "$SUMMARY_FILE"

# Add file location
echo "" >> "$SUMMARY_FILE"
echo "FULL LOG: $LOG_FILE" >> "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo ""
cat "$SUMMARY_FILE"
