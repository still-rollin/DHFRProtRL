#!/bin/bash
# Monitor DHFR training progress

echo "====================================================================="
echo "DHFR Training Monitor"
echo "====================================================================="

# Start training in background
WANDB_MODE=offline nohup python optimize.py --protein DHFR --level medium --device cuda --ent_coef 0.12 --not_sparse > dhfr_training.log 2>&1 &
PID=$!

echo "Training started with PID: $PID"
echo "Log file: dhfr_training.log"
echo ""
echo "Monitoring improvements..."
echo ""

# Monitor for 5 minutes, then show summary
sleep 300

# Count improvements
SUCCESS_COUNT=$(grep -c "SUCCESS" dhfr_training.log)
TOTAL_STEPS=$(grep -c "DEBUG_STEP" dhfr_training.log)

echo "====================================================================="
echo "Progress after 5 minutes:"
echo "====================================================================="
echo "Total steps: $TOTAL_STEPS"
echo "Improvements found: $SUCCESS_COUNT"
echo ""
echo "Recent improvements:"
grep "SUCCESS" dhfr_training.log | tail -10
echo ""
echo "Best fitness achieved:"
grep "from_start=" dhfr_training.log | tail -1
echo ""
echo "Training is still running (PID: $PID)"
echo "To stop: kill $PID"
echo "To view full log: tail -f dhfr_training.log"
