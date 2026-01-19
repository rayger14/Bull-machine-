#!/bin/bash
# CPCV Optimization Progress Monitor
# Usage: ./bin/monitor_cpcv_progress.sh [task_id]
# Default task_id: be1435c (current optimization)

TASK_ID=${1:-"be1435c"}
OUTPUT_FILE="/private/tmp/claude/-Users-raymondghandchi-Bull-machine--Bull-machine-/tasks/${TASK_ID}.output"

echo "========================================"
echo "CPCV OPTIMIZATION PROGRESS MONITOR"
echo "========================================"
echo "Task ID: $TASK_ID"
echo "Output: $OUTPUT_FILE"
echo ""

# Check if task is still running
if ps aux | grep -q "[p]ython3 bin/optimize_constrained_cpcv.py"; then
    echo "Status: RUNNING ✓"
else
    echo "Status: COMPLETED or NOT FOUND"
fi

echo ""
echo "----------------------------------------"
echo "LATEST PROGRESS (Last 30 lines)"
echo "----------------------------------------"
tail -30 "$OUTPUT_FILE"

echo ""
echo "----------------------------------------"
echo "TRIAL SUMMARY"
echo "----------------------------------------"
grep -E "Trial [0-9]+ finished" "$OUTPUT_FILE" | tail -10

echo ""
echo "----------------------------------------"
echo "BEST TRIAL SO FAR"
echo "----------------------------------------"
grep "Best is trial" "$OUTPUT_FILE" | tail -1

echo ""
echo "----------------------------------------"
echo "FOLD DIAGNOSTICS (Last 5)"
echo "----------------------------------------"
grep -E "(Fold|Total trades|Sortino)" "$OUTPUT_FILE" | tail -15

echo ""
echo "----------------------------------------"
echo "QUICK STATS"
echo "----------------------------------------"
TOTAL_TRIALS=$(grep -c "Trial [0-9]+ finished" "$OUTPUT_FILE")
BEST_VALUE=$(grep "Best is trial" "$OUTPUT_FILE" | tail -1 | grep -oE "value: [-0-9.]+" | cut -d' ' -f2)
echo "Trials completed: $TOTAL_TRIALS / 50"
echo "Best objective: $BEST_VALUE"

# Check for new behavior
echo ""
UNIQUE_VALUES=$(grep -E "Trial [0-9]+ finished" "$OUTPUT_FILE" | grep -oE "value: [-0-9.]+" | sort -u | wc -l)
if [ "$UNIQUE_VALUES" -gt 1 ]; then
    echo "✓ GRADIENT DETECTED: $UNIQUE_VALUES unique objective values"
else
    echo "⚠ FLAT SURFACE: All trials have identical objective"
fi

echo ""
echo "========================================"
echo "To watch live: tail -f $OUTPUT_FILE"
echo "To check full output: cat $OUTPUT_FILE"
echo "========================================"
