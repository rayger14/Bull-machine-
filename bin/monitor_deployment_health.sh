#!/bin/bash
# DEPLOYMENT HEALTH MONITORING SCRIPT
# Monitors system resources during System B0 deployment
# Usage: ./monitor_deployment_health.sh [interval_seconds] [duration_minutes]

INTERVAL=${1:-5}  # Default 5 seconds
DURATION=${2:-60} # Default 60 minutes
ITERATIONS=$((DURATION * 60 / INTERVAL))
OUTPUT_FILE="/Users/raymondghandchi/Bull-machine-/Bull-machine-/deployment_health_monitor_$(date +%Y%m%d_%H%M%S).log"

echo "=== DEPLOYMENT HEALTH MONITORING ===" | tee -a "$OUTPUT_FILE"
echo "Started: $(date)" | tee -a "$OUTPUT_FILE"
echo "Interval: ${INTERVAL}s, Duration: ${DURATION}m, Iterations: ${ITERATIONS}" | tee -a "$OUTPUT_FILE"
echo "Output: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
echo "========================================" | tee -a "$OUTPUT_FILE"
echo "" | tee -a "$OUTPUT_FILE"

for i in $(seq 1 $ITERATIONS); do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")
    echo "=== SAMPLE $i of $ITERATIONS at $TIMESTAMP ===" >> "$OUTPUT_FILE"

    # CPU and Memory
    echo "--- CPU & Memory ---" >> "$OUTPUT_FILE"
    top -l 1 | head -n 10 >> "$OUTPUT_FILE"

    # Process count
    echo "" >> "$OUTPUT_FILE"
    echo "--- Active Processes ---" >> "$OUTPUT_FILE"
    echo "Python processes: $(ps aux | grep python | grep -v grep | wc -l)" >> "$OUTPUT_FILE"
    echo "Optuna processes: $(pgrep -fl optuna | wc -l)" >> "$OUTPUT_FILE"
    echo "Backtest processes: $(pgrep -fl backtest | wc -l)" >> "$OUTPUT_FILE"

    # Disk I/O
    echo "" >> "$OUTPUT_FILE"
    echo "--- Disk I/O ---" >> "$OUTPUT_FILE"
    iostat -d -c 1 | tail -n 5 >> "$OUTPUT_FILE"

    # Database file changes
    echo "" >> "$OUTPUT_FILE"
    echo "--- Database Status ---" >> "$OUTPUT_FILE"
    ls -lh /Users/raymondghandchi/Bull-machine-/Bull-machine-/optuna_production_v2_*.db 2>/dev/null | awk '{print $9, $5, $6, $7, $8}' >> "$OUTPUT_FILE"

    # Check for file locks
    echo "" >> "$OUTPUT_FILE"
    echo "--- File Locks ---" >> "$OUTPUT_FILE"
    lsof +D /Users/raymondghandchi/Bull-machine-/Bull-machine- 2>/dev/null | grep -E "\.db$" | wc -l | xargs echo "Active DB locks:" >> "$OUTPUT_FILE"

    # System B0 directory size
    echo "" >> "$OUTPUT_FILE"
    echo "--- System B0 Status ---" >> "$OUTPUT_FILE"
    du -sh /Users/raymondghandchi/Bull-machine-/Bull-machine-/results/system_b0/ 2>/dev/null >> "$OUTPUT_FILE"

    echo "========================================" >> "$OUTPUT_FILE"
    echo "" >> "$OUTPUT_FILE"

    # Display progress
    echo "[$i/$ITERATIONS] $TIMESTAMP - Sample recorded"

    # Sleep between iterations (except on last iteration)
    if [ $i -lt $ITERATIONS ]; then
        sleep $INTERVAL
    fi
done

echo "" | tee -a "$OUTPUT_FILE"
echo "=== MONITORING COMPLETE ===" | tee -a "$OUTPUT_FILE"
echo "Ended: $(date)" | tee -a "$OUTPUT_FILE"
echo "Output saved to: $OUTPUT_FILE" | tee -a "$OUTPUT_FILE"
