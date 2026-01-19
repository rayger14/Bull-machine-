#!/bin/bash
# Quick frontier exploration progress checker

echo "================================================================================"
echo "FRONTIER EXPLORATION PROGRESS"
echo "================================================================================"
echo ""

# Check if process is running
PID=$(ps aux | grep "optuna_frontier" | grep -v grep | awk '{print $2}')
if [ -z "$PID" ]; then
    echo "❌ Process not running"
else
    echo "✅ Process running (PID: $PID)"
    RUNTIME=$(ps aux | grep "optuna_frontier" | grep -v grep | awk '{print $10}')
    echo "   Runtime: $RUNTIME"
fi
echo ""

# Check current frontier and progress
echo "Current Status:"
tail -5 results/frontier_exploration.log | grep -E "(FRONTIER|Best|Trial)" | tail -3
echo ""

# Check database files created
echo "Completed Frontiers:"
for db in results/frontier_exploration/frontier_*.db; do
    if [ -f "$db" ]; then
        frontier=$(basename "$db" .db | sed 's/frontier_//')
        trials=$(sqlite3 "$db" "SELECT COUNT(*) FROM trials WHERE state='COMPLETE'" 2>/dev/null || echo "0")
        best=$(sqlite3 "$db" "SELECT MAX(value) FROM trial_values" 2>/dev/null || echo "N/A")
        echo "  $frontier: $trials trials complete, best PF = $best"
    fi
done
echo ""

# Estimate time remaining
if [ ! -z "$PID" ]; then
    completed=$(ls results/frontier_exploration/frontier_*.db 2>/dev/null | wc -l | tr -d ' ')
    remaining=$((5 - completed))
    echo "Frontiers: $completed/5 complete, $remaining remaining"
    echo "Estimated time remaining: ~$((remaining * 105)) minutes (~$((remaining * 2)) hours)"
fi

echo ""
echo "================================================================================"
echo "To monitor live: tail -f results/frontier_exploration.log"
echo "================================================================================"
