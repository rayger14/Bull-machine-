#!/bin/bash
# QUICK HEALTH CHECK SCRIPT
# Provides instant snapshot of system health
# Usage: ./quick_health_check.sh

echo "=================================="
echo "   SYSTEM HEALTH QUICK CHECK"
echo "=================================="
echo "Timestamp: $(date +"%Y-%m-%d %H:%M:%S")"
echo ""

# CPU and Memory Summary
echo "--- RESOURCES ---"
top -l 1 | grep -E "Processes|CPU usage|PhysMem" | head -3
echo ""

# Active Python Processes
echo "--- ACTIVE PROCESSES ---"
PYTHON_COUNT=$(ps aux | grep python | grep -v grep | wc -l | tr -d ' ')
OPTUNA_COUNT=$(pgrep -fl optuna | wc -l | tr -d ' ')
BACKTEST_COUNT=$(pgrep -fl backtest | wc -l | tr -d ' ')
echo "Python processes: $PYTHON_COUNT"
echo "Optuna processes: $OPTUNA_COUNT"
echo "Backtest processes: $BACKTEST_COUNT"
echo ""

# Disk Usage
echo "--- DISK USAGE ---"
df -h /Users/raymondghandchi/Bull-machine-/Bull-machine- | tail -1 | awk '{print "Used: " $3 " / " $2 " (" $5 ")"}'
echo ""

# Disk I/O
echo "--- DISK I/O ---"
iostat -d -c 1 | tail -n 2
echo ""

# Database Status
echo "--- OPTUNA DATABASES ---"
DB_COUNT=$(ls -1 /Users/raymondghandchi/Bull-machine-/Bull-machine-/optuna_production_v2_*.db 2>/dev/null | wc -l | tr -d ' ')
LOCK_COUNT=$(lsof +D /Users/raymondghandchi/Bull-machine-/Bull-machine- 2>/dev/null | grep -E "\.db$" | wc -l | tr -d ' ')
echo "Production databases: $DB_COUNT"
echo "Active file locks: $LOCK_COUNT"
echo ""

# System B0 Status
echo "--- SYSTEM B0 ---"
if [ -d "/Users/raymondghandchi/Bull-machine-/Bull-machine-/results/system_b0" ]; then
    B0_SIZE=$(du -sh /Users/raymondghandchi/Bull-machine-/Bull-machine-/results/system_b0 2>/dev/null | awk '{print $1}')
    B0_FILES=$(ls -1 /Users/raymondghandchi/Bull-machine-/Bull-machine-/results/system_b0 2>/dev/null | wc -l | tr -d ' ')
    B0_LATEST=$(ls -t /Users/raymondghandchi/Bull-machine-/Bull-machine-/results/system_b0 2>/dev/null | head -1)
    echo "Directory size: $B0_SIZE"
    echo "File count: $B0_FILES"
    echo "Latest file: $B0_LATEST"
else
    echo "Directory not found"
fi
echo ""

# Health Score Calculation
HEALTH_SCORE=100

# Check CPU usage (subtract points if >80%)
CPU_IDLE=$(top -l 1 | grep "CPU usage" | awk '{print $7}' | tr -d '%')
if [ $(echo "$CPU_IDLE < 20" | bc -l 2>/dev/null || echo 0) -eq 1 ]; then
    HEALTH_SCORE=$((HEALTH_SCORE - 20))
fi

# Check memory (subtract points if <10% free)
MEM_FREE=$(vm_stat | grep "Pages free" | awk '{print $3}' | tr -d '.')
if [ "$MEM_FREE" -lt 10000 ]; then
    HEALTH_SCORE=$((HEALTH_SCORE - 15))
fi

# Check disk (subtract points if >90% used)
DISK_USAGE=$(df -h /Users/raymondghandchi/Bull-machine-/Bull-machine- | tail -1 | awk '{print $5}' | tr -d '%')
if [ "$DISK_USAGE" -gt 90 ]; then
    HEALTH_SCORE=$((HEALTH_SCORE - 15))
fi

# Check for database locks (subtract points if locked)
if [ "$LOCK_COUNT" -gt 5 ]; then
    HEALTH_SCORE=$((HEALTH_SCORE - 10))
fi

echo "--- HEALTH SCORE ---"
echo "Overall Health: $HEALTH_SCORE / 100"
if [ "$HEALTH_SCORE" -ge 90 ]; then
    echo "Status: EXCELLENT"
elif [ "$HEALTH_SCORE" -ge 75 ]; then
    echo "Status: GOOD"
elif [ "$HEALTH_SCORE" -ge 60 ]; then
    echo "Status: FAIR"
else
    echo "Status: POOR - INVESTIGATION REQUIRED"
fi

echo ""
echo "=================================="
