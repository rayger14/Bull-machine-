#!/bin/bash
# SYSTEM STATE COMPARISON SCRIPT
# Compares current system state with baseline

echo "==================================="
echo "SYSTEM STATE COMPARISON"
echo "==================================="
echo "Baseline: 2025-12-05 12:29:00"
echo "Current:  $(date +"%Y-%m-%d %H:%M:%S")"
echo ""

echo "=== CPU COMPARISON ==="
echo "Baseline CPU Idle: 76.19%"
echo "Current CPU Idle:  $(top -l 1 | grep "CPU usage" | awk '{print $7}')"
echo ""

echo "=== MEMORY COMPARISON ==="
echo "Baseline Memory Free: 74MB"
echo "Current Memory Free:  $(vm_stat | grep "Pages free" | awk '{printf "%.0fMB", $3*4096/1048576}')"
echo ""

echo "=== DISK I/O COMPARISON ==="
echo "Baseline MB/s: 5.93"
echo "Current MB/s:  $(iostat -d -c 1 | tail -n 2 | awk '{print $3}')"
echo ""

echo "=== PROCESS COMPARISON ==="
echo "Baseline Python Processes: 2"
echo "Current Python Processes:  $(ps aux | grep python | grep -v grep | wc -l | tr -d ' ')"
echo ""
echo "Baseline Optuna Processes: 0"
echo "Current Optuna Processes:  $(pgrep -fl optuna | wc -l | tr -d ' ')"
echo ""

echo "=== DATABASE COMPARISON ==="
for db in optuna_production_v2_bos_choch.db \
          optuna_production_v2_long_squeeze.db \
          optuna_production_v2_order_block_retest.db \
          optuna_production_v2_trap_within_trend.db; do
    echo "$db:"
    echo "  Baseline: 1763403... (Nov 17)"
    echo "  Current:  $(stat -f "%m" /Users/raymondghandchi/Bull-machine-/Bull-machine-/$db 2>/dev/null || echo 'N/A')"
done
echo ""

echo "=== SYSTEM B0 COMPARISON ==="
echo "Baseline Size: 468KB"
echo "Current Size:  $(du -sh /Users/raymondghandchi/Bull-machine-/Bull-machine-/results/system_b0/ 2>/dev/null | awk '{print $1}')"
echo ""
echo "Baseline Files: 7"
echo "Current Files:  $(ls -1 /Users/raymondghandchi/Bull-machine-/Bull-machine-/results/system_b0/ 2>/dev/null | wc -l | tr -d ' ')"
echo ""

echo "==================================="
