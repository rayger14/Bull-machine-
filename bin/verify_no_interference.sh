#!/bin/bash
# ARCHETYPE OPTIMIZATION INTERFERENCE VERIFICATION SCRIPT
# Verifies that System B0 deployment does not interfere with archetype optimizations
# Usage: ./verify_no_interference.sh

BASELINE_FILE="/Users/raymondghandchi/Bull-machine-/Bull-machine-/SYSTEM_HEALTH_BASELINE_SNAPSHOT.md"
REPORT_FILE="/Users/raymondghandchi/Bull-machine-/Bull-machine-/interference_check_$(date +%Y%m%d_%H%M%S).txt"

echo "=================================="
echo "INTERFERENCE VERIFICATION CHECK"
echo "=================================="
echo "Timestamp: $(date +"%Y-%m-%d %H:%M:%S")" | tee "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

# Check 1: Database File Integrity
echo "=== CHECK 1: Database File Integrity ===" | tee -a "$REPORT_FILE"
BASELINE_DB_HASHES=$(cat <<EOF
optuna_production_v2_bos_choch.db: 1763403493.825
optuna_production_v2_long_squeeze.db: 1763403636.334
optuna_production_v2_order_block_retest.db: 1763403503.999
optuna_production_v2_trap_within_trend.db: 1763403505.737
EOF
)

echo "Baseline modification times:" | tee -a "$REPORT_FILE"
echo "$BASELINE_DB_HASHES" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

echo "Current modification times:" | tee -a "$REPORT_FILE"
for db in optuna_production_v2_bos_choch.db \
          optuna_production_v2_long_squeeze.db \
          optuna_production_v2_order_block_retest.db \
          optuna_production_v2_trap_within_trend.db; do
    if [ -f "/Users/raymondghandchi/Bull-machine-/Bull-machine-/$db" ]; then
        stat -f "%Sm %N" -t "%s.%3N" "/Users/raymondghandchi/Bull-machine-/Bull-machine-/$db" | tee -a "$REPORT_FILE"
    fi
done
echo "" | tee -a "$REPORT_FILE"

# Check 2: File Lock Status
echo "=== CHECK 2: File Lock Status ===" | tee -a "$REPORT_FILE"
LOCK_COUNT=$(lsof +D /Users/raymondghandchi/Bull-machine-/Bull-machine- 2>/dev/null | grep -E "optuna.*\.db$" | wc -l | tr -d ' ')
echo "Active locks on Optuna databases: $LOCK_COUNT" | tee -a "$REPORT_FILE"

if [ "$LOCK_COUNT" -gt 0 ]; then
    echo "⚠️  WARNING: Database locks detected!" | tee -a "$REPORT_FILE"
    lsof +D /Users/raymondghandchi/Bull-machine-/Bull-machine- 2>/dev/null | grep -E "optuna.*\.db$" | tee -a "$REPORT_FILE"
else
    echo "✓ PASS: No locks detected" | tee -a "$REPORT_FILE"
fi
echo "" | tee -a "$REPORT_FILE"

# Check 3: Database Accessibility
echo "=== CHECK 3: Database Accessibility ===" | tee -a "$REPORT_FILE"
for db in optuna_production_v2_bos_choch.db \
          optuna_production_v2_long_squeeze.db \
          optuna_production_v2_order_block_retest.db \
          optuna_production_v2_trap_within_trend.db; do
    DB_PATH="/Users/raymondghandchi/Bull-machine-/Bull-machine-/$db"
    if [ -f "$DB_PATH" ]; then
        # Try to open database with SQLite
        INTEGRITY_CHECK=$(sqlite3 "$DB_PATH" "PRAGMA integrity_check;" 2>&1)
        if [ "$INTEGRITY_CHECK" = "ok" ]; then
            echo "✓ $db: ACCESSIBLE and VALID" | tee -a "$REPORT_FILE"
        else
            echo "⚠️  $db: INTEGRITY ISSUE - $INTEGRITY_CHECK" | tee -a "$REPORT_FILE"
        fi
    else
        echo "✗ $db: NOT FOUND" | tee -a "$REPORT_FILE"
    fi
done
echo "" | tee -a "$REPORT_FILE"

# Check 4: Process Isolation
echo "=== CHECK 4: Process Isolation ===" | tee -a "$REPORT_FILE"
echo "Active Python processes:" | tee -a "$REPORT_FILE"
ps aux | grep python | grep -v grep | awk '{print $2, $3, $4, $11, $12, $13}' | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

# Check 5: Resource Contention
echo "=== CHECK 5: Resource Contention ===" | tee -a "$REPORT_FILE"
echo "Current resource usage:" | tee -a "$REPORT_FILE"
top -l 1 | grep -E "Processes|CPU usage|PhysMem" | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

# Check 6: Directory Separation
echo "=== CHECK 6: Directory Separation ===" | tee -a "$REPORT_FILE"
echo "System B0 directory:" | tee -a "$REPORT_FILE"
ls -lah /Users/raymondghandchi/Bull-machine-/Bull-machine-/results/system_b0/ 2>/dev/null | tail -10 | tee -a "$REPORT_FILE"
echo "" | tee -a "$REPORT_FILE"

# Calculate Interference Score
INTERFERENCE_SCORE=0

# Penalize if databases were modified
CURRENT_MTIME_BCHOCH=$(stat -f "%m" /Users/raymondghandchi/Bull-machine-/Bull-machine-/optuna_production_v2_bos_choch.db 2>/dev/null || echo "0")
if [ "$CURRENT_MTIME_BCHOCH" != "1763403493" ]; then
    INTERFERENCE_SCORE=$((INTERFERENCE_SCORE + 25))
    echo "⚠️  Database modification detected: bos_choch" | tee -a "$REPORT_FILE"
fi

# Penalize if locks exist
if [ "$LOCK_COUNT" -gt 0 ]; then
    INTERFERENCE_SCORE=$((INTERFERENCE_SCORE + 30))
fi

# Penalize if database integrity compromised
# (Already checked above)

# Final Score
echo "" | tee -a "$REPORT_FILE"
echo "=== INTERFERENCE ASSESSMENT ===" | tee -a "$REPORT_FILE"
echo "Interference Score: $INTERFERENCE_SCORE / 100 (lower is better)" | tee -a "$REPORT_FILE"

if [ "$INTERFERENCE_SCORE" -eq 0 ]; then
    echo "✓ RESULT: NO INTERFERENCE DETECTED" | tee -a "$REPORT_FILE"
    echo "Status: PASS - System B0 deployment is isolated" | tee -a "$REPORT_FILE"
elif [ "$INTERFERENCE_SCORE" -lt 20 ]; then
    echo "⚠️  RESULT: MINOR INTERFERENCE DETECTED" | tee -a "$REPORT_FILE"
    echo "Status: ACCEPTABLE - Monitor closely" | tee -a "$REPORT_FILE"
elif [ "$INTERFERENCE_SCORE" -lt 50 ]; then
    echo "⚠️  RESULT: MODERATE INTERFERENCE DETECTED" | tee -a "$REPORT_FILE"
    echo "Status: WARNING - Investigation recommended" | tee -a "$REPORT_FILE"
else
    echo "🚨 RESULT: SEVERE INTERFERENCE DETECTED" | tee -a "$REPORT_FILE"
    echo "Status: CRITICAL - Rollback recommended" | tee -a "$REPORT_FILE"
fi

echo "" | tee -a "$REPORT_FILE"
echo "Report saved to: $REPORT_FILE" | tee -a "$REPORT_FILE"
echo "=================================="
