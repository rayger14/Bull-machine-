#!/usr/bin/env bash
# =============================================================================
# Bull Machine — Nightly Health Check
# Checks: service status, heartbeat recency, error rate, drawdown, disk
# Writes: results/health_report.txt + appends to results/health_history.log
# Run via cron: 0 6 * * * /home/ubuntu/Bull-machine-/deploy/health_check.sh
# =============================================================================

REPORT=/home/ubuntu/Bull-machine-/results/health_report.txt
LOG=/home/ubuntu/Bull-machine-/results/health_history.log
HEARTBEAT=/home/ubuntu/Bull-machine-/results/coinbase_paper/heartbeat.json
EQUITY_CSV=/home/ubuntu/Bull-machine-/results/coinbase_paper/equity_history.csv
CD=/home/ubuntu/Bull-machine-

NOW=$(date -u '+%Y-%m-%d %H:%M UTC')
ISSUES=0

exec > >(tee "$REPORT") 2>&1

echo '======================================================'
echo "BULL MACHINE HEALTH REPORT — $NOW"
echo '======================================================'

# ---- 1. Service status ----
SERVICE_STATUS=$(sudo systemctl is-active coinbase-paper 2>/dev/null || echo 'dead')
if [ "$SERVICE_STATUS" = "active" ]; then
    echo "[OK]   Service: active (running)"
else
    echo "[FAIL] Service: $SERVICE_STATUS — restarting..."
    sudo systemctl restart coinbase-paper
    sleep 5
    SERVICE_STATUS=$(sudo systemctl is-active coinbase-paper 2>/dev/null || echo 'dead')
    echo "       Post-restart status: $SERVICE_STATUS"
    ISSUES=$((ISSUES + 1))
fi

# ---- 2. Heartbeat recency (engine must have processed a bar in last 3h) ----
if [ -f "$HEARTBEAT" ]; then
    python3 "$CD/deploy/health_check_heartbeat.py" "$HEARTBEAT"
    HB_EXIT=$?
    [ $HB_EXIT -ne 0 ] && ISSUES=$((ISSUES + 1))
else
    echo "[FAIL] Heartbeat file missing"
    ISSUES=$((ISSUES + 1))
fi

# ---- 3. Error rate in last 24h ----
ERROR_COUNT=$(sudo journalctl -u coinbase-paper --since "24 hours ago" --no-pager 2>/dev/null | grep -c " ERROR " || true)
CRITICAL_COUNT=$(sudo journalctl -u coinbase-paper --since "24 hours ago" --no-pager 2>/dev/null | grep -cE "WATCHDOG|CRITICAL|consecutive bar" || true)
if [ "$CRITICAL_COUNT" -gt 0 ]; then
    echo "[FAIL] Critical/watchdog errors in 24h: $CRITICAL_COUNT"
    ISSUES=$((ISSUES + 1))
elif [ "$ERROR_COUNT" -gt 50 ]; then
    echo "[WARN] High error rate in 24h: $ERROR_COUNT errors"
else
    echo "[OK]   Errors in 24h: $ERROR_COUNT"
fi

# ---- 4. Equity drawdown ----
if [ -f "$EQUITY_CSV" ]; then
    python3 "$CD/deploy/health_check_equity.py" "$EQUITY_CSV"
    DD_EXIT=$?
    [ $DD_EXIT -ne 0 ] && ISSUES=$((ISSUES + 1))
fi

# ---- 5. Disk space ----
DISK_PCT=$(df /home/ubuntu --output=pcent | tail -1 | tr -d ' %')
if [ "$DISK_PCT" -gt 85 ]; then
    echo "[FAIL] Disk usage: ${DISK_PCT}%"
    ISSUES=$((ISSUES + 1))
elif [ "$DISK_PCT" -gt 70 ]; then
    echo "[WARN] Disk usage: ${DISK_PCT}%"
else
    echo "[OK]   Disk usage: ${DISK_PCT}%"
fi

# ---- 6. Memory ----
MEM=$(sudo systemctl status coinbase-paper --no-pager 2>/dev/null | grep Memory | awk '{print $2}' || echo '?')
echo "[OK]   Memory (coinbase-paper): $MEM"

# ---- Summary ----
echo '------------------------------------------------------'
if [ "$ISSUES" -eq 0 ]; then
    echo "RESULT: HEALTHY (0 issues)"
else
    echo "RESULT: $ISSUES ISSUE(S) DETECTED"
fi
echo '======================================================'

# Append one-liner to rolling history
echo "[$NOW] issues=$ISSUES $(grep RESULT $REPORT | tail -1)" >> "$LOG"

exit $ISSUES
