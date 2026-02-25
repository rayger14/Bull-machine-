#!/usr/bin/env bash
# =============================================================================
# monitor_coinbase.sh -- Monitor Coinbase paper trading status
#
# Usage:
#   ./deploy/monitor_coinbase.sh           # Full status
#   ./deploy/monitor_coinbase.sh --logs    # Follow live logs
#   ./deploy/monitor_coinbase.sh --errors  # Show recent errors only
# =============================================================================
set -euo pipefail

SERVER_USER="ubuntu"
SERVER_IP="165.1.79.19"
SSH_KEY="~/.ssh/oracle_bullmachine"
SSH_CMD="ssh -i ${SSH_KEY} ${SERVER_USER}@${SERVER_IP}"

MODE="${1:-status}"

case "$MODE" in
    --logs|-l)
        echo "=== Following Coinbase Paper Trader Logs (Ctrl+C to stop) ==="
        ${SSH_CMD} "sudo journalctl -u coinbase-paper -f --no-pager"
        ;;
    --errors|-e)
        echo "=== Recent Errors (last 50 lines) ==="
        ${SSH_CMD} "sudo journalctl -u coinbase-paper --no-pager -p err -n 50 2>/dev/null || echo 'No error logs found'"
        ;;
    *)
        echo "=== Bull Machine Coinbase Paper Trader Status ==="
        echo ""
        echo "--- Service Status ---"
        ${SSH_CMD} "sudo systemctl status coinbase-paper --no-pager -l 2>/dev/null || echo 'Service not installed'"
        echo ""
        echo "--- Memory Usage ---"
        ${SSH_CMD} "systemctl show coinbase-paper --property=MemoryCurrent 2>/dev/null || echo 'N/A'"
        echo ""
        echo "--- Last 20 Log Lines ---"
        ${SSH_CMD} "sudo journalctl -u coinbase-paper --no-pager -n 20 2>/dev/null || echo 'No logs'"
        echo ""
        echo "--- Recent Signals ---"
        ${SSH_CMD} "tail -5 /home/ubuntu/Bull-machine-/results/coinbase_paper/signals.csv 2>/dev/null || echo 'No signals yet'"
        echo ""
        echo "--- Current State ---"
        ${SSH_CMD} "cat /home/ubuntu/Bull-machine-/results/coinbase_paper/state.json 2>/dev/null | python3 -m json.tool 2>/dev/null || echo 'No state file'"
        echo ""
        echo "--- System Resources ---"
        ${SSH_CMD} "free -m | head -2; echo ''; df -h / | tail -1"
        echo ""
        echo "--- Freqtrade Status (for reference) ---"
        ${SSH_CMD} "sudo systemctl is-active freqtrade 2>/dev/null || echo 'not running'"
        ;;
esac
