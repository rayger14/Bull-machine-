#!/bin/bash

# Watch critical processes and notify when complete
# This script runs until both ETH and BTC processes finish

echo "========================================================================"
echo "  BULL MACHINE 3-YEAR VALIDATION - CRITICAL PROCESS MONITOR"
echo "========================================================================"
echo "Started: $(date)"
echo ""

# Track process states
ETH_RUNNING=1
BTC_RUNNING=1

while [ $ETH_RUNNING -eq 1 ] || [ $BTC_RUNNING -eq 1 ]; do
    clear
    echo "========================================================================"
    echo "  CRITICAL PROCESS STATUS - $(date +%H:%M:%S)"
    echo "========================================================================"
    echo ""

    # Check ETH 3-year backtest (PID 60412)
    if ps aux | grep "60412.*ETH" | grep -v grep > /dev/null; then
        CPU_TIME=$(ps aux | grep "60412.*ETH" | grep -v grep | awk '{print $10}')
        echo "✓ ETH 3-year backtest (PID 60412): RUNNING"
        echo "  CPU Time: $CPU_TIME"
        echo "  Processing: 33,067 bars (2022-2025)"
        echo "  Config: threshold=0.62 (max return)"
    else
        if [ $ETH_RUNNING -eq 1 ]; then
            echo "🎉 ETH 3-year backtest: COMPLETED!"
            echo ""
            ETH_RUNNING=0
        fi
    fi

    echo ""

    # Check BTC 3-year feature store (PID 67171)
    if ps aux | grep "67171.*build_feature" | grep -v grep > /dev/null; then
        CPU_TIME=$(ps aux | grep "67171.*build_feature" | grep -v grep | awk '{print $10}')
        echo "✓ BTC 3-year feature store (PID 67171): RUNNING"
        echo "  CPU Time: $CPU_TIME"
        echo "  Building: 33,166 bars (2022-2025)"
        echo "  Computing: All domain scores + macro integration"
    else
        if [ $BTC_RUNNING -eq 1 ]; then
            echo "🎉 BTC 3-year feature store: COMPLETED!"
            echo ""
            BTC_RUNNING=0
        fi
    fi

    echo ""
    echo "------------------------------------------------------------------------"

    # If both done, prepare final report
    if [ $ETH_RUNNING -eq 0 ] && [ $BTC_RUNNING -eq 0 ]; then
        echo ""
        echo "🏁 ALL CRITICAL PROCESSES COMPLETE!"
        echo ""
        echo "Next steps:"
        echo "1. ETH results available in logs/paper_trading/ETH_3year_optimal.log"
        echo "2. BTC feature store ready - running optimization next"
        echo ""
        break
    fi

    sleep 30
done

echo "========================================================================"
echo "MONITORING COMPLETE at $(date)"
echo "========================================================================"
