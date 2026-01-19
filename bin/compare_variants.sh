#!/bin/bash
# Domain Engine Wiring Verification Test
# Compares Core vs Full variants for S1, S4, S5

set -e

ASSET="BTC"
START="2022-01-01"
END="2022-12-31"

echo "================================================================================"
echo "DOMAIN ENGINE WIRING VERIFICATION TEST"
echo "================================================================================"
echo "Asset: $ASSET"
echo "Period: $START to $END"
echo "Feature Store: data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet"
echo "================================================================================"

# Create output directory
mkdir -p results/domain_wiring_test

echo ""
echo "================================================================================"
echo "S1 (LIQUIDITY VACUUM) COMPARISON"
echo "================================================================================"

echo ""
echo "Running S1 CORE (Wyckoff only)..."
python3 bin/backtest_knowledge_v2.py \
  --asset $ASSET \
  --start $START \
  --end $END \
  --config configs/variants/s1_core.json \
  --export-trades results/domain_wiring_test/s1_core_trades.csv \
  > results/domain_wiring_test/s1_core.log 2>&1

echo "Running S1 FULL (All domain engines)..."
python3 bin/backtest_knowledge_v2.py \
  --asset $ASSET \
  --start $START \
  --end $END \
  --config configs/variants/s1_full.json \
  --export-trades results/domain_wiring_test/s1_full_trades.csv \
  > results/domain_wiring_test/s1_full.log 2>&1

echo ""
echo "================================================================================"
echo "S4 (FUNDING DIVERGENCE) COMPARISON"
echo "================================================================================"

echo ""
echo "Running S4 CORE (Wyckoff only)..."
python3 bin/backtest_knowledge_v2.py \
  --asset $ASSET \
  --start $START \
  --end $END \
  --config configs/variants/s4_core.json \
  --export-trades results/domain_wiring_test/s4_core_trades.csv \
  > results/domain_wiring_test/s4_core.log 2>&1

echo "Running S4 FULL (All domain engines)..."
python3 bin/backtest_knowledge_v2.py \
  --asset $ASSET \
  --start $START \
  --end $END \
  --config configs/variants/s4_full.json \
  --export-trades results/domain_wiring_test/s4_full_trades.csv \
  > results/domain_wiring_test/s4_full.log 2>&1

echo ""
echo "================================================================================"
echo "S5 (LONG SQUEEZE) COMPARISON"
echo "================================================================================"

echo ""
echo "Running S5 CORE (Wyckoff only)..."
python3 bin/backtest_knowledge_v2.py \
  --asset $ASSET \
  --start $START \
  --end $END \
  --config configs/variants/s5_core.json \
  --export-trades results/domain_wiring_test/s5_core_trades.csv \
  > results/domain_wiring_test/s5_core.log 2>&1

echo "Running S5 FULL (All domain engines)..."
python3 bin/backtest_knowledge_v2.py \
  --asset $ASSET \
  --start $START \
  --end $END \
  --config configs/variants/s5_full.json \
  --export-trades results/domain_wiring_test/s5_full_trades.csv \
  > results/domain_wiring_test/s5_full.log 2>&1

echo ""
echo "================================================================================"
echo "✅ ALL BACKTESTS COMPLETE"
echo "================================================================================"
echo "Results saved to: results/domain_wiring_test/"
echo ""
echo "Next step: Run analysis script to compare results"
