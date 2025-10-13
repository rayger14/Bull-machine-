#!/bin/bash
# Gate 3: A/B Sanity Test - v1.8.5
# Compare baseline (features OFF) vs treatment (features ON)

set -e

echo "=== Gate 3: v1.8.5 A/B Sanity Test ==="
echo "Period: 2025-09-01 → 2025-09-08 (7 days)"
echo ""

# Test period
START="2025-09-01"
END="2025-09-08"
ASSET="BTC"

# Configs
BASELINE="configs/v185/BTC_baseline_disabled.json"
TREATMENT="configs/v185/BTC_conservative.json"

# Results
RESULTS_A="results/ab_baseline.jsonl"
RESULTS_B="results/ab_treatment.jsonl"

echo "=== A: Baseline (v1.8.5 features OFF) ==="
rm -f results/*.jsonl
python3 bin/live/hybrid_runner.py \
  --asset $ASSET \
  --start $START \
  --end $END \
  --config $BASELINE

# Backup results
cp results/trade_log.jsonl $RESULTS_A
echo "Baseline trades: $(grep -c '"event":"open"' $RESULTS_A || echo 0)"

echo ""
echo "=== B: Treatment (v1.8.5 features ON) ==="
rm -f results/*.jsonl
python3 bin/live/hybrid_runner.py \
  --asset $ASSET \
  --start $START \
  --end $END \
  --config $TREATMENT

# Backup results
cp results/trade_log.jsonl $RESULTS_B
echo "Treatment trades: $(grep -c '"event":"open"' $RESULTS_B || echo 0)"

echo ""
echo "=== A/B Comparison ==="

# Count trades
TRADES_A=$(grep -c '"event":"open"' $RESULTS_A || echo 0)
TRADES_B=$(grep -c '"event":"open"' $RESULTS_B || echo 0)

echo "Trades: A=$TRADES_A, B=$TRADES_B (Δ=$((TRADES_B - TRADES_A)))"

# Parse PNL (simplified - just sum pnl field)
PNL_A=$(python3 -c "
import json
pnl = sum(json.loads(line)['pnl'] for line in open('$RESULTS_A') if 'pnl' in json.loads(line))
print(f'{pnl:.2f}')
" || echo "0.00")

PNL_B=$(python3 -c "
import json
pnl = sum(json.loads(line)['pnl'] for line in open('$RESULTS_B') if 'pnl' in json.loads(line))
print(f'{pnl:.2f}')
" || echo "0.00")

echo "Net PNL: A=\$$PNL_A, B=\$$PNL_B"

# Win rate
WINS_A=$(grep -c '"event":"partial_exit"\|"event":"trail_hit"' $RESULTS_A || echo 0)
WINS_B=$(grep -c '"event":"partial_exit"\|"event":"trail_hit"' $RESULTS_B || echo 0)

if [ $TRADES_A -gt 0 ]; then
  WR_A=$(python3 -c "print(f'{$WINS_A / $TRADES_A * 100:.1f}%')")
else
  WR_A="N/A"
fi

if [ $TRADES_B -gt 0 ]; then
  WR_B=$(python3 -c "print(f'{$WINS_B / $TRADES_B * 100:.1f}%')")
else
  WR_B="N/A"
fi

echo "Win Rate: A=$WR_A, B=$WR_B"

echo ""
echo "✅ Gate 3 A/B comparison complete"
echo ""
echo "Pass criteria:"
echo "  - Both runs complete without errors ✅"
echo "  - Runtime < 10 minutes (check logs)"
echo "  - PNL_B >= PNL_A OR trades similar with better exits"
