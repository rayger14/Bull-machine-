#!/bin/bash
# Benchmark batch screener + focused replay vs full replay

ASSET=ETH
START=2025-06-15
END=2025-09-30
CONFIG=configs/v18/ETH_comprehensive.json

echo "======================================================================"
echo "🏁 Bull Machine v1.8 - Batch Mode Benchmark"
echo "======================================================================"
echo "Asset:  $ASSET"
echo "Period: $START → $END (3.5 months)"
echo "Config: $CONFIG"
echo "======================================================================"

# Clean up old results
rm -f results/*.jsonl
rm -f /tmp/full_replay.log /tmp/batch_screener.log /tmp/focused_replay.log

# Full replay (baseline)
echo ""
echo "1️⃣ Running FULL replay (baseline)..."
echo "----------------------------------------------------------------------"
START_TIME=$(date +%s)
python3 bin/live/hybrid_runner.py \
    --asset $ASSET \
    --start $START \
    --end $END \
    --config $CONFIG \
    > /tmp/full_replay.log 2>&1
FULL_EXIT_CODE=$?
END_TIME=$(date +%s)
FULL_TIME=$((END_TIME - START_TIME))

if [ $FULL_EXIT_CODE -ne 0 ]; then
    echo "❌ Full replay failed!"
    tail -20 /tmp/full_replay.log
    exit 1
fi

# Extract stats from log
FULL_SIGNALS=$(grep -E "signals generated|Signals:" /tmp/full_replay.log | tail -1 | grep -oE '[0-9]+' | head -1)
FULL_TRADES=$(grep -c "\"entry_price\"" results/trade_log.jsonl 2>/dev/null || echo "0")

echo "Full replay time:     ${FULL_TIME}s ($(($FULL_TIME / 60))m $(($FULL_TIME % 60))s)"
echo "Signals generated:    $FULL_SIGNALS"
echo "Trades executed:      $FULL_TRADES"

# Save full replay results
mv results/trade_log.jsonl results/trade_log_full.jsonl 2>/dev/null

# Batch screener
echo ""
echo "2️⃣ Running BATCH screener..."
echo "----------------------------------------------------------------------"
START_TIME=$(date +%s)
python3 bin/research/batch_screener.py \
    --asset $ASSET \
    --start $START \
    --end $END \
    --config $CONFIG \
    --output results/candidates.jsonl \
    > /tmp/batch_screener.log 2>&1
BATCH_EXIT_CODE=$?
END_TIME=$(date +%s)
BATCH_TIME=$((END_TIME - START_TIME))

if [ $BATCH_EXIT_CODE -ne 0 ]; then
    echo "❌ Batch screener failed!"
    tail -20 /tmp/batch_screener.log
    exit 1
fi

CANDIDATES=$(wc -l < results/candidates.jsonl 2>/dev/null || echo "0")

echo "Batch screener time:  ${BATCH_TIME}s"
echo "Candidates generated: $CANDIDATES"

# Focused replay
echo ""
echo "3️⃣ Running FOCUSED replay (candidate-driven)..."
echo "----------------------------------------------------------------------"
START_TIME=$(date +%s)
python3 bin/live/hybrid_runner.py \
    --asset $ASSET \
    --start $START \
    --end $END \
    --config $CONFIG \
    --candidates results/candidates.jsonl \
    > /tmp/focused_replay.log 2>&1
FOCUSED_EXIT_CODE=$?
END_TIME=$(date +%s)
FOCUSED_TIME=$((END_TIME - START_TIME))

if [ $FOCUSED_EXIT_CODE -ne 0 ]; then
    echo "❌ Focused replay failed!"
    tail -20 /tmp/focused_replay.log
    exit 1
fi

FOCUSED_SIGNALS=$(grep -E "signals generated|Signals:" /tmp/focused_replay.log | tail -1 | grep -oE '[0-9]+' | head -1)
FOCUSED_TRADES=$(grep -c "\"entry_price\"" results/trade_log.jsonl 2>/dev/null || echo "0")

echo "Focused replay time:  ${FOCUSED_TIME}s ($(($FOCUSED_TIME / 60))m $(($FOCUSED_TIME % 60))s)"
echo "Signals generated:    $FOCUSED_SIGNALS"
echo "Trades executed:      $FOCUSED_TRADES"

# Calculate total batch time
BATCH_TOTAL=$((BATCH_TIME + FOCUSED_TIME))

echo ""
echo "======================================================================"
echo "📊 RESULTS SUMMARY"
echo "======================================================================"
echo "Full replay:         ${FULL_TIME}s ($(($FULL_TIME / 60))m $(($FULL_TIME % 60))s)"
echo "                     └─ Signals: $FULL_SIGNALS, Trades: $FULL_TRADES"
echo ""
echo "Batch + Focused:     ${BATCH_TOTAL}s ($(($BATCH_TOTAL / 60))m $(($BATCH_TOTAL % 60))s)"
echo "                     ├─ Screener:  ${BATCH_TIME}s (${CANDIDATES} candidates)"
echo "                     └─ Replay:    ${FOCUSED_TIME}s (Signals: $FOCUSED_SIGNALS, Trades: $FOCUSED_TRADES)"
echo ""
if [ $BATCH_TOTAL -gt 0 ]; then
    SPEEDUP=$(awk "BEGIN {printf \"%.1f\", $FULL_TIME / $BATCH_TOTAL}")
    echo "Speedup:             ${SPEEDUP}×"
else
    echo "Speedup:             N/A"
fi
echo "======================================================================"

# Parity check
echo ""
echo "🔬 Parity Check:"
if [ "$FULL_TRADES" == "$FOCUSED_TRADES" ]; then
    echo "✅ Trade count matches: $FULL_TRADES trades"
else
    echo "⚠️  Trade count differs: $FULL_TRADES (full) vs $FOCUSED_TRADES (batch)"
fi
echo "======================================================================"
