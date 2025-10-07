#!/bin/bash
# Bull Machine v1.7.3 - Baseline Paper Trading Tests
# Using actual BTC data available (1H: Aug 6 - Oct 1, 2025)

set -e
echo "🚀 Bull Machine v1.7.3 - Baseline Paper Trading Tests"
echo "======================================================"
echo ""

# Create dated results directory
RESULTS_DIR="results/$(date +%Y%m%d)"
mkdir -p "$RESULTS_DIR"

# Test 1: BTC Short Window (3 days) - Smoke Test
echo "📊 Test 1: BTC 3-Day Smoke Test"
python3 bin/live/paper_trading.py \
  --asset BTC \
  --start 2025-09-01 \
  --end 2025-09-03 \
  --balance 10000 \
  --config configs/live/presets/BTC_vanilla.json \
  > "$RESULTS_DIR/smoke_test_3day.log" 2>&1

echo "   ✓ 3-day smoke test complete"

# Test 2: BTC Medium Window (7 days) - Nightly Sanity
echo "📊 Test 2: BTC 7-Day Nightly Test"
python3 bin/live/paper_trading.py \
  --asset BTC \
  --start 2025-09-01 \
  --end 2025-09-07 \
  --balance 10000 \
  --config configs/live/presets/BTC_vanilla.json \
  > "$RESULTS_DIR/nightly_test_7day.log" 2>&1

echo "   ✓ 7-day nightly test complete"

# Test 3: BTC Extended Window (14 days) - Weekly Regression
echo "📊 Test 3: BTC 14-Day Weekly Test"
python3 bin/live/paper_trading.py \
  --asset BTC \
  --start 2025-09-01 \
  --end 2025-09-14 \
  --balance 10000 \
  --config configs/live/presets/BTC_vanilla.json \
  > "$RESULTS_DIR/weekly_test_14day.log" 2>&1

echo "   ✓ 14-day weekly test complete"

# Test 4: BTC Maximum Window (Full available 1H data)
echo "📊 Test 4: BTC Full Window Test (Aug 6 - Oct 1)"
python3 bin/live/paper_trading.py \
  --asset BTC \
  --start 2025-08-06 \
  --end 2025-10-01 \
  --balance 10000 \
  --config configs/live/presets/BTC_vanilla.json \
  > "$RESULTS_DIR/full_test_56day.log" 2>&1

echo "   ✓ Full window test complete (56 days)"

echo ""
echo "✅ All baseline tests complete!"
echo "📁 Results saved to: $RESULTS_DIR"
echo ""
echo "Running aggregate report..."
python3 bin/tools/aggregate_daily_report.py "$RESULTS_DIR"