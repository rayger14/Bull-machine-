#!/bin/bash
# Bull Machine v1.7.3 - Progressive CI Test Ladder
# Lighter → Heavier testing approach for CI/CD

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Test level from command line (default: smoke)
TEST_LEVEL=${1:-smoke}
ASSET=${2:-BTC}

echo "🚀 Bull Machine v1.7.3 - CI Test Ladder"
echo "========================================"
echo "Test Level: $TEST_LEVEL"
echo "Asset: $ASSET"
echo ""

# Create timestamped results directory
RESULTS_DIR="results/$(date +%Y%m%d_%H%M%S)"
mkdir -p "$RESULTS_DIR"

case $TEST_LEVEL in
  smoke)
    echo "🔥 Running SMOKE Test (3 days - per commit)"
    echo "Goal: Catch breakages quickly"
    python3 bin/live/live_mock_feed.py \
      --asset "$ASSET" \
      --start 2025-09-01 \
      --end 2025-09-03 \
      --config "configs/live/presets/${ASSET}_vanilla.json" \
      > "$RESULTS_DIR/smoke_test.log" 2>&1

    if [ $? -eq 0 ]; then
      echo -e "${GREEN}✅ Smoke test passed${NC}"
    else
      echo -e "${RED}❌ Smoke test failed${NC}"
      exit 1
    fi
    ;;

  nightly)
    echo "🌙 Running NIGHTLY Test (7 days)"
    echo "Goal: Verify signals + health bands"
    python3 bin/live/paper_trading.py \
      --asset "$ASSET" \
      --start 2025-09-01 \
      --end 2025-09-07 \
      --balance 10000 \
      --config "configs/live/presets/${ASSET}_vanilla.json" \
      > "$RESULTS_DIR/nightly_test.log" 2>&1

    if [ $? -eq 0 ]; then
      echo -e "${GREEN}✅ Nightly test passed${NC}"

      # Run aggregate report and check health
      python3 bin/tools/aggregate_daily_report.py "$RESULTS_DIR"

      if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Health checks failed${NC}"
        exit 1
      fi
    else
      echo -e "${RED}❌ Nightly test failed${NC}"
      exit 1
    fi
    ;;

  weekly)
    echo "📅 Running WEEKLY Test (14 days)"
    echo "Goal: Stability and robustness checks"
    python3 bin/live/paper_trading.py \
      --asset "$ASSET" \
      --start 2025-09-01 \
      --end 2025-09-14 \
      --balance 10000 \
      --config "configs/live/presets/${ASSET}_vanilla.json" \
      > "$RESULTS_DIR/weekly_test.log" 2>&1

    if [ $? -eq 0 ]; then
      echo -e "${GREEN}✅ Weekly test passed${NC}"

      # Run aggregate report
      python3 bin/tools/aggregate_daily_report.py "$RESULTS_DIR"

      # Also run determinism check
      echo ""
      echo "Running determinism check..."
      python3 bin/tools/check_determinism.py

      if [ $? -ne 0 ]; then
        echo -e "${RED}❌ Determinism check failed${NC}"
        exit 1
      fi
    else
      echo -e "${RED}❌ Weekly test failed${NC}"
      exit 1
    fi
    ;;

  full)
    echo "🏁 Running FULL Test (56 days - all available data)"
    echo "Goal: Complete validation using all 1H data"
    python3 bin/live/paper_trading.py \
      --asset "$ASSET" \
      --start 2025-08-06 \
      --end 2025-10-01 \
      --balance 10000 \
      --config "configs/live/presets/${ASSET}_vanilla.json" \
      > "$RESULTS_DIR/full_test.log" 2>&1

    if [ $? -eq 0 ]; then
      echo -e "${GREEN}✅ Full test passed${NC}"

      # Run comprehensive checks
      python3 bin/tools/aggregate_daily_report.py "$RESULTS_DIR"
      python3 bin/tools/check_determinism.py

      # Parse and display key metrics
      echo ""
      echo "📊 Full Test Results:"
      python3 -c "
import json
with open('$RESULTS_DIR/daily_aggregate_$(basename $RESULTS_DIR).json') as f:
    data = json.load(f)
    for asset, metrics in data['assets'].items():
        print(f'   {asset}:')
        print(f'     Return: {metrics[\"return_pct\"]:.2f}%')
        print(f'     Trades: {metrics[\"total_trades\"]}')
        print(f'     Win Rate: {metrics[\"win_rate\"]:.1f}%')
        print(f'     PF: {metrics[\"profit_factor\"]:.2f}' if metrics['profit_factor'] != float('inf') else '     PF: ∞')
"
    else
      echo -e "${RED}❌ Full test failed${NC}"
      exit 1
    fi
    ;;

  *)
    echo -e "${RED}Unknown test level: $TEST_LEVEL${NC}"
    echo "Usage: $0 [smoke|nightly|weekly|full] [ASSET]"
    exit 1
    ;;
esac

echo ""
echo "📁 Results saved to: $RESULTS_DIR"
echo -e "${GREEN}✅ CI Test Ladder Complete${NC}"