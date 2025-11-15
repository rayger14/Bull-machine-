#!/bin/bash
#
# Quick Regime Routing Validation Script
# Runs essential checks before production deployment
#
# Usage:
#   ./bin/validate_regime_routing_quick.sh
#

set -e  # Exit on error

echo "=========================================="
echo "REGIME ROUTING QUICK VALIDATION"
echo "=========================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Step 1: Check if config files exist
echo "[1/4] Checking config files..."
if [ -f "configs/regime_routing_production_v1.json" ]; then
    echo -e "${GREEN}✓${NC} regime_routing_production_v1.json exists"
else
    echo -e "${RED}✗${NC} regime_routing_production_v1.json MISSING"
    exit 1
fi

if [ -f "configs/baseline_btc_bull_pf20_biased_20pct_no_ml.json" ]; then
    echo -e "${GREEN}✓${NC} baseline config exists"
else
    echo -e "${RED}✗${NC} baseline config MISSING"
    exit 1
fi

# Step 2: Verify regime classification on 2022
echo ""
echo "[2/4] Validating 2022 regime classification..."
echo "   → This will take ~2 minutes"
echo ""

REGIME_COUNT=$(python3 bin/backtest_knowledge_v2.py \
    --asset BTC --start 2022-01-01 --end 2022-12-31 \
    --config configs/baseline_btc_bull_pf20_biased_20pct_no_ml.json \
    2>&1 | grep -i "regime_label" | wc -l)

if [ "$REGIME_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓${NC} Regime data found in backtest logs"

    # Count risk_off occurrences
    RISK_OFF_COUNT=$(python3 bin/backtest_knowledge_v2.py \
        --asset BTC --start 2022-01-01 --end 2022-12-31 \
        --config configs/baseline_btc_bull_pf20_biased_20pct_no_ml.json \
        2>&1 | grep "regime_label.*risk_off" | wc -l)

    if [ "$RISK_OFF_COUNT" -gt 100 ]; then
        echo -e "${GREEN}✓${NC} 2022 has significant risk_off periods ($RISK_OFF_COUNT occurrences)"
    else
        echo -e "${YELLOW}⚠${NC} 2022 may not have enough risk_off periods ($RISK_OFF_COUNT occurrences)"
        echo "   → Check regime classifier calibration"
    fi
else
    echo -e "${YELLOW}⚠${NC} Could not verify regime classification (logs incomplete)"
    echo "   → Manual verification recommended"
fi

# Step 3: Check simulation script
echo ""
echo "[3/4] Checking simulation script..."
if [ -x "bin/simulate_regime_routing_2022.py" ]; then
    echo -e "${GREEN}✓${NC} simulate_regime_routing_2022.py is executable"
else
    echo -e "${YELLOW}⚠${NC} Making simulation script executable..."
    chmod +x bin/simulate_regime_routing_2022.py
fi

# Step 4: Verify Python dependencies
echo ""
echo "[4/4] Checking Python environment..."
python3 -c "import pandas, numpy, json" 2>/dev/null
if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓${NC} Required Python libraries available"
else
    echo -e "${RED}✗${NC} Missing required Python libraries"
    exit 1
fi

# Summary
echo ""
echo "=========================================="
echo "VALIDATION COMPLETE"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "  1. Run full simulation:"
echo "     python3 bin/simulate_regime_routing_2022.py --scenario all"
echo ""
echo "  2. Review results in:"
echo "     results/regime_routing_simulation/"
echo ""
echo "  3. If validation succeeds (2022 PF >1.2):"
echo "     → Deploy routing config to production"
echo ""
echo "  4. If validation fails:"
echo "     → Run frontier optimization (see REGIME_ROUTING_IMPLEMENTATION_PLAN.md)"
echo ""
