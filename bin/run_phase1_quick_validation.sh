#!/usr/bin/env bash
################################################################################
# Phase 1 Quick Validation Runner
#
# Runs all quick_test configs on 2022 data (bear market) and extracts metrics
# to identify optimal starting points for Phase 2 optimization.
#
# Usage: ./bin/run_phase1_quick_validation.sh
#
# Outputs:
#   - Individual backtest logs
#   - Comparison table (console)
#   - Summary report (results/phase1_quick_validation/summary.txt)
################################################################################

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
RESULTS_DIR="$PROJECT_ROOT/results/phase1_quick_validation"
BACKTEST_SCRIPT="$PROJECT_ROOT/bin/backtest_knowledge_v2.py"

# Test configuration
ASSET="BTC"
START_DATE="2022-01-01"
END_DATE="2022-12-31"
PERIOD_LABEL="2022_bear_market"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo "================================================================================"
echo -e "${BLUE}Phase 1: Quick Validation Test${NC}"
echo "================================================================================"
echo "Asset:       $ASSET"
echo "Period:      $START_DATE to $END_DATE (2022 Bear Market)"
echo "Output:      $RESULTS_DIR"
echo "================================================================================"
echo ""

# Create output directory
mkdir -p "$RESULTS_DIR"

# Find all quick test configs
CONFIGS=(
    "$PROJECT_ROOT/configs/quick_test_optimized.json"
    "$PROJECT_ROOT/configs/quick_test_optimized_v2.json"
    "$PROJECT_ROOT/configs/quick_validation_fixed.json"
    "$PROJECT_ROOT/configs/quick_fix_2022_regime_override.json"
    "$PROJECT_ROOT/configs/mvp_bull_market_v1.json"
)

# Check for missing configs
VALID_CONFIGS=()
for config in "${CONFIGS[@]}"; do
    if [ -f "$config" ]; then
        VALID_CONFIGS+=("$config")
    else
        echo -e "${YELLOW}Warning: Config not found: $config${NC}"
    fi
done

if [ ${#VALID_CONFIGS[@]} -eq 0 ]; then
    echo -e "${RED}Error: No valid configs found${NC}"
    exit 1
fi

echo "Found ${#VALID_CONFIGS[@]} configs to test"
echo ""

# Results storage
declare -A RESULTS_TRADES
declare -A RESULTS_PF
declare -A RESULTS_WR
declare -A RESULTS_DD
declare -A RESULTS_SHARPE
declare -A RESULTS_RET

# Run backtests
for config in "${VALID_CONFIGS[@]}"; do
    config_name=$(basename "$config" .json)
    log_file="$RESULTS_DIR/${config_name}_${PERIOD_LABEL}.log"
    csv_file="$RESULTS_DIR/${config_name}_${PERIOD_LABEL}.csv"

    echo -e "${BLUE}Running: $config_name${NC}"
    echo "  Log: $log_file"

    # Run backtest with timeout
    if timeout 600 python3 "$BACKTEST_SCRIPT" \
        --asset "$ASSET" \
        --start "$START_DATE" \
        --end "$END_DATE" \
        --config "$config" \
        --export-trades "$csv_file" \
        > "$log_file" 2>&1; then

        # Extract metrics from log file
        trades=$(grep -E "^Total Trades:" "$log_file" | awk '{print $3}' || echo "0")
        pf=$(grep -E "^Profit Factor:" "$log_file" | awk '{print $3}' || echo "0.00")
        wr=$(grep -E "^Win Rate:" "$log_file" | awk '{print $3}' | sed 's/%//' || echo "0.00")
        dd=$(grep -E "^Max Drawdown:" "$log_file" | awk '{print $3}' | sed 's/%//' || echo "0.00")
        sharpe=$(grep -E "^Sharpe Ratio:" "$log_file" | awk '{print $3}' || echo "0.00")
        ret=$(grep -E "^Total Return:" "$log_file" | awk '{print $3}' | sed 's/%//' || echo "0.00")

        # Store results
        RESULTS_TRADES[$config_name]=$trades
        RESULTS_PF[$config_name]=$pf
        RESULTS_WR[$config_name]=$wr
        RESULTS_DD[$config_name]=$dd
        RESULTS_SHARPE[$config_name]=$sharpe
        RESULTS_RET[$config_name]=$ret

        echo -e "  ${GREEN}✓ Complete${NC} - Trades: $trades, PF: $pf, WR: ${wr}%"
    else
        echo -e "  ${RED}✗ Failed (timeout or error)${NC}"
        RESULTS_TRADES[$config_name]="ERR"
        RESULTS_PF[$config_name]="ERR"
        RESULTS_WR[$config_name]="ERR"
        RESULTS_DD[$config_name]="ERR"
        RESULTS_SHARPE[$config_name]="ERR"
        RESULTS_RET[$config_name]="ERR"
    fi

    echo ""
done

# Generate comparison table
echo "================================================================================"
echo -e "${BLUE}Results Summary${NC}"
echo "================================================================================"
echo ""

# Print header
printf "%-40s %8s %8s %8s %8s %8s %10s\n" \
    "Config" "Trades" "PF" "WR%" "DD%" "Sharpe" "Return%"
echo "--------------------------------------------------------------------------------"

# Print results
for config in "${VALID_CONFIGS[@]}"; do
    config_name=$(basename "$config" .json)

    trades="${RESULTS_TRADES[$config_name]}"
    pf="${RESULTS_PF[$config_name]}"
    wr="${RESULTS_WR[$config_name]}"
    dd="${RESULTS_DD[$config_name]}"
    sharpe="${RESULTS_SHARPE[$config_name]}"
    ret="${RESULTS_RET[$config_name]}"

    # Color code based on trade count (target: 25-40)
    if [ "$trades" != "ERR" ] && [ "$trades" -ge 25 ] && [ "$trades" -le 40 ]; then
        color=$GREEN
    elif [ "$trades" != "ERR" ] && [ "$trades" -gt 0 ]; then
        color=$YELLOW
    else
        color=$RED
    fi

    printf "${color}%-40s %8s %8s %8s %8s %8s %10s${NC}\n" \
        "$config_name" "$trades" "$pf" "$wr" "$dd" "$sharpe" "$ret"
done

echo "================================================================================"
echo ""

# Identify best config in target range
echo -e "${BLUE}Recommendations:${NC}"
echo ""

best_config=""
best_pf=0
target_range_configs=0

for config in "${VALID_CONFIGS[@]}"; do
    config_name=$(basename "$config" .json)
    trades="${RESULTS_TRADES[$config_name]}"
    pf="${RESULTS_PF[$config_name]}"

    # Skip errors
    if [ "$trades" = "ERR" ] || [ "$pf" = "ERR" ]; then
        continue
    fi

    # Check if in target range (25-40 trades)
    if [ "$trades" -ge 25 ] && [ "$trades" -le 40 ]; then
        target_range_configs=$((target_range_configs + 1))

        # Track best PF in target range
        if [ $(echo "$pf > $best_pf" | bc -l) -eq 1 ]; then
            best_pf=$pf
            best_config=$config_name
        fi

        echo -e "  ${GREEN}✓${NC} $config_name: $trades trades (PF: $pf) - IN TARGET RANGE"
    else
        if [ "$trades" -lt 25 ]; then
            echo -e "  ${YELLOW}○${NC} $config_name: $trades trades - Too few (need 25-40)"
        else
            echo -e "  ${YELLOW}○${NC} $config_name: $trades trades - Too many (need 25-40)"
        fi
    fi
done

echo ""
if [ $target_range_configs -eq 0 ]; then
    echo -e "${YELLOW}Warning: No configs in target range (25-40 trades)${NC}"
    echo "Consider adjusting thresholds or testing different configs."
else
    echo -e "${GREEN}Found $target_range_configs config(s) in target range${NC}"
    if [ -n "$best_config" ]; then
        echo -e "Best performer: ${GREEN}$best_config${NC} (PF: $best_pf)"
        echo ""
        echo "Recommended for Phase 2 optimization starting point:"
        echo "  Config: configs/${best_config}.json"
        echo "  Baseline PF: $best_pf"
    fi
fi

# Save summary report
SUMMARY_FILE="$RESULTS_DIR/summary_${PERIOD_LABEL}.txt"
{
    echo "Phase 1 Quick Validation Summary"
    echo "================================="
    echo ""
    echo "Test Period: $START_DATE to $END_DATE ($PERIOD_LABEL)"
    echo "Asset: $ASSET"
    echo "Date: $(date)"
    echo ""
    echo "Results:"
    echo "--------"
    printf "%-40s %8s %8s %8s %8s %8s %10s\n" \
        "Config" "Trades" "PF" "WR%" "DD%" "Sharpe" "Return%"
    echo "--------------------------------------------------------------------------------"

    for config in "${VALID_CONFIGS[@]}"; do
        config_name=$(basename "$config" .json)
        printf "%-40s %8s %8s %8s %8s %8s %10s\n" \
            "$config_name" \
            "${RESULTS_TRADES[$config_name]}" \
            "${RESULTS_PF[$config_name]}" \
            "${RESULTS_WR[$config_name]}" \
            "${RESULTS_DD[$config_name]}" \
            "${RESULTS_SHARPE[$config_name]}" \
            "${RESULTS_RET[$config_name]}"
    done

    echo ""
    echo "Target Range Analysis (25-40 trades):"
    echo "--------------------------------------"

    if [ $target_range_configs -eq 0 ]; then
        echo "No configs in target range"
    else
        echo "Configs in range: $target_range_configs"
        if [ -n "$best_config" ]; then
            echo "Best performer: $best_config (PF: $best_pf)"
        fi
    fi

} > "$SUMMARY_FILE"

echo ""
echo "Summary saved to: $SUMMARY_FILE"
echo ""
echo -e "${GREEN}Phase 1 Quick Validation Complete${NC}"
echo "================================================================================"
