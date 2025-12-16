#!/bin/bash
#
# Complete Optuna Optimization Workflow Example
#
# This script demonstrates the full workflow:
# 1. Run integration tests
# 2. Quick exploration (50 trials)
# 3. Full optimization (500 trials)
# 4. Analyze results
# 5. Validate on out-of-sample data
#

set -e  # Exit on error

ASSET="ETH"
BASE_CONFIG="configs/profile_default.json"
OUTPUT_DIR="configs/auto"

echo "=================================="
echo "Optuna Optimization Workflow"
echo "=================================="
echo "Asset: $ASSET"
echo "Base Config: $BASE_CONFIG"
echo "Output: $OUTPUT_DIR"
echo ""

# Step 1: Integration test
echo "Step 1: Running integration tests..."
python3 bin/test_optuna_integration.py

if [ $? -ne 0 ]; then
    echo "ERROR: Integration tests failed!"
    exit 1
fi

echo ""
echo "✓ Integration tests passed"
echo ""

# Step 2: Quick exploration (50 trials, ~50 min)
echo "Step 2: Quick exploration (50 trials)..."
echo "This will take approximately 50 minutes..."
echo ""

python3 bin/optuna_thresholds.py \
    --asset $ASSET \
    --trials 50 \
    --base-config $BASE_CONFIG \
    --start 2024-01-01 \
    --end 2024-09-30 \
    --timeout 60 \
    --output $OUTPUT_DIR/quick_exploration.json

echo ""
echo "✓ Quick exploration complete"
echo ""

# Step 3: Full optimization (500 trials, ~8 hours)
echo "Step 3: Full optimization (500 trials)..."
echo "This will take approximately 8 hours..."
echo "Consider running overnight or in a screen/tmux session"
echo ""

read -p "Continue with full optimization? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Skipping full optimization"
    echo "You can run it later with:"
    echo "  python3 bin/optuna_thresholds.py --asset $ASSET --trials 500"
    exit 0
fi

python3 bin/optuna_thresholds.py \
    --asset $ASSET \
    --trials 500 \
    --base-config $BASE_CONFIG \
    --start 2024-01-01 \
    --end 2024-09-30 \
    --timeout 60 \
    --output $OUTPUT_DIR/best_optuna.json

echo ""
echo "✓ Full optimization complete"
echo ""

# Step 4: Analyze results
echo "Step 4: Analyzing optimization results..."
echo ""

if [ -f "$OUTPUT_DIR/best_optuna_study.pkl" ]; then
    python3 bin/analyze_optuna_results.py $OUTPUT_DIR/best_optuna_study.pkl
else
    echo "WARNING: Study file not found - skipping analysis"
fi

echo ""

# Step 5: Validate on out-of-sample data
echo "Step 5: Out-of-sample validation (Q4 2024)..."
echo ""

echo "Baseline performance (original config):"
python3 bin/backtest_knowledge_v2.py \
    --asset $ASSET \
    --start 2024-10-01 \
    --end 2024-12-31 \
    --config $BASE_CONFIG \
    | grep -E "(Profit Factor|Max Drawdown|Sharpe|Total Trades)"

echo ""
echo "Optimized performance (Optuna config):"
python3 bin/backtest_knowledge_v2.py \
    --asset $ASSET \
    --start 2024-10-01 \
    --end 2024-12-31 \
    --config $OUTPUT_DIR/best_optuna.json \
    | grep -E "(Profit Factor|Max Drawdown|Sharpe|Total Trades)"

echo ""
echo "=================================="
echo "Workflow Complete!"
echo "=================================="
echo ""
echo "Results saved to:"
echo "  Config: $OUTPUT_DIR/best_optuna.json"
echo "  Study:  $OUTPUT_DIR/best_optuna_study.pkl"
echo ""
echo "Next steps:"
echo "1. Review config metadata in $OUTPUT_DIR/best_optuna.json"
echo "2. Compare baseline vs optimized performance above"
echo "3. If satisfied, use optimized config for production"
echo ""
