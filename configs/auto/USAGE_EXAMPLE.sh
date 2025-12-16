#!/bin/bash
# Example workflow for generating and using regime-specific configs

echo "===== REGIME CONFIG GENERATION WORKFLOW ====="
echo ""

# Step 1: Extract thresholds from backtest results
echo "Step 1: Extracting thresholds from winning trades..."
python bin/extract_thresholds.py
echo ""

# Step 2: Generate regime-specific configs
echo "Step 2: Generating regime-specific configs..."
python configs/auto/generate_regime_config.py
echo ""

# Step 3: Validate generated configs
echo "Step 3: Validating generated configs..."
python configs/auto/validate_configs.py
echo ""

# Step 4: View generated thresholds
echo "Step 4: Viewing extracted thresholds..."
cat configs/auto/hmm_thresholds.json | python -m json.tool
echo ""

# Step 5: View config index
echo "Step 5: Viewing config index..."
cat configs/auto/config_index.json | python -m json.tool
echo ""

echo "===== WORKFLOW COMPLETE ====="
echo ""
echo "Generated configs:"
ls -1 configs/auto/config_regime_*.json
echo ""
echo "To run backtests with a specific regime config:"
echo "  python bin/backtest_knowledge_v2.py \\"
echo "    --config configs/auto/config_regime_0.json \\"
echo "    --asset BTC \\"
echo "    --timeframe 1H \\"
echo "    --start 2022-01-01 \\"
echo "    --end 2024-12-31"
