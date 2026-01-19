#!/usr/bin/env bash
#
# One-Click ML Pipeline: Consolidate → Train → Suggest → Validate → Report
#
# This script automates Steps 2-3 of the Meta-Optimizer workflow.
# Run this after Optuna trials complete (Step 1).
#
# Usage:
#   bash bin/pipeline/suggest_and_validate.sh --asset BTC --n-suggestions 12
#

set -e  # Exit on error
set -u  # Exit on undefined variable
set -o pipefail  # Exit on pipe failure

# Parse arguments
ASSET="BTC"
N_SUGGESTIONS=12
BASE_CONFIG="configs/btc_v7_ml_calibrated_2024.json"

while [[ $# -gt 0 ]]; do
  case $1 in
    --asset)
      ASSET="$2"
      shift 2
      ;;
    --n-suggestions)
      N_SUGGESTIONS="$2"
      shift 2
      ;;
    --base-config)
      BASE_CONFIG="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      echo "Usage: $0 --asset BTC --n-suggestions 12 --base-config configs/btc_v7_ml_calibrated_2024.json"
      exit 1
      ;;
  esac
done

# Create timestamped output directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
OUTPUT_DIR="reports/ml/pipeline_${ASSET}_${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"

echo "==========================================="
echo "ML PIPELINE: Meta-Optimizer v2"
echo "==========================================="
echo "Asset: ${ASSET}"
echo "Suggestions: ${N_SUGGESTIONS}"
echo "Base Config: ${BASE_CONFIG}"
echo "Output: ${OUTPUT_DIR}"
echo ""

# Step 2a: Consolidate Trials
echo "========== STEP 2a: Consolidate Trials =========="
python3 bin/consolidate_trials.py \
  --asset "${ASSET}" \
  --output "${OUTPUT_DIR}/config_training_data.csv" \
  --min-rows 200

CONSOLIDATION_STATUS=$?
if [ $CONSOLIDATION_STATUS -ne 0 ]; then
  echo "❌ Consolidation failed!"
  exit 1
fi
echo "✅ Consolidation complete"
echo ""

# Step 2b: Train Meta-Optimizer v2
echo "========== STEP 2b: Train Meta-Optimizer =========="
python3 bin/train/train_config_optimizer.py \
  --data "${OUTPUT_DIR}/config_training_data.csv" \
  --target year_pf \
  --output "${OUTPUT_DIR}/config_optimizer_v2.pkl" \
  --test-size 0.25

TRAIN_STATUS=$?
if [ $TRAIN_STATUS -ne 0 ]; then
  echo "❌ Training failed!"
  exit 1
fi
echo "✅ Training complete"
echo ""

# Step 2c: SHAP Doctrine Check
echo "========== STEP 2c: SHAP Doctrine Check =========="
if [ -f "bin/shap_doctrine_check.py" ]; then
  python3 bin/shap_doctrine_check.py \
    --model "${OUTPUT_DIR}/config_optimizer_v2.pkl" \
    --report "${OUTPUT_DIR}/shap_doctrine_report.json"

  DOCTRINE_STATUS=$?
  if [ $DOCTRINE_STATUS -ne 0 ]; then
    echo "⚠️  SHAP doctrine check failed! Model may not align with trading principles."
    echo "Review: ${OUTPUT_DIR}/shap_doctrine_report.json"
    echo "Proceeding anyway, but validate suggestions carefully."
  else
    echo "✅ SHAP doctrine check passed"
  fi
else
  echo "⚠️  SHAP doctrine check not found (bin/shap_doctrine_check.py)"
  echo "Skipping doctrine validation."
fi
echo ""

# Step 3a: Generate Config Suggestions
echo "========== STEP 3a: Generate Suggestions =========="
python3 bin/suggest_configs.py \
  --model "${OUTPUT_DIR}/config_optimizer_v2.pkl" \
  --n-suggestions "${N_SUGGESTIONS}" \
  --output "${OUTPUT_DIR}/suggested_configs" \
  --base-config "${BASE_CONFIG}" \
  --method differential_evolution

SUGGEST_STATUS=$?
if [ $SUGGEST_STATUS -ne 0 ]; then
  echo "❌ Config suggestion failed!"
  exit 1
fi
echo "✅ Suggestions generated"
echo ""

# Step 3b: Cross-Regime Validation (Top 5)
echo "========== STEP 3b: Cross-Regime Validation =========="
echo "Validating top 5 configs on both 2024 and 2022-2023..."
echo ""

VALIDATION_SUMMARY="${OUTPUT_DIR}/validation_summary.txt"
echo "Config Validation Results" > "${VALIDATION_SUMMARY}"
echo "=========================" >> "${VALIDATION_SUMMARY}"
echo "" >> "${VALIDATION_SUMMARY}"

for CONFIG in "${OUTPUT_DIR}"/suggested_configs/suggested_config_00{1..5}.json; do
  if [ ! -f "${CONFIG}" ]; then
    continue
  fi

  CONFIG_NAME=$(basename "${CONFIG}" .json)
  echo "--- Validating ${CONFIG_NAME} ---"

  # Validate on 2024
  echo "  → 2024 regime..."
  python3 bin/backtest_knowledge_v2.py \
    --asset "${ASSET}" \
    --start 2024-01-01 --end 2024-12-31 \
    --config "${CONFIG}" \
    > "${OUTPUT_DIR}/${CONFIG_NAME}_2024.log" 2>&1

  # Extract metrics from 2024
  TRADES_2024=$(grep -m1 "Trades:" "${OUTPUT_DIR}/${CONFIG_NAME}_2024.log" | awk '{print $2}' || echo "N/A")
  PF_2024=$(grep -m1 "Profit Factor:" "${OUTPUT_DIR}/${CONFIG_NAME}_2024.log" | awk '{print $3}' || echo "N/A")
  WR_2024=$(grep -m1 "Win Rate:" "${OUTPUT_DIR}/${CONFIG_NAME}_2024.log" | awk '{print $3}' || echo "N/A")

  # Validate on 2022-2023
  echo "  → 2022-2023 regime..."
  python3 bin/backtest_knowledge_v2.py \
    --asset "${ASSET}" \
    --start 2022-01-01 --end 2023-12-31 \
    --config "${CONFIG}" \
    > "${OUTPUT_DIR}/${CONFIG_NAME}_2022_2023.log" 2>&1

  # Extract metrics from 2022-2023
  TRADES_2223=$(grep -m1 "Trades:" "${OUTPUT_DIR}/${CONFIG_NAME}_2022_2023.log" | awk '{print $2}' || echo "N/A")
  PF_2223=$(grep -m1 "Profit Factor:" "${OUTPUT_DIR}/${CONFIG_NAME}_2022_2023.log" | awk '{print $3}' || echo "N/A")
  WR_2223=$(grep -m1 "Win Rate:" "${OUTPUT_DIR}/${CONFIG_NAME}_2022_2023.log" | awk '{print $3}' || echo "N/A")

  # Write to summary
  echo "${CONFIG_NAME}:" >> "${VALIDATION_SUMMARY}"
  echo "  2024:      Trades=${TRADES_2024}, PF=${PF_2024}, WR=${WR_2024}" >> "${VALIDATION_SUMMARY}"
  echo "  2022-2023: Trades=${TRADES_2223}, PF=${PF_2223}, WR=${WR_2223}" >> "${VALIDATION_SUMMARY}"
  echo "" >> "${VALIDATION_SUMMARY}"

  echo "  ✓ Complete"
  echo ""
done

echo "✅ Validation complete"
echo ""

# Step 4: Generate README
echo "========== Generating README =========="
cat > "${OUTPUT_DIR}/README.md" <<EOF
# ML Pipeline Results - ${ASSET} - ${TIMESTAMP}

## Pipeline Configuration

- **Asset**: ${ASSET}
- **Base Config**: ${BASE_CONFIG}
- **Suggestions Generated**: ${N_SUGGESTIONS}
- **Validation**: Cross-regime (2024 + 2022-2023)

## Files Generated

- \`config_training_data.csv\` - Consolidated trial data
- \`config_optimizer_v2.pkl\` - Trained meta-optimizer
- \`shap_doctrine_report.json\` - SHAP validation report
- \`suggested_configs/\` - ${N_SUGGESTIONS} ML-suggested configs
- \`validation_summary.txt\` - Cross-regime validation results
- \`suggested_config_*_2024.log\` - 2024 backtest logs
- \`suggested_config_*_2022_2023.log\` - 2022-2023 backtest logs

## Validation Summary

\`\`\`
$(cat "${VALIDATION_SUMMARY}")
\`\`\`

## Promotion Criteria

To promote a config to \`configs/btc_v8_production.json\`:

1. **2024 Performance**:
   - PF ≥ 18.86 (current best - 10%)
   - WR ≥ 75%
   - DD ≈ 0%

2. **2022-2023 Cross-Regime**:
   - PF ≥ 6.0 (baseline historical)
   - WR ≥ 65%
   - DD ≈ 0%

3. **Sensitivity Testing**:
   - Pass local parameter sweep (±15%)
   - Pass ablation tests

## Next Steps

1. Review \`validation_summary.txt\` to identify best config(s)
2. Run sensitivity tests on top 1-2 candidates
3. Promote winner to \`configs/btc_v8_production.json\`
4. Start 30-day paper trading validation

## Model Metrics

See \`analysis/\` directory for:
- Feature importance rankings
- SHAP summary plots
- Training metrics (R², MAE, RMSE)

EOF

echo "✅ README generated"
echo ""

# Final Summary
echo "==========================================="
echo "PIPELINE COMPLETE"
echo "==========================================="
echo "Output directory: ${OUTPUT_DIR}"
echo ""
echo "Review these files:"
echo "  1. ${OUTPUT_DIR}/README.md - Overview"
echo "  2. ${OUTPUT_DIR}/validation_summary.txt - Cross-regime results"
echo "  3. ${OUTPUT_DIR}/analysis/ - SHAP + feature importance"
echo ""
echo "Next: Identify best config and run sensitivity tests."
echo ""
