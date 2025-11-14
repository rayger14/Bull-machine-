#!/bin/bash
# Bull Machine V2 Pre-Commit Validation
# Run before committing documentation changes

set -e  # Exit on first error

echo "=========================================="
echo "Bull Machine V2 Validation Checklist"
echo "=========================================="
echo ""

# 1. Feature Count
echo "[1/4] Validating feature count..."
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet')
feature_count = len(df.columns) - 5
assert feature_count == 114, f'Expected 114 features, got {feature_count}'
print(f'   Total columns: {len(df.columns)}, Features: {feature_count}')
" && echo "✅ Feature count: 114" || exit 1

# 2. Regime Classifier
echo "[2/4] Validating regime classifier config..."
python3 -c "
import json
cfg = json.load(open('configs/frozen/btc_1h_v2_baseline.json'))
regime_features = len(cfg['regime_classifier']['feature_order'])
assert regime_features == 19, f'Expected 19 regime features, got {regime_features}'
print(f'   Regime features: {regime_features}')
" && echo "✅ Regime features: 19" || exit 1

# 3. Gold Standard Test (optional, slow)
if [ "$SKIP_GOLD_STANDARD" != "1" ]; then
    echo "[3/4] Running gold standard test (2024 Q1-Q3)..."
    PYTHONHASHSEED=0 python3 bin/backtest_knowledge_v2.py \
      --asset BTC \
      --start 2024-01-01 \
      --end 2024-09-30 \
      --config configs/frozen/btc_1h_v2_baseline.json \
      --output /tmp/validation_test.json > /dev/null 2>&1

    python3 -c "
import json
results = json.load(open('/tmp/validation_test.json'))
trades = results['total_trades']
pf = results['profit_factor']
assert trades == 17, f'Expected 17 trades, got {trades}'
assert 6.0 <= pf <= 6.5, f'PF out of range: {pf}'
print(f'   Trades: {trades}, PF: {pf:.2f}')
" && echo "✅ Gold standard passed" || exit 1
else
    echo "[3/4] Skipping gold standard test (set SKIP_GOLD_STANDARD=0 to enable)"
fi

# 4. Documentation Consistency
echo "[4/4] Checking documentation consistency..."
OUTDATED=$(grep -r "89 features\|69 features" docs/*.md bin/*.py 2>/dev/null | \
  grep -v "CLEANUP_REPORT.md" | \
  grep -v "VALIDATION_CHECKLIST.md" | \
  grep -v "validate.sh" | \
  grep -v "^Binary" || true)

if [ -n "$OUTDATED" ]; then
    echo "❌ Found outdated feature counts in documentation:"
    echo "$OUTDATED"
    exit 1
else
    echo "✅ Documentation consistent"
fi

echo ""
echo "=========================================="
echo "✅ All validation checks passed!"
echo "=========================================="
