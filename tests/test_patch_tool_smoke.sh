#!/bin/bash
#
# CI Smoke Test for patch_feature_columns.py
#
# Validates:
# - Tool can load existing feature stores
# - Health check mode produces valid JSON
# - JSON contains expected keys and structure
#
# Exit codes:
#   0 = success
#   1 = failure

set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

echo "========================================"
echo "PR#1 Smoke Test: patch_feature_columns.py"
echo "========================================"

# Check if feature store exists
FEATURE_STORE="$PROJECT_ROOT/data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet"
if [ ! -f "$FEATURE_STORE" ]; then
    echo "⚠️  Feature store not found: $FEATURE_STORE"
    echo "Skipping smoke test (not a failure - feature store may not exist in CI)"
    exit 0
fi

# Run health-only mode
echo ""
echo "Running health-only mode..."
TMPFILE=$(mktemp)

cd "$PROJECT_ROOT"
python3 bin/patch_feature_columns.py \
    --asset BTC --tf 1H --start 2024-01-01 --end 2024-12-31 \
    --health-only \
    --json-output "$TMPFILE"

# Validate JSON structure
echo ""
echo "Validating JSON output..."

# Check if file exists and is valid JSON
if [ ! -f "$TMPFILE" ]; then
    echo "❌ FAIL: Health JSON not created"
    exit 1
fi

if ! python3 -c "import json; json.load(open('$TMPFILE'))" 2>/dev/null; then
    echo "❌ FAIL: Invalid JSON output"
    exit 1
fi

# Check required keys exist
REQUIRED_KEYS=("timestamp" "total_rows" "columns_patched" "metrics" "health_checks")
for key in "${REQUIRED_KEYS[@]}"; do
    if ! python3 -c "import json; data = json.load(open('$TMPFILE')); assert '$key' in data" 2>/dev/null; then
        echo "❌ FAIL: Missing required key: $key"
        exit 1
    fi
done

# Check metrics for each P0 column
P0_COLS=("tf4h_boms_displacement" "tf1d_boms_strength" "tf4h_fusion_score")
for col in "${P0_COLS[@]}"; do
    if ! python3 -c "import json; data = json.load(open('$TMPFILE')); assert '$col' in data['metrics']" 2>/dev/null; then
        echo "❌ FAIL: Missing metrics for column: $col"
        exit 1
    fi

    if ! python3 -c "import json; data = json.load(open('$TMPFILE')); assert 'non_zero_pct' in data['metrics']['$col']" 2>/dev/null; then
        echo "❌ FAIL: Missing non_zero_pct in metrics for: $col"
        exit 1
    fi
done

echo ""
echo "✅ PASS: All smoke tests passed"
echo ""
echo "Health summary:"
cat "$TMPFILE" | python3 -m json.tool | grep -E "(timestamp|total_rows|non_zero_pct|health_checks)" | head -20

# Cleanup
rm -f "$TMPFILE"

exit 0
