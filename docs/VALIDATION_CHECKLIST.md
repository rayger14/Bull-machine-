# Bull Machine V2 Validation Checklist
**Purpose:** Automated checks to ensure documentation accuracy and prevent regression
**Branch:** bull-machine-v2-integration
**Last Updated:** 2025-11-13

---

## Quick Validation (Pre-Commit)

Run this before committing documentation changes:

```bash
bash docs/validate.sh
```

---

## Manual Validation Commands

### 1. Feature Count Validation

**Expected:** 114 features (119 total columns)

```bash
# Verify BTC feature store (2022-2023)
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet')
feature_count = len(df.columns) - 5  # Exclude OHLCV metadata
print(f'Total columns: {len(df.columns)}')
print(f'Feature count: {feature_count}')
print(f'Metadata cols: {list(df.columns[:5])}')
assert feature_count == 114, f'Expected 114 features, got {feature_count}'
print('✅ Feature count correct: 114')
"
```

**Expected Output:**
```
Total columns: 119
Feature count: 114
Metadata cols: ['open', 'high', 'low', 'close', 'volume']
✅ Feature count correct: 114
```

---

### 2. Regime Classifier Feature Count

**Expected:** 19 macro features

```bash
# Verify regime classifier configuration
python3 -c "
import json
with open('configs/frozen/btc_1h_v2_baseline.json') as f:
    cfg = json.load(f)
feature_order = cfg['regime_classifier']['feature_order']
print(f'Regime features: {len(feature_order)}')
print(f'Feature list: {feature_order}')
assert len(feature_order) == 19, f'Expected 19 regime features, got {len(feature_order)}'
print('✅ Regime classifier config correct: 19 features')
"
```

**Expected Output:**
```
Regime features: 19
Feature list: ['VIX_Z', 'DXY_Z', 'YC_SPREAD', 'YC_Z', 'BTC.D_Z', 'USDT.D_Z', 'RV_7', 'RV_20', 'RV_30', 'RV_60', 'funding_Z', 'OI_CHANGE', 'TOTAL_RET', 'TOTAL2_RET', 'TOTAL3_RET', 'ALT_ROTATION', 'VOL_TERM', 'SKEW_25D', 'PERP_BASIS']
✅ Regime classifier config correct: 19 features
```

---

### 3. Gold Standard Test (2024 Q1-Q3)

**Expected:** 17 trades, Profit Factor ~6.17

```bash
# Set deterministic seed for reproducibility
export PYTHONHASHSEED=0

# Run gold standard backtest
python3 bin/backtest_knowledge_v2.py \
  --asset BTC \
  --start 2024-01-01 \
  --end 2024-09-30 \
  --config configs/frozen/btc_1h_v2_baseline.json \
  --output /tmp/validation_test.json

# Extract key metrics
python3 -c "
import json
with open('/tmp/validation_test.json') as f:
    results = json.load(f)
print(f\"Trades: {results['total_trades']}\")
print(f\"Profit Factor: {results['profit_factor']:.2f}\")
print(f\"Win Rate: {results['win_rate']:.1%}\")
print(f\"Net PNL: \${results['net_pnl']:.2f}\")

# Validate key metrics
assert results['total_trades'] == 17, f\"Expected 17 trades, got {results['total_trades']}\"
assert 6.0 <= results['profit_factor'] <= 6.5, f\"PF out of range: {results['profit_factor']}\"
print('✅ Gold standard test passed')
"
```

**Expected Output:**
```
Trades: 17
Profit Factor: 6.17
Win Rate: 76.5%
Net PNL: $1285.42
✅ Gold standard test passed
```

**Note:** Profit factor may vary slightly (±0.3) due to floating-point precision, but trade count must be exact.

---

### 4. Documentation Consistency Check

**Check that all documentation references correct feature counts:**

```bash
# Search for outdated "89 features" or "69 features" in docs
echo "Searching for outdated feature counts..."

# Should return NO results from production docs
grep -r "89 features\|69 features" docs/ bin/*.py \
  --include="*.md" --include="*.py" \
  --exclude="CHANGELOG.md" \
  --exclude="CLEANUP_REPORT.md" \
  --exclude="VALIDATION_CHECKLIST.md" 2>/dev/null

# If output is empty, documentation is consistent
# If output is found, update those files to reference "114 features"

echo "✅ Documentation consistency check complete"
```

**Expected:** No output (all references updated)

---

### 5. Config File Validation

**Verify baseline config is valid:**

```bash
# Load and validate baseline config
python3 -c "
import json
from pathlib import Path

config_path = Path('configs/frozen/btc_1h_v2_baseline.json')
assert config_path.exists(), f'Config not found: {config_path}'

with open(config_path) as f:
    cfg = json.load(f)

# Check required sections
required_sections = [
    'regime_classifier',
    'gates_regime_profiles',
    'archetypes',
    'thresholds'
]

for section in required_sections:
    assert section in cfg, f'Missing config section: {section}'
    print(f'✅ {section}: present')

# Check archetype flags
archetype_flags = cfg['archetypes']
enabled_archetypes = [k for k, v in archetype_flags.items() if k.startswith('enable_') and v]
print(f'✅ Enabled archetypes: {len(enabled_archetypes)}')

# Check regime profiles
regime_profiles = cfg['gates_regime_profiles']
assert 'risk_on' in regime_profiles, 'Missing risk_on profile'
assert 'risk_off' in regime_profiles, 'Missing risk_off profile'
print(f'✅ Regime profiles: {list(regime_profiles.keys())}')

print('✅ Config validation passed')
"
```

**Expected Output:**
```
✅ regime_classifier: present
✅ gates_regime_profiles: present
✅ archetypes: present
✅ thresholds: present
✅ Enabled archetypes: 14
✅ Regime profiles: ['risk_on', 'neutral', 'risk_off', 'crisis']
✅ Config validation passed
```

---

### 6. Feature Store Schema Validation

**Verify feature store has expected structure:**

```bash
# Check feature store schema
python3 -c "
import pandas as pd
from pathlib import Path

# Load feature store
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet')

# Check metadata columns
metadata_cols = ['open', 'high', 'low', 'close', 'volume']
for col in metadata_cols:
    assert col in df.columns, f'Missing metadata column: {col}'
print(f'✅ Metadata columns present: {metadata_cols}')

# Check macro features
macro_features = [
    'VIX_Z', 'DXY_Z', 'YC_SPREAD', 'BTC.D_Z', 'USDT.D_Z',
    'funding_Z', 'OI_CHANGE', 'TOTAL_RET', 'PERP_BASIS', 'VOL_TERM'
]
for col in macro_features:
    assert col in df.columns, f'Missing macro feature: {col}'
print(f'✅ Macro features present (sample): {len(macro_features)} checked')

# Check technical features
tech_features = ['atr_14', 'rsi_14', 'adx_14', 'sma_20', 'ema_21']
for col in tech_features:
    assert col in df.columns, f'Missing technical feature: {col}'
print(f'✅ Technical features present (sample): {len(tech_features)} checked')

# Check data integrity
assert len(df) > 0, 'Feature store is empty'
assert df.isnull().sum().sum() < len(df) * 0.05, 'Too many null values (>5%)'
print(f'✅ Data integrity: {len(df)} rows, {df.isnull().sum().sum()} nulls')

print('✅ Feature store schema validation passed')
"
```

**Expected Output:**
```
✅ Metadata columns present: ['open', 'high', 'low', 'close', 'volume']
✅ Macro features present (sample): 10 checked
✅ Technical features present (sample): 5 checked
✅ Data integrity: 17520 rows, 142 nulls
✅ Feature store schema validation passed
```

---

## Automated Validation Script

**Create:** `docs/validate.sh`

```bash
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
assert len(df.columns) - 5 == 114, f'Expected 114 features, got {len(df.columns) - 5}'
" && echo "✅ Feature count: 114" || exit 1

# 2. Regime Classifier
echo "[2/4] Validating regime classifier config..."
python3 -c "
import json
cfg = json.load(open('configs/frozen/btc_1h_v2_baseline.json'))
assert len(cfg['regime_classifier']['feature_order']) == 19
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
assert results['total_trades'] == 17, f\"Expected 17 trades, got {results['total_trades']}\"
assert 6.0 <= results['profit_factor'] <= 6.5, f\"PF out of range: {results['profit_factor']}\"
print(f\"✅ Gold standard: {results['total_trades']} trades, PF {results['profit_factor']:.2f}\")
" || exit 1
else
    echo "[3/4] Skipping gold standard test (set SKIP_GOLD_STANDARD=0 to enable)"
fi

# 4. Documentation Consistency
echo "[4/4] Checking documentation consistency..."
if grep -r "89 features\|69 features" docs/*.md bin/*.py \
   --exclude="CLEANUP_REPORT.md" \
   --exclude="VALIDATION_CHECKLIST.md" 2>/dev/null | grep -v "^Binary"; then
    echo "❌ Found outdated feature counts in documentation"
    exit 1
else
    echo "✅ Documentation consistent"
fi

echo ""
echo "=========================================="
echo "✅ All validation checks passed!"
echo "=========================================="
```

**Usage:**
```bash
# Full validation (including gold standard test)
bash docs/validate.sh

# Quick validation (skip slow gold standard test)
SKIP_GOLD_STANDARD=1 bash docs/validate.sh
```

---

## Regression Detection

### What to Watch For

1. **Feature Count Drift**
   - If feature count changes, update all documentation
   - Run: `python3 -c "import pandas as pd; print(len(pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet').columns) - 5)"`

2. **Gold Standard Deviation**
   - Trade count should always be 17 (exact)
   - Profit factor should be 6.17 ± 0.3
   - If deviates, check for code changes in:
     - `bin/backtest_knowledge_v2.py`
     - `engine/archetypes/logic_v2_adapter.py`
     - `engine/archetypes/threshold_policy.py`

3. **Config File Changes**
   - Never modify `configs/frozen/btc_1h_v2_baseline.json` without documenting
   - Create new config variants in `configs/` instead

4. **Feature Store Corruption**
   - Check for null values: `df.isnull().sum().sum()`
   - Check for constant columns: `df.std() == 0`
   - Check for duplicates: `df.duplicated().sum()`

---

## Troubleshooting

### Issue: Feature count mismatch
```bash
# Expected: 114, Got: XXX

# 1. Check which features are missing/extra
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet')
print('Columns:', sorted(df.columns))
print('Total:', len(df.columns))
print('Features (excl OHLCV):', len(df.columns) - 5)
"

# 2. Rebuild feature store if needed
python3 bin/build_mtf_feature_store.py --asset BTC --start 2022-01-01 --end 2023-12-31
```

### Issue: Gold standard test fails
```bash
# 1. Check PYTHONHASHSEED is set
echo $PYTHONHASHSEED  # Should be 0

# 2. Check for data corruption
python3 -c "
import pandas as pd
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet')
print('Nulls:', df.isnull().sum().sum())
print('Rows:', len(df))
"

# 3. Check config is unchanged
git diff configs/frozen/btc_1h_v2_baseline.json
```

### Issue: Documentation still references old counts
```bash
# Find all occurrences
grep -r "89 features\|69 features" . --include="*.md" --include="*.py"

# Update manually or use sed (CAREFUL!)
# sed -i '' 's/89 features/114 features/g' docs/BULL_MACHINE_V2_PIPELINE.md
```

---

## Pre-Commit Hook (Optional)

Add to `.git/hooks/pre-commit`:

```bash
#!/bin/bash
# Auto-validate before commit

echo "Running Bull Machine V2 validation..."
SKIP_GOLD_STANDARD=1 bash docs/validate.sh

if [ $? -ne 0 ]; then
    echo "❌ Validation failed. Commit aborted."
    echo "Fix issues or run: git commit --no-verify"
    exit 1
fi

echo "✅ Validation passed. Proceeding with commit."
```

**Enable:**
```bash
chmod +x .git/hooks/pre-commit
```

---

## CI/CD Integration

**GitHub Actions** (`.github/workflows/validate.yml`):

```yaml
name: Validate Bull Machine V2
on: [push, pull_request]

jobs:
  validate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: '3.10'
      - name: Install dependencies
        run: pip install pandas numpy
      - name: Run validation
        run: SKIP_GOLD_STANDARD=1 bash docs/validate.sh
```

---

## Summary

✅ **Feature Count:** 114 features (119 columns)
✅ **Regime Classifier:** 19 macro features
✅ **Gold Standard:** 17 trades, PF 6.17
✅ **Documentation:** Consistent across all files

**Run before every commit:**
```bash
SKIP_GOLD_STANDARD=1 bash docs/validate.sh
```

**Last Validated:** 2025-11-13
**Branch:** bull-machine-v2-integration
**Status:** All checks passing ✅
