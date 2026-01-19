# Quick Fix: Data Quality Issue

## Problem
57.2% of your training data (2018-2021) has **ZERO domain features**.
Only 2022-2024 (42.8%) has complete features.

## Impact
- Training on "2018-2024" is actually training on 2022-2024 only
- Testing on 2018-2021 tests on zeros → poor OOS performance
- Model overfits to temporal pattern (features exist = post-2022)

## Solution A: Quick Fix (5 minutes)

### 1. Add to All Training Scripts

```python
def load_data():
    df = pd.read_parquet('data/features_2018_2024_complete.parquet')

    # CRITICAL FIX: Filter to 2022-2024 (features exist)
    df = df[df.index >= '2022-01-01']

    print(f"⚠️  Using 2022-2024 only: {len(df):,} rows")
    print(f"   (2018-2021 excluded: missing domain features)")

    return df
```

### 2. Scripts to Update

```bash
# Add filter to these scripts:
bin/train_logistic_regime_v4.py
bin/train_continuous_risk_score_v2.py
bin/validate_logistic_regime.py
bin/walk_forward_validation.py

# Search and replace:
grep -l "features_2018_2024" bin/*.py | xargs -I {} echo "Update: {}"
```

### 3. Verify

```bash
python3 bin/quick_data_quality_check.py
# Should show: Using 26,236 rows (2022-2024)
```

## Solution B: Proper Fix (2-3 hours)

### 1. Run Domain Feature Backfill

```bash
# Step 1: Backfill domain features
python3 bin/backfill_domain_features_fast.py

# Step 2: Backfill liquidity features
python3 bin/backfill_liquidity_score.py

# Step 3: Backfill Wyckoff events
python3 bin/backfill_wyckoff_events.py

# Step 4: Combine datasets
python3 bin/combine_historical_datasets.py

# Step 5: Validate
python3 bin/quick_data_quality_check.py
# Should show: 2018-2021 >90% complete
```

### 2. Generate Regime Labels for 2018-2021

```python
# Option 1: Train regime model on 2022-2024, predict on 2018-2021
from engine.context.logistic_regime_model import LogisticRegimeModel

# Load 2022-2024, train model
df_train = pd.read_parquet('data/features_2022_2024_MTF_with_signals.parquet')
model = LogisticRegimeModel()
model.train(df_train)

# Predict on 2018-2021
df_old = pd.read_parquet('data/features_2018_2021_backfilled.parquet')
df_old['regime_label'] = model.predict(df_old)
df_old['regime_confidence'] = model.predict_proba(df_old)

# Save
df_old.to_parquet('data/features_2018_2021_with_regime.parquet')
```

### 3. Validate Quality

```bash
python3 bin/validate_data_quality_2018_2024.py
# Should show: All critical features >90% complete across all years
```

## Recommendation

**Start with Solution A TODAY** (immediate fix, 5 minutes)
**Schedule Solution B for TOMORROW** (proper fix, 2-3 hours)

## Verification

Before training ANY model:

```bash
# Quick check
python3 bin/quick_data_quality_check.py

# Should see:
# ✓ PASSED: All periods have sufficient feature coverage
# OR
# Using 2022-2024 only (26,236 rows)
```

## Files Created

- `/tmp/overfitting_investigation_data_quality.md` - Full analysis
- `bin/validate_data_quality_2018_2024.py` - Comprehensive validation
- `bin/quick_data_quality_check.py` - Quick pre-flight check
- `bin/show_data_quality_summary.sh` - Visual summary

## Key Metrics

| Period | Rows | Features Complete | Usable |
|--------|------|-------------------|--------|
| 2018-2021 | 35,041 | 0% | ✗ NO |
| 2022-2024 | 26,236 | 100% | ✓ YES |

**Effective Training Data:** 26,236 rows (not 61,277)
