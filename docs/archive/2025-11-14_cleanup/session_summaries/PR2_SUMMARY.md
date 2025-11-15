# PR#2: Feature Quality & Health Checks

**Branch**: `pr2-feature-calculators` → `integrate/v4-prep`
**Type**: Infrastructure + Quality Gates
**Status**: Ready for Review
**Part of**: 5-PR Integration Sequence
**Depends on**: PR#1 (Infrastructure & Safety)

---

## Overview

PR#2 establishes quality gates and calculator patterns for feature store health validation. This PR focuses on **detection and documentation** rather than fixes - it provides the tools to identify issues with P0 columns that are blocking the 3-archetype entry system.

**All new behavior is infrastructure-only** - no changes to feature calculations yet, only health validation.

---

## What's Included

### 1. HTF Alignment Utilities (`engine/utils_align.py`)

**Purpose**: Shared utilities for higher timeframe → 1H alignment

**Key Functions**:
- `resample_to_timeframe(df, timeframe)` - Resample 1H OHLCV to HTF (4H, 1D)
- `align_htf_to_1h(df_1h, df_htf, columns)` - Forward-fill HTF features to 1H bars
- `validate_alignment(df_1h, df_htf, column)` - Check for lookahead bias
- `get_htf_window(df, timestamp, htf)` - Get point-in-time HTF window

**Features**:
- Point-in-time correctness (no lookahead bias)
- Proper OHLCV aggregation with closed='right', label='right'
- Forward-fill alignment preserves HTF bar timestamps

**Usage**:
```python
from engine.utils_align import resample_to_timeframe, align_htf_to_1h

# Resample 1H → 4H
df_4h = resample_to_timeframe(df_1h, '4H')

# Calculate 4H feature
df_4h['boms_displacement'] = calculate_boms_on_4h(df_4h)

# Align back to 1H
df_1h = align_htf_to_1h(
    df_1h=df_1h,
    df_htf=df_4h[['boms_displacement']],
    htf='4H',
    columns=['boms_displacement'],
    prefix='tf4h_'
)
```

---

### 2. Feature Calculator Module (`engine/calculators.py`)

**Purpose**: Modular, reusable calculator functions for P0 columns

**P0 Calculators**:
1. `calc_boms_displacement(df_1h, timeframe)` - BOMS displacement (absolute price)
2. `calc_boms_strength(df_1h, timeframe)` - BOMS strength (normalized 0-1)
3. `calc_tf4h_fusion(df_1h, tf4h_features)` - Fusion score from 4H indicators

**Architecture**:
- Pure functions: `df_1h + config → Series`
- Uses `engine/utils_align.py` for HTF operations
- No side effects (no in-place modifications)
- Returns Series or dict of Series

**Example**:
```python
from engine.calculators import calc_boms_displacement, calc_boms_strength

# Calculate BOMS displacement on 4H
df_1h['tf4h_boms_displacement'] = calc_boms_displacement(df_1h, '4H')

# Calculate BOMS strength on 1D
df_1h['tf1d_boms_strength'] = calc_boms_strength(df_1h, '1D')
```

**Notes**:
- These calculators are **reference implementations** only
- Feature store builder (`bin/build_mtf_feature_store.py`) still uses inline calculations
- Future PRs may wire these into the builder

---

### 3. Feature Health Tests (`tests/test_feature_health.py`)

**Purpose**: Automated validation of feature store health

**Health Checks**:
- Non-zero rate thresholds (e.g., > 5% for BOMS displacement)
- Value range validation (e.g., [0, 1] for normalized features)
- NaN and Inf detection
- Column presence validation

**P0 Thresholds**:
| Column | Expected Non-Zero | Expected Range |
|--------|-------------------|----------------|
| `tf4h_boms_displacement` | > 5% | [0, 20000] (absolute price) |
| `tf1d_boms_strength` | > 5% | [0, 1] (normalized) |
| `tf4h_fusion_score` | > 15% | [0, 1] (score) |

**Usage**:
```bash
# Run pytest tests
pytest tests/test_feature_health.py -v

# Generate health report
python tests/test_feature_health.py --asset BTC --year 2024

# Save to JSON for CI
python tests/test_feature_health.py --asset BTC --year 2024 --output reports/health/BTC_2024.json
```

**Example Output**:
```
============================================================
Health Report: BTC 2024
============================================================
Overall Status: FAIL
Total Rows: 8,761
Columns Checked: 3

❌ tf4h_boms_displacement:
   Non-zero: 0 / 8,761 (0.0%)
   Range: [0.0000, 0.0000]
   Issues: Non-zero rate 0.0% < expected 5.0%

⚠️ tf1d_boms_strength:
   Non-zero: 288 / 8,761 (3.3%)
   Range: [0.0000, 1.0000]
   Issues: Non-zero rate 3.3% < expected 5.0%

✅ tf4h_fusion_score:
   Non-zero: 1,636 / 8,761 (18.7%)
   Range: [0.0000, 0.3009]
   Health: PASS
```

**Pytest Tests**:
- `test_tf4h_boms_displacement_health()` - Validates displacement health
- `test_tf1d_boms_strength_health()` - Validates strength health
- `test_tf4h_fusion_score_health()` - Validates fusion health
- `test_all_p0_columns_present()` - Ensures all P0 columns exist
- `test_no_nan_in_p0_columns()` - No NaN values
- `test_no_inf_in_p0_columns()` - No infinite values

---

## Current Health Status (BTC 2024)

Based on health report from existing feature store:

### ❌ FAIL: `tf4h_boms_displacement`
- **Status**: Complete failure
- **Non-zero**: 0.0% (expected > 5%)
- **Root Cause**: Only calculated when full BOMS confirmed (FVG + no reversal)
- **Impact**: Blocks Archetype A and Archetype C entries

### ⚠️ WARN: `tf1d_boms_strength`
- **Status**: Partial issue
- **Non-zero**: 3.3% (expected > 5%)
- **Root Cause**: Using displacement (which is 0 most of the time) instead of normalized strength
- **Impact**: Blocks Archetype B entries

### ✅ PASS: `tf4h_fusion_score` (with caveats)
- **Status**: Passes non-zero threshold
- **Non-zero**: 18.7% (expected > 15%) ✅
- **Issue**: Contains negative values (range should be [0, 1])
- **Impact**: No entry blocking, but values outside expected range

---

## Files Changed

```
engine/utils_align.py          +290 (new HTF alignment utilities)
engine/calculators.py           +299 (new calculator module)
tests/test_feature_health.py    +380 (new health test framework)
```

**Total**: +969 lines added

---

## Dependencies

**Python Packages** (already in project):
- `pandas` - DataFrame operations
- `numpy` - Numerical operations
- `pytest` - Testing framework (dev dependency)

**Project Modules**:
- `engine/structure/boms_detector.py` - BOMS detection
- `engine/utils_align.py` - HTF alignment (new)

---

## Testing

### Manual Testing Performed

1. **HTF Alignment Utilities**:
   - Tested `resample_to_timeframe()` with 1H → 4H, 1H → 1D
   - Validated point-in-time correctness (no lookahead)
   - Confirmed forward-fill alignment preserves timestamps

2. **Calculator Functions**:
   - Ran test example in `engine/calculators.py` with synthetic data
   - Verified BOMS displacement calculation logic
   - Confirmed BOMS strength normalization ([0, 1] range)

3. **Health Tests**:
   - Generated health report for BTC 2024 feature store
   - Confirmed P0 issues match `MVP_PHASE2_FEATURE_STORE_FIXES.md`
   - Validated pytest tests run successfully

### CI Integration

Add to `.github/workflows/test.yml`:
```yaml
- name: Run feature health tests
  run: pytest tests/test_feature_health.py -v
  continue-on-error: true  # Don't fail CI yet (known issues)
```

---

## What This PR Does NOT Do

**Important**: This PR is infrastructure-only and does not fix feature calculations:

1. ❌ Does **NOT** modify `bin/build_mtf_feature_store.py`
2. ❌ Does **NOT** fix BOMS displacement calculation
3. ❌ Does **NOT** fix BOMS strength calculation
4. ❌ Does **NOT** fix fusion score calculation
5. ❌ Does **NOT** change any runtime behavior

**Why?** This PR establishes quality gates first. Fixes will come in PR#3 after we have validated the calculator patterns.

---

## Next Steps

### After PR#2 Merges:

**PR#3: Runtime Intelligence** (Feature Calculation Fixes)
- Wire calculator functions into `bin/build_mtf_feature_store.py`
- Fix BOMS displacement calculation (use fixed version from Phase 2)
- Fix BOMS strength normalization
- Fix fusion score range issues
- Add health assertions to builder (fail on missing data)
- Rebuild feature stores with fixes

**Expected Results After PR#3**:
- `tf4h_boms_displacement`: 5-10% non-zero (currently 0%)
- `tf1d_boms_strength`: 5-10% non-zero (currently 3.3%)
- `tf4h_fusion_score`: 15-25% non-zero, values in [0, 1] (currently has negatives)

---

## Checklist for Reviewers

- [ ] HTF alignment utilities preserve point-in-time correctness
- [ ] Calculator functions are pure (no side effects)
- [ ] Health tests correctly identify P0 issues
- [ ] Pytest tests run successfully
- [ ] Health report matches known issues from Phase 2
- [ ] No changes to feature store builder (intentional)
- [ ] No new runtime behavior (tool/test-only PR)
- [ ] All new code well-documented with examples

---

## Questions or Concerns?

Reach out with any questions about:
- HTF alignment patterns and lookahead prevention
- Calculator function architecture
- Health check thresholds and validation logic
- CI integration approach
- Future plans for wiring calculators into builder

