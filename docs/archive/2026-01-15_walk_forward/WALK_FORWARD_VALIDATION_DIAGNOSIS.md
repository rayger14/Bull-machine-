# Walk-Forward Validation Diagnosis Report

**Date**: 2026-01-16
**Archetype**: S1 (Liquidity Vacuum)
**Dataset**: `data/features_2018_2024_UPDATED.parquet` (61,277 rows, 2018-2024)

---

## Executive Summary

**FINDING**: The 82% OOS degradation issue is **NOT caused by missing features**. The root cause is that the walk-forward validation script (`bin/walk_forward_validation.py`) uses **simplified signal generation logic** that does NOT match the actual archetype implementation.

**STATUS**: ❌ Walk-forward validation is INVALID for production assessment

---

## Test Results

### Walk-Forward Validation (Current Run)
```
Total Windows: 39
Windows with 0 trades: 21 (54%)
OOS Degradation: 82.06%
OOS Sharpe: 0.27
Profitable Windows: 9/39 (23%)
Total Trades: 41
```

### Key Metrics Comparison

| Metric | Target | Actual | Status |
|--------|--------|--------|--------|
| OOS Degradation | <20% | 82.1% | ❌ FAIL |
| Profitable Windows | >60% | 23% | ❌ FAIL |
| OOS Sharpe | >0.5 | 0.27 | ❌ FAIL |
| Zero Trade Windows | 0 | 21 | ❌ FAIL |
| Catastrophic Failures | 0 | 0 | ✅ PASS |

---

## Root Cause Analysis

### Issue 1: Simplified Signal Logic
The walk-forward script uses this logic:
```python
# From bin/walk_forward_validation.py lines 428-440
if all(col in data.columns for col in ['liquidity_score', 'volume_z']):
    conditions = (
        (data['liquidity_score'] <= liquidity_max) &
        (data['volume_z'] >= volume_z_min) &
        (data['close'] < data['close'].shift(1))  # Selling pressure
    )
    signals[conditions] = 1
```

**PROBLEM**: This does NOT match the actual archetype!

### Issue 2: Feature Mismatch
The real S1 archetype (`engine/strategies/archetypes/bear/liquidity_vacuum.py`) uses:

| Feature | Walk-Forward Uses | Real Archetype Uses | Available in Dataset |
|---------|-------------------|---------------------|---------------------|
| Volume | `volume_z` | `volume_zscore` | ❌ NO (only `volume_z`) |
| Liquidity | `liquidity_score` | `liquidity_drain_pct` (primary) | ✅ YES |
| Wick | ❌ NOT USED | `wick_lower_ratio` | ❌ NO |
| Crisis | ❌ NOT USED | `VIX_Z`, `DXY_Z`, `funding_Z` | ✅ YES |

### Issue 3: Missing Fusion Score
The real archetype computes a **weighted fusion score**:
```python
fusion_score = (
    0.40 * liquidity_score +
    0.30 * volume_score +
    0.20 * wick_score +
    0.10 * crisis_score
)
```

The walk-forward script uses **simple boolean conditions** instead.

---

## Feature Coverage Analysis

### Dataset Statistics (2018-2021)
```
Total rows: 35,041
liquidity_score: 168 NaN (0.5%)
volume_z: 167 NaN (0.5%)
Valid data coverage: 99.5%
```

### Feature Availability
```
✅ liquidity_drain_pct: YES
✅ liquidity_score: YES (fallback)
✅ VIX_Z: YES
✅ DXY_Z: YES
✅ funding_Z: YES
❌ volume_zscore: NO (have volume_z instead)
❌ wick_lower_ratio: NO
```

---

## Signal Generation Test

Using walk-forward simplified logic on 2018-2021:
```python
liquidity_max = 0.192
volume_z_min = 1.695

signal_conditions = (
    (data['liquidity_score'] <= liquidity_max) &
    (data['volume_z'] >= volume_z_min)
)
```

**Result**: **0 signals generated** (out of 34,873 valid rows)

This proves the simplified logic is fundamentally broken.

---

## Impact Assessment

### Why This Matters
1. **Invalid Production Readiness Assessment**: The 82% degradation metric is meaningless because it's testing the wrong thing
2. **Cannot Trust Walk-Forward Results**: The script doesn't validate what will actually run in production
3. **Wasted Optimization Effort**: Any parameter tuning based on this script will not improve real performance

### What We Learned
1. ✅ Features ARE backfilled successfully (99.5% coverage in 2018-2021)
2. ❌ Walk-forward script does NOT implement real archetype logic
3. ❌ Missing critical features: `volume_zscore`, `wick_lower_ratio`

---

## Recommendations

### Immediate Actions (Priority 1)
1. **Fix Walk-Forward Script**: Update signal generation to match real archetype implementation
2. **Add Missing Features**: Engineer `volume_zscore` and `wick_lower_ratio` to the dataset
3. **Re-run Validation**: Test again with correct logic

### Alternative Approach (Priority 2)
Instead of fixing the walk-forward script, use the **real engine** for validation:
```bash
# Use bin/backtest_with_real_signals.py instead
python3 bin/backtest_with_real_signals.py \
  --archetype S1 \
  --data data/features_2018_2024_UPDATED.parquet \
  --config configs/s1_multi_objective_production.json
```

This would test the ACTUAL production code, not a simplified proxy.

### Feature Engineering (Priority 3)
Add missing features to enable proper validation:
```python
# volume_zscore (different from volume_z?)
data['volume_zscore'] = (data['volume'] - data['volume'].rolling(168).mean()) / \
                        data['volume'].rolling(168).std()

# wick_lower_ratio
data['wick_lower_ratio'] = (data['low'] - data['close']) / (data['high'] - data['low'])
```

---

## GO/NO-GO Decision

### Current Status: ⚠️ INCONCLUSIVE

**Reasoning**:
- ❌ Cannot assess OOS degradation because walk-forward script is invalid
- ✅ Feature backfilling WAS successful (99.5% coverage)
- ❌ Missing critical features needed by real archetype
- ❌ Cannot validate production readiness without correct testing

### Next Steps Before Re-Evaluation
1. Engineer missing features (`volume_zscore`, `wick_lower_ratio`)
2. Use real backtest script instead of walk-forward proxy
3. Validate against 2018-2021 with actual archetype logic
4. THEN make GO/NO-GO decision based on real performance

---

## Files Changed

### New Diagnostic Files
- `WALK_FORWARD_VALIDATION_DIAGNOSIS.md` (this file)
- `walk_forward_validation_results.txt` (validation output log)

### Existing Files Tested
- `bin/walk_forward_validation.py` (found to be invalid)
- `configs/s1_multi_objective_production.json` (config used)
- `data/features_2018_2024_UPDATED.parquet` (dataset tested)

---

## Conclusion

The walk-forward validation FAILED, but **not because the features are missing**. The test itself is flawed because it uses simplified logic that doesn't match production.

**Bottom Line**: We need to either:
1. Fix the walk-forward script to use real archetype logic, OR
2. Use the actual backtest engine for validation

The feature backfilling work WAS successful. The validation methodology needs to be fixed before we can assess OOS degradation.
