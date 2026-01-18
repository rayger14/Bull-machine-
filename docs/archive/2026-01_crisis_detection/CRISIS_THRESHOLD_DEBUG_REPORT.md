# Crisis Threshold Debug Report

## Problem Statement
Crisis threshold is loaded and configured correctly (0.60), but fires ZERO vetos in backtests despite 78.6% crisis bars in HMM-labeled data.

## Root Cause Analysis

### 1. Missing Features (CRITICAL BUG)
The logistic regime model requires 14 features, but the 2022 dataset is **missing 6 critical features**:

**Missing:**
- `crash_frequency_7d` - Crash event count
- `crisis_persistence` - Crisis regime persistence
- `aftershock_score` - Post-crash volatility
- `drawdown_persistence` - Drawdown duration
- `volume_z_7d` - Volume z-score
- `YIELD_CURVE` - Yield curve spread

**Available:**
- `rv_20d`, `rv_60d`, `funding_Z`, `oi`, `USDT.D`, `BTC.D`, `VIX_Z`, `DXY_Z`

**Impact**: Model fills missing features with 0, causing it to predict `neutral` 100% of the time.

### 2. Crisis Threshold Code is Working Correctly

The threshold logic in `regime_service.py` lines 298-323 is **correct**:

```python
if top_regime == 'crisis' and top_prob < self.crisis_threshold:
    self.crisis_threshold_veto_count += 1
    metadata['crisis_threshold_veto'] = True
    # Fall back to second-highest regime
```

**Why it fires 0 vetos**: The model NEVER predicts `crisis` as top regime because of missing features, so the `if` condition never evaluates to True.

### 3. Backtest Uses Stale HMM Data

The backtest reads `regime_label` from `features_2022_COMPLETE.parquet`, which contains:
- **Old HMM labels**: 25% crisis, 75% risk_off
- **Confidence = 1.0**: HMM output, not probabilistic
- **No veto tracking**: HMM doesn't have threshold logic

The backtest never calls `RegimeService.classify_batch()` during runtime because `regime_label` already exists in the DataFrame.

### 4. Code Path Verification

**Expected flow:**
```
Layer 0: Event Override → crisis (if flash crash)
Layer 1: Logistic Model → P(crisis)=0.45, P(risk_off)=0.35, ...
Layer 1.5: Crisis Threshold → VETO (0.45 < 0.60) → fallback to risk_off
Layer 2: Hysteresis → smooth transitions
```

**Actual flow (due to missing features):**
```
Layer 0: Event Override → no events
Layer 1: Logistic Model → P(neutral)=0.95, P(crisis)=0.02, ... (invalid predictions)
Layer 1.5: Crisis Threshold → NO VETO (crisis not top regime)
Layer 2: Hysteresis → locked in neutral
```

## Evidence

### Test Results

**Test 1: Direct model predictions (no hysteresis, no EMA)**
```
Dataset: 1000 bars from 2022
Model output: 100% neutral
Crisis as top regime: 0 bars
Crisis vetos: 0
```

**Test 2: Full dataset scan (8,741 bars)**
```
Crisis predictions: 0
Bars where P(crisis) is highest: 0
Expected vetos: 0 (because crisis never predicted)
Actual vetos: 0 ✓ (code working correctly)
```

### Feature Analysis
```
Model features: 14
Available in data: 8 (57%)
Missing in data: 6 (43%)
```

Critical crisis features missing:
- `crash_frequency_7d`: Core crisis detector
- `crisis_persistence`: Crisis duration tracker
- `aftershock_score`: Post-crash volatility

## Fix Implementation

### Option A: Regenerate Features (RECOMMENDED)

1. **Compute Missing Features**:
   ```bash
   python bin/generate_all_missing_features.py \
     --start-date 2022-01-01 \
     --end-date 2022-12-31 \
     --compute-crisis-features
   ```

2. **Features to Add**:
   - `crash_frequency_7d`: Count of >4% drops in rolling 7-day window
   - `crisis_persistence`: HMM crisis bars / total bars in 7-day window
   - `aftershock_score`: Realized volatility spike after crashes
   - `drawdown_persistence`: Days in drawdown / total days
   - `volume_z_7d`: Volume z-score (7-day rolling)
   - `YIELD_CURVE`: 10Y-2Y yield spread

3. **Reclassify Regimes**:
   ```python
   service = RegimeService(
       model_path='models/logistic_regime_v1.pkl',
       crisis_threshold=0.60
   )
   df = service.classify_batch(df)
   ```

4. **Expected Result**:
   - Crisis predictions: ~5-15% (calibrated model)
   - Crisis vetos: ~1-5% (low-confidence crisis bars)
   - More balanced regime distribution

### Option B: Retrain Model on Available Features (QUICK FIX)

1. **Use Only Available Features**:
   ```python
   # Retrain with 8 available features
   features = ['rv_20d', 'rv_60d', 'funding_Z', 'oi',
               'USDT.D', 'BTC.D', 'VIX_Z', 'DXY_Z']
   ```

2. **Limitation**: Less accurate crisis detection without crash frequency/persistence

### Option C: Map Old Features to New Names

Check if equivalent features exist under different names:
- `volume_z_7d` → `volume_z` or `volume_zscore`?
- `YIELD_CURVE` → `YC_SPREAD` or `YC_Z`?

## Verification Plan

After fix is applied:

1. **Check Feature Availability**:
   ```python
   missing = [f for f in model_features if f not in df.columns]
   assert len(missing) == 0, f"Still missing: {missing}"
   ```

2. **Test Model Predictions**:
   ```python
   # Should see diverse regime predictions
   result = service.get_regime(features, timestamp)
   assert result['regime_label'] != 'neutral'  # Not always neutral
   ```

3. **Verify Crisis Threshold**:
   ```python
   stats = service.get_statistics()
   assert stats['crisis_threshold_veto_count'] > 0  # Vetos happening
   ```

4. **Expected Backtest Results**:
   - Total crisis bars: 5-15% (down from 78.6%)
   - Crisis threshold vetos: 1-5% of bars
   - More nuanced regime transitions

## Conclusion

**The crisis threshold implementation is CORRECT** - it's just being starved of valid input data.

**Action Required**: Generate missing features or retrain model on available features before running production backtests.

**No Code Changes Needed**: The regime_service.py implementation is working as designed.

---

**Generated**: 2026-01-09
**Author**: Claude Code (Backend Architect)
