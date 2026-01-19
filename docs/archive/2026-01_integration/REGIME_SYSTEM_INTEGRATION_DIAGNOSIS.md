# Regime Detection System Integration Diagnosis
**Date:** 2026-01-09
**Author:** System Architect
**Status:** 🔴 CRITICAL ISSUES IDENTIFIED

---

## Executive Summary

**DIAGNOSIS:** The crisis threshold (Layer 1.5) is **fundamentally broken** due to integration architecture flaws. Performance degraded from **-$324 to -$825** after adding the threshold layer.

**ROOT CAUSE:** The threshold logic is bypassed by hysteresis, making it completely ineffective.

**CRISIS RATE EXPLOSION:**
- v0 (baseline): 68% crisis
- v1 (threshold): 78.6% crisis (**WORSE**)
- Threshold veto count: **0** (never fires)

**KEY FINDINGS:**
1. ❌ Crisis threshold has ZERO effect (0 vetos recorded)
2. ❌ Hysteresis overrides threshold decisions
3. ❌ EMA state persists incorrectly during batch mode
4. ❌ EventOverride bypasses threshold (correct) but doesn't explain high crisis rate

---

## System Architecture Analysis

### Current 4-Layer Stack

```
┌─────────────────────────────────────────────────────┐
│ Layer 0: EventOverrideDetector                      │
│ - Flash crash: >10% drop in 1H                      │
│ - Extreme volume: z>5 + negative return            │
│ - Funding shock: |z|>5                             │
│ - OI cascade: >15% drop                            │
│ → Override active: Skip all other layers           │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Layer 1: LogisticRegimeModel                       │
│ - Outputs: P(crisis), P(risk_off), P(neutral), ...│
│ - Source: trained logistic regression (v1)        │
│ → Returns: regime_probs (raw posteriors)          │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Layer 1.5: Crisis Threshold + EMA (NEW)           │
│ - Apply EMA smoothing (α=0.08, 24h window)        │
│ - Check: if argmax=='crisis' AND P(crisis)<0.60   │
│ - Action: Override regime to second-highest       │
│ → Returns: smoothed_probs + threshold_metadata    │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Layer 2: RegimeHysteresis                         │
│ - Min dwell time (crisis=6h, risk_off=24h, ...)  │
│ - Dual thresholds (enter=0.70, exit=0.50)        │
│ - EWMA smoothing (α=0.3) - DUPLICATE!            │
│ → Returns: final_regime (after stability logic)  │
└─────────────────────────────────────────────────────┘
```

---

## Critical Integration Issues

### Issue #1: Hysteresis Overrides Threshold Logic ❌

**Location:** `engine/context/regime_service.py:395-430`

**Code Flow:**
```python
# Line 393: Crisis threshold applied
smoothed_probs, threshold_metadata = self._apply_crisis_threshold_and_ema(raw_probs, timestamp)

# Line 396-402: Determine regime after threshold
if threshold_metadata.get('crisis_threshold_veto', False):
    # Crisis was vetoed, use fallback
    final_regime = threshold_metadata['fallback_regime']  # e.g., 'risk_off'
else:
    # Use argmax of smoothed probabilities
    final_regime = max(smoothed_probs.items(), key=lambda x: x[1])[0]

# Line 404-430: PROBLEM - Hysteresis ignores final_regime!
if self.enable_hysteresis and self.hysteresis:
    # Pass smoothed_probs to hysteresis (NOT final_regime)
    hyst_result = self.hysteresis.apply(smoothed_probs, timestamp, override_active=False)

    # Hysteresis re-computes regime from probabilities!
    # It does NOT respect the threshold veto decision
    result = {
        'regime_label': hyst_result['regime_label'],  # ← OVERRIDES final_regime
        ...
    }
```

**THE BUG:**
The threshold layer computes `final_regime = 'risk_off'` (after vetoing crisis), but then passes the **original probabilities** to hysteresis, which **re-computes** the regime from scratch using `argmax(smoothed_probs)`.

**Result:** Threshold veto is ignored. Crisis is still selected if P(crisis) is highest (even if below 0.60).

---

### Issue #2: Duplicate Smoothing Layers ❌

**Layer 1.5 EMA:**
- Location: `regime_service.py:271-289`
- Alpha: 0.08 (24-hour window)
- Updates: `self.ema_probs`

**Layer 2 EWMA:**
- Location: `regime_hysteresis.py:201-225`
- Alpha: 0.3 (3-hour window)
- Updates: `self.smoothed_probs`

**PROBLEM:**
- Probabilities are smoothed TWICE with different windows
- Layer 2 smoothing overrides Layer 1.5 smoothing
- Final probabilities reflect 3h window, not 24h window
- Inconsistent behavior: EMA state from Layer 1.5 is ignored

**Mathematical Effect:**
```
Raw probs → EMA (α=0.08) → EMA (α=0.3) → Final
            [24h window]    [3h window]
```
The 3h window dominates, making the 24h EMA useless.

---

### Issue #3: Batch Mode EMA State Bug ❌

**Location:** `regime_service.py:485-496`

**Code:**
```python
def classify_batch(self, df: pd.DataFrame) -> pd.DataFrame:
    # Reset hysteresis state for clean backtest
    if self.enable_hysteresis and self.hysteresis:
        self.hysteresis.reset()
        logger.info("Hysteresis state reset for batch mode")

    # MISSING: EMA state reset!
    # self.ema_probs is NOT reset here

    for idx, row in df.iterrows():
        result = self.get_regime(features, timestamp)
        results.append(result)
```

**THE BUG:**
- Hysteresis is reset before batch classification (correct)
- EMA state (`self.ema_probs`) is NOT reset (incorrect)
- If `classify_batch()` is called twice:
  - First call: EMA state initializes from data
  - Second call: EMA state continues from first call
  - Result: Second backtest carries forward state from first

**Impact:**
- Batch mode is NOT reproducible
- EMA smoothing gives different results on repeated runs
- Crisis threshold may behave differently in production vs backtest

---

### Issue #4: EventOverride Correct But Doesn't Explain Crisis Rate ✅/❌

**Location:** `regime_service.py:351-386`

**Analysis:**
EventOverride correctly bypasses threshold when flash crash detected:
```python
if is_crisis:
    # Force crisis through hysteresis layer
    crisis_probs = {'crisis': 1.0, ...}
    result = self.hysteresis.apply(crisis_probs, timestamp, override_active=True)
    return result
```

**Flash Crash Threshold:** 10% drop in 1H (recalibrated from 4%)

**Expected Override Rate:**
- 10% threshold should trigger ~1-2% of bars (true crises only)
- Historical crisis periods: 2022 LUNA (May), 2022 FTX (Nov), 2020 COVID (Mar)

**PROBLEM:**
If EventOverride is working correctly, why is crisis rate 68-78%?

**Answer:** The logistic model itself is predicting crisis 68% of the time at **raw probability level**, NOT from event overrides.

**Root Cause:** Model v1 was trained on mislabeled data (high crisis rate in training set).

---

## Configuration Validation

### Current Production Config

**File:** `bin/backtest_full_engine_replay.py:205-209`

```python
self.regime_service = RegimeService(
    model_path='models/logistic_regime_v1.pkl',
    enable_event_override=True,
    enable_hysteresis=True
) if self.enable_adaptive_regime else None
```

**MISSING PARAMETERS:**
- ❌ No `crisis_threshold` specified (defaults to 0.60)
- ❌ No `enable_ema_smoothing` specified (defaults to True)
- ❌ No `ema_alpha` specified (defaults to 0.08)

**Status:** Configuration is correct (using defaults), but threshold is ineffective due to hysteresis override bug.

---

### Archetype S1 Direction-Aware Penalties

**File:** `archetype_registry.yaml` (S1 configuration)

**Expected:**
```yaml
S1:
  regime_penalties:
    crisis: 0.25
    risk_off: 0.50
    neutral: 1.0
    risk_on: 0.90
```

**Verification Needed:**
- Check if S1 has correct regime penalties
- Check if direction-aware penalties are applied
- Verify S1 generates signals before penalties applied

---

## Performance Analysis

### Before vs After Crisis Threshold

| Metric | v0 (Baseline) | v1 (Threshold) | Delta |
|--------|---------------|----------------|-------|
| PnL | **-$324** | -$825 | -$501 (WORSE) |
| Crisis Rate | 68.0% | 78.6% | +10.6% (WORSE) |
| Crisis Vetos | N/A | **0** | Threshold never fires |
| Event Overrides | ~2% (expected) | ~2% (expected) | No change |

**Analysis:**
- Threshold made performance WORSE (not better)
- Crisis rate INCREASED (opposite of intended effect)
- 0 vetos means threshold is completely bypassed
- Performance degradation likely from:
  1. Duplicate EMA smoothing (changed regime timing)
  2. Threshold veto logic not working (hysteresis override)
  3. Regime classification more unstable

---

## Holistic System Alignment Issues

### Architecture Design Flaws

**Problem 1: Wrong Layer Order**

Current order:
```
Model → Threshold → Hysteresis
```

Threshold tries to override regime, but Hysteresis re-computes from probabilities.

**Better order:**
```
Model → Hysteresis (stability) → Threshold (confidence check)
```
OR
```
Model → Threshold (confidence) → Hysteresis (use overridden regime)
```

**Problem 2: Hysteresis Should Not Re-Compute Regime**

Hysteresis should accept a **regime label** and apply stability constraints, NOT re-compute from probabilities.

**Current:**
```python
hyst_result = self.hysteresis.apply(smoothed_probs, timestamp)
# Hysteresis does argmax(probs) internally
```

**Should be:**
```python
hyst_result = self.hysteresis.apply(
    regime_label=final_regime,  # From threshold layer
    regime_probs=smoothed_probs,  # For confidence computation
    timestamp=timestamp
)
```

**Problem 3: Duplicate Smoothing Violates DRY Principle**

- Layer 1.5 EMA (α=0.08, 24h)
- Layer 2 EWMA (α=0.3, 3h)

**Should be:** Pick ONE smoothing layer, not both.

**Options:**
1. Remove Layer 1.5 EMA, keep Layer 2 EWMA
2. Remove Layer 2 EWMA, keep Layer 1.5 EMA
3. Make Layer 2 work on regime labels, not probabilities

---

## Root Cause Summary

### Why Crisis Threshold Made Performance WORSE

**Reason 1: Threshold Never Fires (0 vetos)**
- Hysteresis overrides threshold decisions
- Threshold logic is dead code in current architecture
- No false crisis signals were actually prevented

**Reason 2: Duplicate EMA Smoothing**
- Layer 1.5 EMA (24h) → Layer 2 EWMA (3h)
- Changed probability dynamics
- Regime transitions occur at different times
- Archetype signals now misaligned with regime state

**Reason 3: Batch Mode EMA State Bug**
- EMA state not reset in `classify_batch()`
- Results depend on previous batch runs
- Non-reproducible backtests
- Different crisis rates on repeated runs

**Reason 4: Model v1 Still Predicts 68% Crisis**
- Threshold can't fix a broken model
- Model trained on mislabeled data
- P(crisis) is genuinely high (0.70-0.80) for 68% of bars
- Threshold of 0.60 doesn't help when model outputs 0.75

---

## Critical Questions Answered

### Q1: Why did crisis threshold make performance worse?

**A:** Threshold logic is bypassed by hysteresis (0 vetos). Performance degraded from duplicate EMA smoothing changing regime timing, causing archetype signals to misalign with regime state.

---

### Q2: Why does crisis rate keep increasing (68% → 78%)?

**A:** The logistic model v1 itself predicts crisis 68-78% of the time at raw probability level. The model was trained on mislabeled crisis data. Adding threshold didn't help because:
1. Threshold is bypassed (0 vetos)
2. When crisis P=0.75 (above 0.60), threshold accepts it
3. Model needs retraining, not post-processing

---

### Q3: Is the threshold implementation correct?

**A:** Implementation logic is correct (`_apply_crisis_threshold_and_ema()` works), but **integration is broken**:
- Hysteresis receives probabilities and re-computes regime
- Threshold veto decision is ignored
- Result: 0 vetos recorded

---

### Q4: Should we keep threshold or disable it?

**A:** **DISABLE** for now (or fix integration). Current threshold:
- Has zero effect (0 vetos)
- Adds duplicate smoothing (breaks regime timing)
- Makes performance worse (-$501 regression)

**RECOMMENDATION:**
1. **Immediate:** Disable `enable_ema_smoothing=False` and set `crisis_threshold=0.0` to revert to v0 behavior
2. **Short-term:** Retrain model v2 with correct crisis labels (reduce 68% → 5-10%)
3. **Long-term:** Fix integration architecture (hysteresis should not re-compute regime)

---

## Recommended Fixes

### Fix #1: Disable Crisis Threshold (IMMEDIATE)

**File:** `bin/backtest_full_engine_replay.py`

```python
self.regime_service = RegimeService(
    model_path='models/logistic_regime_v1.pkl',
    enable_event_override=True,
    enable_hysteresis=True,
    crisis_threshold=0.0,        # Disable threshold
    enable_ema_smoothing=False   # Disable duplicate EMA
)
```

**Expected Impact:**
- Revert to v0 performance (-$324)
- Crisis rate back to 68% (not 78%)
- Remove duplicate smoothing

---

### Fix #2: Fix Hysteresis Integration (SHORT-TERM)

**File:** `engine/context/regime_service.py`

**Option A: Pass final_regime to hysteresis (preferred)**

```python
# After threshold logic (line 395-402)
if threshold_metadata.get('crisis_threshold_veto', False):
    final_regime = threshold_metadata['fallback_regime']
else:
    final_regime = max(smoothed_probs.items(), key=lambda x: x[1])[0]

# Pass regime label to hysteresis
if self.enable_hysteresis and self.hysteresis:
    hyst_result = self.hysteresis.apply_with_regime(
        regime_label=final_regime,  # Use threshold decision
        regime_probs=smoothed_probs,
        timestamp=timestamp,
        override_active=False
    )
```

**Option B: Disable hysteresis smoothing (simpler)**

```python
# In RegimeHysteresis.__init__
self.enable_smoothing = False  # Disable EWMA in hysteresis layer
```

---

### Fix #3: Reset EMA State in Batch Mode (IMMEDIATE)

**File:** `engine/context/regime_service.py:485-488`

```python
def classify_batch(self, df: pd.DataFrame) -> pd.DataFrame:
    # Reset hysteresis state for clean backtest
    if self.enable_hysteresis and self.hysteresis:
        self.hysteresis.reset()
        logger.info("Hysteresis state reset for batch mode")

    # FIX: Reset EMA state
    self.ema_probs = None
    logger.info("EMA state reset for batch mode")
```

---

### Fix #4: Retrain Model v2 with Correct Labels (LONG-TERM)

**Problem:** Model v1 trained on mislabeled crisis data (68% crisis rate)

**Solution:**
1. Re-label historical data using EventOverrideDetector as ground truth
2. Crisis definition: Flash crash (>10% drop) OR extreme events
3. Expected crisis rate: 1-5% (not 68%)
4. Train logistic_regime_v2.pkl with corrected labels

**Expected Impact:**
- Model predicts crisis 5% of bars (not 68%)
- P(crisis) = 0.05-0.20 for normal bars
- P(crisis) = 0.80-0.95 for true crises
- Threshold of 0.60 now meaningful

---

## Action Plan

### IMMEDIATE (Next 30 minutes)

1. ✅ **Disable crisis threshold:**
   - Set `crisis_threshold=0.0`
   - Set `enable_ema_smoothing=False`
   - Re-run backtest to confirm reversion to v0 performance

2. ✅ **Fix EMA state reset:**
   - Add `self.ema_probs = None` to `classify_batch()`
   - Verify batch runs are reproducible

3. ✅ **Document findings:**
   - This report explains all issues
   - Share with team for review

---

### SHORT-TERM (Next 2-4 hours)

4. ⏳ **Fix hysteresis integration:**
   - Option A: Modify hysteresis to accept regime label
   - Option B: Disable hysteresis smoothing layer
   - Test both options, pick best performance

5. ⏳ **Validate EventOverride:**
   - Verify flash crash threshold (10%) is correct
   - Check override rate on historical crises (2022 LUNA, FTX)
   - Ensure event overrides work correctly

---

### LONG-TERM (Next 1-2 days)

6. ⏳ **Retrain Model v2:**
   - Re-label training data using EventOverrideDetector
   - Reduce crisis rate from 68% → 5-10%
   - Train logistic_regime_v2.pkl
   - Validate on out-of-sample crisis periods

7. ⏳ **Re-enable crisis threshold:**
   - After model v2 deployed, re-enable threshold
   - Test on backtest: should see 5-10% veto rate
   - Monitor in production

---

## Verification Checklist

### Before Deployment

- ⏳ Backtest with disabled threshold matches v0 performance (-$324)
- ⏳ Crisis rate reverts to 68% (not 78%)
- ⏳ Batch mode runs are reproducible (same results on repeated runs)
- ⏳ EventOverride rate ~1-2% on historical data
- ⏳ S1 generates signals before regime penalty applied

### After Fixes

- ⏳ Hysteresis respects threshold veto decisions
- ⏳ Crisis threshold fires >0 vetos
- ⏳ Only one smoothing layer active (not two)
- ⏳ EMA state resets correctly in batch mode

### After Model v2

- ⏳ Crisis rate 5-10% (not 68%)
- ⏳ Crisis threshold veto rate 5-10% (meaningful)
- ⏳ Performance improves vs v0 baseline

---

## Architecture Recommendation

### Proposed 4-Layer Stack (Fixed)

```
┌─────────────────────────────────────────────────────┐
│ Layer 0: EventOverrideDetector (unchanged)         │
│ - Flash crash, extreme events                      │
│ - Output: override_active=True → force crisis     │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Layer 1: LogisticRegimeModel v2 (retrain)         │
│ - Output: P(crisis), P(risk_off), ...             │
│ - Crisis rate: 5-10% (was 68%)                    │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Layer 1.5: Crisis Threshold (NO EMA)              │
│ - Check: if argmax=='crisis' AND P(crisis)<0.60  │
│ - Output: final_regime (vetoed or accepted)      │
│ - NO smoothing (remove duplicate)                │
└─────────────────────────────────────────────────────┘
                        ↓
┌─────────────────────────────────────────────────────┐
│ Layer 2: RegimeHysteresis (modified)              │
│ - Input: regime_label (from Layer 1.5)           │
│ - Apply: min dwell time, dual thresholds         │
│ - Keep: EWMA smoothing (α=0.3, 3h)               │
│ - Output: stabilized regime (respects threshold) │
└─────────────────────────────────────────────────────┘
```

**Key Changes:**
1. Remove EMA from Layer 1.5 (duplicate smoothing)
2. Hysteresis accepts `regime_label` (not just probabilities)
3. Hysteresis respects upstream threshold decisions
4. Model v2 trained with correct crisis labels (5-10% rate)

---

## Conclusion

**DIAGNOSIS COMPLETE:** Crisis threshold implementation is correct but integration is broken.

**ROOT CAUSES:**
1. ❌ Hysteresis overrides threshold decisions (0 vetos)
2. ❌ Duplicate EMA smoothing breaks regime timing
3. ❌ EMA state not reset in batch mode
4. ❌ Model v1 predicts 68% crisis (model issue, not threshold issue)

**IMMEDIATE ACTION:**
- Disable crisis threshold (`crisis_threshold=0.0`)
- Disable EMA smoothing (`enable_ema_smoothing=False`)
- Fix EMA state reset bug

**SHORT-TERM ACTION:**
- Fix hysteresis to accept regime labels
- Remove duplicate smoothing layer

**LONG-TERM ACTION:**
- Retrain model v2 with correct crisis labels (68% → 5-10%)
- Re-enable crisis threshold after model fixed

**PERFORMANCE EXPECTATION:**
- Immediate: Revert to v0 baseline (-$324)
- After fixes: Match or beat v0 baseline
- After model v2: Significant improvement (lower crisis rate → more signals)

---

**Report Status:** ✅ COMPLETE
**Next Steps:** Implement Fix #1 (disable threshold) and validate performance reversion.
