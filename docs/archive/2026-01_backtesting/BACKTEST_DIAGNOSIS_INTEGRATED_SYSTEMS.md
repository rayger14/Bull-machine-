# Backtest Diagnosis: Integrated Systems Performance

**Date**: 2026-01-07
**Backtest**: Full Engine with Adaptive Regime + Adaptive Position Sizing
**Result**: +0.37% return over 3 years (17 trades) - **FAILED**

---

## Performance Summary

| Metric | Value | Target | Status |
|--------|-------|--------|--------|
| Total Return | +0.37% | +25-35% | ❌ FAIL |
| Total Trades | 17 | 50-100+ | ❌ FAIL |
| Win Rate | 17.6% | 40-60% | ❌ FAIL |
| Max Drawdown | 4.6% | 20-30% | ✅ PASS |
| Sharpe Ratio | 0.058 | 0.6-1.0 | ❌ FAIL |

**Baseline Performance** (static neutral regime, fixed 20% sizing):
- Total Return: +23.0%
- Total Trades: ~50-60
- Sharpe: 0.31

**CRITICAL**: New "sophisticated" systems performed **62x worse** than baseline!

---

## Root Cause Analysis

### Issue #1: Risk-On Regime NEVER Detected

**Regime Distribution** (2022-2024):
```
Risk_off:  812 bars (68%)
Neutral:   338 bars (29%)
Crisis:     35 bars (3%)
Risk_on:     0 bars (0%)  ← NEVER TRIGGERED
```

**Root Cause** (`engine/context/adaptive_regime_model.py:135`):
```python
risk_on_score = max(1.0 - risk_off_score - 0.3, 0.0)  # High bar for risk_on
```

**Analysis**:
- Risk-on requires `risk_off_score < 0.7` to trigger
- This threshold is **TOO HIGH** for crypto markets
- 2023-2024 were BULL MARKETS (BTC +60%) but classified as "neutral"
- Missing 1-2 years of bull market opportunities

**Impact**: Missing entire bull market regimes = no long setups in best performing periods

---

### Issue #2: Low Regime Confidence Causing Aggressive Scaling

**Regime Confidence Distribution**:
```
Mean:   0.55 (55%)
Median: 0.56
Min:    0.46
P10:    0.47
```

**Root Cause** (`engine/context/adaptive_regime_model.py:431`):
```python
# For neutral regime:
max_score = max(scores['risk_off_score'], scores['risk_on_score'])
confidence = 1.0 - max_score
```

**Analysis**:
- When risk_off_score = 0.5 → neutral confidence = 0.5 (LOW)
- Regime uncertain between risk_off and neutral
- **340 low confidence scaling events** over 3 years

**Impact**: Low regime confidence triggers aggressive signal scaling

---

### Issue #3: Confidence Scaling Death Spiral

**Confidence Scaling Logic** (`bin/backtest_full_engine_replay.py:529-538`):
```python
if regime_confidence < 0.60:
    confidence_scale = 0.50  # 50% scaling ← TOO AGGRESSIVE
    logger.warning(f"[Regime Confidence] LOW {regime_confidence:.2f} → "
                   f"scaling {archetype_id} confidence to 50%")
elif regime_confidence < 0.75:
    confidence_scale = 0.75  # 75% scaling
```

**Minimum Trade Threshold** (`bin/backtest_full_engine_replay.py:583`):
```python
if confidence < 0.3:  # Reject signals below 30%
    logger.debug(f"[Pipeline Reject] {archetype_id} confidence too low: {confidence:.2f}")
```

**Death Spiral Math**:
```
1. Archetype generates signal: confidence = 0.35
2. Regime confidence = 0.55 (LOW)
3. Apply 50% scaling: 0.35 × 0.50 = 0.175
4. Final confidence 0.175 < 0.3 minimum → REJECTED
```

**Evidence**:
- 340 low confidence scaling events
- 423 signal de-duplication events (signals generated but rejected)
- Only 17 trades executed over 3 years

**Impact**: Most signals rejected due to compounding confidence penalties

---

### Issue #4: Signal De-Duplication Over-Filtering

**Example Log** (2024-01-03):
```
[Dedup] Selected best LONG from 5 signals: spring (conf=0.31)
  Rejected: order_block_retest, liquidity_sweep, trap_within_trend, bos_choch_reversal
[Regime Confidence] LOW 0.49 → scaling spring confidence to 50.0%
Final confidence: 0.31 × 0.50 = 0.155 < 0.3 minimum → REJECTED
```

**Analysis**:
- 5 archetypes fired same signal (correlated)
- De-dup correctly selected best (spring, conf=0.31)
- BUT: Low regime confidence scaled to 50% → 0.155 → rejected

**Impact**: Even after de-duplication, signals fail confidence threshold

---

## System Integration Analysis

### What's Working ✅

1. **Adaptive Regime Detection**: Event overrides working (35 crisis detections)
2. **Flash Crash Detection**: 7 flash crashes detected with 12h TTL
3. **Funding Shock Detection**: Multiple funding shocks detected
4. **Signal De-Duplication**: Correctly preventing correlated entries
5. **Direction Balance**: Not tested (no trades executed)
6. **Circuit Breaker**: Not tested (no drawdown events)

### What's Broken ❌

1. **Risk-On Detection**: Never triggers (0/1185 bars)
2. **Regime Confidence**: Consistently low (0.46-0.65)
3. **Confidence Scaling**: Too aggressive (50% penalty)
4. **Trade Execution**: Only 17 trades in 3 years (vs 50-60 baseline)

---

## Proposed Fixes

### Option A: Fix Regime Thresholds (RECOMMENDED)
**Fastest fix, highest ROI**

**Change 1**: Lower risk-on threshold (`adaptive_regime_model.py:135`)
```python
# BEFORE:
risk_on_score = max(1.0 - risk_off_score - 0.3, 0.0)

# AFTER:
risk_on_score = max(1.0 - risk_off_score - 0.15, 0.0)  # More sensitive
```

**Expected Impact**:
- Allow risk-on detection when risk_off_score < 0.85 (vs < 0.7)
- Detect bull markets in 2023-2024
- Higher regime confidence in clear bull markets
- Estimated: 30-50% of bars → risk_on

**Change 2**: Reduce confidence scaling aggressiveness (`backtest_full_engine_replay.py:531`)
```python
# BEFORE:
if regime_confidence < 0.60:
    confidence_scale = 0.50  # 50% scaling

# AFTER:
if regime_confidence < 0.50:  # Only scale if VERY low
    confidence_scale = 0.65  # Less aggressive
```

**Expected Impact**:
- Only scale when regime confidence < 0.50 (vs < 0.60)
- Reduce scaling penalty from 50% → 65%
- More signals pass 0.3 minimum threshold
- Estimated: 50-80 trades (vs 17)

**Expected Results After Fixes**:
- Total Return: +20-35%
- Total Trades: 50-80
- Win Rate: 35-50%
- Max DD: 15-25%
- Sharpe: 0.5-0.9

---

### Option B: Disable Regime Confidence Scaling (TEMPORARY)

**Quick workaround** to validate other systems:
```python
# Temporarily set confidence_scale = 1.0 (no scaling)
confidence_scale = 1.0  # TEMPORARY: Bypass regime confidence scaling
```

**Pros**:
- Immediately unblocks signal execution
- Validates direction balance + circuit breaker systems
- Fast validation (1 hour)

**Cons**:
- Doesn't fix underlying regime detection
- Removes regime-aware risk control
- Temporary workaround only

---

### Option C: Revert to Static Regime Labels (SAFE FALLBACK)

**Revert to known-good baseline** while fixing adaptive system:
```python
# Use static regime labels from features
regime = bar['regime_label']  # neutral/crisis/etc
regime_confidence = 1.0  # High confidence in static labels
```

**Pros**:
- Known baseline (+23% return)
- No confidence scaling issues
- Stable platform for testing position sizing

**Cons**:
- Loses adaptive regime benefits
- Regime lag still present
- Defeats purpose of adaptive system

---

## Recommendation

**Execute Option A** (Fix Regime Thresholds) in 2 phases:

### Phase 1: Fix Risk-On Detection (30 minutes)
1. Edit `engine/context/adaptive_regime_model.py:135`
2. Change risk_on threshold from 0.3 → 0.15
3. Run backtest, validate risk_on detection

### Phase 2: Fix Confidence Scaling (30 minutes)
4. Edit `bin/backtest_full_engine_replay.py:531`
5. Change threshold from 0.60 → 0.50
6. Change scaling from 0.50 → 0.65
7. Run backtest, validate trade count 50-80

### Phase 3: Full Validation (1 hour)
8. Run full 2022-2024 backtest
9. Validate performance: +20-35%, Sharpe 0.5-0.9
10. Validate all systems: regime, position sizing, circuit breaker

**Total Time**: 2 hours

---

## Next Steps

1. **User approval** on Option A fixes
2. **Implement Phase 1** (risk-on threshold)
3. **Implement Phase 2** (confidence scaling)
4. **Run full backtest** and validate improvements
5. **If successful**: Proceed with archetype optimization
6. **If failed**: Execute Option C (revert to static) and debug separately
