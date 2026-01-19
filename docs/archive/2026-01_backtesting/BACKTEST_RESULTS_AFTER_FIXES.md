# Backtest Results After Threshold Fixes

**Date**: 2026-01-07
**Fixes Applied**: Risk-on threshold (0.3 → 0.15), Confidence scaling (0.60/50% → 0.50/65%)

---

## Performance Comparison

| Metric | Baseline (Static) | Before Fixes | After Fixes | Target | Status |
|--------|-------------------|--------------|-------------|--------|--------|
| Total Return | +23.0% | +0.37% | **+1.24%** | +25-35% | ❌ |
| Total Trades | ~50-60 | 17 | **22** | 50-80 | ❌ |
| Win Rate | ~45% | 17.6% | **27.3%** | 40-60% | ❌ |
| Sharpe Ratio | 0.31 | 0.058 | **0.136** | 0.6-1.0 | ❌ |
| Max Drawdown | ~25% | 4.6% | 5.9% | 20-30% | ✅ |
| Profit Factor | ~1.5 | 1.07 | **1.17** | 1.5-2.0 | ❌ |

**Summary**: Improvements in right direction but still **18x worse** than baseline (+1.24% vs +23%)

---

## What's Working Now ✅

1. **Risk-On Detection**: 175 bars detected (was 0 before)
   - Dec 2024: Correctly identified as risk_on
   - Regime transitions working properly

2. **Direction Balance Tracking**: Active and monitoring
   - Detects 100% directional imbalance
   - Warning system operational

3. **Confidence Scaling**: Less aggressive
   - 259 scaling events (vs 340 before)
   - Threshold 0.50 (vs 0.60 before)
   - Scaling 65% (vs 50% before)

4. **All Systems Integrated**:
   - Adaptive regime: ✅ Working
   - Signal de-duplication: ✅ Working
   - Circuit breaker: ✅ Ready (not triggered, no drawdowns)
   - Position limits: ✅ Working

---

## What's Still Broken ❌

### Issue: Only 22 Trades in 3 Years

**Root Cause**: Regime confidence consistently LOW (0.31-0.48), still triggering 65% scaling

**Example from Logs** (2024-12-20):
```
Signal: funding_divergence, confidence = 0.48
Regime: neutral, confidence = 0.38 (LOW)
Scaling applied: 65%
Final confidence: 0.48 × 0.65 = 0.312
Result: Barely above 0.3 threshold → ACCEPTED (lucky)

Signal: spring, confidence = 0.31
Regime: neutral, confidence = 0.33 (LOW)
Scaling applied: 65%
Final confidence: 0.31 × 0.65 = 0.202
Result: Below 0.3 threshold → REJECTED
```

**The Math**:
- 259 signals scaled down due to low regime confidence
- Many signals 0.30-0.35 base confidence
- After 65% scaling → 0.195-0.228 final confidence
- Below 0.3 minimum → REJECTED

---

## Regime Confidence Analysis

### Neutral Regime Confidence Calculation

**Current Code** (`adaptive_regime_model.py:431`):
```python
# For neutral regime:
max_score = max(scores['risk_off_score'], scores['risk_on_score'])
confidence = 1.0 - max_score
```

**Problem**:
- If risk_off_score = 0.5, risk_on_score = 0.2
- max_score = 0.5
- confidence = 1.0 - 0.5 = **0.50** (LOW)

This makes sense conceptually: "If you're 50% bear, you're not confidently neutral."

BUT: This causes constant low confidence in transitional periods → signal rejection

### Regime Distribution

```
Risk-off:  812 bars (68%)  ← Bear market 2022
Neutral:   163 bars (14%)  ← Transitions
Risk-on:   175 bars (15%)  ← Bull market late 2023-2024
Crisis:     35 bars (3%)   ← Flash crashes
```

**Most of 2022-2024 was risk_off** (68%), which is actually correct given -80% BTC drawdown in 2022.

**The Real Issue**: Even when regime IS confident (risk_off with 1.0 confidence), we're not getting enough signals from archetypes!

---

## Deep Dive: Why So Few Signals?

Let me check the trade log to see what archetypes are actually firing...

**Checking trades CSV**:
- 22 trades over 3 years
- Most trades in early 2022 (bear market)
- Very few trades in 2023-2024 (bull market)

**Hypothesis**: Archetypes might not be optimized for adaptive regime + low confidence signals

---

## Three Options to Fix

### Option A: Disable Regime Confidence Scaling (QUICK FIX)

**Change** (`backtest_full_engine_replay.py`):
```python
# BEFORE:
if regime_confidence < 0.50:
    confidence_scale = 0.65

# AFTER:
confidence_scale = 1.0  # Disable regime confidence scaling entirely
```

**Pros**:
- Immediate unblock (5 minutes)
- Test if archetypes + direction balance can perform without regime penalties
- Validate other systems (circuit breaker, position limits, de-dup)

**Cons**:
- Loses regime-aware risk control
- Signals fire even during uncertain regime transitions
- May increase drawdowns during transitions

**Expected Result**: 50-100 trades, +15-25% return

---

### Option B: Improve Neutral Confidence Calculation (PROPER FIX)

**Change** (`adaptive_regime_model.py:429-431`):
```python
# BEFORE:
else:  # neutral
    max_score = max(scores['risk_off_score'], scores['risk_on_score'])
    confidence = 1.0 - max_score

# AFTER:
else:  # neutral
    # Confidence = how close scores are (balanced = high confidence)
    score_diff = abs(scores['risk_off_score'] - scores['risk_on_score'])
    confidence = max(0.6, 1.0 - score_diff)  # Min 0.6 confidence for neutral
```

**Logic**:
- If risk_off = 0.5, risk_on = 0.4 → diff = 0.1 → confidence = 0.90 (HIGH)
- If risk_off = 0.7, risk_on = 0.2 → diff = 0.5 → confidence = 0.60 (MODERATE)

**Pros**:
- Better models "neutral = balanced scores"
- Reduces low confidence scaling events
- Keeps regime-aware risk control

**Cons**:
- More complex logic
- Need to validate on edge cases
- 1-2 hours work + testing

**Expected Result**: 40-70 trades, +18-28% return

---

### Option C: Revert to Static Regime Labels (SAFE FALLBACK)

**Change** (`backtest_full_engine_replay.py`):
```python
# Use static regime labels instead of adaptive
regime = bar.get('regime_label', 'neutral')
regime_confidence = 1.0  # High confidence in static labels
```

**Pros**:
- Known baseline (+23% return)
- Stable platform for testing position sizing
- No regime confidence issues

**Cons**:
- Regime lag still present (static 2022=crisis, 2023=neutral, 2024=risk_on)
- Defeats purpose of adaptive regime system
- Gives up on adaptive regime benefits

**Expected Result**: +20-25% return (baseline), 50-60 trades

---

## Recommendation

**Execute Option B** (Improve Neutral Confidence) for best long-term solution:

### Implementation Plan (1 hour)

**Step 1**: Fix neutral confidence calculation (15 min)
```python
# File: engine/context/adaptive_regime_model.py:429-431
else:  # neutral
    score_diff = abs(scores['risk_off_score'] - scores['risk_on_score'])
    confidence = max(0.60, 1.0 - score_diff)
```

**Step 2**: Run backtest and validate (20 min)
- Expected: 40-70 trades
- Expected: +18-28% return
- Expected: < 100 low confidence scaling events

**Step 3**: If results still poor → Execute Option A (disable scaling) as temporary workaround (5 min)

**Step 4**: If Option A/B both fail → Execute Option C (revert to static) and debug adaptive system separately

---

## Alternative Theory: Archetype Issue?

**Possibility**: The problem might not be regime confidence, but **archetypes not optimized**!

Looking at logs:
- funding_divergence: confidence 0.25-0.48 (LOW)
- spring: confidence 0.30-0.32 (LOW)
- order_block_retest: confidence 0.31-0.34 (LOW)

**All archetypes generating LOW confidence signals (0.25-0.35)!**

This suggests: **Archetypes need optimization** (Option C from previous agent recommendations)

### If Archetype Optimization is the Real Issue:

1. Even without regime scaling, 0.30 confidence barely passes 0.3 minimum
2. With ANY scaling, 0.30 → 0.24 → rejected
3. Need to optimize archetypes to generate 0.5-0.7 confidence signals

**This would explain**:
- Why baseline (+23%) worked with fixed 20% position sizing
- Why adaptive systems are struggling (compound penalties)
- Why we're getting so few trades

---

## Final Decision Matrix

| Option | Time | Expected Trades | Expected Return | Risk |
|--------|------|-----------------|-----------------|------|
| A: Disable scaling | 5 min | 50-100 | +15-25% | Medium (loses regime control) |
| B: Fix neutral conf | 1 hour | 40-70 | +18-28% | Low (proper fix) |
| C: Revert to static | 10 min | 50-60 | +20-25% | Low (known baseline) |
| D: Optimize archetypes | 8 hours | 60-100 | +30-50% | Medium (time intensive) |

**Recommended Path**:
1. Try Option B (1 hour) - proper fix for neutral confidence
2. If still poor → Try Option A (5 min) - disable scaling entirely
3. If still poor → Execute Option C (10 min) - revert to baseline
4. Then execute Option D (8 hours) - optimize archetypes properly

This validates each layer systematically to isolate the true bottleneck.
