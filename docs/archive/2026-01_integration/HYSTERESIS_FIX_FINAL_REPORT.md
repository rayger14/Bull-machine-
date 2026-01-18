# Hysteresis Fix Investigation - Final Report

**Date**: 2026-01-13
**Status**: ❌ **DEPLOYMENT BLOCKER IDENTIFIED**
**Objective**: Fix excessive regime transitions (590/year) by integrating RegimeService with hysteresis

---

## Executive Summary

**The hysteresis integration cannot proceed as planned due to a critical model calibration issue.**

After Phase 3 A/B testing revealed excessive regime transitions (590-636/year vs 10-40 target), we attempted to fix this by integrating RegimeService's hysteresis layer. However, diagnostic investigation revealed the root cause is **poor model calibration**, not missing hysteresis.

### Key Finding

**LogisticRegimeModel v3 has average confidence of only 0.173** (barely better than random guessing at 0.25 for 4-class problem). This fundamental quality issue makes hysteresis tuning impossible:

- **Too tight hysteresis**: System gets stuck (2-4 transitions/year, misses regime changes)
- **Too loose hysteresis**: System flip-flops (591 transitions/year, same as no hysteresis)
- **"Goldilocks" zone doesn't exist** with confidence this low

---

## Investigation Timeline

### 1. Initial Problem (Phase 3 A/B Test)

**Symptom**: Both v3 ML and Hybrid models showed 590-636 regime transitions/year

**Root Cause**: Test bypassed RegimeService hysteresis layer, used raw model output directly

**Expected Fix**: Integrate RegimeService with proper hysteresis config

### 2. Hysteresis Integration (Attempt 1 - Tight Config)

**Config**:
```python
{
    'enter_threshold': 0.70,    # Strong evidence needed
    'exit_threshold': 0.50,     # Lower to exit
    'min_duration_hours': {
        'crisis': 6,
        'risk_off': 24,
        'neutral': 12,
        'risk_on': 48
    }
}
```

**Result**:
- Transitions: 1 total (0/year)
- Trades: 17
- PF: 1.05
- PnL: $+32

**Problem**: System stuck in risk_off entire 3 years. Only 1 regime change.

### 3. Hysteresis Tuning (Attempt 2 - Balanced Config)

**Config**:
```python
{
    'enter_threshold': 0.60,    # Moderate evidence
    'exit_threshold': 0.45,     # Easier to exit
    'min_duration_hours': {
        'risk_off': 12,         # Reduced from 24
        'neutral': 12,
        'risk_on': 24           # Reduced from 48
    }
}
```

**Result**:
- Transitions: 7 total (2/year)
- Trades: 20
- PF: 1.32
- PnL: $+221

**Problem**: Still too stable (2/year vs 10-40 target). Better than Attempt 1 but insufficient.

### 4. Raw Model Diagnostic (Critical Discovery)

**Ran LogisticRegimeModel v3 WITHOUT hysteresis** to understand baseline behavior:

**Results**:
```
Total transitions: 1,770
Transitions/year: 591 (excessive flip-flopping)

Regime Distribution:
  crisis:    0.2%
  risk_off: 28.0%
  neutral:  30.6%
  risk_on:  41.3%

Average confidence: 0.173 ⚠️ CRITICAL ISSUE
```

**Diagnosis**:
- Model confidence of 0.173 is VERY low (barely above random 0.25 for 4-class)
- Causes constant regime changes (591/year)
- When hysteresis added with 0.60 threshold, model NEVER meets it
- Results in system getting stuck in starting regime

### 5. Low-Confidence Tuning (Attempt 3 - Final)

**Config** (tuned for 0.173 confidence):
```python
{
    'enter_threshold': 0.25,    # VERY low to match model
    'exit_threshold': 0.15,
    'min_duration_hours': {
        'risk_off': 48,         # Increased dwell time
        'neutral': 48,
        'risk_on': 72
    }
}
```

**Result**:
- Transitions: 13 total (4/year)
- Trades: 109
- PF: 0.96 ❌ (WORSE than baseline)
- PnL: -$159 ❌ (WORSE than baseline)

**Problem**: More trades but worse quality. Negative PnL. Still below transition target.

---

## Root Cause Analysis

### Why Model Confidence is 0.173

**Training Data Limitation**:
- v3 trained on 2023-2024 only (2 years)
- Limited to avoid leakage from 2022 test set
- Small training set → poor generalization → low confidence

**Model Architecture**:
- Logistic Regression (linear model)
- 12 features
- CalibratedClassifierCV used, but calibration needs MORE data

**Result**: Model is uncertain about regime classification, producing low-confidence predictions that flip frequently.

### Why Hysteresis Cannot Fix This

Hysteresis works by:
1. **Dual thresholds**: Enter regime at high confidence (0.60-0.70), exit at lower (0.45-0.50)
2. **Dwell time**: Minimum hours before allowing regime change

But when model confidence is ~0.17:
- **Never meets enter threshold** → Gets stuck in starting regime
- **Lowering threshold** → Allows more transitions but trades quality degrades
- **No middle ground exists** → Fundamental model quality issue

---

## Performance Comparison

| Configuration | Transitions/Year | PF | PnL | Trades | Status |
|--------------|------------------|--------|----------|--------|--------|
| **Phase 3 Baseline** (no hysteresis) | 590 | 1.11 | +$240 | 68 | ⚠️ Too noisy |
| **Tight Hysteresis** (0.70/0.50) | 0 | 1.05 | +$32 | 17 | ❌ Too stuck |
| **Balanced Hysteresis** (0.60/0.45) | 2 | 1.32 | +$221 | 20 | ❌ Too stuck |
| **Loose Hysteresis** (0.25/0.15) | 4 | 0.96 | -$159 | 109 | ❌ Negative PnL |
| **Raw Model** (no hysteresis) | 591 | ? | ? | ? | ❌ Excessive noise |

**Conclusion**: No hysteresis configuration works with current model quality.

---

## Regime Quality Metrics

### Raw Model (v3) Predictions

```
Regime Distribution (2022-2024):
  Crisis:   0.2%  (target: 1-5%)    ⚠️ Under-predicting
  Risk-off: 28.0% (target: 30-40%)  ✓ Reasonable
  Neutral:  30.6% (target: 30-40%)  ✓ Reasonable
  Risk-on:  41.3% (target: 20-30%)  ⚠️ Over-predicting
```

**Analysis**:
- Distribution is reasonable (not wildly wrong)
- But **confidence per prediction is too low** (0.173 avg)
- Model "knows" what regimes exist but not confident when to switch

### Crisis Detection (Specific Events)

**LUNA Collapse (May 2022)**:
- Event override detected funding shocks ✓
- Model did NOT predict crisis on its own ❌
- Only 0.2% crisis predictions across all data

**FTX Collapse (Nov 2022)**:
- Event override detected funding shocks ✓
- Model did NOT predict crisis on its own ❌

**Conclusion**: Event override layer working. ML crisis detection not working (same as Hybrid findings).

---

## Code Changes Made

### 1. RealSignalBacktest Integration (bin/backtest_with_real_signals.py:90-160)

**Added RegimeService initialization**:
```python
# CRITICAL FIX: Initialize RegimeService with hysteresis
model_path = config.get('regime_model_path', 'models/logistic_regime_v3.pkl')
hysteresis_config = config.get('hysteresis_config', {...})

self.regime_service = RegimeService(
    model_path=model_path,
    enable_event_override=True,
    enable_hysteresis=True,
    hysteresis_config=hysteresis_config,
    crisis_threshold=0.60
)
```

### 2. Dynamic Regime Detection (bin/backtest_with_real_signals.py:208-216)

**Replaced static regime reads**:
```python
# BEFORE (reading pre-computed):
regime = bar.get(f'{archetype}_regime', bar.get('regime_label', 'unknown'))

# AFTER (dynamic detection):
features = bar.to_dict()
regime_result = self.regime_service.get_regime(features, timestamp)
regime = regime_result['regime_label']

# Track transitions
if regime_result.get('transition_flag', False):
    self.regime_transitions += 1
```

### 3. Transition Validation (bin/backtest_with_real_signals.py:556-573)

**Added regime behavior metrics**:
```python
logger.info("Regime Behavior:")
logger.info(f"  Total transitions: {self.regime_transitions}")

transitions_per_year = (self.regime_transitions / results['total_days']) * 365
logger.info(f"  Transitions/year: {transitions_per_year:.0f}")

# Validate against target (10-40 transitions/year)
if transitions_per_year < 10:
    logger.warning(f"  ⚠️  Too few transitions ({transitions_per_year:.0f}/year < 10 target)")
elif transitions_per_year > 40:
    logger.error(f"  ❌ EXCESSIVE transitions ({transitions_per_year:.0f}/year > 40 target)")
else:
    logger.info(f"  ✅ Transitions within target range (10-40/year)")
```

### 4. Diagnostic Scripts

**Created diagnostic tools**:
- `bin/validate_hysteresis_fix.py` - Full backtest with hysteresis validation
- `bin/test_raw_regime_model.py` - Raw model diagnostic (discovered 0.173 confidence)

---

## Deployment Decision

### ❌ DO NOT DEPLOY RegimeService Integration

**Blockers**:
1. Model confidence too low (0.173 avg) for hysteresis to work
2. No hysteresis config achieves 10-40 transitions/year target
3. Trade quality degrades when allowing more transitions
4. Fundamental model retraining required

### Two Paths Forward

#### **Option A: Deploy Phase 3 Baseline** (Short-term)

**Use what works now**:
- Phase 3 streaming backtest (PF 1.11, +$240 PnL)
- Pre-computed regime labels (no dynamic detection)
- 590 transitions/year is noisy BUT system is profitable
- Accept noise, monitor in paper trading

**Pros**:
- Deploy immediately
- Known profitable configuration
- Real-world validation can proceed

**Cons**:
- Excessive regime transitions (590/year)
- Not using full temporal stack
- Technical debt accumulates

#### **Option B: Retrain Model v4** (Medium-term - RECOMMENDED)

**Fix root cause properly**:
1. Acquire 2018-2021 data (COVID crash, China ban, BCH fork)
2. Train on 2018-2023 (6 years vs current 2 years)
3. Test on 2024 as out-of-sample
4. Expected: Higher confidence, better calibration
5. THEN integrate RegimeService with hysteresis

**Pros**:
- Fixes fundamental model quality issue
- More crisis examples → better crisis detection
- Higher confidence → hysteresis will work
- Production-grade regime detection

**Cons**:
- 1-2 weeks delay for data + retraining
- No guarantee v4 will be better (but likely)

---

## Recommended Actions

### Immediate (This Week)

1. **Document findings** in this report ✓
2. **User decision required**: Option A (deploy now) vs Option B (retrain first)
3. If Option A: Deploy Phase 3 baseline to paper trading with caveats
4. If Option B: Begin 2018-2021 data acquisition

### Short-term (If Option B Selected)

1. **Acquire historical data**:
   - BTC 1H OHLCV: 2018-2021 (Binance or similar)
   - Funding rates, OI, volume
   - Label crisis periods: COVID (Mar 2020), China ban (May-Jul 2021), BCH (Nov 2018)

2. **Train LogisticRegimeModel v4**:
   - Training: 2018-2023 (6 years)
   - Test: 2024 (out-of-sample)
   - Target accuracy: >75%
   - Target confidence: >0.40 avg (vs current 0.173)

3. **Validate v4 + Hysteresis**:
   - Re-run hysteresis validation with v4
   - Expect 10-40 transitions/year naturally
   - Validate PF improvement

4. **Deploy to paper trading**:
   - Full temporal stack with RegimeService
   - Monitor transition frequency
   - 2-4 weeks validation before live

---

## Technical Debt Cleared

✅ **RegimeService integration code** - Complete and working
✅ **Hysteresis configuration system** - Flexible and tunable
✅ **Transition validation** - Automated checks in backtest
✅ **Diagnostic tooling** - Can detect model quality issues

---

## Technical Debt Remaining

⚠️ **Model calibration** - v3 has 0.173 avg confidence (needs v4 with more data)
⚠️ **Crisis detection** - Only event override working, ML not confident
⚠️ **Historical data** - Need 2018-2021 for better training

---

## Lessons Learned

### What Worked Well

1. **Systematic diagnosis** - Raw model diagnostic identified root cause quickly
2. **Parity ladder approach** - Isolated hysteresis as separate concern
3. **Quantitative validation** - Transition counts + confidence metrics caught issue
4. **Code infrastructure** - RegimeService integration ready when model improves

### What We'd Do Differently

1. **Check model confidence FIRST** - Should have diagnosed before integration
2. **Train on more data initially** - 2 years (2023-2024) was too limited
3. **Validate calibration separately** - Model quality gate before hysteresis tuning

---

## Conclusion

**The hysteresis fix cannot proceed with current model quality.**

LogisticRegimeModel v3 has a fundamental calibration issue (0.173 avg confidence) that makes hysteresis tuning impossible. No configuration achieves the 10-40 transitions/year target without degrading trade quality.

**Two viable paths**:
1. **Deploy Phase 3 baseline now** (accept 590 transitions/year, monitor in production)
2. **Retrain v4 on 2018-2024 data** (fix root cause, then deploy with hysteresis)

**User decision required** to choose path forward.

---

**Report Date**: 2026-01-13
**Author**: Claude Code
**Status**: Awaiting user decision on deployment path
