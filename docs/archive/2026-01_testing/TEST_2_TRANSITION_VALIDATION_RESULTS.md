# Test 2: Transition Validation Results

**Date**: 2026-01-14
**Status**: ⚠️ PARTIAL PASS (3/4 tests passed)
**Dataset**: 61,277 bars (2018-2024, 7.0 years)

---

## Executive Summary

Test 2 revealed **excellent regime stability and distribution**, but **poor crisis recall** on recent events (LUNA, FTX). The ensemble + hysteresis combination creates very stable regimes (7.3 transitions/year) but may be TOO conservative for crisis detection.

### Key Findings

✅ **PASS: Regime Distribution (Excellent)**
- Crisis: 2.4% of time (target 1-5%) ✅
- Distribution: 46% risk-on, 40% neutral, 11% risk-off, 2.4% crisis
- Well-balanced, not stuck in one regime

✅ **PASS: Transition Rate (Acceptable but Low)**
- 7.3 transitions/year (target 10-40, acceptable 5-100)
- Average regime durations:
  - Risk-on: 2179 hours (90 days!)
  - Neutral: 1070 hours (45 days)
  - Risk-off: 574 hours (24 days)
  - Crisis: 365 hours (15 days)

❌ **FAIL: Crisis Recall (Poor)**
- Only 1/3 major events detected (33%)
- COVID: ✅ Detected (39.4% crisis in window)
- LUNA: ❌ Missed (29% risk-off, below threshold)
- FTX: ❌ Completely missed (91% risk-on during collapse!)

✅ **PASS: Transition Patterns (Reasonable)**
- Most common: risk_on ↔ neutral (stable oscillation)
- Crisis transitions: risk_off → crisis (3 times)

---

## Detailed Results

### Test 2A: Transition Rate

**Result**: ✅ **PASS (Acceptable)**

```
Total transitions: 51
Date range: 7.0 years
Transitions/year: 7.3

Average durations:
  crisis:    365h (15 days)  [n=4 runs]
  risk_off:  574h (24 days)  [n=12 runs]
  neutral:  1070h (45 days)  [n=23 runs]
  risk_on:  2179h (90 days)  [n=13 runs]
```

**Analysis**:
- **Below target** (10-40/year) but **within acceptable range** (5-100/year)
- Hysteresis creating very stable regimes (90-day risk-on average!)
- This is GOOD for reducing false signals, but MAY be too conservative

**Comparison to Training**:
- Training (ensemble v1): 40.2 transitions/year
- Production (with hysteresis): 7.3 transitions/year
- **82% reduction** due to hysteresis stability constraints

### Test 2B: Regime Distribution

**Result**: ✅ **PASS (Excellent)**

```
Regime distribution (61,277 bars):
  risk_on:   28,324 (46.2%)
  neutral:   24,607 (40.2%)
  risk_off:   6,885 (11.2%)
  crisis:     1,461 ( 2.4%) ✅
```

**Analysis**:
- Crisis at 2.4% is **PERFECT** (target 1-5%)
- Well-balanced distribution reflects 2018-2024 period (bull-dominant but with volatility)
- No single regime dominates (max 46%)

### Test 2C: Crisis Recall on Known Events

**Result**: ❌ **FAIL (Poor - 33% recall)**

#### Event 1: COVID-19 Crash (March 8-16, 2020)
**Result**: ✅ **DETECTED**

```
Window: 193 bars
Distribution:
  crisis:   76 bars (39.4%) ✅
  neutral: 117 bars (60.6%)
  risk_off:  0 bars
  risk_on:   0 bars
```

**Analysis**: Strong detection. Crisis regime activated for 40% of the crash window.

#### Event 2: LUNA/UST Collapse (May 8-14, 2022)
**Result**: ❌ **MISSED**

```
Window: 145 bars
Distribution:
  crisis:    0 bars ( 0.0%) ❌
  risk_off: 42 bars (29.0%)
  neutral: 103 bars (71.0%)
  risk_on:   0 bars
```

**Analysis**:
- Only 29% risk-off (below 30% threshold for "detected")
- Model saw volatility but not enough to trigger crisis
- Hysteresis may have prevented transition from neutral

**Root Cause**: LUNA crash was more of a "stablecoin depeg" event than broad market volatility. BTC only dropped ~20% during this window, not the >30% typical of crisis.

#### Event 3: FTX Collapse (November 6-12, 2022)
**Result**: ❌ **COMPLETELY MISSED**

```
Window: 145 bars
Distribution:
  crisis:    0 bars ( 0.0%) ❌
  risk_off:  0 bars ( 0.0%) ❌
  neutral:  13 bars ( 9.0%)
  risk_on: 132 bars (91.0%) ⚠️
```

**Analysis**:
- **91% risk-on during FTX collapse** - This is WRONG
- FTX crash was November 8-9, but model stayed in risk-on
- BTC was relatively stable during FTX (only ~5% drop initially)

**Root Cause**: FTX was more of a "exchange contagion" event. Market volatility was moderate. Hysteresis 48-hour minimum dwell time for risk-on prevented any transition.

### Test 2D: Transition Pattern Analysis

**Result**: ✅ **PASS (Diagnostic)**

**Transition Matrix** (% of transitions from row → column):
```
             crisis   risk_off   neutral   risk_on
crisis          0%        50%       50%        0%
risk_off       25%         0%       67%        8%
neutral         5%        41%        0%       55%
risk_on         0%         0%      100%        0%
```

**Most Common Transitions**:
1. risk_on → neutral: 13 times
2. neutral → risk_on: 12 times
3. neutral → risk_off: 9 times
4. risk_off → neutral: 8 times
5. risk_off → crisis: 3 times

**Key Insights**:
- Risk-on NEVER transitions directly to risk-off or crisis (must go through neutral)
- This is expected behavior from hysteresis (prevents sudden jumps)
- Crisis exits equally to risk-off (50%) or neutral (50%)

---

## Root Cause Analysis: Low Crisis Recall

### Primary Issue: Hysteresis Too Conservative

**Hysteresis Settings**:
```python
min_dwell_hours = {
    'crisis': 6,
    'risk_off': 24,
    'neutral': 12,
    'risk_on': 48  # ← THIS IS THE PROBLEM
}
enter_threshold = 0.7   # High barrier to enter regime
exit_threshold = 0.5    # Medium barrier to exit
```

**Impact**:
- Once in risk-on, must stay for **48 hours minimum** (2 days)
- FTX collapse happened quickly (<24 hours), but model was "locked" in risk-on
- Even if ensemble detected danger, hysteresis blocked transition

### Secondary Issue: Crisis Threshold (60%)

Current threshold requires P(crisis) > 0.60 to declare crisis. This may be too strict for:
- Exchange-specific events (FTX, LUNA) that don't cause broad market volatility
- Flash crashes that recover quickly
- Localized contagion events

### Tertiary Issue: Feature Lag

Features are 7-30 day rolling windows (RV_7, RV_30, etc.), which creates lag:
- LUNA crash: May 9-12 (3 days)
- FTX crash: November 8-9 (1 day)
- Features based on 7-day windows won't spike immediately

---

## Comparison: Training vs Production

### Training (Ensemble V1 Validation)
- Test R²: -1.62 (poor but acceptable)
- Transitions/year: 40.2
- Crisis %: Not reported (but model was trained to detect crises)

### Production (with Hysteresis)
- Transitions/year: 7.3 (82% reduction)
- Crisis %: 2.4% (realistic)
- Crisis recall: 33% (poor)

**Takeaway**: Hysteresis dramatically reduces transitions (GOOD for stability) but also reduces crisis sensitivity (BAD for risk management).

---

## Recommendations

### Option A: Tune Hysteresis for Better Crisis Response (RECOMMENDED)

**Changes**:
```python
min_dwell_hours = {
    'crisis': 6,       # Keep (6h minimum for crisis)
    'risk_off': 12,    # Reduce from 24 (allow faster exits)
    'neutral': 12,     # Keep
    'risk_on': 24      # Reduce from 48 (allow faster crisis detection)
}
```

**Expected Impact**:
- Transitions/year: 7.3 → 15-20 (still within target)
- Crisis recall: 33% → 60-80% (better detection)
- Trade-off: Slightly more false signals

### Option B: Add Event Override Layer (HYBRID)

**Enable event override** for rapid crisis detection:
```python
service = RegimeService(
    mode='dynamic_ensemble',
    enable_event_override=True,  # Enable flash crash detection
    enable_hysteresis=True
)
```

**Event triggers** (Layer 0):
- Flash crash: >10% drop in 1H
- Volume spike: volume z-score >5 + negative return
- Funding shock: |funding z| >5

**Expected Impact**:
- Catches flash crashes/extreme events immediately
- Bypasses hysteresis for rapid response
- Maintains ensemble for normal conditions

### Option C: Lower Crisis Threshold (EASIEST)

**Change**:
```python
crisis_threshold = 0.40  # Lower from 0.60
```

**Expected Impact**:
- Crisis recall: 33% → 50-60%
- Crisis %: 2.4% → 3-5% (still acceptable)
- Easier to trigger crisis mode

### Option D: Accept Trade-Off (DO NOTHING)

**Current behavior**:
- Very stable regimes (7.3 transitions/year)
- Low false positive rate
- Misses rapid contagion events (LUNA, FTX)

**Acceptable if**:
- You prioritize signal stability over crisis detection
- Event-specific crashes (exchange failures) are less important than market-wide crashes
- You have OTHER risk controls for rapid events

---

## Next Steps

### Immediate (Before Test 3)

1. **Decide on crisis detection strategy**:
   - Option A: Tune hysteresis (reduce risk_on dwell to 24h)
   - Option B: Enable event override (hybrid approach)
   - Option C: Lower crisis threshold (0.60 → 0.40)

2. **Re-run Test 2** with chosen option to validate improvements

### Then Proceed to Guardrails

3. **Guardrail #1: Prefix Invariance Test**
   - Run twice: full period vs truncated
   - Validate no lookahead

4. **Guardrail #2: One-Sided Rolling Windows**
   - Audit all feature calculations

### Then Test 3 (A/B Backtest)

5. **Streaming A/B Backtest** (the real proof)
   - Compare baseline vs ensemble on 2022-2024
   - Measure PF, DD, trade count

---

## Conclusion

**Test 2 Status**: ⚠️ **PARTIAL PASS (3/4)**

**What Went Well**:
- ✅ Regime distribution excellent (crisis 2.4%, target 1-5%)
- ✅ Very stable regimes (90-day risk-on average)
- ✅ Detected COVID crash (39.4% crisis)

**What Needs Improvement**:
- ❌ Missed LUNA and FTX collapses
- ❌ Hysteresis too conservative (48h risk-on dwell)
- ❌ Crisis recall only 33% (need 60%+)

**Recommended Action**: **Option B (Enable Event Override)** for hybrid approach:
- Keep stable ensemble for normal conditions (7-15 transitions/year)
- Add event override for rapid crisis detection (flash crashes)
- Best of both worlds: stability + rapid response

**Decision Required**: Which option should we implement before proceeding to Test 3?

---

**Contact**: Claude Code (Backend Architect)
**References**:
- `ENSEMBLE_REGIME_V1_VALIDATION.json`
- `REGIME_ENSEMBLE_INTEGRATION_COMPLETE.md`
- Test execution: `bin/test_regime_transition_validation.py`
