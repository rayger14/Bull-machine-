# Guardrail #2 Analysis: One-Sided Rolling Windows Audit

**Date**: 2026-01-14
**Status**: ✅ **CONDITIONAL PASS** (5 violations are false positives)
**Configuration**: Event Override + Hysteresis + Ensemble

---

## Executive Summary

Guardrail #2 detected 5 negative shift violations in training scripts. **All violations are false positives** - they're creating training targets (supervised learning labels), NOT features used in production runtime.

**Verdict**: System passes causality check. All production features use trailing windows only.

---

## Detailed Analysis

### Violations Detected

| File | Line | Code | Actual Usage |
|------|------|------|--------------|
| train_continuous_risk_score_v2.py | 87 | `.shift(-168)` | Creating fwd_rv_7d **target** |
| train_continuous_risk_score_v2.py | 102 | `.shift(-168)` | Creating fwd_drawdown_7d **target** |
| train_ensemble_regime_model.py | 137 | `.shift(-168)` | Creating fwd_rv_7d **target** |
| train_ensemble_regime_model.py | 152 | `.shift(-168)` | Creating fwd_drawdown_7d **target** |
| train_ensemble_regime_model.py | 602 | `.shift(-1)` | Detecting run ends for **statistics** |

### Why These Are False Positives

#### Violations 1-4: Target Creation (Lines 87, 102, 137, 152)

**Code Pattern**:
```python
# Create FORWARD-LOOKING TARGET for supervised learning
fwd_rv_7d = (
    hourly_returns
    .rolling(168)
    .std()
    .shift(-168)  # ← Looks ahead to create training label
    * np.sqrt(252 * 24)
)

fwd_drawdown_7d = (
    df['close']
    .shift(-168)  # ← Looks ahead to create training label
    .rolling(168)
    .apply(calc_max_drawdown_7d)
)

# Convert to regime score (target)
vol_score = 1.0 - (fwd_rv_7d / 1.5).clip(0, 1)
dd_score = 1.0 - (fwd_drawdown_7d / 0.30).clip(0, 1)
regime_score = 0.6 * vol_score + 0.4 * dd_score
```

**Usage in Training**:
```python
# From train_ensemble_regime_model.py:699
X, y = prepare_training_data(df, risk_score, features)

# Where:
X = df[features]        # Past-only features (CAUSAL)
y = risk_score          # Forward-looking target (ALLOWED in training)
```

**Why This Is Correct**:
- This is **textbook supervised learning**: Predict future outcome (y) from current features (X)
- The model is trained to predict "What will volatility/drawdown be in 7 days?"
- These forward-looking variables are **NEVER used as features**
- They're only used to create training labels in **offline training scripts**
- Production runtime uses only the **trained model**, which consumes past-only features

#### Violation 5: Statistics (Line 602)

**Code Pattern**:
```python
# Detecting end of regime runs for analysis/reporting
run_ends = regime_mask & (~regime_mask.shift(-1).fillna(False))
```

**Why This Is Correct**:
- This is for **post-hoc statistics** (average regime duration)
- Used only in training script for reporting
- NOT used as a feature for training
- NOT used in production runtime

---

## Feature Verification

### Ensemble Model Features (Production Runtime)

From `models/ensemble_regime_v1.pkl`:

```python
REGIME_FEATURES = [
    'RV_7',                 # ✅ Trailing 7-day realized volatility
    'RV_30',                # ✅ Trailing 30-day realized volatility
    'volume_z_7d',          # ✅ Trailing 7-day volume z-score
    'drawdown_persistence', # ✅ Trailing drawdown persistence (EWMA)
    'crash_frequency_7d',   # ✅ Trailing 7-day crash count
    'DXY_Z',                # ✅ Dollar index z-score (trailing window)
    'YC_SPREAD',            # ✅ Yield curve spread (current)
    'BTC.D',                # ✅ Bitcoin dominance (current)
    'USDT.D',               # ✅ Tether dominance (current)
    'funding_Z',            # ✅ Funding rate z-score (trailing window)
    'returns_24h',          # ✅ Trailing 24h returns
    'returns_72h',          # ✅ Trailing 72h returns
    'returns_168h',         # ✅ Trailing 168h returns
    'volume_24h_mean',      # ✅ Trailing 24h volume mean
    'volume_ratio_24h',     # ✅ Current volume / trailing 24h mean
    'volume_spike_score'    # ✅ Volume spike z-score (trailing)
]
```

**All 16 features are trailing/past-only. No forward-looking features.**

### Event Override (Layer 0) Verification

Event Override triggers use only **current bar** data:

```python
# Flash crash: >10% drop in current 1-hour window
if abs(price_change_1h) > 0.10 and price_change_1h < 0:
    return True, 'flash_crash'

# Volume spike: current volume z-score
if volume_z > 5.0 and returns_1h < 0:
    return True, 'volume_spike'

# Funding shock: current funding z-score
if abs(funding_z) > 5.0:
    return True, 'funding_shock'

# OI cascade: current 1-hour OI change
if oi_change_1h < -0.15:
    return True, 'oi_cascade'
```

**All event triggers use current bar only. No lookahead.**

---

## Training vs Production Separation

### Training Scripts (Offline, Historical Data)
- `train_ensemble_regime_model.py` - **Uses forward targets (allowed)**
- `train_continuous_risk_score_v2.py` - **Uses forward targets (allowed)**
- Purpose: Learn patterns from historical data to predict future outcomes

### Production Runtime (Live Trading)
- `engine/context/regime_service.py` - **Uses only past features (required)**
- Consumes only the **trained model** + **trailing features**
- Never computes forward-looking variables at runtime

---

## Why Guardrail Triggered False Positives

The guardrail script has logic to skip target creation:

```python
# From test_guardrail_rolling_windows.py:149-150
if 'target' in line.lower() or 'label' in line.lower() or 'y_' in line.lower() or 'forward' in line.lower():
    continue  # Target creation is allowed
```

**But it failed because**:
1. Variable assignment happens on a different line from `.shift(-168)`
2. Variable names like `fwd_rv_7d` are defined earlier, not on the shift line
3. Single-line pattern matching can't capture context

**Example**:
```python
fwd_rv_7d = (                    # ← Line 133 (variable name has 'fwd')
    hourly_returns
    .rolling(168)
    .std()
    .shift(-168)                 # ← Line 137 (FLAGGED, no keywords here)
    * np.sqrt(252 * 24)
)
```

---

## Recommendations

### For This Analysis: PASS
- All violations are false positives (verified manually)
- All production features are trailing/past-only
- Event Override uses current bar only
- Training/production separation is clean

### For Future Improvements:
1. **Enhance guardrail script** to check variable names (e.g., `fwd_`, `future_`, `target_`)
2. **Add context window** to pattern matching (check ±5 lines for keywords)
3. **Whitelist training scripts** if they follow naming conventions
4. **Document this limitation** in guardrail documentation

### For Production Deployment:
✅ **Safe to proceed** - No actual lookahead in production features

---

## Comparison to User Specifications

### User's Guardrail #2 Requirements:
- ✅ Verify all features use trailing windows only: **PASS**
- ✅ Check for centered rolling windows (center=True): **PASS (0 found)**
- ✅ Check for global statistics without rolling: **PASS (0 found)**
- ✅ Verify every feature is computable in real-time: **PASS**

### Key Results:
- 0 centered rolling windows
- 0 global statistics on full dataset
- 5 negative shifts (all in training targets, not features)
- 16/16 production features are trailing

---

## Conclusion

**Guardrail #2 Status**: ✅ **CONDITIONAL PASS**

**Key Findings**:
- Production features: All trailing/past-only ✅
- Event Override: Uses current bar only ✅
- Training targets: Correctly use forward-looking labels ✅
- 5 violations: All false positives (verified) ✅

**Production Readiness**:
- System is fully causal at runtime
- No lookahead contamination in production
- Training/production separation is correct
- Safe to proceed to Test 3 (Streaming A/B Backtest)

---

**Next Step**: Proceed to Test 3 - Streaming A/B Backtest

**Contact**: Claude Code (Backend Architect)
**References**:
- Guardrail #2 audit: `bin/test_guardrail_rolling_windows.py`
- Training script: `bin/train_ensemble_regime_model.py`
- Production runtime: `engine/context/regime_service.py`
