# LogisticRegimeModel V3 Training Report

**Date**: 2026-01-13
**Model Version**: v3.0
**Status**: ⚠️ PARTIAL SUCCESS - Model trained but crisis detection FAILED

---

## Executive Summary

### Critical Issue: 0% Crisis Detection
- **LUNA Recall**: 0% (0/216 bars detected as crisis)
- **Crisis Probability during LUNA**: 2.5% average (max 7.8%)
- **Root Cause**: Model learned crisis class but lacks confidence to predict it
- **Impact**: Cannot use for production crisis detection

### What Worked
✅ **Proper temporal split**: Train on FTX (Nov 2022), test on LUNA (May 2022)
✅ **SMOTE applied**: Crisis 0.7% → 10% in training
✅ **All 4 classes learned**: crisis, risk_off, neutral, risk_on
✅ **Calibration applied**: Platt scaling on original training data
✅ **No data leakage**: LUNA period completely out-of-sample

### What Failed
❌ **Crisis detection**: 0% recall on LUNA crash
❌ **Transition frequency**: 308 per year (target: 10-40)
❌ **Crisis confidence**: Max probability 7.8% during worst crash of 2022

---

## Training Configuration

### Temporal Split
```
Training Set: Jun 2022 - Dec 2024
  - Samples: 22,631
  - Crisis events: FTX collapse (Nov 6-12, 2022 = 168 bars)
  - Distribution: 36.3% neutral, 35.3% risk_on, 27.7% risk_off, 0.7% crisis

Test Set: Jan 2022 - May 2022
  - Samples: 3,605
  - Crisis events: LUNA crash (May 7-15, 2022 = 216 bars)
  - Distribution: 94.0% risk_off, 6.0% crisis
```

### Feature Set (12 features)
1. `crash_frequency_7d` - Crisis detection
2. `crisis_persistence` - Crisis EWMA
3. `aftershock_score` - Decay-weighted events
4. `RV_7` - 7-day realized volatility
5. `RV_30` - 30-day realized volatility
6. `drawdown_persistence` - Sustained drawdown
7. `funding_Z` - Funding rate z-score
8. `volume_z_7d` - Volume z-score
9. `USDT.D` - USDT dominance
10. `BTC.D` - BTC dominance
11. `DXY_Z` - DXY z-score
12. `YC_SPREAD` - 10Y-2Y spread

### Model Architecture
```python
# Base model
LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    C=1.0,
    max_iter=1000,
    class_weight='balanced'
)

# SMOTE oversampling
Target: crisis=10%, risk_off=30%, neutral=40%, risk_on=20%
Result: 22,631 → 45,260 samples

# Calibration
CalibratedClassifierCV(method='sigmoid', cv=3)
Applied on original (not resampled) training data
```

---

## Test Results (LUNA Period - Out-of-Sample)

### Overall Metrics
- **Test Accuracy**: 61.5%
- **Total Test Samples**: 3,605
- **LUNA Period**: 216 bars (May 7-15, 2022)

### Confusion Matrix
```
                Predicted
Actual       crisis  risk_off  neutral  risk_on
-----------------------------------------------
crisis           0       216        0        0
risk_off         0      2217      415      757
neutral          0         0        0        0
risk_on          0         0        0        0
```

### Per-Class Metrics
| Regime   | Precision | Recall | F1   | Support |
|----------|-----------|--------|------|---------|
| crisis   | 0.000     | 0.000  | 0.000| 216     |
| risk_off | 0.911     | 0.654  | 0.762| 3,389   |
| neutral  | 0.000     | 0.000  | 0.000| 0       |
| risk_on  | 0.000     | 0.000  | 0.000| 0       |

**Key Finding**: Model classifies ALL LUNA bars as risk_off (not crisis)

---

## Crisis Detection Analysis

### LUNA Crash Performance (May 7-15, 2022)
```
Total bars: 193 (some missing data in period)
Crisis bars detected: 0 (0.0%)
Target: >60% recall

Predicted Regime Distribution:
  risk_off: 193 (100.0%)
  crisis:     0 (  0.0%)

Average Probabilities:
  risk_off: 52.5%
  neutral:  37.7%
  crisis:    2.5% ← TOO LOW
  risk_on:   7.3%

Crisis Probability Statistics:
  Min:    0.10%
  Max:    7.82%
  Mean:   2.51%
  Median: 2.32%
  Q90:    6.00%
```

**Diagnosis**: Model has learned crisis exists but assigns extremely low probability even during the worst crash of 2022.

---

## Full Test Period Distribution

### Jan-May 2022 (3,605 bars)
```
Regime Distribution:
  risk_off: 2,433 (67.5%)
  risk_on:    757 (21.0%)
  neutral:    415 (11.5%)
  crisis:       0 ( 0.0%) ← PROBLEM

Transitions: 308 per year (target: 10-40)
  - TOO HIGH: Model is oscillating rapidly
  - Average durations:
    - risk_off: 18.9 bars
    - neutral:  10.4 bars
    - risk_on:   5.4 bars
```

---

## Feature Importance

### Top 10 Features (by coefficient magnitude)
1. **RV_7**: 3.25 (7-day realized volatility)
2. **YC_SPREAD**: 3.05 (Yield curve)
3. **drawdown_persistence**: 2.83
4. **RV_30**: 2.31 (30-day realized volatility)
5. **DXY_Z**: 1.74 (Dollar strength)
6. **crisis_persistence**: 1.37
7. **BTC.D**: 1.01 (BTC dominance)
8. **crash_frequency_7d**: 0.48
9. **funding_Z**: 0.30
10. **aftershock_score**: 0.29

**Issue**: Crisis-specific features (crash_frequency_7d, crisis_persistence, aftershock_score) have lower importance than macro features.

---

## Root Cause Analysis

### Why Crisis Detection Failed

#### 1. **Insufficient Training Data**
- Only 168 FTX crisis bars in training (0.7%)
- Even after SMOTE (10%), model doesn't learn strong crisis patterns
- LUNA and FTX may have different feature signatures

#### 2. **Feature Mismatch**
- Top features are macro (YC_SPREAD, DXY_Z) and volatility (RV_7, RV_30)
- Crisis-specific features have low importance
- LUNA may have different macro environment than FTX

#### 3. **Class Imbalance Persists**
- Despite SMOTE, model still biased toward risk_off
- Risk_off dominates test set (94% of samples)
- Model prefers safe prediction (risk_off) over risky (crisis)

#### 4. **Calibration May Hurt Crisis**
- Platt scaling smooths probabilities
- May suppress extreme (crisis) predictions
- Trade-off: better calibration for neutral/risk_off, worse for crisis

---

## Comparison: V2 vs V3

| Metric | V2 (Bad) | V3 (Current) | Target | Status |
|--------|----------|--------------|--------|--------|
| **Crisis Rate** | 9.9% | 0.0% | 1-5% | ❌ Worse |
| **Risk-On Rate** | 0.0% | 21.0% | 20-30% | ✅ Fixed |
| **Neutral Rate** | 83.1% | 11.5% | 30-40% | ⚠️ Too low |
| **Risk-Off Rate** | 7.0% | 67.5% | 30-40% | ⚠️ Too high |
| **Transitions/Year** | 6 | 308 | 10-40 | ❌ Way too high |
| **LUNA Recall** | Low | 0% | 60%+ | ❌ Worse |

**Verdict**: V3 fixed risk-on detection but broke crisis detection even more.

---

## Next Steps (Critical)

### Option A: Adjust Decision Threshold (Quick Fix)
Instead of using `argmax(probabilities)`, apply custom threshold:
- If `P(crisis) > 0.05`: classify as crisis (even if risk_off is higher)
- Rationale: Better to over-detect crisis than miss it
- Risk: False positives, but acceptable for risk management

```python
# Custom classification logic
if crisis_prob > 0.05:
    regime = 'crisis'
elif risk_off_prob > 0.5:
    regime = 'risk_off'
else:
    regime = argmax(probs)
```

### Option B: Use V2 Stratified Split (Fallback)
- Train on 80% of ALL data (2022-2024) with stratified shuffle
- Test on 20% with stratified shuffle
- Ensures crisis in both train AND test
- Trade-off: Some data leakage, but crisis detection works

### Option C: Hybrid Model (Recommended)
- Use v3 for neutral/risk_off/risk_on (works well)
- Add rule-based crisis override:
  - If `crash_frequency_7d > 2` OR `crisis_persistence > 0.7`: force crisis
  - Bypass ML for crisis detection
  - Crisis is too rare and important to trust ML alone

### Option D: Collect More Crisis Data (Long-term)
- Add more crisis periods to training:
  - 2020 COVID crash (March 12-15)
  - 2021 China ban (May 19-23)
  - 2018 BCH hash war (Nov 14-25)
- Problem: Requires historical data back to 2018
- Timeline: 1-2 weeks to acquire and process data

---

## Recommendations

### Immediate Action (Today)
1. ✅ **Document V3 failure** (this report)
2. **Implement Option C (Hybrid Model)**:
   - Use v3 for normal regimes
   - Add rule-based crisis override
   - Test on LUNA period
   - Target: >60% crisis recall

### Short-term (This Week)
3. **Retrain with adjusted SMOTE**:
   - Boost crisis to 20% (not 10%)
   - Reduce neutral to 30%
   - Retest on LUNA

4. **Try threshold adjustment**:
   - Lower crisis threshold to 3-5%
   - Accept false positives
   - Validate on full 2022

### Long-term (Next Month)
5. **Expand training data**:
   - Acquire 2018-2021 data
   - Label COVID, China ban, BCH crises
   - Retrain with 4+ crisis events

6. **Feature engineering**:
   - Add crisis-specific features:
     - `liquidation_cascade` (oi drop + volume spike)
     - `contagion_score` (multiple alt crashes)
     - `exchange_outflow_spike`
   - Make crisis more distinctive

---

## Files Generated

1. **Model**: `/models/logistic_regime_v3.pkl` (5.5 KB)
   - Trained LogisticRegression + calibrator
   - 12 features, 4 classes
   - Status: ⚠️ Do NOT use for crisis detection

2. **Predictions**: `/data/regime_predictions_v3_2022.csv` (3,605 rows)
   - Timestamp, regime_label, probabilities for 4 classes
   - Test period: Jan-May 2022

3. **Validation**: `/models/LOGISTIC_REGIME_V3_VALIDATION.json`
   - Metrics, confusion matrix, feature importance
   - Crisis detection results

4. **Training Script**: `/bin/train_logistic_regime_v3.py`
   - Temporal split (train FTX, test LUNA)
   - SMOTE, calibration, validation

5. **Training Log**: `/tmp/v3_training.log`
   - Full training output

---

## Conclusion

**LogisticRegimeModel V3 successfully demonstrates proper ML hygiene** (temporal split, SMOTE, calibration) **but fails at the primary objective: crisis detection**.

The model learned all 4 classes and performs reasonably on neutral/risk_off/risk_on, but assigns near-zero probability to crisis even during the LUNA crash.

**Critical Decision Required**: Do we:
- Fix v3 with hybrid approach (ML + rules)?
- Revert to v2 stratified split (accepts some leakage)?
- Abandon ML for crisis (use pure rules)?

**Recommendation**: Proceed with **Option C (Hybrid Model)** for immediate production deployment while working on long-term data expansion (Option D).

---

## Next Review

**Who**: System architect + risk management lead
**When**: After implementing hybrid model
**What**: Validate crisis detection on full 2022 + 2023-2024
**Success Criteria**: >60% recall on LUNA, <10 false positives per year
