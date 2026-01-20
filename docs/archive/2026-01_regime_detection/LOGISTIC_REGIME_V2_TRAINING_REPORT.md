# Logistic Regime Model V2 - Training Report
**Date:** 2026-01-09
**Model Path:** `/Users/raymondghandchi/Bull-machine-/Bull-machine-/models/logistic_regime_v2.pkl`
**Status:** ✅ TRAINED (Partial Success - Crisis recall improved but below target)

---

## Executive Summary

Successfully implemented ALL 5 critical fixes to create `logistic_regime_v2.pkl`. The model shows significant improvement over v1:

**V1 Baseline:**
- Crisis Recall: 2.2% (unusable)
- LUNA Detection: 0%
- Test set crisis samples: 0 (all in train set)
- VIX_Z had inverse signal

**V2 Results:**
- Crisis Recall: 18.2% (**8x improvement**)
- LUNA Detection: 12% (vs 0% baseline)
- Test set crisis samples: 77 (stratified split working)
- VIX_Z removed, features fixed

**Issue:** Crisis recall is 18% vs 50% target. The model is learning but still conservative (81% of crisis samples misclassified as risk_off).

---

## Fixes Implemented

### Fix #1: Stricter Crisis Labels (1-2% of bars)
**Status:** ✅ COMPLETE

**Implementation:**
- LUNA Collapse: May 7-15, 2022 (9 days only, not full month)
- FTX Collapse: Nov 6-12, 2022 (7 days only)
- Other periods: risk_off, neutral, risk_on

**Results:**
- Crisis samples: 384/26,236 (1.46%) ✅ Within 1-2% target
- Precise labeling (crash days only, not recovery periods)

**Ground Truth Labels:**
- `/Users/raymondghandchi/Bull-machine-/Bull-machine-/models/regime_ground_truth_v2.csv`

---

### Fix #2: Remove VIX_Z Feature
**Status:** ✅ COMPLETE + EXTENDED

**Implementation:**
- Removed VIX_Z (inverse signal: high VIX predicted false crisis)
- Also removed:
  - `oi` (Open Interest): 67% NaN, ALL NaN during LUNA crash
  - `RV_20`, `RV_60`: Zero during LUNA crash (not useful)

**Replaced with:**
- `RV_7` (7-day realized volatility): NON-ZERO during LUNA ✅
- `RV_30` (30-day realized volatility): NON-ZERO during LUNA ✅

**Final Feature Set (12 features):**
1. `crisis_persistence` - #1 most important (2.16)
2. `RV_7` - #2 most important (1.34)
3. `BTC.D` - Bitcoin dominance (1.14)
4. `DXY_Z` - Dollar strength (0.79)
5. `RV_30` - 30d realized vol (0.57)
6. `YIELD_CURVE` - 10Y-2Y spread (0.44)
7. `crash_frequency_7d` - Flash crash count (0.37)
8. `aftershock_score` - Event decay (0.21)
9. `drawdown_persistence` - Sustained drawdown (0.20)
10. `funding_Z` - Funding rate z-score (0.12)
11. `volume_z_7d` - Volume z-score (0.08)
12. `USDT.D` - USDT dominance (0.06)

---

### Fix #3: SMOTE Oversampling (crisis 1-2% → 10%)
**Status:** ✅ COMPLETE

**Implementation:**
- Applied SMOTE to training set only (not test set)
- Original training samples: 20,988
- After SMOTE: 41,974 samples (2x increase)

**Target Distribution:**
- Crisis: 10%
- Risk-off: 30%
- Neutral: 40%
- Risk-on: 20%

**Results:**
- SMOTE successfully generated synthetic crisis samples
- Training accuracy: 67.8%
- CV accuracy: 68.1% ± 0.8%

---

### Fix #4: Stratified Train/Test Split
**Status:** ✅ COMPLETE

**Implementation:**
- Used `StratifiedShuffleSplit` (80/20 split)
- Ensures crisis samples in BOTH train and test sets

**Results:**
- Train set: 20,988 samples
  - Crisis: 307 (1.5%)
  - Risk-off: 7,722 (36.8%)
  - Neutral: 6,564 (31.3%)
  - Risk-on: 6,395 (30.5%)

- Test set: 5,248 samples
  - Crisis: 77 (1.5%) ✅ NON-ZERO (v1 had 0)
  - Risk-off: 1,931 (36.8%)
  - Neutral: 1,641 (31.3%)
  - Risk-on: 1,599 (30.5%)

**Verification:** ✅ Test set has 77 crisis samples (v1 had 0)

---

### Fix #5: Same Model Architecture (Production-Ready)
**Status:** ✅ COMPLETE

**Model Configuration:**
```python
LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    C=1.0,                    # L2 regularization
    class_weight='balanced',  # Handle class imbalance
    max_iter=1000,
    random_state=42
)
```

**Probability Calibration:**
```python
CalibratedClassifierCV(
    estimator=model,
    method='sigmoid',  # Platt scaling
    cv=3
)
```

**Model Artifact Structure:**
- Same interface as v1 (production-compatible)
- `feature_order`: List of 12 features
- `scaler`: StandardScaler (fitted on training data)
- `model`: LogisticRegression
- `calibrator`: CalibratedClassifierCV
- `use_calibration`: True

---

## Test Set Performance

### Overall Metrics
- **Test Accuracy:** 64.9%
- **CV Accuracy:** 68.1% ± 0.8%
- **Test Samples:** 5,248

### Confusion Matrix
```
                crisis  risk_off  neutral  risk_on
  crisis            14        62        1        0
  risk_off          11      1772      121       27
  neutral            0      1054      283      304
  risk_on            0       163      101     1335
```

### Per-Class Metrics

| Regime    | Precision | Recall | F1 Score | Support |
|-----------|-----------|--------|----------|---------|
| **Crisis**    | 56.0%     | **18.2%** | 27.5%    | 77      |
| Risk-off  | 58.1%     | 91.8%  | 71.1%    | 1,931   |
| Neutral   | 55.9%     | 17.2%  | 26.4%    | 1,641   |
| Risk-on   | 80.1%     | 83.5%  | 81.8%    | 1,599   |

**Key Findings:**
- **Crisis Recall: 18.2%** (v1: 2.2% - **8x improvement**)
- **Crisis Precision: 56.0%** (when model predicts crisis, it's correct 56% of the time)
- **Crisis F1: 27.5%** (still low, but much better than v1's near-zero)

**Problem:** Out of 77 crisis samples, only 14 detected correctly (18%), 62 misclassified as risk_off (81%)

---

## LUNA Crash Detection Test

### Test Period: May 7-15, 2022 (216 hours)

**Results:**
- Crisis bars detected: 26/216 (12.0%)
- Average crisis probability: 30.4%
- Prediction distribution:
  - Crisis: 26 bars (12.0%)
  - Risk-off: 190 bars (88.0%)

**Comparison:**
- V1 baseline: 0% detection (FAILED)
- V2 result: 12% detection (IMPROVED but below 50% target)

**Feature Values During LUNA:**
```
crash_frequency_7d:    mean=0.50, max=1.00
crisis_persistence:    mean=0.07, max=0.19
aftershock_score:      mean=0.17, max=0.69
RV_7:                  mean=0.19, max=0.22 ✅
RV_30:                 mean=0.12, max=0.13 ✅
drawdown_persistence:  mean=1.00, max=1.00
funding_Z:             mean=0.15, max=3.29
volume_z_7d:           mean=0.21, max=6.08
```

---

## Deliverables

### 1. Model Artifact
**Path:** `/Users/raymondghandchi/Bull-machine-/Bull-machine-/models/logistic_regime_v2.pkl`
**Size:** 5.1 KB
**Format:** Pickle (same as v1)

**Contents:**
- Trained LogisticRegression model
- CalibratedClassifierCV calibrator
- StandardScaler
- Feature order (12 features)
- Regime labels
- Training metadata

### 2. Validation Report
**Path:** `/Users/raymondghandchi/Bull-machine-/Bull-machine-/models/logistic_regime_v2_validation.json`

### 3. Ground Truth Labels
**Path:** `/Users/raymondghandchi/Bull-machine-/Bull-machine-/models/regime_ground_truth_v2.csv`
**Records:** 26,236 hourly labels

### 4. Training Script
**Path:** `/Users/raymondghandchi/Bull-machine-/Bull-machine-/bin/train_logistic_regime_v2.py`
**Features:**
- All 5 fixes implemented
- Comprehensive logging
- LUNA crash validation
- Reproducible (random_state=42)

---

## Critical Insights

### Why Crisis Recall is Still Low (18% vs 50% target)

**Root Causes:**
1. **Feature Discriminability:** Crisis features (crash_frequency_7d, crisis_persistence) have low mean values during LUNA (0.07-0.50), making them hard to separate from risk_off
2. **Class Imbalance:** Despite SMOTE, only 384 true crisis samples in 26K bars (1.5%)
3. **Label Overlap:** Crisis and risk_off are similar - high volatility, drawdowns, funding stress
4. **Calibration Effect:** Platt scaling may be too conservative, pushing crisis probabilities down

**Confusion Analysis:**
- 81% of crisis samples predicted as risk_off (62/77)
- This suggests the model sees them as "high stress" but not "crisis level"
- Average crisis probability during LUNA: 30.4% (not confident enough to predict crisis)

### What Worked

1. **Feature Selection:** RV_7 and RV_30 are actually predictive (ranked #2 and #5)
2. **SMOTE:** Boosted training samples from 20K → 42K without breaking the model
3. **Stratified Split:** Successfully ensured crisis in test set (77 samples vs 0 in v1)
4. **VIX_Z Removal:** Eliminated inverse signal issue

### What Needs Improvement

1. **Threshold Tuning:** Model might benefit from lower crisis threshold (e.g., 25% probability instead of max probability)
2. **Feature Engineering:** Need stronger crisis-specific features (e.g., price velocity, liquidation cascades)
3. **Temporal Context:** Current model is timestamp-agnostic; could add "hours since last crisis" or "crisis clustering" features
4. **Label Refinement:** May need to relabel some risk_off periods as crisis to increase training signal

---

## Comparison: V1 vs V2

| Metric | V1 | V2 | Change |
|--------|----|----|--------|
| **Crisis Recall** | 2.2% | **18.2%** | **+16.0pp** (8x) |
| **Crisis Precision** | N/A | **56.0%** | New |
| **Crisis F1** | ~0% | **27.5%** | New |
| **LUNA Detection** | 0% | **12.0%** | **+12.0pp** |
| **Test Crisis Samples** | 0 | **77** | **Fixed** |
| **Features** | 14 (VIX_Z bad) | **12** (clean) | **-2** |
| **Test Accuracy** | Unknown | **64.9%** | - |
| **CV Accuracy** | Unknown | **68.1%** | - |

**Overall:** V2 is a significant improvement over V1, but still needs threshold tuning or feature engineering to reach 50% crisis recall target.

---

## Next Steps (Recommendations)

### Immediate (Production Deployment)

1. **Deploy V2 with Threshold Tuning:**
   - Instead of `max(proba)`, use `proba['crisis'] > 0.25` as crisis trigger
   - This could boost crisis recall from 18% → 40-50%
   - Accept some false positives (better safe than sorry in risk management)

2. **Monitor Performance:**
   - Track crisis detection rate in production
   - Compare with ground truth during next crash event
   - Log feature values during detected crises

### Short-Term (Model Improvement)

3. **Feature Engineering:**
   - Add `price_velocity_1h`: Extreme price moves in last hour
   - Add `liquidation_cascade`: OI + Volume + Price change combo
   - Add `crisis_clustering`: Exponential decay of recent crisis signals

4. **Label Refinement:**
   - Review the 62 crisis samples misclassified as risk_off
   - Check if labels are correct (some may be borderline)
   - Consider adding "severe risk_off" intermediate class

5. **Ensemble Approach:**
   - Combine LogisticRegression with RandomForest
   - Use crisis_composite_score as additional input
   - Average predictions for more robust detection

### Long-Term (Research)

6. **Sequential Model:**
   - Use LSTM/GRU to capture temporal patterns
   - Crisis events often have buildup → peak → decay pattern
   - Current model ignores time series structure

7. **Multi-Task Learning:**
   - Predict regime AND crisis_composite_score simultaneously
   - Forces model to learn crisis features explicitly

---

## Conclusion

The Logistic Regime Model V2 successfully implements all 5 critical fixes and shows **8x improvement** in crisis recall (2.2% → 18.2%). However, it still falls short of the 50% target for LUNA crash detection (12% vs 50%).

**Key Achievements:**
- ✅ Fixed VIX_Z inverse signal issue
- ✅ Removed broken features (oi, RV_20/60)
- ✅ Implemented SMOTE oversampling
- ✅ Stratified split ensures crisis in test set
- ✅ Production-ready model artifact

**Remaining Challenges:**
- ⚠️ Crisis recall 18% (need threshold tuning or more features)
- ⚠️ LUNA detection 12% (below 50% target)
- ⚠️ 81% of crisis samples misclassified as risk_off

**Recommendation:**
Deploy V2 with probability threshold tuning (`crisis_prob > 0.25`) to boost recall, then iterate on feature engineering in V3.

---

**Model Version:** v2.0
**Training Date:** 2026-01-09
**Author:** Claude Code (Backend Architect)
**Production Status:** Ready for deployment with threshold tuning
