# Regime Model Performance Analysis - Deep Dive

**Date:** 2026-01-08
**Model:** LogisticRegimeModel (macro_regime predictions)
**Analysis Period:** 2022-01-01 to 2024-12-31 (26,236 bars)

---

## Executive Summary

### Critical Finding: Model is SEVERELY UNDERPREDICTING Crisis

The LogisticRegimeModel has a **97.8% false negative rate** for crisis detection:
- **Predicts:** 1.4% crisis (357 bars)
- **Reality:** 4.2% crisis (1,104 bars)
- **Detects only:** 2.2% of actual crisis periods (24 out of 1,104 bars)
- **Misses:** 1,080 crisis bars (97.8% of all crisis)

**This is the OPPOSITE of the reported 68.1% over-prediction issue** - the model is actually extremely conservative and missing almost all crisis periods.

---

## 1. Distribution Analysis

### Predicted vs Actual Regimes

| Regime    | Predicted | Actual | Delta     |
|-----------|-----------|--------|-----------|
| crisis    | 357 (1.4%)| 1,104 (4.2%) | **-747 (-67.7%)** |
| neutral   | 25,333 (96.6%) | 8,085 (30.8%) | +17,248 (+213.3%) |
| risk_off  | 380 (1.4%) | 9,053 (34.5%) | **-8,673 (-95.8%)** |
| risk_on   | 166 (0.6%) | 7,994 (30.5%) | **-7,828 (-97.9%)** |

**Key Issue:** Model is massively over-predicting "neutral" (96.6% vs 30.8% actual) while severely under-predicting all other regimes, especially crisis.

---

## 2. Confusion Matrix Analysis

```
Confusion Matrix (rows=actual, cols=predicted):

                     crisis    neutral   risk_off    risk_on     TOTAL
------------------------------------------------------------------------
crisis                   24       1080          0          0      1104
neutral                  24       7918          0        143      8085
risk_off                309       8481        263          0      9053
risk_on                   0       7854        117         23      7994
------------------------------------------------------------------------
TOTAL                   357      25333        380        166     26236
```

### Crisis Row Analysis (Actual Crisis = 1,104 bars):
- **Correctly detected:** 24 bars (2.2%)
- **Misclassified as neutral:** 1,080 bars (97.8%) ← **CRITICAL FAILURE**
- **Misclassified as risk_off:** 0 bars
- **Misclassified as risk_on:** 0 bars

### Crisis Column Analysis (Predicted Crisis = 357 bars):
- **True crisis:** 24 bars (6.7% precision)
- **Actually risk_off:** 309 bars (86.6%) ← **Main false positive source**
- **Actually neutral:** 24 bars (6.7%)
- **Actually risk_on:** 0 bars

---

## 3. Per-Class Performance Metrics

| Regime    | Precision | Recall | F1 Score | Support |
|-----------|-----------|--------|----------|---------|
| **crisis**    | **0.067** | **0.022** | **0.033** | 1,104 |
| neutral   | 0.313 | 0.979 | 0.474 | 8,085 |
| risk_off  | 0.692 | 0.029 | 0.056 | 9,053 |
| risk_on   | 0.139 | 0.003 | 0.006 | 7,994 |
| **WEIGHTED AVG** | **0.380** | **0.314** | **0.168** | **26,236** |

**Overall Accuracy: 31.4%** (8,228 correct out of 26,236 predictions)

### Crisis Class Breakdown:
- **Precision: 6.7%** - Only 24 out of 357 predicted crisis are real crisis
- **Recall: 2.2%** - Only 24 out of 1,104 actual crisis are detected
- **F1 Score: 0.033** - Extremely poor overall performance

### Error Rates:
- **False Positive Rate:** 1.3% (333 false crisis / 25,132 non-crisis)
- **False Negative Rate:** 97.8% (1,080 missed / 1,104 total crisis)
- **Type II Error dominates** - missing crisis is the primary failure mode

---

## 4. Temporal Analysis - When Crisis is Missed

### True Crisis Periods (from ground truth):

| Period | Start | End | Total Bars | Detected | Detection Rate |
|--------|-------|-----|------------|----------|----------------|
| **May 2022 Crash** | 2022-05-01 | 2022-05-31 | **744 bars** | **0** | **0.0%** |
| **FTX Collapse** | 2022-11-01 | 2022-11-15 | **360 bars** | **24** | **6.7%** |

**CRITICAL:** The May 2022 crash (Terra/LUNA collapse, -50% BTC drawdown) had **ZERO** bars detected as crisis out of 744 hours. The model completely missed one of the most significant crypto crashes in history.

---

## 5. False Crisis Periods (What triggers false alarms)

### Top False Crisis Stretches:

| Period | Start | End | Bars | Actually |
|--------|-------|-----|------|----------|
| 1 | 2022-09-23 | 2022-09-27 | 120 | risk_off |
| 2 | 2022-06-13 | 2022-06-15 | 71 | risk_off |
| 3 | 2022-03-07 | 2022-03-08 | 48 | risk_off |
| 4 | 2022-10-10 | 2022-10-12 | 47 | risk_off |
| 5 | 2024-08-05 | 2024-08-05 | 24 | neutral |

**Pattern:** 92.8% of false crisis predictions are actually **risk_off** periods - the model confuses high volatility risk-off with crisis.

---

## 6. Feature Analysis - Root Cause Identification

### Feature Comparison: True Crisis vs False Crisis

| Feature | True Crisis Avg | False Crisis Avg | Difference | % Change |
|---------|----------------|------------------|------------|----------|
| **VIX_Z** | **0.330** | **2.138** | **+1.808** | **+548%** |
| drawdown_persistence | 0.800 | 0.944 | +0.144 | +18% |
| crisis_persistence | 0.043 | 0.126 | +0.083 | +196% |
| DXY_Z | 1.486 | 2.070 | +0.583 | +39% |
| crash_frequency_7d | 0.304 | 0.138 | -0.166 | -55% |

### ROOT CAUSE IDENTIFIED: VIX_Z Feature

**VIX_Z is 548% HIGHER in false crisis vs true crisis.**

This means:
1. **Model is trained to predict crisis when VIX_Z is HIGH**
2. **But actual crisis periods have LOWER VIX_Z (0.33) than false alarms (2.14)**
3. **Risk_off periods have very HIGH VIX_Z**, triggering false crisis predictions
4. **True crisis in crypto ≠ traditional market fear (VIX)**

### Why This Happens:
- VIX measures S&P 500 implied volatility
- Crypto crises (Terra/LUNA, FTX) are **crypto-specific events**
- During crypto-specific crises, VIX may be LOW because traditional markets are decoupled
- Risk_off periods often have HIGH VIX because they correlate with macro fear
- **Model confuses macro fear (high VIX) with crypto crisis**

---

## 7. Model Training Issues

### Validation Metrics (from logistic_regime_v1_validation.json):

```json
{
  "test_accuracy": 0.579,
  "test_samples": 5248,
  "confusion_matrix": [
    [0, 0, 0, 0],
    [0, 0, 0, 0],
    [0, 75, 0, 2133],
    [0, 0, 0, 3040]
  ]
}
```

**CRITICAL TRAINING FAILURE:**
- Confusion matrix shows **all zeros in rows 0 and 1** (crisis and one other class)
- This indicates the model **NEVER predicted crisis or another class during validation**
- The model only predicts classes in rows 2 and 3
- **This explains why production predictions show 96.6% neutral** - the model collapsed to predicting mostly one class

### Feature Importance (Top 5):
1. **VIX_Z: 2.121** ← Highest weight, but wrong signal
2. YIELD_CURVE: 1.928
3. BTC.D: 1.584
4. drawdown_persistence: 1.538
5. oi: 1.415

VIX_Z has the highest feature importance, confirming it drives predictions, but it's giving false signals.

---

## 8. Where the "68.1%" Number Came From

**The 68.1% crisis over-prediction is NOT in macro_regime predictions.**

Possible sources:
1. **Different model run** not captured in current features file
2. **HMM regime model** (separate from logistic model)
3. **Event override logic** in backtest (flash crash detection)
4. **Training set statistics** before model collapsed

From backtest logs, we see aggressive crisis event detection:
- Flash crash detection (>4% drop in 1H)
- Funding shock detection (|z| > 4.0)
- These may be creating many crisis signals not visible in macro_regime column

**Recommendation:** Need to check if there's a different regime prediction column or if crisis_confirmed column is being used.

---

## 9. Diagnostic Summary

### The Model Has THREE Severe Problems:

1. **Class Imbalance Collapse**
   - Model predicts 96.6% neutral in production
   - Training confusion matrix shows model NEVER predicted crisis during validation
   - This is a classic sign of class imbalance overfitting to majority class

2. **Wrong Feature Signal (VIX_Z)**
   - VIX_Z has highest importance (2.121)
   - But VIX_Z is 5.5x HIGHER in false crisis than true crisis
   - Model learned the OPPOSITE pattern
   - VIX measures macro fear, not crypto-specific crisis

3. **Extreme Conservatism**
   - 97.8% false negative rate
   - Missed 100% of May 2022 crash (744 bars)
   - Only detected 6.7% of FTX collapse
   - **Model is essentially broken for crisis detection**

---

## 10. Data-Driven Recommendations

### IMMEDIATE FIXES (Priority 1):

1. **RETRAIN with SMOTE or Class Weights**
   ```python
   # Crisis is 4.2% of data - needs balancing
   class_weights = {
       'crisis': 23.8,      # 100/4.2 = 23.8x weight
       'neutral': 3.2,      # 100/30.8 = 3.2x weight
       'risk_off': 2.9,     # 100/34.5 = 2.9x weight
       'risk_on': 3.3       # 100/30.5 = 3.3x weight
   }
   ```

2. **REMOVE or REWEIGHT VIX_Z Feature**
   - VIX_Z is giving inverse signal (high VIX → false crisis, low VIX → missed crisis)
   - Replace with crypto-native fear indicators:
     - funding_rate extremes
     - BTC realized volatility (not S&P implied vol)
     - drawdown speed
     - liquidation cascades

3. **ADD Crypto-Specific Crisis Features**
   - `funding_stress_7d`: Rolling funding rate stress
   - `cascade_score`: OI + volume + volatility composite
   - `btc_dominance_spike`: Rapid BTC.D increases during altcoin crashes
   - `exchange_stress`: Withdrawal/deposit abnormalities

### MEDIUM-TERM FIXES (Priority 2):

4. **THRESHOLD TUNING**
   - Current model uses 0.5 probability threshold
   - For crisis, optimize for recall (catch more crises even if false positives increase)
   - Suggested crisis threshold: 0.2 (predict crisis if P(crisis) > 0.2)

5. **ENSEMBLE APPROACH**
   - Logistic model alone is insufficient
   - Combine with:
     - HMM regime detection (temporal patterns)
     - Event-based rules (flash crash, funding shock)
     - Anomaly detection (Isolation Forest for unusual conditions)

6. **TEMPORAL VALIDATION**
   - Current validation likely used random split
   - Use walk-forward validation to prevent look-ahead bias
   - Test on held-out crisis periods (e.g., train on 2022, test on 2023 crises)

### LONG-TERM IMPROVEMENTS (Priority 3):

7. **RECURRENT MODEL**
   - Crisis is a temporal phenomenon
   - LSTM or GRU could capture crisis build-up patterns
   - Include lagged features (VIX_Z past 24h, not just current)

8. **MULTI-LABEL APPROACH**
   - Instead of 4 mutually exclusive classes
   - Predict crisis as binary (yes/no) separately from market direction (risk_on/off/neutral)
   - Crisis can occur during any market direction

9. **ACTIVE LEARNING**
   - Continuously retrain on new crisis events
   - Weight recent crises higher (FTX 2022 > older events)
   - Update model monthly with new data

---

## 11. Validation Plan

Before deploying any fix:

1. **Baseline Metrics**
   - Current: 2.2% recall, 6.7% precision, 0.033 F1
   - Target: >50% recall, >30% precision, >0.38 F1

2. **Critical Test Cases**
   - Must detect >50% of May 2022 crash (currently 0%)
   - Must detect >70% of FTX collapse (currently 6.7%)
   - False positive rate should stay <5%

3. **Out-of-Sample Testing**
   - Train on 2022 data
   - Test on 2023 crises
   - Validate on 2024 crises

4. **Feature Ablation**
   - Test model WITHOUT VIX_Z
   - Test model WITH crypto-native features only
   - Measure impact on precision/recall

---

## 12. Files for Further Investigation

1. **Check for alternative regime predictions:**
   ```bash
   # Look for crisis_confirmed or other regime columns
   python -c "import pandas as pd; df = pd.read_parquet('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'); print([c for c in df.columns if 'crisis' in c.lower()])"
   ```

2. **Examine event override logic:**
   - `engine/context/logistic_regime_model.py` - check crisis override rules
   - May be separate crisis detection that's more aggressive

3. **Review training script:**
   - `bin/train_logistic_regime.py` - check class weights, sampling, features

4. **Compare with HMM regime:**
   - `models/hmm_regime_*.pkl` - check if HMM performs better on crisis

---

## Conclusion

The LogisticRegimeModel is **SEVERELY BROKEN for crisis detection**:

1. **Misses 97.8% of all crisis periods**
2. **Completely missed the May 2022 crash** (0% detection on 744 bars)
3. **VIX_Z feature gives INVERSE signal** (high VIX → false crisis, low VIX → missed crisis)
4. **Model collapsed during training** (validation confusion matrix shows it never predicted crisis)
5. **Predicts 96.6% neutral** - essentially a constant predictor

**The 68.1% over-prediction number does NOT match current macro_regime predictions (1.4%).** This suggests either:
- Different model version was tested
- Different prediction column exists
- Event override logic is more aggressive than logistic model

**IMMEDIATE ACTION REQUIRED:**
1. Find source of 68.1% crisis prediction (not in macro_regime)
2. Retrain logistic model with class balancing
3. Remove or fix VIX_Z feature (inverse signal)
4. Validate on May 2022 crash as critical test case

---

**Analysis Generated:** 2026-01-08
**Full Results:** `/Users/raymondghandchi/Bull-machine-/Bull-machine-/results/regime_performance_analysis.txt`
