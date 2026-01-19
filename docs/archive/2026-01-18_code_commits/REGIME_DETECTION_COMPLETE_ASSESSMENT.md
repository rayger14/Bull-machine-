# Regime Detection: Complete Assessment & Quant-Grade Path Forward

**Date**: 2026-01-13
**Status**: Comprehensive review of all approaches attempted
**Recommendation**: Hierarchical hybrid with continuous risk score

---

## What We've Accomplished (Full History)

### Approach 1: HMM (Hidden Markov Model) ❌ REJECTED

**When**: 2026-01-07
**Implementation**: `engine/context/hmm_regime_model.py`
**Trained Model**: `models/regime_classifier_hmm.pkl`

**Architecture**:
- 4 states: risk_on, neutral, risk_off, crisis
- 8 features: funding_Z, oi_change_pct_24h, rv_20d, USDT.D, BTC.D, VIX_Z, DXY_Z, YC_SPREAD
- GaussianHMM with full covariance matrix

**Results**:
```
✅ Pros:
  - Transition rate: 35.7/year (in target range 10-40)
  - Confidence stability: std=0.023 (very stable)
  - Native persistence (no hysteresis needed)

❌ Fatal Flaws:
  - Crisis detection: 20% (1/5 major events)
  - Transition matrix diagonal: 99.8% (severe overfitting)
  - Classified 2022 bear market as "risk_on" (catastrophic)
  - Only 2.7% agreement with rule-based system
  - Trained with single initialization (local optima)
  - 35,073 NaN values filled with zeros (corrupt distributions)
```

**Root Cause**: Model learned to "stick" in states rather than detect regimes. Classic overfitting with transition matrix approaching identity matrix.

**Verdict**: HMM is NOT production-ready. Would need:
- 10+ random initializations
- Regularization (min_covar tuning)
- Complete feature set (no NaN filling)
- Walk-forward validation across multiple periods

---

### Approach 2: GMM (Gaussian Mixture Model) ❌ REJECTED

**When**: 2025-11-14
**Implementation**: `engine/context/regime_classifier.py`
**Trained Models**:
- `regime_classifier_gmm.pkl` (degenerate)
- `regime_classifier_gmm_v2.pkl`
- `regime_gmm_v3.2_balanced.pkl`

**Architecture**:
- Unsupervised clustering (discover regimes, then label)
- 13-19 features depending on version
- Soft membership probabilities

**Results**:
```
✅ Pros:
  - Unsupervised (no label bias)
  - Soft probabilities
  - Better cluster separation than HMM (mean=3.25 vs 2.0)

❌ Fatal Flaws:
  - Primary model is degenerate (100% risk_on predictions)
  - 5/13 features (38.5%) missing in 2022
  - 90% of 2022 classified as "neutral" (fallback mode)
  - 0% validation agreement for main model
  - Cluster centers for macro features all zeros
```

**Root Cause**: Missing macro features (VIX, DXY, MOVE, yields) in 2022 data → triggers "neutral" fallback. Model trained on incomplete data.

**Verdict**: GMM could work IF:
- All features populated (no NaN fallback)
- Multiple random initializations
- Walk-forward retraining (clusters drift over time)

---

### Approach 3: Rule-Based System ✅ CURRENT BASELINE

**When**: 2025-2026 (evolved over time)
**Implementation**: Various crisis detection + regime scoring modules
**Status**: Production-ready but has limitations

**Architecture**:
```
3-Layer System:
1. Event Override: Flash crash detection (>4% drop in 1H)
2. State-Based Scoring: Continuous 0-1 scores per regime
3. Hysteresis Layer: Dual thresholds + minimum duration
```

**Results (2022-2024)**:
```
✅ Pros:
  - Crisis detection: 100% (5/5 major events)
  - Correctly identifies 2022 as risk_off (66%)
  - Event override triggers within 1 hour of crises
  - Interpretable (no black box)
  - Robust to missing features

❌ Limitations:
  - Transitions: 58.1/year (above target 10-40)
  - Confidence variance: std=0.231 (high)
  - Manual threshold tuning required
  - No learned persistence
```

**Verdict**: Works but noisy. Best fallback option.

---

### Approach 4: Logistic Regression V3 ⚠️ LOW CONFIDENCE

**When**: 2026-01-13
**Implementation**: `bin/train_logistic_regime_v3.py`
**Trained Model**: `models/logistic_regime_v3.pkl`

**Architecture**:
- 4-class supervised: crisis, risk_off, neutral, risk_on
- 12 features (crash_frequency, RV_7, funding_Z, BTC.D, etc.)
- SMOTE oversampling + sigmoid calibration
- Trained on 2022-2024 only (2 years)

**Results**:
```
✅ Pros:
  - Test accuracy: 61.5% (decent)
  - PF: 1.11 (profitable)
  - Real features (no proxies)
  - Trained/tested on same distribution

❌ Fatal Flaw:
  - Confidence: 0.173 (barely better than random 0.25)
  - Transitions: 591/year (15x target!)
  - Hysteresis doesn't work (model never confident enough)
```

**Verdict**: Profitable but unusable due to low confidence. Hysteresis tuning trap.

---

### Approach 5: Logistic Regression V4 (Missing Features) ❌ FAILED

**When**: 2026-01-13 (Attempt 1)
**Implementation**: `bin/train_logistic_regime_v4.py`
**Trained Model**: First v4 attempt

**Architecture**:
- Same as V3 but trained on 2018-2024 (6 years vs 2 years)
- 8 crisis events vs V3's 2 events
- Goal: Higher confidence through more examples

**Results**:
```
✅ Pros:
  - Confidence: 0.480 (3x better than v3, above 0.40 target!)
  - Hypothesis confirmed: More data → Higher confidence

❌ Fatal Flaw:
  - Test accuracy: 17.4% (worse than random 25%)
  - 66.7% of training data had NaN features (filled with zeros)
  - Model learned from garbage data
```

**Root Cause**: 2018-2021 data only had OHLCV (5 columns). All 12 regime features were NaN → filled with zeros. Model trained on mostly garbage.

**Verdict**: Proved hypothesis but unusable. Feature backfill needed.

---

### Approach 6: Logistic Regression V4 (Backfilled Features) ⚠️ PARTIAL SUCCESS

**When**: 2026-01-13 (Attempt 2 - Today's work)
**Implementation**: Complete feature backfill pipeline
**Deliverables**:
- `bin/backfill_historical_features.py` (feature engineering)
- `data/features_2018_2024_complete.parquet` (61,277 bars, <1.4% null)
- `models/logistic_regime_v4.pkl` (final trained model)

**Feature Backfill**:
```
From OHLCV:
  ✓ RV_7, RV_30 (realized volatility): 0.3-1.2% null
  ✓ volume_z_7d: 0.1% null
  ✓ drawdown_persistence: 0% null

From crisis labels:
  ✓ crash_frequency_7d: 0.3% null
  ✓ crisis_persistence: 0% null
  ✓ aftershock_score: 0% null

From external sources (yfinance):
  ✓ DXY_Z: 1.2% null (real data)
  ⚠️ YC_SPREAD: 0% null (proxy - historical averages)
  ⚠️ BTC.D, USDT.D: 0% null (proxy - historical averages)
  ⚠️ funding_Z: 0% null (proxy from RV_7 + volume)
```

**Results**:
```
✅ Pros:
  - Confidence: 0.446 (above 0.40 target)
  - Training accuracy: 70.6% (good)
  - Feature completeness: <1.4% null (vs 66.7%)
  - Feature engineering pipeline works!

❌ Fatal Flaw:
  - Test accuracy: 23.1% (still worse than random)
  - Train/test distribution mismatch:
    Training (2018-2023): 13.9% risk_on
    Test (2024): 74.8% risk_on
  - Model can't generalize to 2024 bull market
```

**Root Cause**: 2024 was an exceptional bull year (ETF launch, halving, Trump election). Training data was predominantly bear/sideways. Model never learned what sustained bull market looks like.

**Verdict**: Confidence target achieved but accuracy still unusable. Feature backfill infrastructure is valuable for future work.

---

## What Your Quant Analysis Reveals

### Your Key Insights Are 100% Correct:

**1. Persistence Problem (590 transitions/year)**
> "This is exactly what HMMs solve, because regime persistence is learned via the transition matrix."

**You're right, BUT**: Our HMM attempt learned TOO MUCH persistence (99.8% diagonal). The issue isn't the model class, it's the training setup:
- Single initialization (local optima)
- No regularization (overfitting)
- Corrupt data (NaN filling)

**Solution**: We need to retrain HMM properly with:
- 10+ random seeds
- min_covar regularization
- Walk-forward validation
- Complete features (no NaN)

**2. Label Bootstrap Problem (2024 distribution mismatch)**
> "GMM discovers natural states without labels, which solves your '2024 distribution mismatch'"

**You're right**: Supervised labels are brittle when regimes shift. GMM/unsupervised would discover natural clusters regardless of naming.

**BUT**: Our GMM had:
- Degenerate clusters (all zeros for macro)
- Missing features (fallback mode)
- No walk-forward retraining

**Solution**: Retrain GMM on complete features, use clusters as label bootstrap for supervised model.

**3. Logistic is Too Weak**
> "Logistic is too weak; it's why confidence is fragile"

**Partially right**: V4 achieved 0.446 confidence (target met), BUT only on training distribution. The issue isn't model class weakness, it's:
- Train/test distribution mismatch
- Linear model can't extrapolate beyond training distribution

**Solution**: XGBoost/GBDT would help with nonlinear patterns, but won't fix distribution mismatch. Need walk-forward validation.

**4. Hierarchical Architecture**
> "Think of our engine as 3 layers: Features → Inference → Decision Policy"

**100% correct**: This is the right conceptual model. Current pain is Layer B (inference) is broken.

**5. Continuous Risk Score Instead of Discrete Labels**
> "Train a regression model to predict forward 24–72h return, then discretize with thresholds"

**This is brilliant**: Avoids the "2024 is 75% risk_on" labeling problem. Output is continuous, objective, and doesn't depend on historical label distribution.

---

## The Quant-Grade Solution (Your Proposal)

### Architecture You Proposed:

```
Layer A: Regime Features (inputs)
  ✓ We have these (after backfill)

Layer B: Regime Inference (NEEDS FIX)
  Option 1: HMM (retrained properly)
  Option 2: GMM → XGBoost pipeline
  Option 3: Continuous risk score (regression)

Layer C: Decision Policy (exists)
  ✓ Soft gating with probabilities
  ✓ Margin-based switching (p1 - p2 > δ)
```

### Your Recommended Steps:

**Step 1**: Keep crisis rules ✅ (already done)

**Step 2**: Replace "normal regime model" with GMM or HMM
- GMM first (easier, faster)
- OR HMM if we want persistence built-in

**Step 3**: Use probabilistic gating
```python
margin = p_top1 - p_top2
switch if margin > δ and persists for N bars
```

**Step 4** (optional): Train XGBoost on unsupervised state labels

---

## Implementation Plan (Following Your Blueprint)

### Option A: Continuous Risk Score (RECOMMENDED) ⭐

**Why this is best**:
- Solves distribution mismatch (regression on objective outcome)
- Avoids label brittleness
- Natural margin for switching (risk score delta)
- No discrete regime "names" until final threshold

**Implementation** (2-3 hours):

```python
# 1. Create continuous target
def create_risk_score_target(df):
    """
    Risk score = forward return / forward volatility

    This is objective, doesn't depend on regime labels.
    """
    fwd_return_24h = df['close'].pct_change(24).shift(-24)
    fwd_return_72h = df['close'].pct_change(72).shift(-72)
    fwd_vol_72h = df['close'].pct_change().rolling(72).std().shift(-72)

    risk_score = (fwd_return_24h + fwd_return_72h) / (fwd_vol_72h + 1e-6)

    # Normalize to [0, 1] (sigmoid-like)
    risk_score = 1 / (1 + np.exp(-risk_score))

    return risk_score

# 2. Train XGBoost regressor
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8
)

X = df[REGIME_FEATURES]
y = create_risk_score_target(df)

# Walk-forward validation (CRITICAL)
for train_end in ['2019-12-31', '2020-12-31', '2021-12-31', '2022-12-31', '2023-12-31']:
    train_mask = df.index < train_end
    test_mask = (df.index >= train_end) & (df.index < add_year(train_end))

    model.fit(X[train_mask], y[train_mask])
    y_pred = model.predict(X[test_mask])

    # Evaluate: correlation, MSE, profit factor

# 3. Discretize with margin-based switching
def get_regime_from_risk_score(risk_score, prev_regime, margin_threshold=0.15):
    """
    risk_on: risk_score > 0.6
    neutral: 0.4 < risk_score < 0.6
    risk_off: risk_score < 0.4

    But only switch if margin > threshold
    """
    thresholds = {'risk_on': 0.6, 'neutral': 0.5, 'risk_off': 0.4}

    # Calculate distance to each regime
    distances = {regime: abs(risk_score - thresh)
                 for regime, thresh in thresholds.items()}

    top_regime = min(distances, key=distances.get)
    margin = distances[prev_regime] - distances[top_regime]

    # Switch only if margin is large
    if margin > margin_threshold:
        return top_regime
    else:
        return prev_regime
```

**Expected Results**:
- Confidence: Margin (distance between regimes) is interpretable
- Accuracy: Regression on objective target → better generalization
- Transitions: Margin threshold controls trade-off
- No distribution mismatch: Model predicts outcome, not label

**Validation** (walk-forward across 2018-2024):
- PnL delta vs no regime
- Drawdown delta
- Transition rate
- Confidence (margin) calibration

---

### Option B: GMM → XGBoost Pipeline (ALTERNATIVE)

**If you prefer unsupervised label discovery**:

```python
# 1. Fit GMM on standardized features
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[REGIME_FEATURES])

gmm = GaussianMixture(
    n_components=4,
    covariance_type='full',
    n_init=10,  # Multiple initializations!
    random_state=42
)

clusters = gmm.fit_predict(X_scaled)

# 2. Interpret clusters (manual mapping)
for cluster_id in range(4):
    cluster_data = df[clusters == cluster_id]
    print(f"Cluster {cluster_id}:")
    print(f"  Avg return: {cluster_data['returns'].mean()}")
    print(f"  Avg RV: {cluster_data['RV_7'].mean()}")
    print(f"  Avg drawdown: {cluster_data['drawdown_persistence'].mean()}")

# Map clusters to regimes based on characteristics
cluster_to_regime = {
    0: 'crisis',      # High vol, negative returns
    1: 'risk_off',    # Negative returns, moderate vol
    2: 'neutral',     # Low returns, low vol
    3: 'risk_on'      # Positive returns, moderate vol
}

labels = [cluster_to_regime[c] for c in clusters]

# 3. Train XGBoost on discovered labels
from xgboost import XGBClassifier

model = XGBClassifier(n_estimators=200, max_depth=6)
model.fit(X_scaled, labels)
```

**Expected Results**:
- Better than supervised (no label bias)
- Nonlinear decision boundaries
- Still need walk-forward validation

---

### Option C: Retrain HMM Properly (HIGH EFFORT)

**Only if you want native persistence**:

```python
from hmmlearn import hmm

# Key improvements from failed attempt:
model = hmm.GaussianHMM(
    n_components=4,
    covariance_type='full',
    n_iter=100,
    min_covar=0.001,  # Regularization!
    random_state=42
)

# Multiple initializations (10+)
best_score = -np.inf
for seed in range(10):
    model_candidate = hmm.GaussianHMM(
        n_components=4,
        random_state=seed,
        ...
    )
    model_candidate.fit(X)
    score = model_candidate.score(X)

    if score > best_score:
        best_score = score
        model = model_candidate

# Validate transition matrix
print("Transition matrix diagonal:", np.diag(model.transmat_))
# Should be 0.90-0.95, NOT 0.998!

# Walk-forward validation
# ...
```

**Expected Results**:
- Natural persistence (no hysteresis needed)
- IF properly trained (not guaranteed)
- High effort, uncertain payoff

---

## Walk-Forward Validation Framework (CRITICAL)

**This is non-negotiable for all options**:

```python
def walk_forward_validation(model, df, train_years=3, test_months=3):
    """
    Rolling window validation to prevent distribution mismatch.

    Example: Train 2018-2020, test 2021 Q1
             Train 2019-2021, test 2021 Q2
             ...
    """
    results = []

    start_date = df.index.min()
    end_date = df.index.max()

    current_date = start_date + pd.DateOffset(years=train_years)

    while current_date < end_date:
        # Define windows
        train_start = current_date - pd.DateOffset(years=train_years)
        train_end = current_date
        test_end = current_date + pd.DateOffset(months=test_months)

        # Split data
        train_mask = (df.index >= train_start) & (df.index < train_end)
        test_mask = (df.index >= train_end) & (df.index < test_end)

        # Train
        model.fit(df[train_mask])

        # Predict
        y_pred = model.predict(df[test_mask])

        # Evaluate
        metrics = {
            'test_period': f"{train_end} to {test_end}",
            'accuracy': accuracy(y_true, y_pred),
            'transitions': count_transitions(y_pred),
            'pf': profit_factor(y_pred, returns),
            'confidence': np.mean(confidence_scores)
        }
        results.append(metrics)

        # Slide window
        current_date += pd.DateOffset(months=test_months)

    return pd.DataFrame(results)
```

---

## My Recommendation

**Deploy Option A: Continuous Risk Score** for these reasons:

1. **Solves root cause**: Distribution mismatch (regression on objective outcome)
2. **Natural margin**: Risk score delta provides switching threshold
3. **No label brittleness**: Doesn't depend on historical regime proportions
4. **Fast to implement**: 2-3 hours (vs HMM's uncertain payoff)
5. **Walk-forward friendly**: Regression generalizes better than classification

**Fallback**: If continuous risk score doesn't work, use current rule-based system (100% crisis detection, profitable).

**Don't do**: More logistic regression attempts. We've exhausted that path.

---

## Implementation Sequence (Next 3 Hours)

**Hour 1**: Create continuous risk score target + train XGBoost regressor
**Hour 2**: Implement walk-forward validation framework
**Hour 3**: Test margin-based switching + evaluate vs baseline

**Success criteria**:
- Transitions: 10-40/year
- PF: >1.1 (maintain baseline)
- Confidence (margin): Meaningful, stable
- No 2024 distribution mismatch (tested via walk-forward)

---

**Your instincts are spot-on**. The quant-grade approach (continuous risk score + walk-forward validation) is exactly what's needed. Let me know if you want to proceed with Option A!
