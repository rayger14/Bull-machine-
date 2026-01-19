# LogisticRegimeModel v4 Training Plan

**Date**: 2026-01-13
**Objective**: Train v4 with 2018-2024 data to fix v3's low confidence (0.173 avg)
**Expected Outcome**: Confidence >0.40, enable proper hysteresis integration

---

## Executive Summary

**Problem**: V3 has average confidence of only 0.173 (barely better than random 0.25 for 4-class problem), causing:
- 591 regime transitions/year (vs 10-40 target)
- Hysteresis cannot stabilize (model never confident enough)
- Production deployment blocked

**Solution**: Train v4 on 6 years (2018-2023) vs v3's 2 years (2023-2024)
- More crisis examples (COVID, China ban, BCH fork, LUNA, FTX)
- Better calibration from larger dataset
- Higher confidence predictions
- Hysteresis will work naturally

---

## Crisis Period Labels for 2018-2021

### Major Crisis Events to Label

| Event | Start Date | End Date | Duration | Description |
|-------|------------|----------|----------|-------------|
| **2018 Bear Market** | 2018-11-14 | 2018-12-24 | 40 days | 52% drawdown, capitulation phase |
| **COVID-19 Crash** | 2020-03-08 | 2020-03-16 | 8 days | 63% drop in 2 days, extreme volatility |
| **COVID Recovery** | 2020-03-17 | 2020-03-30 | 13 days | Bounce from $3,800 to $6,400 |
| **China Ban Phase 1** | 2021-05-12 | 2021-05-23 | 11 days | Mining ban announcement, -50% |
| **China Ban Phase 2** | 2021-06-18 | 2021-06-30 | 12 days | Continued enforcement, -40% |
| **September Dip** | 2021-09-06 | 2021-09-08 | 2 days | Sharp correction from ATH approach |

### Crisis Period Labels (For Training)

```python
CRISIS_PERIODS_2018_2021 = [
    # 2018 Bear Market Capitulation
    {
        'start': '2018-11-14',
        'end': '2018-12-24',
        'label': 'crisis',
        'event': '2018_bear_capitulation',
        'confidence': 1.0,
        'notes': 'Final capitulation phase of 2018 bear market, 52% DD from $6k to $3.2k'
    },

    # COVID-19 Crash (Most severe crisis)
    {
        'start': '2020-03-08',
        'end': '2020-03-16',
        'label': 'crisis',
        'event': 'covid_crash',
        'confidence': 1.0,
        'notes': 'Black Thursday, 63% drop in 48 hours, liquidation cascade'
    },

    # COVID Recovery (Risk-Off → Risk-On transition)
    {
        'start': '2020-03-17',
        'end': '2020-03-30',
        'label': 'risk_off',
        'event': 'covid_recovery',
        'confidence': 0.8,
        'notes': 'Aggressive bounce after capitulation, high volatility'
    },

    # China Mining Ban (Phase 1)
    {
        'start': '2021-05-12',
        'end': '2021-05-23',
        'label': 'crisis',
        'event': 'china_ban_phase1',
        'confidence': 0.9,
        'notes': 'Initial mining ban announcement, -50% drop'
    },

    # China Ban Continuation (Phase 2)
    {
        'start': '2021-06-18',
        'end': '2021-06-30',
        'label': 'risk_off',
        'event': 'china_ban_phase2',
        'confidence': 0.8,
        'notes': 'Enforcement continues, secondary leg down'
    },

    # September 2021 Flash Correction
    {
        'start': '2021-09-06',
        'end': '2021-09-08',
        'label': 'risk_off',
        'event': 'sept_2021_correction',
        'confidence': 0.7,
        'notes': 'Sharp but brief correction from ATH approach'
    }
]
```

### Regime Heuristics for Unlabeled Periods

For periods without explicit crisis labels, use rules:

```python
def infer_regime_2018_2021(row):
    """
    Infer regime for unlabeled periods using heuristics

    Args:
        row: DataFrame row with features

    Returns:
        regime_label: 'crisis', 'risk_off', 'neutral', 'risk_on'
    """

    # Crisis indicators
    if row['RV_7'] > 3.0 or row['crash_frequency_7d'] >= 2:
        return 'crisis'

    # Risk-off indicators
    if row['DXY_Z'] > 1.5 or row['VIX_Z'] > 2.0:
        return 'risk_off'

    # Risk-on indicators
    if row['VIX_Z'] < -1.0 and row['returns_30d'] > 0.10:
        return 'risk_on'

    # Default: neutral
    return 'neutral'
```

---

## Training Dataset Construction

### Data Sources

**Primary**: CryptoCompare (2018-2021) + Existing Parquet (2022-2024)

| Period | Source | Bars | Status |
|--------|--------|------|--------|
| 2018-2021 | CryptoCompare | ~35,064 | ✓ Downloading |
| 2022-2024 | Existing parquet | ~26,236 | ✓ Available |
| **Total** | Combined | ~61,300 | 6.7 years |

### Train/Test Split

**Temporal Split** (no leakage):

```
Training:   2018-01-01 to 2023-12-31 (6 years)  → 52,584 bars
Test:       2024-01-01 to 2024-12-31 (1 year)   →  8,762 bars

Validation: 5-fold temporal CV on training set
```

**Rationale**:
- 2024 is true out-of-sample (not in training)
- Includes all major crises except FTX (2022)
- 6 years of data should significantly improve confidence

### Feature Set (Same as v3)

**12 Regime Features**:
```python
REGIME_FEATURES_V4 = [
    # Crisis/Volatility
    'crash_frequency_7d',
    'crisis_persistence',
    'aftershock_score',
    'rv_20d',               # Realized volatility
    'rv_60d',

    # Macro Sentiment
    'VIX_Z',                # VIX z-score
    'DXY_Z',                # Dollar strength
    'YC_SPREAD',            # Yield curve (2y-10y)

    # Derivatives
    'funding_Z',            # Funding rate z-score
    'oi_change_pct_24h',    # Open interest change

    # Price Momentum
    'returns_7d',
    'returns_30d'
]
```

**Note**: OI data only available 2024+, will be NaN for 2018-2023 (model should handle gracefully via imputation)

---

## Model Configuration

### Architecture (Same as v3)

```python
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from imblearn.over_sampling import SMOTE

# Base model
base_model = LogisticRegression(
    C=1.0,
    class_weight='balanced',
    max_iter=1000,
    random_state=42
)

# SMOTE oversampling (crisis is rare class)
smote = SMOTE(
    sampling_strategy={'crisis': 0.05},  # Oversample to 5% (from ~1%)
    random_state=42
)

# Calibration (critical for confidence)
calibrated_model = CalibratedClassifierCV(
    base_model,
    method='sigmoid',
    cv=5
)
```

### Target Metrics for v4

| Metric | V3 Actual | V4 Target | Importance |
|--------|-----------|-----------|------------|
| **Accuracy** | 61.5% | >75% | Medium |
| **Average Confidence** | 0.173 | **>0.40** | **CRITICAL** |
| **Crisis Recall** | 0% | >60% | High |
| **Risk-On Recall** | 21% | >60% | Medium |
| **Regime Transitions/Year** | 591 | N/A (raw model) | N/A |

**Most Important**: Average confidence >0.40 (enables hysteresis to work)

---

## Training Procedure

### Step 1: Data Preparation

```bash
# Combine historical and existing data
python3 bin/combine_historical_data.py \
  --historical data/raw/historical_2018_2021/CRYPTOCOMPARE_BTCUSD_1h_OHLCV.parquet \
  --existing data/features_2022_2024_streaming_signals.parquet \
  --output data/features_2018_2024_combined.parquet
```

### Step 2: Create Ground Truth Labels

```python
# In training script
def create_ground_truth_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create ground truth regime labels for training

    Combines:
    1. Explicit crisis period labels (CRISIS_PERIODS_2018_2021 + 2022-2024)
    2. Heuristic inference for unlabeled periods
    """

    # Initialize all as unlabeled
    df['regime_label'] = None

    # Apply explicit crisis labels
    for period in CRISIS_PERIODS_2018_2021:
        mask = (df.index >= period['start']) & (df.index <= period['end'])
        df.loc[mask, 'regime_label'] = period['label']

    # Add 2022-2024 crisis labels (LUNA, FTX, etc.)
    for period in CRISIS_PERIODS_2022_2024:
        mask = (df.index >= period['start']) & (df.index <= period['end'])
        df.loc[mask, 'regime_label'] = period['label']

    # Infer remaining labels using heuristics
    unlabeled = df['regime_label'].isna()
    df.loc[unlabeled, 'regime_label'] = df[unlabeled].apply(infer_regime_2018_2021, axis=1)

    return df
```

### Step 3: Train v4 Model

```bash
# Run training script (adapted from v3)
python3 bin/train_logistic_regime_v4.py \
  --data data/features_2018_2024_combined.parquet \
  --train-start 2018-01-01 \
  --train-end 2023-12-31 \
  --test-start 2024-01-01 \
  --test-end 2024-12-31 \
  --output models/logistic_regime_v4.pkl
```

Expected runtime: 5-10 minutes

### Step 4: Validation

```bash
# Validate on 2024 out-of-sample data
python3 bin/validate_logistic_regime_v4.py \
  --model models/logistic_regime_v4.pkl \
  --data data/features_2018_2024_combined.parquet \
  --test-year 2024
```

**Critical Checks**:
1. Average confidence >0.40 ✓
2. Crisis recall on 2024 events >60% ✓
3. No obvious biases (check confusion matrix)
4. Confidence histogram (should be well-distributed, not all at 0.25)

---

## Integration with RegimeService

Once v4 trained and validated:

### Step 1: Update Configuration

```python
# In bin/backtest_with_real_signals.py or production code
config = {
    'regime_model_path': 'models/logistic_regime_v4.pkl',  # Updated
    'hysteresis_config': {
        'enter_threshold': 0.60,     # Should work now with higher confidence
        'exit_threshold': 0.45,
        'min_duration_hours': {
            'crisis': 6,
            'risk_off': 24,
            'neutral': 12,
            'risk_on': 48
        },
        'enable_ema': True,
        'ema_alpha': 0.15
    }
}
```

### Step 2: Run Hysteresis Validation

```bash
# Re-run validation that failed with v3
python3 bin/validate_hysteresis_fix.py

# Expected results:
#   Transitions/year: 10-40 (vs v3's 2-4 or 591)
#   PF: >1.2 (vs v3's 0.96)
#   Trades: Similar to Phase 3 baseline
```

### Step 3: Paper Trading Deployment

If validation passes:

```bash
# Deploy to paper trading with v4
python3 bin/deploy_paper_trading.py \
  --model models/logistic_regime_v4.pkl \
  --capital 5000 \
  --duration 30  # 30 days validation
```

---

## Success Criteria

### Required for v4 Approval

- [ ] Average confidence >0.40 (vs v3's 0.173)
- [ ] Accuracy >75% on 2024 test set
- [ ] Crisis recall >60% on major events (COVID, China, LUNA, FTX)
- [ ] Regime transitions 10-40/year with hysteresis
- [ ] No catastrophic failures (all events detected)

### Nice to Have

- [ ] Risk-on recall >60%
- [ ] Confidence >0.50 for crisis predictions
- [ ] Smooth regime distribution (not stuck in one regime)

---

## Timeline

| Task | Effort | Dependency |
|------|--------|------------|
| Data download (2018-2021) | 10 min | ✓ In progress |
| Combine datasets | 15 min | After download |
| Create crisis labels | 30 min | Parallel |
| Adapt training script (v3→v4) | 1 hour | Parallel |
| Train v4 model | 10 min | After data prep |
| Validate v4 | 30 min | After training |
| Integrate with hysteresis | 30 min | After validation |
| **Total** | **3-4 hours** | Sequential |

**Target completion**: End of today (2026-01-13)

---

## Rollback Plan

If v4 doesn't meet success criteria:

**Option 1**: Deploy Phase 3 baseline (accept 591 transitions/year)
- Known profitable (PF 1.11, +$240 PnL)
- Noisy but functional
- Monitor in paper trading

**Option 2**: Use Hybrid model instead
- Crisis rules + ML
- Already validated (75% LUNA recall)
- Trade-off: More complex, similar transitions

**Option 3**: Acquire more data (2015-2017)
- Extend to 9 years instead of 6
- More halving cycles
- Diminishing returns (data quality degrades)

---

## Risk Mitigation

### Risk 1: v4 confidence still low (<0.40)

**Mitigation**:
- Check calibration is working (plot calibration curve)
- Try alternative calibration methods (isotonic vs sigmoid)
- Increase SMOTE oversampling for rare classes

### Risk 2: Crisis recall still 0%

**Mitigation**:
- Fall back to Hybrid model (crisis rules + ML)
- Adjust crisis labeling (may be too conservative)
- Add crisis features (higher weight in model)

### Risk 3: Model overfits to training data

**Mitigation**:
- Use 5-fold temporal CV (built into calibration)
- Add L2 regularization (tune C parameter)
- Check performance on each year 2018-2023 separately

---

## Next Actions

1. **Wait for data download to complete** (~5 min remaining)
2. **Combine 2018-2021 with 2022-2024 data**
3. **Create training script for v4** (adapt v3 script)
4. **Train v4 model**
5. **Validate confidence >0.40**
6. **Deploy if successful**

---

**Prepared by**: Claude Code
**Date**: 2026-01-13
**Status**: Ready for execution after data download
