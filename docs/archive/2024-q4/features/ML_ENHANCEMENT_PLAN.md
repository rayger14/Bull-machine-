# XGBoost Meta-Filter Enhancement Plan

## Executive Summary

**Goal**: Add ML meta-layer to further refine BTC v7's already excellent 89.5% WR → target 93-95% WR

**Approach**: XGBoost binary classifier predicts trade success probability at each candidate entry

**Expected Impact**:
- Win Rate: 89.5% → 93-95% (+3-5pp)
- Profit Factor: 15.73 → 18-22 (+15-40%)
- Trades: 19 → 12-15 (more selective)
- Sharpe: Improvement expected

---

## Architecture

```
┌─────────────────────────────────────────┐
│ Phase 1: Archetype Engine (EXISTING)   │
│ • 11 archetypes with fusion thresholds │
│ • Regime filters + dynamic sizing      │
│ • Result: 19 candidate entries         │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Phase 2: XGBoost Meta-Filter (NEW)     │
│ • Input: 50+ features per entry        │
│ • Output: Trade quality score [0-1]    │
│ • Threshold: Accept if score >= 0.65   │
│ • Result: 12-15 high-quality entries   │
└──────────────┬──────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────┐
│ Phase 3: Enhanced Sizing (OPTIONAL)    │
│ • Base: Fusion + regime multipliers    │
│ • ML bonus: 0.8x-1.2x based on ML conf │
└─────────────────────────────────────────┘
```

---

## Implementation Steps

### Step 1: Generate Training Data ⏳

We need historical trades (2022-2023) with full feature sets to train the model.

**Action Required**: Create trade export functionality

```bash
# Proposed: Modify backtest to export trades with features
python3 bin/backtest_knowledge_v2.py \
    --asset BTC \
    --start 2022-01-01 \
    --end 2023-12-31 \
    --config reports/optuna_btc_frontier_v7/BTC_best_config.json \
    --export-trades reports/ml/btc_trades_2022_2023.csv
```

**Required Features** (50+):

1. **Archetype Context** (12)
   - One-hot: `archetype_{A-M}`
   - `entry_fusion_score`
   - `entry_liquidity_score`

2. **Market State** (15)
   - `macro_regime_*` (one-hot: risk_on, neutral, risk_off, crisis)
   - `vix_level`, `vix_z_score`
   - `btc_volatility_percentile`
   - `volume_zscore`, `atr_percentile`
   - `adx_14`, `rsi_14`, `macd_histogram`

3. **Multi-Timeframe** (12)
   - `tf1h_fusion`, `tf4h_fusion`, `tf1d_fusion`
   - `tf4h_trend_aligned`, `tf1d_trend_aligned`
   - `nested_structure_quality`

4. **Microstructure** (10)
   - `boms_strength`, `fvg_quality`
   - `wyckoff_phase_score`, `poc_distance`
   - `lvn_trap_risk`, `liquidity_sweep_strength`

5. **Recent Performance** (8)
   - `last_3_trades_wr`
   - `bars_since_last_trade`
   - `recent_dd_pct`
   - `streak_length` (win/loss)

6. **Timing** (4)
   - `hour_of_day`, `day_of_week`, `days_into_quarter`

**Target Variable**: `trade_won` = 1 if r_multiple > 0, else 0

---

### Step 2: Train XGBoost Model ✅

**Script Created**: `bin/train/train_trade_quality_filter.py`

**Algorithm**: XGBoost Classifier
- Objective: Binary logistic
- Max depth: 5 (prevent overfitting)
- Learning rate: 0.05 (conservative)
- N estimators: 300 with early stopping
- Scale pos weight: Auto-balance classes

**Validation**: Time-Series Cross-Validation (5 folds)
- Prevents lookahead bias
- Each fold trains on past, validates on future
- Final model = best F1 score across folds

**Threshold Optimization**:
- Precision-recall curve analysis
- Maximize F1 score OR
- Target precision @ recall >= 0.90

**Usage**:
```bash
python3 bin/train/train_trade_quality_filter.py \
    --data reports/ml/btc_trades_2022_2023.csv \
    --output models/btc_trade_quality_filter_v1.pkl \
    --n-splits 5
```

**Output**:
- Trained model (`.pkl` file)
- Optimal threshold
- CV performance metrics
- Feature importance rankings

---

### Step 3: Wire ML Filter into Backtest ⏳

**Required Changes to** `bin/backtest_knowledge_v2.py`:

#### 3a. Add ML Model Loading

```python
def __init__(self, ...):
    ...
    self._load_ml_model()

def _load_ml_model(self):
    """Load XGBoost model if ML enabled in config"""
    ml_config = self.runtime_config.get('ml', {})
    if ml_config.get('enabled') and ml_config.get('model_path'):
        import joblib
        model_data = joblib.load(ml_config['model_path'])
        self.ml_model = model_data['model']
        self.ml_feature_names = model_data['feature_names']
        self.ml_threshold = ml_config.get('entry_threshold', model_data['threshold'])
        self.ml_sizing = ml_config.get('use_for_sizing', True)
        logger.info(f"ML model loaded: threshold={self.ml_threshold:.3f}")
    else:
        self.ml_model = None
```

#### 3b. Feature Extraction

```python
def _extract_ml_features(self, row: pd.Series, context: Dict) -> np.ndarray:
    """Extract 50+ features for ML prediction"""
    features = {}

    # Archetype one-hot
    archetype = context.get('entry_archetype', 'none')
    for arch in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M', 'N']:
        features[f'archetype_{arch.lower()}'] = 1 if archetype == arch else 0

    # Fusion & liquidity
    features['entry_fusion_score'] = context.get('fusion_score', 0.0)
    features['entry_liquidity_score'] = context.get('liquidity_score', 0.0)

    # Market state
    regime = context.get('macro_regime', 'neutral')
    features['macro_regime_risk_on'] = 1 if regime == 'risk_on' else 0
    features['macro_regime_neutral'] = 1 if regime == 'neutral' else 0
    features['macro_regime_risk_off'] = 1 if regime == 'risk_off' else 0
    features['macro_regime_crisis'] = 1 if regime == 'crisis' else 0

    # Technical indicators
    features['atr_percentile'] = row.get('atr_percentile', 0.5)
    features['adx_14'] = row.get('adx', 0.0)
    features['rsi_14'] = row.get('rsi', 50.0)
    # ... extract all 50+ features

    # Return as ordered array matching training feature names
    return np.array([features.get(name, 0.0) for name in self.ml_feature_names])
```

#### 3c. ML Filter Check

```python
def check_ml_filter(self, row: pd.Series, context: Dict) -> tuple:
    """
    Apply ML meta-filter.
    Returns: (ml_score, size_multiplier)
    """
    if not self.ml_model:
        return 1.0, 1.0  # Pass-through if ML disabled

    # Extract features
    features = self._extract_ml_features(row, context)

    # Predict probability
    ml_proba = self.ml_model.predict_proba([features])[0][1]

    # Optional: Adjust position size based on ML confidence
    if self.ml_sizing:
        size_mult = 0.8 + 0.4 * ml_proba  # Range: [0.8x, 1.2x]
    else:
        size_mult = 1.0

    return ml_proba, size_mult
```

#### 3d. Integration into Entry Logic

```python
def check_entry_signal(...):
    ...
    if archetype_name:
        # Existing archetype logic passes

        # NEW: ML meta-filter
        ml_score, ml_size_mult = self.check_ml_filter(row, context)

        if ml_score < self.ml_threshold:
            logger.info(f"ML FILTER REJECT: {archetype_name} | ml_score={ml_score:.3f} < {self.ml_threshold:.3f}")
            return None

        logger.info(f"ML FILTER ACCEPT: {archetype_name} | ml_score={ml_score:.3f} | size_mult={ml_size_mult:.2f}x")

        # Store ML metadata
        context['ml_entry_score'] = ml_score
        context['ml_size_mult'] = ml_size_mult

        return (entry_type, entry_price)
```

---

### Step 4: Configuration

**New Config Section**:

```json
{
  "ml": {
    "enabled": true,
    "model_path": "models/btc_trade_quality_filter_v1.pkl",
    "entry_threshold": 0.65,
    "use_for_sizing": true,
    "sizing_range": [0.8, 1.2]
  }
}
```

**Create ML-enabled config**:
```bash
cp reports/optuna_btc_frontier_v7/BTC_best_config.json \
   configs/btc_v7_ml_enabled.json
# Then add ML section above
```

---

### Step 5: Validation on 2024 ⏳

**Baseline (No ML)**:
```bash
python3 bin/backtest_knowledge_v2.py \
    --asset BTC \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --config reports/optuna_btc_frontier_v7/BTC_best_config.json
```

Result: 19 trades, 89.5% WR, PF 15.73

**ML-Enhanced**:
```bash
python3 bin/backtest_knowledge_v2.py \
    --asset BTC \
    --start 2024-01-01 \
    --end 2024-12-31 \
    --config configs/btc_v7_ml_enabled.json
```

Expected: 12-15 trades, 93-95% WR, PF 18-22

---

## Acceptance Criteria

| Metric | Baseline | ML Target | Status |
|--------|----------|-----------|--------|
| Win Rate | 89.5% | ≥92.5% (+3pp min) | ⏳ |
| Profit Factor | 15.73 | ≥17.3 (+10% min) | ⏳ |
| Sharpe | TBD | > Baseline | ⏳ |
| Trade Count | 19 | ≥10 (sig. sample) | ⏳ |
| Max DD | 0.0% | ≤ Baseline | ⏳ |

**Pass Gates**:
- ✅ If ML meets all criteria → Deploy
- ⚠️ If marginal improvement → Iterate on features/threshold
- ❌ If regression → Debug or abandon

---

## Feature Importance Analysis

Post-training, analyze which features drive predictions:

```python
# Top predictors (hypothesis):
# 1. entry_fusion_score (quality signal)
# 2. macro_regime (market context)
# 3. tf4h_trend_aligned (MTF confirmation)
# 4. last_3_trades_wr (momentum/streak)
# 5. liquidity_sweep_strength (trap detection)
```

Use insights to:
- Refine archetype thresholds
- Adjust fusion weights
- Improve regime classification

---

## Next Actions

- [ ] **CRITICAL**: Implement trade export in backtest (add `--export-trades` flag)
- [ ] Generate 2022-2023 BTC training data
- [ ] Train XGBoost model with cross-validation
- [ ] Wire ML filter into backtest engine
- [ ] Validate on BTC 2024
- [ ] Compare metrics vs baseline
- [ ] Feature importance analysis
- [ ] Document final results

---

## Risk Mitigation

**Overfitting Prevention**:
- Time-series CV (no lookahead)
- Max depth = 5 (conservative)
- Early stopping
- Holdout validation (2024)

**Production Safeguards**:
- ML is **additive** - can disable via config
- Fallback to baseline if model fails to load
- Threshold tuning post-deployment

**Monitoring**:
- Track ML rejection rate
- Monitor feature drift over time
- Retrain quarterly with new data

---

## References

**Training Script**: `bin/train/train_trade_quality_filter.py`

**Expected Model**: `models/btc_trade_quality_filter_v1.pkl`

**Training Data** (TBD): `reports/ml/btc_trades_2022_2023.csv`

**Live Config**: `configs/btc_v7_ml_enabled.json`
