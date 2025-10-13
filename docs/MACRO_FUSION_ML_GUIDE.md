# Bull Machine v1.8.6 - Macro Fusion + ML Framework

## Overview

Bull Machine v1.8.6 introduces two major enhancements:

1. **8-Factor Macro Fusion**: Composite score that adjusts fusion signals based on macro regime (DXY, Oil, Gold, Yields, Breadth, USDT.D)
2. **ML-Based Config Suggestion**: LightGBM/XGBoost models that suggest optimal configs based on current regime features

Both features are **opt-in** with feature flags and designed for safe rollout.

---

## Part 1: Macro Fusion (8-Factor Composite)

### What It Does

Macro Fusion adjusts the domain fusion score (Wyckoff + SMC + HOB + Momentum) by analyzing 8 macro factors:

| Factor | Weight | Bullish Signal | Bearish Signal |
|--------|--------|----------------|----------------|
| **DXY** (Dollar) | 15% | Breakdown (<100, -1.5σ) | Breakout (>105, +1.5σ) |
| **Oil** (WTI) | 10% | Cooling (ROC <-15%) | Hot (ROC >+15%, inflation) |
| **Gold** (XAUUSD) | 5% | Neutral | Flight-to-safety (ROC >+10%) |
| **Yields** (US10Y) | 10% | Neutral | Spike (ROC >+10%, bond stress) |
| **Curve** (10Y-2Y) | 5% | Steepening (+30bps vs EMA) | Flattening (-30bps, recession) |
| **Breadth** (TOTAL/TOTAL2) | 5% | Strong (BTC.D <50%) | Weak (BTC.D >60%, flight) |
| **USDT.D** (Stablecoin) | 5% | Breakdown (-1.5σ) | Breakout (+1.5σ, risk-off) |

**Composite Score**: Weighted sum of adjustments, capped at **±0.10** (max 10% boost/penalty to fusion score).

### Configuration

Located in [`configs/v18/context_defaults.json`](../configs/v18/context_defaults.json):

```json
{
  "macro_fusion": {
    "enabled": false,  // Feature flag - set to true to enable

    "weights": {
      "dxy_weight": 0.15,
      "oil_weight": 0.10,
      "gold_weight": 0.05,
      "yield_weight": 0.10,
      "curve_weight": 0.05,
      "breadth_weight": 0.05,
      "usdt_d_weight": 0.05
    },

    "thresholds": {
      "dxy_breakout_z": 1.5,       // Z-score threshold for DXY breakout
      "oil_hot_z": 1.5,             // Z-score threshold for oil surge
      "gold_flight_z": 1.5,         // Z-score threshold for gold flight
      "yield_spike_roc": 10.0,      // ROC% threshold for yield spike
      "curve_steep_threshold": 0.30, // Spread threshold for steepening
      "breadth_dominance_low": 0.50, // BTC.D < 50% = alt season
      "breadth_dominance_high": 0.60, // BTC.D > 60% = flight
      "usdt_d_breakout_z": 1.5      // Z-score threshold for USDT.D breakout
    },

    "caps": {
      "max_boost": 0.10,     // Max positive adjustment
      "max_penalty": -0.10   // Max negative adjustment
    }
  }
}
```

### How It Works

1. **Macro Engine** ([`engine/context/macro_engine.py`](../engine/context/macro_engine.py)) calls `build_macro_composite()` from [`engine/context/macro_signals.py`](../engine/context/macro_signals.py)

2. **State Detection**: Each factor analyzed independently:
   ```python
   # Example: DXY breakout detection
   if dxy_zscore > 1.5 and dxy_ema_fast > dxy_ema_slow:
       state = 'breakout'
       adjustment = -0.10  # Risk-off for crypto
   ```

3. **Composite Score**: Weighted sum of all adjustments:
   ```python
   composite = (dxy_adj * 0.15) + (oil_adj * 0.10) + ...
   composite = np.clip(composite, -0.10, 0.10)  # Cap at ±10%
   ```

4. **Fusion Boost**: Applied in [`engine/fusion/domain_fusion.py`](../engine/fusion/domain_fusion.py):
   ```python
   if macro_fusion_enabled:
       fusion_score = fusion_score + composite  # e.g., 0.68 + 0.05 = 0.73
   ```

### Example

**Scenario**: Bull market with weak dollar and strong breadth

- DXY: 98.5 (-1.8σ) → **+0.05** (breakdown)
- Oil: $72 (flat) → **0.0** (neutral)
- Gold: $2480 (flat) → **0.0** (neutral)
- Yields: Stable → **0.0** (neutral)
- Curve: 10Y-2Y = +0.65% → **+0.03** (steepening)
- Breadth: BTC.D = 48% → **+0.05** (alt season)
- USDT.D: 4.2% (falling) → **+0.03** (risk-on)

**Composite**: `(0.05*0.15) + (0*0.10) + ... = +0.08` → **+8% fusion boost**

If base fusion score = 0.68, adjusted = **0.76** → More aggressive entries.

---

## Part 2: ML-Based Config Suggestion

### What It Does

Uses historical optimization results to train a model that predicts config performance (Sharpe/PF) based on:

- **Regime features**: VIX, MOVE, DXY, Oil, Gold, Yields, Curve, Breadth (30+ features)
- **Config parameters**: Fusion threshold, stop ATR, trail ATR, ADX threshold, etc.

The model suggests top-N configs for the **current regime** without running expensive backtests.

### Workflow

```
1. Run optimizations → Collect results
   ↓
2. Train ML model on results
   ↓
3. Use model to suggest configs for current regime
   ↓
4. Backtest suggested config
   ↓
5. Deploy if metrics meet criteria
```

### Step 1: Collect Training Data

Run optimizations and save results:

```bash
# Grid search on BTC (3 years)
python bin/optimize_v19.py --mode grid --asset BTC --years 3 --output results/btc_grid.json

# Quick search on ETH (1 year)
python bin/optimize_v19.py --mode quick --asset ETH --years 1 --output results/eth_quick.json
```

This generates JSON files with optimization results:
```json
[
  {
    "config": {...},
    "sharpe": 2.14,
    "profit_factor": 1.39,
    "max_drawdown": -0.08,
    "total_trades": 40,
    "win_rate": 0.65,
    "total_return_pct": 0.23
  },
  ...
]
```

### Step 2: Train ML Model

Use [`bin/research/train_ml.py`](../bin/research/train_ml.py) to train LightGBM model:

```bash
# Train Sharpe predictor (default)
python bin/research/train_ml.py --target sharpe --asset BTC --min-trades 50 --max-dd 0.15

# Train Profit Factor predictor with XGBoost
python bin/research/train_ml.py --target pf --model xgboost --min-trades 100

# Train with normalization
python bin/research/train_ml.py --target sharpe --normalize
```

**Output**:
```
📂 Loading dataset...
   Total rows: 222

🔍 Filtering dataset...
   Criteria: asset=BTC, min_trades=50, max_dd<=15.0%
   222 → 87 rows

📈 Features: 47 columns
   Target: sharpe

🤖 Training lightgbm model...
   Train RMSE: 0.1234, R²: 0.8567
   Val RMSE: 0.1456, R²: 0.7821

🔍 Top 10 important features:
   vix: 1234.5
   dxy_zscore: 987.3
   config_fusion_threshold: 856.2
   ...

💾 Model saved to models/sharpe_model
```

### Step 3: Suggest Configs

Use [`bin/research/suggest_config.py`](../bin/research/suggest_config.py) to get recommendations:

```bash
# Suggest top 5 configs for BTC in current regime
python bin/research/suggest_config.py \
  --model models/sharpe_model \
  --asset BTC \
  --vix 24.5 \
  --move 85.0 \
  --dxy 103.2 \
  --top-n 5

# Save best config to file
python bin/research/suggest_config.py \
  --model models/sharpe_model \
  --asset BTC \
  --vix 24.5 \
  --dxy 103.2 \
  --output configs/v18/BTC_ml_suggested.json
```

**Output**:
```
🌍 Current Regime:
   VIX: 24.5 (Elevated)
   MOVE: 85.0
   DXY: 103.2 (Strong)
   Oil: 70.0
   Gold: 2500
   Yield Spread: +0.35%

🤖 Ranking 240 configs...

✅ TOP 5 RECOMMENDED CONFIGS
═══════════════════════════

Rank 1: Predicted Sharpe = 2.34
  Fusion Threshold: 0.70
  Stop ATR: 1.20
  Trail ATR: 1.40
  ADX Threshold: 22
  Base Risk: 0.75%

Rank 2: Predicted Sharpe = 2.28
  Fusion Threshold: 0.68
  Stop ATR: 1.00
  Trail ATR: 1.20
  ADX Threshold: 20
  Base Risk: 0.75%

...
```

### Step 4: Backtest Suggested Config

```bash
# Backtest ML-suggested config
python bin/optimize_v19.py \
  --config configs/v18/BTC_ml_suggested.json \
  --asset BTC \
  --years 2 \
  --output results/ml_backtest.json
```

### Step 5: Deploy if Valid

If backtest metrics meet criteria (Sharpe > 1.5, MaxDD < 15%, Trades > 100):

```bash
# Copy to live config
cp configs/v18/BTC_ml_suggested.json configs/v18/BTC_live.json

# Paper trade for 1-3 days
python bin/bull-live-paper \
  --config configs/v18/BTC_live.json \
  --balance 25000 \
  --start 2025-10-01 \
  --end 2025-10-03
```

---

## Feature Engineering

### Regime Features (30+ features)

Extracted from macro snapshot in [`engine/ml/featurize.py`](../engine/ml/featurize.py):

**Level Features**:
- `vix`, `move`, `dxy`, `oil`, `gold`, `us2y`, `us10y`
- `total_mc`, `total2_mc`, `total3_mc`, `usdt_d`, `btc_d`
- `yield_spread` (10Y-2Y)
- `btc_dominance_calc` (1 - TOTAL2/TOTAL)

**Time-Series Features** (if lookback window provided):
- `vix_roc_5`, `move_roc_5`, `dxy_roc_10`, `oil_roc_10`, `gold_roc_10`
- `vix_zscore`, `move_zscore`, `dxy_zscore`
- `dxy_ema_10`, `dxy_ema_50`, `oil_ema_20`, `gold_ema_20`

**Regime Classification**:
- `vix_regime`: 0=Calm (<18), 1=Elevated (18-30), 2=Panic (>30)
- `dxy_regime`: 0=Weak (<100), 1=Neutral (100-105), 2=Strong (>105)
- `curve_regime`: 0=Inverted (<-0.2), 1=Flat, 2=Steep (>0.5)

### Config Features

From config dict:
- `config_fusion_threshold`
- `config_wyckoff_weight`, `config_smc_weight`, `config_hob_weight`, `config_momentum_weight`
- `config_stop_atr`, `config_trail_atr`, `config_tp1_r`
- `config_base_risk_pct`, `config_adx_threshold`

### Target Metrics

- `sharpe`: Sharpe ratio (primary)
- `pf`: Profit factor
- `total_return_pct`: Total return %
- `max_dd`: Max drawdown (constraint)
- `total_trades`: Trade count (constraint)
- `win_rate`: Win rate %

---

## Model Architecture

### LightGBM (Default)

Gradient-boosted decision trees for tabular data:

```python
model_params = {
    'objective': 'regression',
    'metric': 'rmse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'min_data_in_leaf': 10,
    'max_depth': 6
}
```

**Advantages**:
- Fast training (<1 min for 200 samples)
- Handles missing values
- Feature importance built-in
- Robust to overfitting

### XGBoost (Alternative)

Similar to LightGBM but different tree construction:

```python
model_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'rmse',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8
}
```

### Linear Regression (Baseline)

Ridge regression for comparison:

```python
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
```

---

## Walk-Forward Validation

To prevent overfitting, use walk-forward splits in [`engine/ml/dataset.py`](../engine/ml/dataset.py):

```python
# Split data into 5 folds with 20% test size
splits = dataset.walk_forward_split(df, n_splits=5, test_size=0.2)

for fold, (train_df, test_df) in enumerate(splits):
    print(f"Fold {fold+1}: Train={len(train_df)}, Test={len(test_df)}")

    X_train, y_train, _ = dataset.get_feature_target_split(train_df)
    X_test, y_test, _ = dataset.get_feature_target_split(test_df)

    model.train(X_train, y_train, X_test, y_test)
    # Evaluate...
```

**Result**: Model trained on past data, tested on future data → More realistic performance.

---

## Integration with Optimizer

### Modify [`bin/optimize_v19.py`](../bin/optimize_v19.py)

Add ML dataset appending:

```python
from engine.ml.dataset import OptimizationDataset, convert_optimization_results_to_training_rows
from engine.ml.featurize import build_regime_vector

# After optimization completes
dataset = OptimizationDataset()

# Build macro snapshot (current regime)
macro_snapshot = {
    'VIX': {'value': 24.5, 'stale': False},
    'MOVE': {'value': 85.0, 'stale': False},
    ...
}

# Convert results to training rows
training_rows = convert_optimization_results_to_training_rows(
    results=all_results,
    macro_snapshot=macro_snapshot,
    metadata={'asset': 'BTC', 'start_date': '2024-01-01', 'end_date': '2025-01-01'}
)

# Append to dataset
dataset.append_results(training_rows)
```

### Nightly Workflow

```bash
#!/bin/bash
# nightly_optimize.sh

# 1. Run optimization
python bin/optimize_v19.py --mode grid --asset BTC --years 3 --output results/nightly.json

# 2. Train model (optional - weekly)
if [ $(date +%u) -eq 1 ]; then  # Monday
    python bin/research/train_ml.py --target sharpe --asset BTC
fi

# 3. Suggest config for current regime
python bin/research/suggest_config.py \
  --model models/sharpe_model \
  --asset BTC \
  --vix $(curl -s https://api.example.com/vix) \
  --dxy $(curl -s https://api.example.com/dxy) \
  --output configs/v18/BTC_auto.json

# 4. Backtest suggested config
python bin/optimize_v19.py --config configs/v18/BTC_auto.json --years 1 --output results/auto_backtest.json

# 5. Email report
python bin/email_report.py results/auto_backtest.json
```

---

## Safe Rollout Plan

### Phase 1: Alpha (Dev Only)

- ✅ Macro Fusion: `enabled: false` (feature flag off)
- ✅ ML Framework: Built, tested locally
- ✅ Backtest validation: 3 months of data
- ✅ A/B test: Compare macro fusion ON vs OFF

**Success Criteria**:
- Sharpe improvement > 0.2 (e.g., 2.14 → 2.34+)
- Max drawdown < 15%
- No catastrophic failures

### Phase 2: Beta (Paper Trading)

- Enable macro fusion: `enabled: true`
- Paper trade ML-suggested configs for 1-3 days
- Monitor daily PnL, win rate, drawdowns
- Compare vs baseline config

**Success Criteria**:
- Paper trading Sharpe > 1.5
- Max drawdown < 10% in paper
- No major divergence from backtest

### Phase 3: Production (Live)

- Deploy ML-suggested config to live trading
- Start with small position sizes (0.5% risk)
- Monitor for 1 week
- Scale up if metrics hold

**Success Criteria**:
- Live Sharpe > 1.5 over 1 month
- Max drawdown < 15%
- Win rate > 60%

---

## Troubleshooting

### Issue: Empty Dataset

**Error**: `Dataset is empty! Run optimization first.`

**Fix**:
```bash
# Run optimization to collect training data
python bin/optimize_v19.py --mode grid --asset BTC --years 3
```

### Issue: LightGBM Not Available

**Error**: `LightGBM not available - install with: pip install lightgbm`

**Fix**:
```bash
pip install lightgbm
# or
pip install xgboost  # Use XGBoost instead
python bin/research/train_ml.py --model xgboost
```

### Issue: Low R² Score

**Problem**: Model R² < 0.5 (poor predictions)

**Possible Causes**:
1. **Not enough data**: Need 100+ optimization results
2. **High variance**: Too many outliers (filter with `--max-dd 0.15`)
3. **Weak features**: Regime features not predictive

**Fixes**:
- Collect more optimization results (run on multiple assets)
- Filter outliers: `--min-trades 100 --max-dd 0.15 --min-sharpe 1.0`
- Add more features (technical indicators, volatility regimes)

### Issue: Macro Fusion Not Applying

**Problem**: Fusion score unchanged despite macro fusion enabled

**Debug**:
1. Check feature flag: `configs/v18/BTC_conservative.json` → `"macro_fusion": {"enabled": true}`
2. Check macro snapshot: Ensure VIX/MOVE/DXY/etc. are not stale
3. Check logs: `logger.debug(f"Macro fusion composite: {fusion_composite:+.3f}")`

### Issue: All Configs Have Same Prediction

**Problem**: Model predicts similar scores for all configs

**Possible Causes**:
1. **Insufficient feature variance**: All configs too similar
2. **Model underfitting**: Learning rate too high, not enough trees

**Fixes**:
- Generate more diverse configs (wider grid)
- Increase num_boost_round: `500 → 1000`
- Lower learning_rate: `0.05 → 0.01`

---

## API Reference

### [`engine/ml/featurize.py`](../engine/ml/featurize.py)

```python
def build_regime_vector(macro_snapshot: Dict, lookback_window: Optional[pd.DataFrame] = None) -> Dict
```
Build feature vector from macro snapshot.

**Args**:
- `macro_snapshot`: Dict of `{symbol: {'value': float, 'stale': bool}}`
- `lookback_window`: Optional DataFrame with historical macro data

**Returns**:
```python
{
    'features': Dict[str, float],           # All features as dict
    'feature_vector': np.ndarray,           # Feature array for model
    'feature_names': List[str]              # Feature names in order
}
```

---

```python
def build_training_row(regime_vector: Dict, config: Dict, metrics: Dict, metadata: Dict = None) -> Dict
```
Build training row combining regime + config + metrics.

---

### [`engine/ml/dataset.py`](../engine/ml/dataset.py)

```python
class OptimizationDataset:
    def __init__(dataset_path: str = "data/ml/optimization_results.parquet")
    def append_results(results: List[Dict]) -> None
    def filter(asset: str = None, min_trades: int = 50, max_dd_threshold: float = 0.20) -> pd.DataFrame
    def get_feature_target_split(df: pd.DataFrame, target_col: str = 'sharpe') -> Tuple[X, y, feature_names]
    def walk_forward_split(df: pd.DataFrame, n_splits: int = 5) -> List[Tuple[train_df, test_df]]
```

---

### [`engine/ml/models.py`](../engine/ml/models.py)

```python
class ConfigSuggestionModel:
    def __init__(model_type: str = 'lightgbm', target: str = 'sharpe', model_params: Dict = None)
    def train(X_train, y_train, X_val=None, y_val=None) -> Dict[str, float]
    def predict(X: pd.DataFrame) -> np.ndarray
    def save(model_path: str) -> None
    @classmethod
    def load(model_path: str) -> ConfigSuggestionModel
```

---

```python
def rank_configs_by_prediction(model, candidate_configs, current_regime, top_n=10) -> List[Tuple[config, score]]
```
Rank configs by predicted performance in current regime.

---

## Performance Benchmarks

### Training Time

| Dataset Size | Model Type | Train Time | Test R² |
|--------------|------------|------------|---------|
| 100 samples | LightGBM | 15s | 0.76 |
| 500 samples | LightGBM | 45s | 0.82 |
| 100 samples | XGBoost | 20s | 0.74 |
| 500 samples | XGBoost | 60s | 0.81 |
| 100 samples | Linear | 1s | 0.45 |

### Prediction Speed

| Candidate Configs | Model Type | Prediction Time |
|-------------------|------------|-----------------|
| 50 configs | LightGBM | <1s |
| 500 configs | LightGBM | 2s |
| 50 configs | XGBoost | <1s |
| 500 configs | XGBoost | 3s |

---

## Future Enhancements

### v1.8.7 Roadmap

1. **Multi-Asset Models**: Train separate models for BTC, ETH, SOL
2. **Ensemble Methods**: Combine LightGBM + XGBoost predictions
3. **Online Learning**: Update model weights as new data arrives
4. **Feature Selection**: Automated feature importance analysis
5. **Hyperparameter Tuning**: Optuna/GridSearchCV for optimal params
6. **Multi-Target Models**: Predict Sharpe + PF + MaxDD jointly
7. **Risk-Aware Ranking**: Suggest configs with Sharpe/MaxDD trade-off

### v1.9.0 (Deep Learning)

1. **LSTM Models**: Temporal patterns in regime transitions
2. **Attention Mechanisms**: Focus on most important macro factors
3. **Reinforcement Learning**: RL agent for config adaptation
4. **Transformers**: BERT-style models for regime embeddings

---

## References

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [Scikit-learn Regression](https://scikit-learn.org/stable/modules/linear_model.html)
- [Walk-Forward Validation](https://en.wikipedia.org/wiki/Walk_forward_optimization)

---

## Contact & Support

- GitHub Issues: [Bull Machine Issues](https://github.com/username/bull-machine/issues)
- Documentation: [Bull Machine Docs](./README.md)
- Feature Requests: Create issue with `[Feature Request]` tag

---

**Last Updated**: October 13, 2025
**Version**: 1.8.6
**Status**: Alpha (Dev Only)
