# Bull Machine: TARGET STATE QUICK START

**Version:** 1.0.0
**Date:** 2025-12-03
**For:** Developers implementing the target architecture

This is a **quick reference guide** for the target state architecture. For detailed design, see [TARGET_STATE_ARCHITECTURE.md](./TARGET_STATE_ARCHITECTURE.md).

---

## TL;DR - What's Changing?

**OLD WAY (Current):**
```python
# Features scattered: v1 parquet + runtime enrichment
df = pd.read_parquet('v1.parquet')  # 167 cols, 27 with NaNs
enricher = LiquidityVacuumRuntimeFeatures()
df = enricher.enrich_dataframe(df)  # SLOW - recompute every time
```

**NEW WAY (Target):**
```python
# All features in v2 parquet (precomputed)
df = pd.read_parquet('v2.parquet')  # 195 cols, 0 NaNs
# NO enrichment needed - features already there!
```

**Benefits:**
- 60% faster backtests
- No code duplication
- ML models unblocked
- Fair model comparison

---

## Key Concepts

### 1. Versioned Feature Store

```
data/features_mtf/
├── BTC_1H_2022-2024_v1.parquet   # Current (167 cols, broken)
├── BTC_1H_2022-2024_v2.parquet   # Target (195 cols, 100% coverage)
└── BTC_1H_2022-2024_v3.parquet   # Future (220+ cols, ML features)
```

**Rule:** Features are computed ONCE (offline), used MANY times (all models)

### 2. No Runtime Enrichment

**OLD (Deprecated):**
```python
from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import (
    LiquidityVacuumRuntimeFeatures
)
enricher = LiquidityVacuumRuntimeFeatures()
df = enricher.enrich_dataframe(df)  # ✗ Don't do this
```

**NEW (Recommended):**
```python
df = pd.read_parquet('data/features_mtf/BTC_1H_2022-2024_v2.parquet')
# Features already present:
assert 'fvg_below' in df.columns  # S1 feature
assert 'wick_ratio' in df.columns  # S2 feature
```

### 3. Model-Agnostic Backtesting

**All models use SAME v2 parquet:**
```python
df = pd.read_parquet('BTC_1H_2022-2024_v2.parquet')

# Model 1: S2 Archetype
model1 = ArchetypeModel('configs/s2.json', 'S2')

# Model 2: XGBoost
model2 = XGBoostModel(features=['rsi_14', 'adx_14', 'liquidity_score'])

# Both use SAME data → fair comparison
comparison = ModelComparison(models=[model1, model2], data=df)
```

---

## Common Tasks

### Task 1: Add New Feature to v2

**Steps:**
1. Add feature calculation to `engine/features/derived.py`
2. Update `bin/build_feature_store_v2.py` to call it
3. Rebuild parquet: `python bin/build_feature_store_v2.py`
4. Validate: `python bin/validate_feature_store.py`
5. Use in models (features already present)

**Example:**
```python
# 1. Add to engine/features/derived.py
def calculate_my_new_feature(df: pd.DataFrame) -> pd.DataFrame:
    df['my_new_feature'] = df['close'] / df['sma_50']
    return df

# 2. Update bin/build_feature_store_v2.py
df = calculate_derived_features(df)  # Existing call
df = calculate_my_new_feature(df)    # Add this line

# 3. Rebuild
python bin/build_feature_store_v2.py --asset BTC --timeframe 1H

# 4. Use in model
df = pd.read_parquet('BTC_1H_2022-2024_v2.parquet')
assert 'my_new_feature' in df.columns  # ✓ Present
```

### Task 2: Create New Model

**Steps:**
1. Inherit from `BaseModel`
2. Implement `fit()`, `predict()`, `get_position_size()`
3. Load v2 parquet in `fit()`
4. Use features (no enrichment)

**Example:**
```python
from engine.models.base import BaseModel, Signal

class MyCustomModel(BaseModel):
    def fit(self, train_data: pd.DataFrame, **kwargs):
        # Calibrate on training data
        self.threshold = train_data['rsi_14'].quantile(0.3)
        self._is_fitted = True

    def predict(self, bar: pd.Series, position=None) -> Signal:
        if bar['rsi_14'] < self.threshold:
            return Signal(
                direction='long',
                confidence=0.8,
                entry_price=bar['close'],
                stop_loss=bar['close'] - 2 * bar['atr_14']
            )
        return Signal(direction='hold', confidence=0, entry_price=bar['close'])

    def get_position_size(self, bar: pd.Series, signal: Signal) -> float:
        risk = abs(signal.entry_price - signal.stop_loss)
        return (10000 * 0.02) / risk  # 2% risk

# Usage
df = pd.read_parquet('BTC_1H_2022-2024_v2.parquet')
model = MyCustomModel(name='MyCustom')
model.fit(df['2022':'2023'])

engine = BacktestEngine(model, df['2024'])
results = engine.run()
print(results.summary())
```

### Task 3: Compare Multiple Models

**Steps:**
1. Load v2 parquet (once)
2. Create list of models
3. Run `ModelComparison`

**Example:**
```python
from engine.backtesting.comparison import ModelComparison

# Load data (once)
df = pd.read_parquet('BTC_1H_2022-2024_v2.parquet')

# Define models
models = [
    ArchetypeModel('configs/s2.json', 'S2', name='S2-Optimized'),
    ArchetypeModel('configs/s5.json', 'S5', name='S5-Optimized'),
    XGBoostModel(features=['rsi_14', 'adx_14'], name='XGB-Baseline'),
    SimpleClassifier(features=['rsi_14'], rules={'rsi_14': ('<', 30)}, name='RSI-Oversold')
]

# Run comparison
comparison = ModelComparison(
    models=models,
    data=df,
    train_period=('2022-01-01', '2023-12-31'),
    test_period=('2024-01-01', '2024-12-31')
)

results = comparison.run_comparison()
print(results)
# Output: Comparison table (Model, Trades, WR%, PF, Sharpe, MaxDD, PnL)
```

### Task 4: Run Validation

**Validate v2 feature store:**
```bash
python bin/validate_feature_store.py \
    --input data/features_mtf/BTC_1H_2022-2024_v2.parquet \
    --schema v2 \
    --strict

# Output:
# ✓ No NaN values (195/195 columns)
# ✓ All ranges valid
# ✓ Logical consistency passed
# ✓ Timestamp continuity verified
# ✓ All data types correct
#
# Feature Store Status: VALIDATED (v2.0)
```

**Validate backtest regression (v2 vs v1):**
```bash
pytest tests/integration/test_v2_vs_v1_regression.py

# Checks:
# - S2 archetype: Same PF/WR on v2 vs v1 (within 1%)
# - S5 archetype: Same PF/WR on v2 vs v1 (within 1%)
# - No crashes on v2 data
```

### Task 5: Backfill Historical Data (Fix NaNs)

**Backfill macro data (2022-2023):**
```bash
python bin/backfill_macro_data.py \
    --start 2022-01-01 \
    --end 2023-12-31 \
    --output data/macro/macro_backfilled_2022-2023.csv

# Sources:
# - DXY: TradingView API
# - VIX: TradingView API
# - Yields: TradingView API
# - BTC.D, USDT.D: TradingView API
```

**Backfill OI data (2022-2023):**
```bash
python bin/backfill_oi_data.py \
    --start 2022-01-01 \
    --end 2023-12-31 \
    --output data/derivatives/oi_backfilled_2022-2023.csv

# Sources:
# - Binance Futures API
# - OKX API (fallback)
```

---

## File Locations

**Feature Store:**
```
data/features_mtf/
├── BTC_1H_2022-2024_v2.parquet       # Main store
├── manifests/v2_manifest.json         # Metadata
└── validation_reports/v2_validation.json  # Validation
```

**Build Scripts:**
```
bin/
├── build_feature_store_v2.py          # v2 builder
├── validate_feature_store.py          # Validator
├── backfill_macro_data.py             # Macro backfill
└── backfill_oi_data.py                # OI backfill
```

**Feature Modules:**
```
engine/features/
├── base.py         # Tier 1: OHLCV + technical
├── mtf.py          # Tier 2: Multi-timeframe
├── smc.py          # Tier 3: SMC (OB, FVG, BOS)
├── wyckoff.py      # Tier 4: Wyckoff events
├── macro.py        # Tier 5: Macro features
├── fusion.py       # Tier 6: Fusion scores
├── derived.py      # Tier 7: Derived features (NEW)
└── registry.py     # Feature metadata
```

**Models:**
```
engine/models/
├── base.py                  # BaseModel interface
├── archetype_model.py       # Archetype wrapper
├── simple_classifier.py     # Baselines
├── xgboost_model.py         # ML model (future)
└── ensemble_model.py        # Ensemble (future)
```

**Backtesting:**
```
engine/backtesting/
├── engine.py           # Backtest engine (model-agnostic)
├── comparison.py       # Multi-model comparison
├── validator.py        # Walk-forward validation
└── metrics.py          # Performance metrics
```

---

## Migration Checklist

### Phase 1: Fix Broken Columns (Week 1)
- [ ] Backfill macro data (2022-2023)
- [ ] Backfill OI data (2022-2023)
- [ ] Rebuild v1_fixed parquet (167 cols, 0 NaNs)
- [ ] Validate: `pytest tests/integration/test_v1_fixed.py`

### Phase 2: Build v2 (Week 2)
- [ ] Implement `engine/features/derived.py` (28 new columns)
- [ ] Update `bin/build_feature_store_v2.py`
- [ ] Build v2 parquet (195 cols, 0 NaNs)
- [ ] Validate: `python bin/validate_feature_store.py --schema v2`

### Phase 3: Eliminate Runtime Enrichment (Week 3)
- [ ] Update archetype configs (point to v2 features)
- [ ] Deprecate runtime modules (add warnings)
- [ ] Update backtest scripts (remove enrichment calls)
- [ ] Run regression tests: `pytest tests/integration/test_v2_vs_v1.py`

### Phase 4: Model Comparison (Week 4)
- [ ] Implement `engine/backtesting/comparison.py`
- [ ] Create comparison dashboard script
- [ ] Run baseline comparison (S2, S5, XGB, Simple)
- [ ] Document results: `docs/backtests/model_comparison_baseline.md`

### Phase 5: ML Models (Week 5-6)
- [ ] Implement `engine/models/xgboost_model.py`
- [ ] Train XGBoost on v2 data
- [ ] Backtest XGBoost vs archetypes
- [ ] Feature importance analysis

### Phase 6: Live Trading Prep (Week 7-8)
- [ ] Implement `engine/live/streaming_features.py`
- [ ] Implement `engine/live/paper_executor.py`
- [ ] Run 2-week paper trading test
- [ ] Validate: paper trading ≈ backtest (within 5%)

---

## Testing

**Unit Tests:**
```bash
# Test feature calculations
pytest tests/unit/test_derived_features.py

# Test model interface
pytest tests/unit/test_base_model.py

# Test validation
pytest tests/unit/test_feature_validation.py
```

**Integration Tests:**
```bash
# Test v2 vs v1 regression
pytest tests/integration/test_v2_vs_v1_regression.py

# Test multi-model comparison
pytest tests/integration/test_model_comparison.py

# Test walk-forward validation
pytest tests/integration/test_walk_forward.py
```

**Run All Tests:**
```bash
pytest tests/  # All tests (55+)
```

---

## Debugging

### Issue: "Column not found" error

**Problem:**
```python
KeyError: 'fvg_below'
```

**Solution:**
```python
# Check if using v2 parquet
df = pd.read_parquet('BTC_1H_2022-2024_v2.parquet')  # Not v1!
assert 'fvg_below' in df.columns

# If using v1, rebuild v2:
python bin/build_feature_store_v2.py
```

### Issue: NaN values in v2

**Problem:**
```
AssertionError: NaN values detected in v2 parquet
```

**Solution:**
```bash
# Run validation to identify broken columns
python bin/validate_feature_store.py --input v2.parquet --strict

# Fix broken calculation in engine/features/derived.py
# Rebuild v2
python bin/build_feature_store_v2.py
```

### Issue: Backtest results differ from v1

**Problem:**
```
S2 archetype: v1 PF=1.8, v2 PF=1.2 (regression!)
```

**Solution:**
```python
# Debug feature differences
python bin/debug_v2_diff.py \
    --v1 BTC_1H_2022-2024_v1_fixed.parquet \
    --v2 BTC_1H_2022-2024_v2.parquet \
    --columns fvg_below,wick_ratio,ob_retest

# Check for calculation errors in derived.py
# Fix and rebuild
```

---

## FAQ

**Q: Can I still use runtime enrichment?**

A: Yes (for now), but it's deprecated. v2 eliminates the need for runtime enrichment. Runtime modules will be removed in v3.

**Q: Do I need to rebuild v2 parquet every time I change a feature?**

A: Yes. Features are computed offline (once per day in production). This is the trade-off for 60% faster backtests.

**Q: How do I version my own custom features?**

A: Add them to `engine/features/derived.py` and increment version (e.g., v2.1, v2.2). Update manifest to track changes.

**Q: Can I use v1 and v2 parquets simultaneously?**

A: Yes, during migration. But models should use ONE version (not mix). Final state: v2 only.

**Q: What if v2 breaks production?**

A: Rollback to v1_fixed (see [Rollback Plan](./TARGET_STATE_ARCHITECTURE.md#83-rollback-plan)).

**Q: How do I add ML models?**

A: See [Task 2: Create New Model](#task-2-create-new-model). Inherit from `BaseModel`, load v2 parquet in `fit()`.

**Q: How do I prepare for live trading?**

A: See [Phase 6: Live Trading Prep](#phase-6-live-trading-prep-week-7-8). Implement streaming feature pipeline that matches v2 builder logic.

---

## References

- **Architecture:** [TARGET_STATE_ARCHITECTURE.md](./TARGET_STATE_ARCHITECTURE.md)
- **Diagrams:** [TARGET_STATE_DIAGRAMS.md](./TARGET_STATE_DIAGRAMS.md)
- **Schema:** [FEATURE_STORE_SCHEMA_v2.md](./FEATURE_STORE_SCHEMA_v2.md)
- **Current Architecture:** [ARCHITECTURE.md](./ARCHITECTURE.md)

---

**Last Updated:** 2025-12-03
**Status:** Ready for Implementation
