# Feature Store Design - Unified Architecture

**Version:** 1.0
**Date:** 2025-11-06
**Status:** Design Phase

---

## 1. Current State Analysis

### Problems with Current Architecture

**Scattered Data Files:**
```
data/features/v18/          # Old technical indicators
data/features_mtf/          # Multi-timeframe features
data/macro/                 # Macro indicators
data/cached/                # Ad-hoc caching
```

**Issues:**
- No single source of truth per asset
- Redundant feature computation across scripts
- Index inconsistencies (RangeIndex vs DatetimeIndex)
- No schema validation
- No versioning system
- Manual cache invalidation

---

## 2. Proposed Unified Architecture

### Directory Structure

```
data/
  feature_store/
    btc/
      raw/
        btc_1h_raw_v1.0_2022-2024.parquet          # Tier 1: OHLCV + basic indicators
      mtf/
        btc_1h_mtf_v1.0_2022-2024.parquet          # Tier 2: Multi-timeframe features
      full/
        btc_1h_full_v1.0_2022-2024.parquet         # Tier 3: Complete (all tiers + regime)
      metadata/
        schema_v1.0.json                           # Column definitions
        manifest.json                              # Build history
        validation_report.json                     # Last validation results

    eth/
      [same structure]

    shared/
      models/
        regime_gmm_v3.1_fixed.pkl                  # Shared regime detector
        regime_scaler_v3.1.pkl
        btc_trade_quality_filter_v1.pkl
      macro/
        macro_history.parquet                      # DXY, VIX, yields, etc.
      schemas/
        tier1_schema_v1.0.json                     # Raw + technical
        tier2_schema_v1.0.json                     # MTF features
        tier3_schema_v1.0.json                     # Full with regime
```

### Feature Tiers

**Tier 1 - Raw + Technical** (Always Present):
- **Source**: Exchange OHLCV data
- **Features**:
  - Price: open, high, low, close
  - Volume: volume, volume_ma_20
  - Volatility: atr_20, bb_upper, bb_lower, bb_width
  - Trend: adx_14, +di, -di, ema_20, ema_50, ema_200
  - Momentum: rsi_14, macd, macd_signal, macd_hist
  - Structure: swing_highs, swing_lows
- **Build Time**: ~2-5 minutes
- **Dependencies**: None

**Tier 2 - Multi-Timeframe** (Requires MTF Computation):
- **Source**: Tier 1 + timeframe aggregation
- **Features**:
  - `tf4h_*`: 4H timeframe features (fusion, alignment, bos)
  - `tf1d_*`: Daily timeframe features (trend, volatility)
  - `tf1h_*`: 1H structure features (fvg, ob, bos, liquidity)
  - Cross-timeframe: mtf_alignment_score, htf_trend_strength
- **Build Time**: ~10-20 minutes
- **Dependencies**: Tier 1

**Tier 3 - Regime + Macro** (Requires External Models):
- **Source**: Tier 2 + regime models + macro data
- **Features**:
  - Regime: regime_label, regime_confidence, regime_prob_*
  - Macro context: dxy_close, vix_close, yields_10y
  - Composite: market_stress_score, risk_appetite_index
- **Build Time**: ~2-5 minutes
- **Dependencies**: Tier 2, regime models, macro data

---

## 3. Schema Versioning

### Version Format: `{major}.{minor}.{patch}`

**Major**: Breaking changes (column removal, type changes)
**Minor**: Additive changes (new columns)
**Patch**: Bug fixes (data corrections)

### Schema Validation

```python
{
  "version": "1.0.0",
  "tier": "full",
  "columns": {
    "timestamp": {
      "type": "datetime64[ns]",
      "index": true,
      "required": true,
      "description": "Bar timestamp (1H resolution)"
    },
    "close": {
      "type": "float64",
      "required": true,
      "range": [0, null],
      "description": "Closing price"
    },
    "regime_label": {
      "type": "category",
      "required": true,
      "values": ["risk_on", "risk_off", "neutral", "crisis"],
      "description": "Market regime classification"
    }
  },
  "constraints": {
    "no_duplicates": ["timestamp"],
    "no_nulls": ["close", "volume", "regime_label"],
    "monotonic_increasing": ["timestamp"]
  }
}
```

---

## 4. Builder Pipeline

### Single Entry Point

```python
# bin/build_feature_store.py --asset BTC --tier full --period 2022-2024

class FeatureStoreBuilder:
    """Unified feature store builder with validation."""

    def __init__(self, asset: str, version: str = "1.0"):
        self.asset = asset
        self.version = version
        self.schema_validator = SchemaValidator(version)

    def build_tier1(self, start: str, end: str) -> pd.DataFrame:
        """Build Tier 1: Raw + Technical indicators."""
        # 1. Load OHLCV from exchange API or CSV
        # 2. Compute technical indicators
        # 3. Validate against tier1_schema
        # 4. Set DatetimeIndex
        # 5. Save to data/feature_store/{asset}/raw/

    def build_tier2(self, tier1_df: pd.DataFrame) -> pd.DataFrame:
        """Build Tier 2: Multi-timeframe features."""
        # 1. Aggregate to 4H and 1D
        # 2. Compute MTF features
        # 3. Downcast to 1H resolution
        # 4. Validate against tier2_schema
        # 5. Save to data/feature_store/{asset}/mtf/

    def build_tier3(self, tier2_df: pd.DataFrame) -> pd.DataFrame:
        """Build Tier 3: Regime + Macro."""
        # 1. Load regime models
        # 2. Classify regimes
        # 3. Merge macro data
        # 4. Validate against tier3_schema
        # 5. Save to data/feature_store/{asset}/full/

    def build_incremental(self, new_start: str) -> pd.DataFrame:
        """Incremental update (append new data only)."""
        # 1. Load existing full store
        # 2. Determine gap: last_date → new_start
        # 3. Build only missing data
        # 4. Append and save
```

### Validation at Each Stage

```python
class SchemaValidator:
    """Validate dataframes against versioned schemas."""

    def validate(self, df: pd.DataFrame, tier: str) -> ValidationReport:
        schema = self.load_schema(tier)

        # Check required columns
        # Check data types
        # Check value ranges
        # Check constraints (no nulls, no duplicates)
        # Check index type

        return ValidationReport(passed, errors, warnings)
```

---

## 5. Migration Plan

### Phase 1: Establish Infrastructure (Week 1)

**Deliverables:**
- [ ] Schema definitions for all 3 tiers
- [ ] `FeatureStoreBuilder` class with tier1 implementation
- [ ] `SchemaValidator` with unit tests
- [ ] Migration script: `bin/migrate_to_feature_store.py`

**Steps:**
1. Create schema files from existing data
2. Implement basic builder for Tier 1
3. Migrate existing data to new structure
4. Validate all existing functionality still works

### Phase 2: Build Multi-Timeframe (Week 2)

**Deliverables:**
- [ ] Tier 2 builder (MTF features)
- [ ] Consolidate `bin/build_mtf_feature_store.py` logic
- [ ] Incremental update support
- [ ] Performance benchmarks

### Phase 3: Add Regime & Versioning (Week 3)

**Deliverables:**
- [ ] Tier 3 builder (regime + macro)
- [ ] Version management system
- [ ] Cache invalidation logic
- [ ] Documentation

### Phase 4: Multi-Asset Support (Week 4)

**Deliverables:**
- [ ] ETH feature store
- [ ] Cross-asset validation
- [ ] Shared regime models
- [ ] Performance optimization

---

## 6. Integration with Existing Code

### Backtest Integration

```python
# Before (scattered):
df = pd.read_parquet('data/features/v18/BTC_1H.parquet')
df_mtf = pd.read_parquet('data/features_mtf/BTC_1H_2022-2024.parquet')
# ... manual merging ...

# After (unified):
from engine.feature_store import FeatureStore

fs = FeatureStore(asset='BTC', version='1.0')
df = fs.load('full', start='2022-01-01', end='2024-12-31')
# Single call, validated, cached
```

### Optimizer Integration

```python
# Optimizer can request specific tier based on needs
fs = FeatureStore(asset='BTC', version='1.0')

# Fast loading for optimizers (use cached)
df_cached = fs.load_cached('full', '2022-01-01', '2024-12-31')

# Incremental updates (only fetch new data)
df_updated = fs.update_incremental()
```

---

## 7. Testing Strategy

### Unit Tests

```python
# tests/test_feature_store_builder.py
def test_tier1_schema_validation():
    """Tier 1 output matches schema."""

def test_tier2_mtf_features():
    """MTF features computed correctly."""

def test_tier3_regime_labels():
    """Regime labels match expected distribution."""

def test_incremental_update():
    """Incremental update produces same result as full rebuild."""
```

### Integration Tests

```python
# tests/test_feature_store_integration.py
def test_backtest_with_feature_store():
    """Backtest produces same results with feature store."""

def test_optimizer_with_feature_store():
    """Optimizer can load and slice feature store efficiently."""
```

### Performance Benchmarks

```python
# Benchmark: Build 3 years of BTC data
# Target: < 30 minutes for full build (all tiers)
# Target: < 2 minutes for incremental update
# Target: < 1 second for cached load
```

---

## 8. Benefits

**For Development:**
- Single source of truth per asset
- Reproducible builds with version control
- Faster iteration (cached features)
- Schema validation catches errors early

**For Optimization:**
- Faster trial execution (cached features)
- Consistent features across trials
- Easy A/B testing (version comparison)

**For Production:**
- Incremental updates only
- Version rollback capability
- Schema evolution without breaking changes

---

## 9. Open Questions

1. **Regime Model Sharing**: Per-asset models or shared across BTC/ETH?
2. **Macro Data Updates**: How frequently to refresh macro indicators?
3. **Cache Invalidation**: Automatic or manual when upstream changes?
4. **Cloud Storage**: S3/GCS for large feature stores?
5. **Parallelization**: Multi-asset builds in parallel?

---

## 10. Next Steps

**Immediate (This Session):**
1. Create schema files for current data
2. Implement basic `FeatureStoreBuilder` for Tier 1
3. Migrate one dataset (BTC 2022-2024) to new structure
4. Validate with existing backtest

**Short Term (This Week):**
1. Complete Tier 2 and Tier 3 builders
2. Add incremental update support
3. Write comprehensive tests
4. Document API

**Medium Term (Next 2 Weeks):**
1. Add ETH support
2. Implement version management
3. Performance optimization
4. CI/CD integration

---

**Generated:** 2025-11-06
**Author:** Claude (with Raymond)
**Status:** Design phase - ready for review and implementation
