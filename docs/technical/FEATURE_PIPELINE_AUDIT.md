# Feature Pipeline Audit - Bear Pattern Requirements

**Date**: 2025-11-13
**Scope**: MTF Feature Store (BTC 2022-2024)
**Purpose**: Comprehensive audit of feature availability for bear archetype implementation

---

## Executive Summary

**Current State**:
- MTF Store: 113 features (26,236 rows)
- Available: 108 features (95.6%) ✅
- Broken: 3 features (2.7%) ❌ (OI derivatives)
- Missing: 1 feature (0.9%) ❌ (liquidity_score)
- Partially Available: 1 feature (0.9%) ⚠️ (OI raw data, 2024 only)

**Blockers**:
1. **CRITICAL**: oi_change_24h, oi_change_pct_24h, oi_z (all NaN) - Blocks S5
2. **CRITICAL**: liquidity_score (not in store) - Blocks S1, S4, S5
3. **HIGH**: OI raw data (2024 only) - Limits bear market analysis to 2024

**Impact**:
- S5 (Long Squeeze): 100% blocked ❌
- S1 (Liquidity Vacuum): 100% blocked ❌
- S4 (Distribution Climax): 80% blocked ⚠️ (can run without OI/liquidity)
- S2 (Failed Rally): 60% functional ⚠️ (needs minor additions)

---

## Current MTF Feature Store (113 Features)

### Available Features ✅ (108 features)

#### 1. Base OHLCV (5 features)
```
high, low, close, open, volume
```
- **Coverage**: 100% (26,236 / 26,236)
- **Status**: ✅ Fully available

#### 2. Macro Features (19 features)
```
BTC.D, DXY, VIX, MOVE, USDT.D, TOTAL, TOTAL2, TOTAL3
YIELD_2Y, YIELD_10Y, YIELD_CURVE, WTI
BTC.D_Z, DXY_Z, VIX_Z, MOVE_Z
RV_20, RV_30, RV_60
```
- **Coverage**:
  - 2024: 100% (8,761 / 8,761)
  - 2022-2023: 0% (0 / 17,475) ⚠️
- **Status**: ✅ Available for 2024, ⚠️ Missing for 2022-2023
- **Note**: Macro features were added after 2022-2023 backtest window

#### 3. Funding Features (3 features)
```
funding, funding_rate, funding_Z
```
- **Coverage**:
  - 2024: 100%
  - 2022-2023: Unknown (needs verification)
- **Status**: ✅ Column exists, needs coverage validation

#### 4. Volume Features (3 features)
```
volume, volume_zscore, volume_z
```
- **Coverage**: 100%
- **Status**: ✅ Fully available

#### 5. Indicator Features (15+ features)
```
rsi_14, adx_14, atr_14
macd, macd_signal, macd_hist
bbands_upper, bbands_lower, bbands_mid
ema_20, ema_50, sma_200
```
- **Coverage**: 100%
- **Status**: ✅ Fully available

#### 6. Multi-Timeframe (MTF) Features (20+ features)
```
tf4h_trend, tf1d_trend, tf1w_trend
tf4h_boms_strength, tf1d_boms_strength
tf4h_fusion_score, tf1d_fusion_score
```
- **Coverage**: 100%
- **Status**: ✅ Fully available

#### 7. Wyckoff Features (10+ features)
```
wyckoff_phase, wyckoff_score
wyckoff_event, wyckoff_strength
```
- **Coverage**: 100%
- **Status**: ✅ Fully available

#### 8. Order Block / FVG Features (10+ features)
```
tf1h_fvg_present, tf1h_fvg_quality, tf1h_fvg_low, tf1h_fvg_high
tf1h_ob_high, tf1h_ob_low
tf4h_fvg_present, tf4h_fvg_quality
```
- **Coverage**: 100%
- **Status**: ✅ Fully available

#### 9. Regime Features (5+ features)
```
regime_gmm, regime_confidence, regime_name
market_state, volatility_regime
```
- **Coverage**: 100%
- **Status**: ✅ Fully available

---

### Broken Features ❌ (3 features)

#### 1. oi_change_24h
```python
# Column exists but ALL VALUES ARE NaN
mtf_df['oi_change_24h'].describe()
# count: 0 / 26,236
# All NaN
```
- **Formula**: `df['oi'].diff(24)`
- **Expected Range**: -5B to +5B USD (absolute change)
- **Root Cause**: Calculation never run after OI column was added
- **Impact**: S5 cannot detect OI spikes
- **Fix**: `bin/fix_oi_change_pipeline.py` Phase 2

#### 2. oi_change_pct_24h
```python
# Column exists but ALL VALUES ARE NaN
mtf_df['oi_change_pct_24h'].describe()
# count: 0 / 26,236
# All NaN
```
- **Formula**: `df['oi'].pct_change(24) * 100`
- **Expected Range**: -25% to +20% (percentage change)
- **Expected Normal**: -5% to +5% (90% of data)
- **Root Cause**: Calculation never run after OI column was added
- **Impact**: S5 liquidation cascade detection impossible
- **Fix**: `bin/fix_oi_change_pipeline.py` Phase 2

#### 3. oi_z
```python
# Column exists but ALL VALUES ARE NaN
mtf_df['oi_z'].describe()
# count: 0 / 26,236
# All NaN
```
- **Formula**: `(oi - rolling_mean_252h) / rolling_std_252h`
- **Expected Range**: -3.5 to +3.5 (z-score)
- **Expected Distribution**: mean=0, std=1
- **Root Cause**: Calculation never run after OI column was added
- **Impact**: S5 cannot detect crowded positioning
- **Fix**: `bin/fix_oi_change_pipeline.py` Phase 2

---

### Missing Features ❌ (1 feature)

#### 1. liquidity_score
```python
# Column DOES NOT EXIST in MTF store
'liquidity_score' in mtf_df.columns
# False
```
- **Formula**: `compute_liquidity_score(ctx, side)` (from `engine/liquidity/score.py`)
- **Expected Range**: 0.0 to 1.0
- **Expected Distribution**:
  - median: 0.45–0.55
  - p75: 0.68–0.75
  - p90: 0.80–0.90
- **Root Cause**: Runtime-only feature, never persisted
- **Impact**:
  - S1 (Liquidity Vacuum): 100% blocked
  - S4 (Distribution Climax): 80% blocked
  - S5 (Long Squeeze): Optional fallback logic needed
- **Fix**: `bin/backfill_liquidity_score.py`

---

### Partially Available Features ⚠️ (1 feature)

#### 1. oi (Open Interest)
```python
# Column exists but PARTIAL COVERAGE
mtf_df['oi'].describe()
# count: 8,761 / 26,236 (33.4%)
```
- **Coverage by Year**:
  - 2022: 0 / 8,741 (0.0%) ❌
  - 2023: 0 / 8,734 (0.0%) ❌
  - 2024: 8,761 / 8,761 (100%) ✅
- **Root Cause**: Macro features (OI source) only available from 2024-01-05
- **Impact**:
  - Cannot analyze 2022 bear market (Terra, FTX collapses)
  - Limits bear archetype validation to 2024 only
- **Fix**: `bin/fix_oi_change_pipeline.py` Phase 1 (fetch 2022-2023 OI)

---

## Pattern-Feature Dependency Matrix

### S1: Liquidity Vacuum

**Required Features**:
```python
required = ['liquidity_score', 'fvg_below', 'volume_z']
```

| Feature | Status | Coverage | Blocker |
|---------|--------|----------|---------|
| liquidity_score | ❌ Missing | 0% | CRITICAL |
| fvg_below | 🔨 Derivable | 100% | Can derive from tf1h_fvg_low |
| volume_z | ✅ Available | 100% | None |

**Overall Status**: ❌ **BLOCKED** (liquidity_score missing)

**Derived Feature Logic**:
```python
# fvg_below: FVG below current price
fvg_below = (mtf_df['tf1h_fvg_high'] < mtf_df['close'])
```

---

### S2: Failed Rally

**Required Features**:
```python
required = ['ob_retest', 'rsi_divergence', 'vol_fade', 'wick_ratio']
```

| Feature | Status | Coverage | Blocker |
|---------|--------|----------|---------|
| ob_retest | 🔨 Derivable | 100% | Can derive from tf1h_ob_high/low |
| rsi_divergence | 🔨 Derivable | 100% | Can derive from rsi_14 + close |
| vol_fade | 🔨 Derivable | 100% | Can derive from volume_z rolling |
| wick_ratio | 🔨 Derivable | 100% | Can derive from OHLC |

**Overall Status**: ⚠️ **PARTIAL** (all features derivable, need calculation)

**Derived Feature Logic**:
```python
# ob_retest: Price touches order block
ob_retest = (mtf_df['high'] >= mtf_df['tf1h_ob_low']) & (mtf_df['low'] <= mtf_df['tf1h_ob_high'])

# rsi_divergence: Price makes higher high, RSI makes lower high
price_hh = (mtf_df['close'] > mtf_df['close'].shift(1))
rsi_lh = (mtf_df['rsi_14'] < mtf_df['rsi_14'].shift(1))
rsi_divergence = price_hh & rsi_lh

# vol_fade: Current volume_z < 4H ago volume_z
vol_fade = (mtf_df['volume_z'] < mtf_df['volume_z'].shift(4))

# wick_ratio: Upper wick / total range
wick_ratio = (mtf_df['high'] - mtf_df['close']) / (mtf_df['high'] - mtf_df['low'])
```

---

### S4: Distribution Climax

**Required Features**:
```python
required = ['volume_spike', 'liquidity_score', 'oi_z']
optional = ['wyckoff_phase']
```

| Feature | Status | Coverage | Blocker |
|---------|--------|----------|---------|
| volume_spike | 🔨 Derivable | 100% | Can derive from volume_z > 2.0 |
| liquidity_score | ❌ Missing | 0% | HIGH priority |
| oi_z | ❌ Broken | 0% | HIGH priority |
| wyckoff_phase | ✅ Available | 100% | None |

**Overall Status**: ⚠️ **PARTIAL** (can run without liquidity/OI, degraded accuracy)

**Fallback Logic**:
```python
# Without liquidity_score and oi_z, rely on volume + wyckoff
if liquidity_score is None:
    # Use volume spike + wyckoff phase as proxy
    volume_spike = (mtf_df['volume_z'] > 2.0)
    distribution_phase = (mtf_df['wyckoff_phase'] == 'Distribution')
    s4_signal = volume_spike & distribution_phase
```

---

### S5: Long Squeeze Cascade

**Required Features**:
```python
required = ['funding_Z', 'oi_z', 'rsi_14', 'liquidity_score']
```

| Feature | Status | Coverage | Blocker |
|---------|--------|----------|---------|
| funding_Z | ✅ Available | 100%? | Needs coverage validation |
| oi_z | ❌ Broken | 0% | CRITICAL |
| rsi_14 | ✅ Available | 100% | None |
| liquidity_score | ❌ Missing | 0% | CRITICAL |

**Overall Status**: ❌ **BLOCKED** (oi_z and liquidity_score missing)

**No Fallback**: S5 requires OI data for liquidation cascade detection.

---

## Derived Features (Need Calculation) 🔨

These features can be computed from existing MTF columns:

### 1. fvg_below (for S1)
```python
mtf_df['fvg_below'] = (mtf_df['tf1h_fvg_high'] < mtf_df['close'])
```
- **Depends on**: tf1h_fvg_high, close
- **Availability**: 100%
- **Complexity**: Trivial (1-line)

### 2. ob_retest (for S2)
```python
mtf_df['ob_retest'] = (
    (mtf_df['high'] >= mtf_df['tf1h_ob_low']) &
    (mtf_df['low'] <= mtf_df['tf1h_ob_high'])
)
```
- **Depends on**: OHLC, tf1h_ob_high/low
- **Availability**: 100%
- **Complexity**: Trivial (1-line)

### 3. rsi_divergence (for S2)
```python
# Bearish divergence: price higher high, RSI lower high
def detect_rsi_divergence(df, lookback=5):
    price_hh = df['close'] > df['close'].shift(lookback).rolling(lookback).max()
    rsi_lh = df['rsi_14'] < df['rsi_14'].shift(lookback).rolling(lookback).max()
    return price_hh & rsi_lh

mtf_df['rsi_divergence'] = detect_rsi_divergence(mtf_df)
```
- **Depends on**: close, rsi_14
- **Availability**: 100%
- **Complexity**: Medium (rolling window logic)

### 4. vol_fade (for S2)
```python
# Volume fading: current volume_z < 4H ago
mtf_df['vol_fade'] = (mtf_df['volume_z'] < mtf_df['volume_z'].shift(4))
```
- **Depends on**: volume_z
- **Availability**: 100%
- **Complexity**: Trivial (1-line)

### 5. wick_ratio (for S2)
```python
# Upper wick ratio (for bearish patterns)
mtf_df['wick_ratio'] = (mtf_df['high'] - mtf_df['close']) / (mtf_df['high'] - mtf_df['low'] + 1e-9)
```
- **Depends on**: OHLC
- **Availability**: 100%
- **Complexity**: Trivial (1-line)

### 6. volume_spike (for S4)
```python
# Volume spike: z-score > 2.0 (2-sigma event)
mtf_df['volume_spike'] = (mtf_df['volume_z'] > 2.0).astype(int)
```
- **Depends on**: volume_z
- **Availability**: 100%
- **Complexity**: Trivial (1-line)

---

## Coverage Analysis by Year

### 2022 (8,741 rows)
```
Base OHLCV:        100% ✅
Indicators:        100% ✅
MTF/Wyckoff:       100% ✅
Macro features:      0% ❌
OI features:         0% ❌
Funding features:   ?? ⚠️ (needs verification)
Liquidity score:     0% ❌
```

**Implication**: 2022 bear market (Terra, FTX) cannot be fully analyzed without OI data.

### 2023 (8,734 rows)
```
Base OHLCV:        100% ✅
Indicators:        100% ✅
MTF/Wyckoff:       100% ✅
Macro features:      0% ❌
OI features:         0% ❌
Funding features:   ?? ⚠️ (needs verification)
Liquidity score:     0% ❌
```

**Implication**: 2023 recovery cannot be analyzed with macro context.

### 2024 (8,761 rows)
```
Base OHLCV:        100% ✅
Indicators:        100% ✅
MTF/Wyckoff:       100% ✅
Macro features:    100% ✅
OI (raw):          100% ✅
OI (derived):        0% ❌ (oi_change, oi_z)
Funding features:  100% ✅
Liquidity score:     0% ❌
```

**Implication**: 2024 data is mostly complete, only needs OI derivatives + liquidity_score.

---

## Recommended Fixes (Priority Order)

### Phase 1: Unblock S2 (Quick Wins - 1 day)

**Goal**: Get S2 (Failed Rally) fully functional

**Tasks**:
1. Add wick_ratio calculation
2. Add vol_fade detection
3. Add rsi_divergence calculation
4. Add ob_retest flag
5. Validate S2 on 2022 data

**Script**: Create `bin/add_s2_derived_features.py`

**Deliverable**: S2 fully functional, validated on 2022 bear market

**Estimated Time**: 4 hours

---

### Phase 2: Fix OI Pipeline (Critical - 2 days)

**Goal**: Restore OI features for all years

**Tasks**:
1. Fetch 2022-2023 OI data from OKX API
2. Merge into MTF store (fill 2022-2023 gap)
3. Calculate oi_change_24h, oi_change_pct_24h, oi_z
4. Validate against Terra/FTX collapses
5. Re-export MTF store

**Script**: `bin/fix_oi_change_pipeline.py` (already created)

**Deliverable**: OI features available for 2022-2024

**Estimated Time**: 8 hours (including API fetch time)

---

### Phase 3: Backfill Liquidity Score (Complex - 3 days)

**Goal**: Add liquidity_score to MTF store

**Tasks**:
1. Batch compute liquidity_score for all rows
2. Validate distribution (median ~0.5, p90 ~0.85)
3. Add column to MTF store
4. Validate S1 pattern detection

**Script**: `bin/backfill_liquidity_score.py` (already created)

**Deliverable**: S1, S4, S5 fully functional

**Estimated Time**: 12 hours (including validation)

---

### Phase 4: Validation (1 day)

**Goal**: Validate all bear patterns on 2022 bear market

**Tasks**:
1. Run S2 on 2022 data (Failed Rally during Terra collapse)
2. Run S5 on 2022 data (Long Squeeze during FTX collapse)
3. Run S1 on 2023 data (Liquidity Vacuum during recovery)
4. Measure PF, win rate, trade count

**Script**: Create `bin/validate_bear_patterns_2022.py`

**Deliverable**: Performance metrics for all bear patterns

**Estimated Time**: 8 hours

---

## Total Feature Count Projection

### Current State
```
MTF Store: 113 features
- Available: 108 (95.6%)
- Broken: 3 (2.7%)
- Missing: 1 (0.9%)
- Partial: 1 (0.9%)
```

### After All Fixes
```
MTF Store: 120 features (+7)
- liquidity_score: +1 (backfilled)
- oi_change_24h: +1 (fixed)
- oi_change_pct_24h: +1 (fixed)
- oi_z: +1 (fixed)
- fvg_below: +1 (derived)
- ob_retest: +1 (derived)
- rsi_divergence: +1 (derived)
- vol_fade: +1 (derived)
- wick_ratio: +1 (derived)
- volume_spike: +1 (derived)

Total: 120 features (100% coverage for bear patterns)
```

---

## Success Criteria

### Quantitative
✅ All 113 features have > 99% non-null coverage
✅ oi_change_pct_24h shows Terra collapse < -15%
✅ oi_change_pct_24h shows FTX collapse < -20%
✅ liquidity_score distribution: median = 0.45–0.55, p90 = 0.80–0.90
✅ S2 pattern runs without errors on 2022 data
✅ S5 pattern detects May/Nov 2022 liquidation cascades

### Qualitative
✅ Bear archetype backtests complete end-to-end
✅ No KeyError exceptions for required features
✅ Pattern detection aligns with known market events (Terra, FTX)

---

## Related Documentation

- **OI Pipeline Diagnosis**: `/docs/OI_CHANGE_FAILURE_DIAGNOSIS.md`
- **Fix Scripts**:
  - `/bin/fix_oi_change_pipeline.py`
  - `/bin/backfill_liquidity_score.py`
- **Implementation Roadmap**: `/docs/BEAR_FEATURE_PIPELINE_ROADMAP.md`
- **Runtime Liquidity Logic**: `/engine/liquidity/score.py`
- **Bear Pattern Specs**: `/engine/archetypes/bear_patterns_phase1.py`

---

## Appendix: Full Feature List (113 features)

<details>
<summary>Click to expand full feature list</summary>

```
Base OHLCV (5):
  high, low, close, open, volume

Macro Features (19):
  BTC.D, BTC.D_Z, DXY, DXY_Z, VIX, VIX_Z, MOVE, MOVE_Z
  USDT.D, TOTAL, TOTAL2, TOTAL3
  YIELD_2Y, YIELD_10Y, YIELD_CURVE, WTI
  RV_20, RV_30, RV_60

Funding/OI Features (7):
  funding, funding_rate, funding_Z
  oi, oi_change_24h, oi_change_pct_24h, oi_z

Volume Features (3):
  volume, volume_zscore, volume_z

Indicators (15):
  rsi_14, adx_14, atr_14
  macd, macd_signal, macd_hist
  bbands_upper, bbands_lower, bbands_mid
  ema_20, ema_50, sma_200
  stoch_k, stoch_d
  williams_r

Multi-Timeframe (20+):
  tf4h_trend, tf1d_trend, tf1w_trend
  tf4h_boms_strength, tf1d_boms_strength
  tf4h_fusion_score, tf1d_fusion_score
  tf4h_boms_displacement, tf1d_boms_displacement
  (... and more HTF indicators)

Wyckoff Features (10):
  wyckoff_phase, wyckoff_score
  wyckoff_event, wyckoff_strength
  wyckoff_composite_operator
  wyckoff_accumulation, wyckoff_distribution
  (... and more Wyckoff states)

Order Block / FVG (10):
  tf1h_fvg_present, tf1h_fvg_quality, tf1h_fvg_low, tf1h_fvg_high
  tf1h_ob_high, tf1h_ob_low
  tf4h_fvg_present, tf4h_fvg_quality
  (... and more OB/FVG metrics)

Regime Features (5):
  regime_gmm, regime_confidence, regime_name
  market_state, volatility_regime

Other (20+):
  rolling_high, rolling_low, range_eq
  fresh_bos_flag, tod_boost
  (... and more runtime features)
```

</details>
