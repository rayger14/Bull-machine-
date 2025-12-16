# Feature Store Schema v2.0

**Version:** 2.0.0
**Date:** 2025-11-19
**Purpose:** Complete schema definition for Bull Machine feature store with validation rules
**Target:** 140+ columns (116 current + 24+ new)

---

## Executive Summary

This document defines the complete schema for the Bull Machine feature store, including all current columns (116), new columns being added in the Ghost → Live v2 upgrade (24+), and comprehensive validation rules to ensure data integrity.

**Schema Evolution:**
- **Current (v1.0):** 116 columns, 97.4% valid (3 broken columns)
- **Target (v2.0):** 140+ columns, 100% valid (all columns validated)

**Validation Policy:** NO NaNs ALLOWED. All features must have 100% coverage.

---

## 1. Current Schema (116 columns)

### 1.1 Tier 1: Base OHLCV (6 columns)

| Column | Type | Range | Description | Validation |
|--------|------|-------|-------------|------------|
| `timestamp` | datetime64[ns] | 2022-01-01 to 2024-12-31 | UTC timestamp | No gaps, hourly continuity |
| `open` | float64 | > 0 | Opening price | open > 0 |
| `high` | float64 | > 0 | Highest price | high >= max(open, close) |
| `low` | float64 | > 0 | Lowest price | low <= min(open, close) |
| `close` | float64 | > 0 | Closing price | close > 0 |
| `volume` | float64 | >= 0 | Trading volume | volume >= 0 |

**Logical Constraints:**
```python
assert (df['high'] >= df['low']).all()
assert (df['high'] >= df['close']).all()
assert (df['high'] >= df['open']).all()
assert (df['low'] <= df['close']).all()
assert (df['low'] <= df['open']).all()
assert (df['volume'] >= 0).all()
```

---

### 1.2 Tier 1: Core Indicators (8 columns)

| Column | Type | Range | Description | Validation |
|--------|------|-------|-------------|------------|
| `atr_14` | float64 | > 0 | Average True Range (14 periods) | atr_14 > 0 |
| `atr_20` | float64 | > 0 | Average True Range (20 periods) | atr_20 > 0 |
| `adx_14` | float64 | 0-100 | Average Directional Index (14 periods) | 0 <= adx_14 <= 100 |
| `rsi_14` | float64 | 0-100 | Relative Strength Index (14 periods) | 0 <= rsi_14 <= 100 |
| `sma_20` | float64 | > 0 | Simple Moving Average (20 periods) | sma_20 > 0 |
| `sma_50` | float64 | > 0 | Simple Moving Average (50 periods) | sma_50 > 0 |
| `sma_100` | float64 | > 0 | Simple Moving Average (100 periods) | sma_100 > 0 |
| `sma_200` | float64 | > 0 | Simple Moving Average (200 periods) | sma_200 > 0 |

**Calculation Details:**
- ATR: Wilder's smoothing, 14-period lookback
- ADX: Directional movement index, 14-period
- RSI: Wilder's RSI, 14-period
- SMAs: Simple moving average, no gaps

---

### 1.3 Tier 1: 1D Timeframe Features (14 columns)

| Column | Type | Range | Description | Validation |
|--------|------|-------|-------------|------------|
| `tf1d_wyckoff_score` | float64 | 0.0-1.0 | Wyckoff phase score (1D) | 0.0 <= score <= 1.0 |
| `tf1d_wyckoff_phase` | int64 | 0-4 | Wyckoff phase (0=Accumulation, 4=Distribution) | phase in [0,1,2,3,4] |
| `tf1d_boms_detected` | int64 | 0-1 | Break of Market Structure detected | boms in [0, 1] |
| `tf1d_boms_strength` | float64 | 0.0-1.0 | BOMS strength score | 0.0 <= strength <= 1.0 |
| `tf1d_boms_direction` | int64 | -1, 0, 1 | BOMS direction (-1=bearish, 0=none, 1=bullish) | direction in [-1, 0, 1] |
| `tf1d_range_outcome` | int64 | 0-3 | Range outcome (0=none, 1=breakout, 2=breakdown, 3=consolidation) | outcome in [0,1,2,3] |
| `tf1d_range_confidence` | float64 | 0.0-1.0 | Range confidence score | 0.0 <= confidence <= 1.0 |
| `tf1d_range_direction` | int64 | -1, 0, 1 | Range direction | direction in [-1, 0, 1] |
| `tf1d_frvp_poc` | float64 | > 0 | Fixed Range Volume Profile - Point of Control | poc > 0 |
| `tf1d_frvp_va_high` | float64 | > 0 | FRVP - Value Area High | va_high > 0 |
| `tf1d_frvp_va_low` | float64 | > 0 | FRVP - Value Area Low | va_low > 0 |
| `tf1d_frvp_position` | float64 | 0.0-1.0 | Price position in value area (0=low, 1=high) | 0.0 <= position <= 1.0 |
| `tf1d_pti_score` | float64 | 0.0-1.0 | Potential Trap Index score | 0.0 <= pti <= 1.0 |
| `tf1d_pti_reversal` | int64 | 0-1 | PTI reversal likely | reversal in [0, 1] |

**Logical Constraints:**
```python
assert (df['tf1d_frvp_va_high'] >= df['tf1d_frvp_va_low']).all()
assert (df['tf1d_frvp_poc'] >= df['tf1d_frvp_va_low']).all()
assert (df['tf1d_frvp_poc'] <= df['tf1d_frvp_va_high']).all()
```

---

### 1.4 Tier 1: Macro Features (7 columns)

| Column | Type | Range | Description | Validation |
|--------|------|-------|-------------|------------|
| `macro_regime` | int64 | 0-3 | Macro regime (0=risk_on, 1=risk_off, 2=neutral, 3=crisis) | regime in [0,1,2,3] |
| `macro_dxy_trend` | int64 | -1, 0, 1 | DXY trend (-1=down, 0=sideways, 1=up) | trend in [-1, 0, 1] |
| `macro_yields_trend` | int64 | -1, 0, 1 | Yields trend | trend in [-1, 0, 1] |
| `macro_oil_trend` | int64 | -1, 0, 1 | Oil trend | trend in [-1, 0, 1] |
| `macro_vix_level` | int64 | 0-3 | VIX level (0=low, 1=normal, 2=elevated, 3=panic) | level in [0,1,2,3] |
| `macro_correlation_score` | float64 | -1.0 to 1.0 | Macro correlation score | -1.0 <= score <= 1.0 |
| `macro_exit_recommended` | int64 | 0-1 | Exit recommended by macro | exit in [0, 1] |

**Coverage:**
- 2024: 100% (8,761 rows)
- 2022-2023: 0% (17,475 rows) ← TO BE FIXED in Phase 1

---

### 1.5 Tier 1: 4H Timeframe Features (15 columns)

| Column | Type | Range | Description | Validation |
|--------|------|-------|-------------|------------|
| `tf4h_internal_phase` | int64 | 0-4 | Wyckoff internal phase (4H) | phase in [0,1,2,3,4] |
| `tf4h_external_trend` | int64 | -1, 0, 1 | External trend (4H) | trend in [-1, 0, 1] |
| `tf4h_structure_alignment` | float64 | 0.0-1.0 | Structure alignment score | 0.0 <= score <= 1.0 |
| `tf4h_conflict_score` | float64 | 0.0-1.0 | Internal/external conflict | 0.0 <= score <= 1.0 |
| `tf4h_squiggle_stage` | int64 | 0-4 | Squiggle wave stage | stage in [0,1,2,3,4] |
| `tf4h_squiggle_direction` | int64 | -1, 0, 1 | Squiggle direction | direction in [-1, 0, 1] |
| `tf4h_squiggle_entry_window` | int64 | 0-1 | Entry window open | window in [0, 1] |
| `tf4h_squiggle_confidence` | float64 | 0.0-1.0 | Squiggle confidence | 0.0 <= confidence <= 1.0 |
| `tf4h_choch_flag` | int64 | 0-1 | Change of Character detected | choch in [0, 1] |
| `tf4h_boms_direction` | int64 | -1, 0, 1 | BOMS direction (4H) | direction in [-1, 0, 1] |
| `tf4h_boms_displacement` | float64 | >= 0 | BOMS displacement (ATR multiple) | displacement >= 0 |
| `tf4h_fvg_present` | int64 | 0-1 | FVG present (4H) | fvg in [0, 1] |
| `tf4h_range_outcome` | int64 | 0-3 | Range outcome (4H) | outcome in [0,1,2,3] |
| `tf4h_range_breakout_strength` | float64 | 0.0-1.0 | Breakout strength | 0.0 <= strength <= 1.0 |
| `tf4h_fusion_score` | float64 | 0.0-1.0 | Fusion score (4H) | 0.0 <= score <= 1.0 |

---

### 1.6 Tier 1: 1H Timeframe Features (24 columns)

| Column | Type | Range | Description | Validation |
|--------|------|-------|-------------|------------|
| `tf1h_pti_score` | float64 | 0.0-1.0 | PTI score (1H) | 0.0 <= score <= 1.0 |
| `tf1h_pti_trap_type` | int64 | 0-3 | Trap type (0=none, 1=spring, 2=upthrust, 3=both) | type in [0,1,2,3] |
| `tf1h_pti_confidence` | float64 | 0.0-1.0 | PTI confidence | 0.0 <= confidence <= 1.0 |
| `tf1h_pti_reversal_likely` | int64 | 0-1 | Reversal likely | reversal in [0, 1] |
| `tf1h_frvp_poc` | float64 | > 0 | FRVP POC (1H) | poc > 0 |
| `tf1h_frvp_va_high` | float64 | > 0 | FRVP VA High (1H) | va_high > 0 |
| `tf1h_frvp_va_low` | float64 | > 0 | FRVP VA Low (1H) | va_low > 0 |
| `tf1h_frvp_position` | float64 | 0.0-1.0 | Price position in VA | 0.0 <= position <= 1.0 |
| `tf1h_frvp_distance_to_poc` | float64 | >= 0 | Distance to POC (ATR multiple) | distance >= 0 |
| `tf1h_fakeout_detected` | int64 | 0-1 | Fakeout detected | fakeout in [0, 1] |
| `tf1h_fakeout_intensity` | float64 | 0.0-1.0 | Fakeout intensity | 0.0 <= intensity <= 1.0 |
| `tf1h_fakeout_direction` | int64 | -1, 0, 1 | Fakeout direction | direction in [-1, 0, 1] |
| `tf1h_kelly_atr_pct` | float64 | 0.0-1.0 | Kelly position size (% of capital) | 0.0 <= kelly <= 1.0 |
| `tf1h_kelly_volatility_ratio` | float64 | > 0 | Volatility ratio for Kelly | ratio > 0 |
| `tf1h_kelly_hint` | float64 | 0.0-1.0 | Kelly hint (confidence-adjusted) | 0.0 <= hint <= 1.0 |
| `tf1h_ob_low` | float64 | > 0 | Order Block low boundary | ob_low > 0 |
| `tf1h_ob_high` | float64 | > 0 | Order Block high boundary | ob_high > 0 |
| `tf1h_bb_low` | float64 | > 0 | Bollinger Band low | bb_low > 0 |
| `tf1h_bb_high` | float64 | > 0 | Bollinger Band high | bb_high > 0 |
| `tf1h_fvg_low` | float64 | >= 0 | Fair Value Gap low boundary | fvg_low >= 0 (0 if not present) |
| `tf1h_fvg_high` | float64 | >= 0 | Fair Value Gap high boundary | fvg_high >= 0 |
| `tf1h_fvg_present` | int64 | 0-1 | FVG present (1H) | fvg in [0, 1] |
| `tf1h_bos_bearish` | int64 | 0-1 | Bearish BOS detected | bos in [0, 1] |
| `tf1h_bos_bullish` | int64 | 0-1 | Bullish BOS detected | bos in [0, 1] |

**Logical Constraints:**
```python
assert (df['tf1h_ob_high'] >= df['tf1h_ob_low']).all()
assert (df['tf1h_bb_high'] >= df['tf1h_bb_low']).all()
assert (df['tf1h_fvg_high'] >= df['tf1h_fvg_low']).all()
```

---

### 1.7 Tier 1: Multi-Timeframe Coordination (7 columns)

| Column | Type | Range | Description | Validation |
|--------|------|-------|-------------|------------|
| `mtf_alignment_ok` | int64 | 0-1 | MTF alignment check passed | alignment in [0, 1] |
| `mtf_conflict_score` | float64 | 0.0-1.0 | MTF conflict score (0=aligned, 1=conflict) | 0.0 <= score <= 1.0 |
| `mtf_governor_veto` | int64 | 0-1 | Governor veto (HTF prevents entry) | veto in [0, 1] |
| `volume_zscore` | float64 | -5.0 to 5.0 | Volume z-score (typical range) | -10.0 <= zscore <= 10.0 |
| `tf1h_fusion_score` | float64 | 0.0-1.0 | Fusion score (1H) | 0.0 <= score <= 1.0 |
| `tf1d_fusion_score` | float64 | 0.0-1.0 | Fusion score (1D) | 0.0 <= score <= 1.0 |
| `k2_fusion_score` | float64 | 0.0-1.0 | K2 fusion score (multi-domain) | 0.0 <= score <= 1.0 |

---

### 1.8 Tier 2: K2 Fusion Metrics (2 columns)

| Column | Type | Range | Description | Validation |
|--------|------|-------|-------------|------------|
| `k2_threshold_delta` | float64 | -1.0 to 1.0 | Distance from K2 threshold | -1.0 <= delta <= 1.0 |
| `k2_score_delta` | float64 | -1.0 to 1.0 | K2 score change vs previous | -1.0 <= delta <= 1.0 |

---

### 1.9 Tier 2: Macro Raw Data (10 columns)

| Column | Type | Range | Description | Validation |
|--------|------|-------|-------------|------------|
| `VIX` | float64 | 0-100+ | CBOE Volatility Index | VIX > 0 |
| `DXY` | float64 | 0-200 | US Dollar Index | DXY > 0 |
| `MOVE` | float64 | 0-500 | Bond Market Volatility Index | MOVE > 0 |
| `YIELD_2Y` | float64 | 0-20 | 2-Year Treasury Yield (%) | 0 <= yield <= 20 |
| `YIELD_10Y` | float64 | 0-20 | 10-Year Treasury Yield (%) | 0 <= yield <= 20 |
| `USDT.D` | float64 | 0-100 | Tether Dominance (%) | 0 <= dominance <= 100 |
| `BTC.D` | float64 | 0-100 | Bitcoin Dominance (%) | 0 <= dominance <= 100 |
| `TOTAL` | float64 | > 0 | Total Crypto Market Cap ($B) | TOTAL > 0 |
| `TOTAL2` | float64 | > 0 | Altcoin Market Cap ($B) | TOTAL2 > 0 |
| `funding` | float64 | -1.0 to 1.0 | Funding rate (%) | -1.0 <= funding <= 1.0 (typical) |

**Coverage:**
- 2024: 100%
- 2022-2023: 0% ← TO BE FIXED

---

### 1.10 Tier 2: Derivatives Raw Data (2 columns)

| Column | Type | Range | Description | Validation |
|--------|------|-------|-------------|------------|
| `oi` | float64 | > 0 | Open Interest ($B) | oi > 0 |
| `funding_rate` | float64 | -1.0 to 1.0 | Funding rate (duplicate of `funding`?) | -1.0 <= rate <= 1.0 |

**Coverage:**
- 2024: 100%
- 2022-2023: 0% ← TO BE FIXED

**Note:** `funding_rate` may be duplicate of `funding`. Needs investigation.

---

### 1.11 Tier 2: Realized Volatility (6 columns)

| Column | Type | Range | Description | Validation |
|--------|------|-------|-------------|------------|
| `rv_20d` | float64 | 0-500 | Realized volatility (20 days, %) | rv >= 0 |
| `rv_60d` | float64 | 0-500 | Realized volatility (60 days, %) | rv >= 0 |
| `RV_7` | float64 | 0-500 | Realized volatility (7 days, %) | rv >= 0 |
| `RV_20` | float64 | 0-500 | Realized volatility (20 days, %) | rv >= 0 |
| `RV_30` | float64 | 0-500 | Realized volatility (30 days, %) | rv >= 0 |
| `RV_60` | float64 | 0-500 | Realized volatility (60 days, %) | rv >= 0 |

**Note:** Apparent duplicates (`rv_20d` vs `RV_20`, `rv_60d` vs `RV_60`). Needs consolidation.

---

### 1.12 Tier 2: Macro Z-Scores (7 columns)

| Column | Type | Range | Description | Validation |
|--------|------|-------|-------------|------------|
| `funding_Z` | float64 | -5.0 to 5.0 | Funding rate z-score (typical) | -10.0 <= z <= 10.0 |
| `VIX_Z` | float64 | -5.0 to 5.0 | VIX z-score | -10.0 <= z <= 10.0 |
| `DXY_Z` | float64 | -5.0 to 5.0 | DXY z-score | -10.0 <= z <= 10.0 |
| `YC_SPREAD` | float64 | -5.0 to 5.0 | Yield curve spread (10Y - 2Y) | -5.0 <= spread <= 5.0 |
| `YC_Z` | float64 | -5.0 to 5.0 | Yield curve z-score | -10.0 <= z <= 10.0 |
| `BTC.D_Z` | float64 | -5.0 to 5.0 | BTC dominance z-score | -10.0 <= z <= 10.0 |
| `USDT.D_Z` | float64 | -5.0 to 5.0 | USDT dominance z-score | -10.0 <= z <= 10.0 |

**Calculation:** Z-score = (value - rolling_mean_252h) / rolling_std_252h

---

### 1.13 Tier 3: Advanced Derivatives (8 columns)

| Column | Type | Range | Description | Validation |
|--------|------|-------|-------------|------------|
| `TOTAL_RET` | float64 | -100 to 100 | Total market cap return (%) | -100 <= ret <= 100 |
| `TOTAL2_RET` | float64 | -100 to 100 | Altcoin market cap return (%) | -100 <= ret <= 100 |
| `PERP_BASIS` | float64 | -50 to 50 | Perpetual futures basis (%) | -50 <= basis <= 50 |
| `OI_CHANGE` | float64 | -100 to 100 | OI change (%) | ❌ BROKEN (all NaN) |
| `VOL_TERM` | float64 | -10 to 10 | Volatility term structure slope | -10 <= term <= 10 |
| `ALT_ROTATION` | float64 | -100 to 100 | Alt rotation indicator (%) | -100 <= rotation <= 100 |
| `TOTAL3_RET` | float64 | -100 to 100 | TOTAL3 market cap return (%) | -100 <= ret <= 100 |
| `SKEW_25D` | float64 | -10 to 10 | 25-delta skew | -10 <= skew <= 10 |

**Known Issue:** `OI_CHANGE` is all NaN (broken calculation). Needs fix in Phase 1.

---

## 2. New Columns (v2.0 Target: +24 columns)

### 2.1 Derived Features for Bear Archetypes (6 columns)

| Column | Type | Range | Description | Validation | Calculation |
|--------|------|-------|-------------|------------|-------------|
| `fvg_below` | int64 | 0-1 | FVG below current price | fvg_below in [0,1] | tf1h_fvg_high < close |
| `ob_retest` | int64 | 0-1 | Order block retest flag | ob_retest in [0,1] | (high >= ob_low) & (low <= ob_high) |
| `rsi_divergence` | int64 | 0-1 | RSI bearish divergence | divergence in [0,1] | (price HH) & (rsi LH) |
| `vol_fade` | int64 | 0-1 | Volume fading flag | vol_fade in [0,1] | volume_z < volume_z.shift(4) |
| `wick_ratio` | float64 | 0.0-1.0 | Upper wick / total range | 0.0 <= ratio <= 1.0 | (high - close) / (high - low) |
| `volume_spike` | int64 | 0-1 | Volume spike (z > 2.0) | volume_spike in [0,1] | volume_z > 2.0 |

**Purpose:** Enable S1, S2, S4 bear archetype detection

---

### 2.2 Fixed OI Derivatives (3 columns)

| Column | Type | Range | Description | Validation | Calculation |
|--------|------|-------|-------------|------------|-------------|
| `oi_change_24h` | float64 | -5B to +5B | OI absolute change (24H) | no NaNs | oi.diff(24) |
| `oi_change_pct_24h` | float64 | -50 to 50 | OI percentage change (24H) | no NaNs | oi.pct_change(24) * 100 |
| `oi_z` | float64 | -5.0 to 5.0 | OI z-score (252H window) | -10 <= z <= 10 | (oi - mean) / std |

**Purpose:** Enable S5 (Long Squeeze Cascade) bear archetype detection
**Status:** BROKEN in v1.0, FIXED in v2.0

---

### 2.3 Liquidity Score (1 column)

| Column | Type | Range | Description | Validation | Calculation |
|--------|------|-------|-------------|------------|-------------|
| `liquidity_score` | float64 | 0.0-1.0 | Composite liquidity score | 0.0 <= score <= 1.0 | See liquidity_score_calculation.md |

**Purpose:** Enable S1 (Liquidity Vacuum), S4 (Distribution), S5 (Long Squeeze)
**Status:** MISSING in v1.0, ADDED in v2.0
**Calculation:** Composite of volume profile, bid/ask spread, market depth

---

### 2.4 Fibonacci Retracements (5 columns)

| Column | Type | Range | Description | Validation | Calculation |
|--------|------|-------|-------------|------------|-------------|
| `fib_0.236` | float64 | > 0 | Fib 0.236 retracement level | fib > 0 | swing_high - 0.236 * (swing_high - swing_low) |
| `fib_0.382` | float64 | > 0 | Fib 0.382 retracement level | fib > 0 | swing_high - 0.382 * (swing_high - swing_low) |
| `fib_0.5` | float64 | > 0 | Fib 0.5 retracement level | fib > 0 | swing_high - 0.5 * (swing_high - swing_low) |
| `fib_0.618` | float64 | > 0 | Fib 0.618 retracement level | fib > 0 | swing_high - 0.618 * (swing_high - swing_low) |
| `fib_0.786` | float64 | > 0 | Fib 0.786 retracement level | fib > 0 | swing_high - 0.786 * (swing_high - swing_low) |

**Purpose:** Tier 2 feature enhancement (fib-based entries)
**Dependencies:** `swing_high_1h`, `swing_low_1h`

---

### 2.5 Swing Detection (4 columns)

| Column | Type | Range | Description | Validation | Calculation |
|--------|------|-------|-------------|------------|-------------|
| `swing_high_1h` | float64 | > 0 | 1H swing high | swing_high > 0 | Last confirmed swing high (5-bar confirmation) |
| `swing_low_1h` | float64 | > 0 | 1H swing low | swing_low > 0 | Last confirmed swing low (5-bar confirmation) |
| `swing_high_4h` | float64 | > 0 | 4H swing high | swing_high > 0 | Last confirmed swing high (4H) |
| `swing_low_4h` | float64 | > 0 | 4H swing low | swing_low > 0 | Last confirmed swing low (4H) |

**Purpose:** Fib calculations, support/resistance, trendlines
**Confirmation:** 5-bar swing (3 bars left, 1 bar peak/trough, 3 bars right)

---

### 2.6 Psychology Features (2 columns)

| Column | Type | Range | Description | Validation | Calculation |
|--------|------|-------|-------------|------------|-------------|
| `fear_greed_index` | float64 | 0-100 | Fear & Greed composite | 0 <= fg <= 100 | Composite of volatility, momentum, volume |
| `crowd_sentiment` | float64 | -1.0 to 1.0 | Crowd behavior metric | -1.0 <= sentiment <= 1.0 | Funding + OI + Volume composite |

**Purpose:** Tier 2 feature enhancement (psychology layer)

---

### 2.7 Structure Features (3 columns)

| Column | Type | Range | Description | Validation | Calculation |
|--------|------|-------|-------------|------------|-------------|
| `support_level` | float64 | > 0 | Nearest support level | support > 0 | Historical swing lows, volume clusters |
| `resistance_level` | float64 | > 0 | Nearest resistance level | resistance > 0 | Historical swing highs, volume clusters |
| `trendline_slope` | float64 | -90 to 90 | Trendline slope (degrees) | -90 <= slope <= 90 | Angle of dominant trendline |

**Purpose:** Tier 2 feature enhancement (structure layer)

---

## 3. Validation Rules

### 3.1 No NaN Policy

**Rule:** ALL columns must have 100% coverage (no NaN values)

**Validation:**
```python
def validate_no_nans(df):
    nan_counts = df.isnull().sum()
    if nan_counts.sum() > 0:
        print("❌ VALIDATION FAILED: NaN values detected")
        print(nan_counts[nan_counts > 0])
        return False
    print("✓ No NaN values detected")
    return True
```

**Action on Failure:**
- Identify source of NaNs (missing data vs calculation error)
- Fix calculation or backfill missing data
- Re-export feature store
- Re-validate

---

### 3.2 Range Validation

**Rule:** All columns must fall within expected ranges

**Validation:**
```python
def validate_ranges(df, schema):
    failures = []
    for col, (min_val, max_val) in schema.items():
        if col not in df.columns:
            failures.append(f"{col}: MISSING COLUMN")
            continue
        if (df[col] < min_val).any():
            failures.append(f"{col}: Values below {min_val}")
        if (df[col] > max_val).any():
            failures.append(f"{col}: Values above {max_val}")
    if failures:
        print("❌ RANGE VALIDATION FAILED:")
        print("\n".join(failures))
        return False
    print("✓ All ranges valid")
    return True

schema = {
    'rsi_14': (0, 100),
    'adx_14': (0, 100),
    'atr_14': (0, float('inf')),
    # ... all columns
}
```

---

### 3.3 Logical Consistency Validation

**Rule:** Logical relationships between columns must hold

**Validation:**
```python
def validate_logical_consistency(df):
    checks = [
        (df['high'] >= df['low'], "high >= low"),
        (df['high'] >= df['close'], "high >= close"),
        (df['high'] >= df['open'], "high >= open"),
        (df['low'] <= df['close'], "low <= close"),
        (df['low'] <= df['open'], "low <= open"),
        (df['volume'] >= 0, "volume >= 0"),
        (df['tf1h_ob_high'] >= df['tf1h_ob_low'], "ob_high >= ob_low"),
        (df['tf1h_bb_high'] >= df['tf1h_bb_low'], "bb_high >= bb_low"),
        (df['tf1h_fvg_high'] >= df['tf1h_fvg_low'], "fvg_high >= fvg_low"),
        (df['tf1d_frvp_va_high'] >= df['tf1d_frvp_va_low'], "va_high >= va_low"),
    ]

    failures = []
    for check, description in checks:
        if not check.all():
            failures.append(f"❌ {description} violated ({(~check).sum()} rows)")

    if failures:
        print("❌ LOGICAL CONSISTENCY FAILED:")
        print("\n".join(failures))
        return False
    print("✓ Logical consistency passed")
    return True
```

---

### 3.4 Timestamp Continuity Validation

**Rule:** Hourly data must have no gaps (except known market closures)

**Validation:**
```python
def validate_timestamp_continuity(df):
    df = df.sort_values('timestamp')
    time_diffs = df['timestamp'].diff()
    expected_diff = pd.Timedelta(hours=1)

    gaps = time_diffs[time_diffs != expected_diff]
    gaps = gaps[gaps.notna()]  # Exclude first row

    if len(gaps) > 0:
        print(f"⚠ {len(gaps)} gaps detected in timestamp continuity")
        print(gaps.head(10))
        return False
    print("✓ Timestamp continuity verified")
    return True
```

---

### 3.5 Data Type Validation

**Rule:** All columns must have correct data types

**Validation:**
```python
expected_dtypes = {
    'timestamp': 'datetime64[ns]',
    'open': 'float64',
    'high': 'float64',
    'close': 'float64',
    'low': 'float64',
    'volume': 'float64',
    'rsi_14': 'float64',
    'adx_14': 'float64',
    'macro_regime': 'int64',
    # ... all columns
}

def validate_dtypes(df, expected_dtypes):
    failures = []
    for col, expected_dtype in expected_dtypes.items():
        if col not in df.columns:
            failures.append(f"{col}: MISSING COLUMN")
            continue
        if str(df[col].dtype) != expected_dtype:
            failures.append(f"{col}: Expected {expected_dtype}, got {df[col].dtype}")

    if failures:
        print("❌ DATA TYPE VALIDATION FAILED:")
        print("\n".join(failures))
        return False
    print("✓ All data types correct")
    return True
```

---

## 4. Feature Store Build Process

### 4.1 Current Process (v1.0)

```
1. Fetch OHLCV (Binance API)
2. Fetch macro (TradingView / CoinGecko) - 2024 only
3. Fetch derivatives (Binance / OKX) - partial coverage
4. Calculate indicators (ATR, RSI, ADX, etc.)
5. Calculate MTF features (4H, 1D)
6. Calculate Wyckoff phases
7. Calculate SMC features (OB, FVG, BOS)
8. Calculate fusion scores
9. Export to parquet
```

**Issues:**
- Liquidity score not persisted
- OI derivatives (oi_change, oi_z) not calculated
- Derived features (fvg_below, etc.) not persisted
- Macro data only 2024

---

### 4.2 Proposed Process (v2.0)

```
1. Fetch OHLCV (Binance API, 1H/4H/1D)
2. Fetch macro (backfill 2022-2023) ← NEW
3. Fetch derivatives (backfill 2022-2023 OI) ← NEW
4. Calculate base indicators
5. Calculate MTF features
6. Calculate Wyckoff phases
7. Calculate SMC features
8. Calculate fusion scores
9. Calculate liquidity scores ← NEW
10. Calculate OI derivatives ← NEW
11. Calculate derived features (fvg_below, ob_retest, etc.) ← NEW
12. Calculate swing detection ← NEW
13. Calculate Fibonacci levels ← NEW
14. Calculate psychology features ← NEW
15. Calculate structure features ← NEW
16. Validate schema (no NaNs, correct ranges) ← NEW
17. Export to parquet
```

---

## 5. Schema Migration Plan

### 5.1 Phase 0: Planning ✓
- [x] Document current schema (116 columns)
- [x] Document target schema (140+ columns)
- [x] Define validation rules
- [x] Create migration plan

### 5.2 Phase 1: Fix Broken Columns
- [ ] Fix `OI_CHANGE`, `oi_change_24h`, `oi_change_pct_24h`, `oi_z`
- [ ] Backfill macro data (2022-2023)
- [ ] Backfill OI data (2022-2023)
- [ ] Validate fixes

### 5.3 Phase 2: Add Derived Features
- [ ] Add `fvg_below`, `ob_retest`, `rsi_divergence`, `vol_fade`, `wick_ratio`, `volume_spike`
- [ ] Validate bear archetype detection (S1, S2, S4)

### 5.4 Phase 3: Add Liquidity Score
- [ ] Backfill `liquidity_score` (26,236 rows)
- [ ] Validate distribution (median ~0.5, p90 ~0.85)

### 5.5 Phase 4: Add Tier 2 Features
- [ ] Add swing detection (`swing_high_1h`, `swing_low_1h`, etc.)
- [ ] Add Fibonacci levels (`fib_0.236`, `fib_0.382`, etc.)
- [ ] Add psychology features (`fear_greed_index`, `crowd_sentiment`)
- [ ] Add structure features (`support_level`, `resistance_level`, `trendline_slope`)

### 5.6 Phase 5: Final Validation
- [ ] Run full schema validation (no NaNs, ranges, logical consistency, etc.)
- [ ] Export v2.0 feature store
- [ ] Run gold standard backtests
- [ ] Merge to main

---

## 6. References

- **Brain Blueprint:** `docs/BRAIN_BLUEPRINT_SNAPSHOT_v2.md`
- **Architecture:** `docs/GHOST_TO_LIVE_ARCHITECTURE.md`
- **Feature Pipeline Audit:** `docs/technical/FEATURE_PIPELINE_AUDIT.md`
- **Liquidity Score Calculation:** `docs/technical/liquidity_score_calculation.md` (TBD)

---

## Appendix A: Validation Script

**Script:** `bin/validate_feature_store_schema.py`

```bash
python bin/validate_feature_store_schema.py \
  --input data/features_mtf/BTC_1H_2022-2024.parquet \
  --schema docs/FEATURE_STORE_SCHEMA_v2.md \
  --strict

# Output:
# ✓ No NaN values detected (116/116 columns)
# ✓ All ranges valid
# ✓ Logical consistency passed
# ✓ Timestamp continuity verified
# ✓ All data types correct
#
# Feature Store Status: VALIDATED (v2.0)
```

---

## Version History

- **v2.0.0** (2025-11-19): Complete schema definition for Ghost → Live v2 upgrade (140+ columns)
- **v1.0.0** (2025-11-14): Initial schema documentation (116 columns)
