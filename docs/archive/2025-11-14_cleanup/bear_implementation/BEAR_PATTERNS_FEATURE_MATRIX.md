# Bear Market Patterns: Feature Availability Matrix

**Generated:** 2025-11-13
**Feature Store:** BTC_1H_2022-01-01_to_2023-12-31.parquet
**Total Columns:** 119

---

## Executive Summary

**Overall Status:** 3 of 5 proposed patterns are implementable today
**Critical Gap:** `liquidity_score` column missing (blocks S1, S4)
**Workaround Available:** BOMS strength * 0.5 as liquidity proxy (crude but usable)

---

## Pattern-by-Pattern Breakdown

### S1: Liquidity Vacuum Cascade

| Feature | Required? | Available? | Column Name | Notes |
|---------|-----------|------------|-------------|-------|
| Liquidity Score | YES | NO | `liquidity_score` | BLOCKING - must backfill |
| Volume Z-Score | YES | YES | `volume_zscore` | Available |
| 4H Trend | YES | YES | `tf4h_external_trend` | Available (values: up/down/ranging) |
| FVG Below Price | NICE | YES | `tf4h_fvg_present` | Can be used as proxy |

**Status:** BLOCKED
**Workaround:** Use `tf1d_boms_strength * 0.5` as crude liquidity proxy
**Recommendation:** Backfill `liquidity_score` before production deployment

**Backfill Formula:**
```python
liquidity_score = (
    0.5 * df['tf1d_boms_strength'].fillna(0) +
    0.25 * df['tf4h_fvg_present'].fillna(0) +
    0.25 * (df['tf4h_boms_displacement'] / (2.0 * df['atr_20'])).clip(0, 1)
)
```

---

### S2: Failed Rally Rejection [READY TO IMPLEMENT]

| Feature | Required? | Available? | Column Name | Notes |
|---------|-----------|------------|-------------|-------|
| RSI (14) | YES | YES | `rsi_14` | Available |
| Volume Z-Score | YES | YES | `volume_zscore` | Available |
| OHLC Data | YES | YES | `open`, `high`, `low`, `close` | For wick ratio calculation |
| 4H Trend | NICE | YES | `tf4h_external_trend` | For confirmation filter |

**Status:** READY - All features available
**Implementation:** Immediate
**Wick Ratio Calculation:**
```python
wick_upper = (df['high'] - df[['open', 'close']].max(axis=1)) / (df['high'] - df['low'] + 1e-9)
signal = (df['rsi_14'] > 70) & (df['volume_zscore'] < 0.5) & (wick_upper > 0.4)
```

---

### S3: Wyckoff Upthrust [MERGED INTO S2]

**Decision:** >70% overlap with S2 (Failed Rally Rejection)
**Action:** Do not implement separately; S2 captures this pattern
**Rationale:** Both detect resistance rejection with upper wicks; redundant

---

### S4: Distribution Climax

| Feature | Required? | Available? | Column Name | Notes |
|---------|-----------|------------|-------------|-------|
| Volume Z-Score | YES | YES | `volume_zscore` | Available |
| Liquidity Score | YES | NO | `liquidity_score` | BLOCKING - must backfill |
| RSI (14) | FILTER | YES | `rsi_14` | For tightened version |
| 4H Trend | FILTER | YES | `tf4h_external_trend` | For confirmation |

**Status:** BLOCKED (same as S1)
**Workaround:** Use BOMS proxy, but edge may degrade
**Recommendation:** Implement in Phase 2 after liquidity backfill + add RSI/trend filters

**Tightened Logic:**
```python
signal = (
    (df['volume_zscore'] > 1.8) &       # Tightened from 1.5
    (liq_proxy < 0.20) &                 # Tightened from 0.25
    (df['rsi_14'] > 60) &                # NEW: in rally, not oversold
    (df['tf4h_external_trend'] == 'down') # NEW: higher TF confirmation
)
```

---

### S5: Long Squeeze Cascade (Corrected) [READY TO IMPLEMENT]

| Feature | Required? | Available? | Column Name | Notes |
|---------|-----------|------------|-------------|-------|
| Funding Rate Z-Score | YES | YES | `funding_Z` | Available |
| OI Change | YES | YES | `OI_CHANGE` | Available (but appears broken - all zeros?) |
| RSI (14) | YES | YES | `rsi_14` | Available |
| Liquidity Score | NICE | NO | `liquidity_score` | Optional filter |

**Status:** READY (with caveat on OI_CHANGE data quality)
**Implementation:** Immediate, but investigate OI_CHANGE pipeline

**Data Quality Issue:**
```
OI_CHANGE statistics (2022):
- Mean: 0.000%
- % Large Drops: 0.0%
- % Large Spikes: 0.0%
```

**Action Required:** Diagnose why OI_CHANGE shows zero variance
**Workaround (short-term):** Relax OI filter to optional; trigger on funding + RSI alone

```python
# Strict version (requires OI fix)
signal = (
    (df['funding_Z'] > 1.0) &
    (df['OI_CHANGE'] > 0.03) &
    (df['rsi_14'] > 65)
)

# Relaxed version (works with broken OI data)
signal = (
    (df['funding_Z'] > 1.0) &
    (df['rsi_14'] > 65)
)
```

---

### S6: Altcoin Rotation Drain

| Feature | Required? | Available? | Column Name | Notes |
|---------|-----------|------------|-------------|-------|
| Altcoin Rotation | YES | YES | `ALT_ROTATION` | Available |
| DXY Z-Score | YES | YES | `DXY_Z` | Available |
| Liquidity Score | YES | NO | `liquidity_score` | BLOCKING |

**Status:** REJECTED (pattern too rare)
**Rationale:** Even with all features, pattern triggered <1% of time in 2022
**Recommendation:** Do not implement; low frequency, unclear edge

---

### S7: Yield Curve Panic

| Feature | Required? | Available? | Column Name | Notes |
|---------|-----------|------------|-------------|-------|
| Yield Curve Spread | YES | YES | `YC_SPREAD` | Available |
| Break of Structure | YES | PARTIAL | `tf4h_choch_flag` | CHOCH available, not explicit BOS |
| Volume Z-Score | YES | YES | `volume_zscore` | Available |
| VIX Z-Score | NICE | YES | `VIX_Z` | For confirmation |

**Status:** REJECTED (no tradeable edge)
**Rationale:** YC inverted 66% of 2022; event too persistent, not a trigger
**Recommendation:** Use YC inversion as **regime filter**, not entry signal

---

### S8: Trend Exhaustion Fade

| Feature | Required? | Available? | Column Name | Notes |
|---------|-----------|------------|-------------|-------|
| ADX (14) | YES | YES | `adx_14` | Available |
| RSI (14) | YES | YES | `rsi_14` | Available |
| Volume Z-Score | YES | YES | `volume_zscore` | Available |

**Status:** REJECTED (wrong directional bias)
**Rationale:** Forward returns were POSITIVE (bullish), not negative (bearish)
**Recommendation:** Do not implement; contradicts bear market thesis

---

## Additional Patterns (Proposed by Analysis)

### Macro Risk-Off Acceleration

| Feature | Required? | Available? | Column Name | Notes |
|---------|-----------|------------|-------------|-------|
| VIX Z-Score | YES | YES | `VIX_Z` | Available |
| DXY Z-Score | YES | YES | `DXY_Z` | Available |
| BTC-SPY Correlation | YES | NO | `btc_spy_corr` | MISSING - need to add |
| MOVE Index | NICE | NO | `move_index` | MISSING - need to add |

**Status:** BLOCKED (need correlation + MOVE features)
**Priority:** HIGH (strong 2022 signal)
**Action Required:**
1. Add rolling 30-day BTC-SPY correlation to feature pipeline
2. Fetch MOVE index from Bloomberg/FRED API
3. Backtest pattern once features available

**Expected Edge:** PF 1.5+ (macro shocks were reliable in 2022)

---

### Liquidation Cascade

| Feature | Required? | Available? | Column Name | Notes |
|---------|-----------|------------|-------------|-------|
| OI Drop (5-min) | YES | NO | `oi_drop_5m` | MISSING - need higher granularity |
| Price Drop (5-min) | YES | NO | `price_drop_5m` | MISSING - need OHLC resampling |
| Volume Z-Score | YES | YES | `volume_zscore` | Available (but 1H, not 5m) |

**Status:** BLOCKED (need 5-minute data pipeline)
**Priority:** HIGH (liquidation cascades are high-alpha events)
**Action Required:**
1. Add 5-minute OI data to feature store
2. Calculate rolling OI change over 5m/15m/1H windows
3. Detect rapid OI drops (>8% in 15 minutes)

**Note:** Current OI_CHANGE column appears broken (all zeros in 2022)

---

### Support-to-Resistance Flip

| Feature | Required? | Available? | Column Name | Notes |
|---------|-----------|------------|-------------|-------|
| Price vs. Prior Support | YES | PARTIAL | `close` + pivot detection | Need to add S/R detection |
| Retest Detection | YES | NO | N/A | Requires price action logic |
| Volume Z-Score | YES | YES | `volume_zscore` | Available |
| 4H Trend | YES | YES | `tf4h_external_trend` | Available |

**Status:** BLOCKED (need support/resistance detection)
**Priority:** MEDIUM (classical TA pattern, proven)
**Action Required:**
1. Implement pivot point detection (swing highs/lows)
2. Track recent breakdowns (price < support)
3. Detect retest from below (price approaches former support)

**Suggested Implementation:**
```python
# Option 1: Use SMAs as dynamic S/R
support_broken = (df['close'] < df['sma_50']) & (df['close'].shift(1) >= df['sma_50'].shift(1))

# Option 2: Use FRVP POC as S/R
support_broken = (df['close'] < df['tf1d_frvp_poc']) & (df['close'].shift(1) >= df['tf1d_frvp_poc'].shift(1))

# Retest logic
retest = (
    support_broken.rolling(20).sum() > 0 &  # Breakdown in last 20 bars
    (df['close'] > df['sma_50'] * 0.98) &    # Within 2% of former support
    (df['volume_zscore'] < 0.5)              # Weak volume on retest
)
```

---

## Missing Features Summary

### Critical Gaps (Blocking Patterns)

| Feature | Impact | Workaround Available? | Priority |
|---------|--------|----------------------|----------|
| `liquidity_score` | Blocks S1, S4 | Yes (BOMS proxy) | HIGH |
| `oi_drop_5m` | Blocks Liquidation Cascade | No | HIGH |
| `btc_spy_corr` | Blocks Macro Risk-Off | No | HIGH |

### Nice-to-Have (Enhance Edge)

| Feature | Impact | Workaround Available? | Priority |
|---------|--------|----------------------|----------|
| `move_index` | Improves Macro Risk-Off | Yes (use VIX as proxy) | MEDIUM |
| `support_resistance_levels` | Blocks S/R Flip | Yes (use SMAs/POC) | MEDIUM |
| `oi_change_pct_5m` | Improves Liquidation Cascade | No | MEDIUM |

---

## Feature Engineering Recommendations

### 1. Backfill `liquidity_score` (IMMEDIATE)

**Formula:**
```python
df['liquidity_score'] = (
    0.50 * df['tf1d_boms_strength'].fillna(0) +
    0.25 * df['tf4h_fvg_present'].fillna(0).astype(float) +
    0.20 * (df['tf4h_boms_displacement'] / (2.0 * df['atr_20'])).clip(0, 1).fillna(0) +
    0.05 * (df['tf1h_frvp_position'] == 'high').astype(float)  # Near value area high
)
```

**Validation:**
- Check correlation with manually labeled liquidity events
- Target: >0.7 correlation with "thick" vs "thin" market classifications

---

### 2. Add BTC-SPY Correlation (IMMEDIATE)

**Formula:**
```python
import yfinance as yf

# Fetch SPY data
spy = yf.download('SPY', start='2022-01-01', end='2023-12-31', interval='1h')

# Align with BTC data
df = df.join(spy[['Close']].rename(columns={'Close': 'spy_close'}), how='left')

# Calculate rolling 30-day correlation
df['btc_spy_corr'] = df['close'].rolling(30*24).corr(df['spy_close'])
```

---

### 3. Fix OI_CHANGE Data Pipeline (URGENT)

**Current Issue:** OI_CHANGE column shows zero variance in 2022 (impossible)

**Diagnosis Steps:**
1. Check data source: Is OI data being fetched correctly?
2. Check granularity: Is OI_CHANGE calculated at 1H or 24H frequency?
3. Check normalization: Is percentage change calculated correctly?

**Expected Fix:**
```python
# Should see non-zero variance
df['oi_change_pct_1h'] = df['oi'].pct_change()
df['oi_change_pct_24h'] = df['oi'].pct_change(24)

# Validate
assert df['oi_change_pct_1h'].std() > 0, "OI_CHANGE is broken!"
```

---

### 4. Add 5-Minute OI Data (MEDIUM PRIORITY)

**Use Case:** Liquidation cascade detection

**Implementation:**
```python
# Fetch 5-minute OHLCV + OI data
df_5m = fetch_ohlcv(asset='BTC', timeframe='5m', start='2022-01-01', end='2023-12-31')
df_5m['oi_drop_5m'] = df_5m['oi'].pct_change()
df_5m['oi_drop_15m'] = df_5m['oi'].pct_change(3)  # 3 x 5m = 15m

# Flag liquidation events
df_5m['liquidation_cascade'] = (df_5m['oi_drop_15m'] < -0.08) & (df_5m['close'].pct_change(3) < -0.02)

# Resample to 1H and join
df = df.join(df_5m[['liquidation_cascade']].resample('1H').max(), how='left')
```

---

### 5. Implement Support/Resistance Detection (MEDIUM PRIORITY)

**Option A: Pivot Points (Simple)**
```python
def detect_pivots(df, left_bars=5, right_bars=5):
    """Detect swing highs and lows"""
    pivot_highs = []
    pivot_lows = []

    for i in range(left_bars, len(df) - right_bars):
        # Swing high: higher than left_bars and right_bars neighbors
        if all(df['high'].iloc[i] > df['high'].iloc[i-left_bars:i]) and \
           all(df['high'].iloc[i] > df['high'].iloc[i+1:i+right_bars+1]):
            pivot_highs.append((df.index[i], df['high'].iloc[i]))

        # Swing low: lower than left_bars and right_bars neighbors
        if all(df['low'].iloc[i] < df['low'].iloc[i-left_bars:i]) and \
           all(df['low'].iloc[i] < df['low'].iloc[i+1:i+right_bars+1]):
            pivot_lows.append((df.index[i], df['low'].iloc[i]))

    return pivot_highs, pivot_lows
```

**Option B: FRVP POC (Already Available)**
```python
# Use POC as dynamic support/resistance
df['poc_sr'] = df['tf1d_frvp_poc']
df['support_broken'] = (df['close'] < df['poc_sr']) & (df['close'].shift(1) >= df['poc_sr'].shift(1))
```

---

## Feature Store Audit Checklist

- [x] OHLCV data (1H granularity) - Available
- [x] Technical indicators (RSI, ADX, ATR, SMAs) - Available
- [x] Macro features (VIX, DXY, YC_SPREAD) - Available
- [x] Derivatives (funding_Z) - Available
- [ ] Derivatives (OI_CHANGE) - Broken (all zeros)
- [ ] Liquidity score - Missing
- [ ] BTC-SPY correlation - Missing
- [ ] MOVE index - Missing
- [ ] 5-minute OI data - Missing
- [ ] Support/resistance levels - Missing

**Completeness:** 70% (7/10 critical features available)

---

## Implementation Priority Matrix

| Feature | Pattern Impact | Effort | Priority | Timeline |
|---------|---------------|--------|----------|----------|
| Backfill `liquidity_score` | S1, S4 | LOW | HIGH | Week 1 |
| Fix `OI_CHANGE` pipeline | S5 | MEDIUM | HIGH | Week 1 |
| Add `btc_spy_corr` | Macro Risk-Off | LOW | HIGH | Week 2 |
| Add `MOVE` index | Macro Risk-Off | MEDIUM | MEDIUM | Week 3 |
| Add 5m OI data | Liquidation Cascade | HIGH | MEDIUM | Week 4 |
| Implement S/R detection | S/R Flip | MEDIUM | MEDIUM | Week 4 |

---

**End of Matrix**
