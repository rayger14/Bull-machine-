# Open Interest (OI) Pipeline Specification

**Status**: Phase 1 - Critical Data Blocker
**Priority**: CRITICAL (Blocks S5 Long Squeeze archetype)
**Data Availability**: ✅ OI data available via OKX API
**Recommendation**: **Use Option A (Raw OI Data)**

---

## Executive Summary

**Decision: Use Option A (Preferred) - Raw OI data is available**

- **Source**: OKX API (`/api/v5/public/open-interest-history`)
- **Instrument**: `BTC-USDT-SWAP` (perpetual futures)
- **Coverage**: Historical data available from 2022-present
- **Granularity**: 1H bars (matches MTF store)
- **Existing Implementation**: `bin/fix_oi_change_pipeline.py` (production-ready)

**Option B (Contingency)** is NOT needed - raw OI data is accessible.

---

## Option A: Raw OI Data Pipeline (RECOMMENDED)

### 1. Data Source

#### API Details
- **Provider**: OKX Exchange
- **Endpoint**: `https://www.okx.com/api/v5/public/open-interest-history`
- **Instrument**: `BTC-USDT-SWAP` (perpetual futures)
- **Granularity**: 1H (hourly bars)
- **Rate Limit**: 200ms between requests (enforced in script)
- **Pagination**: 100 records per request, cursor-based

#### Data Coverage
- **Historical Availability**: 2022-01-01 to present
- **Critical Periods**:
  - Terra collapse (May 9-12, 2022): ✅ Available
  - FTX collapse (Nov 8-10, 2022): ✅ Available
- **Expected Records**: ~26,000 rows (2022-2024)

#### Existing CSV
- **File**: `data/OI_1H.csv`
- **Status**: Basic OHLC data (no OI field directly visible)
- **Action**: Fetch fresh OI data from API (more reliable)

### 2. Derived Features

#### Feature Columns

| Column Name | Formula | NaN Handling | Description |
|-------------|---------|--------------|-------------|
| `oi` | Raw API field | Forward fill | Open Interest (USD notional or BTC) |
| `oi_change_24h` | `oi.diff(24)` | First 24 bars → NaN | Absolute change vs 24H ago |
| `oi_change_pct_24h` | `oi.pct_change(24) * 100` | First 24 bars → NaN | Percentage change vs 24H ago |
| `oi_z` | `(oi - rolling_mean) / rolling_std` | First 252 bars → 0.0 | Z-score (252H window ≈ 10.5 days) |
| `oi_spike` | `abs(oi_z) > 2.0` | Derived from `oi_z` | Binary flag (2-sigma event) |

#### Derivation Logic

**1. Raw OI (from API)**
```python
# OKX API returns OI in contracts or BTC
# Convert to USD notional if needed
df['oi'] = df['oi'].astype(float)  # Raw field from API
```

**2. 24H Change (Absolute)**
```python
df['oi_change_24h'] = df['oi'].diff(24)
# First 24 bars will be NaN (no lookback)
```

**3. 24H Change (Percentage)**
```python
df['oi_change_pct_24h'] = df['oi'].pct_change(24) * 100
# First 24 bars will be NaN
```

**4. Z-Score (Rolling 252H Window)**
```python
window = 252  # ~10.5 days
rolling_mean = df['oi'].rolling(window=window, min_periods=100).mean()
rolling_std = df['oi'].rolling(window=window, min_periods=100).std()
df['oi_z'] = (df['oi'] - rolling_mean) / rolling_std
df['oi_z'] = df['oi_z'].fillna(0.0)  # First 100-252 bars → 0.0
```

**5. Spike Detection**
```python
df['oi_spike'] = (df['oi_z'].abs() > 2.0).astype(int)
# 2-sigma threshold (expect ~5% of data to be spikes)
```

### 3. NaN Handling Strategy

#### Expected NaN Patterns

| Feature | NaN Location | Count (3yr data) | Handling |
|---------|--------------|------------------|----------|
| `oi` | None (API always provides) | 0 | Forward fill as fallback |
| `oi_change_24h` | First 24 bars | 24 | **Keep as NaN** (no valid lookback) |
| `oi_change_pct_24h` | First 24 bars | 24 | **Keep as NaN** |
| `oi_z` | First ~100-252 bars | 100-252 | **Fill with 0.0** (neutral) |
| `oi_spike` | Derived from `oi_z` | 0 | Computed from filled `oi_z` |

#### Gap Filling Strategy
1. **Raw OI**: Forward fill (assume OI persists if API returns gaps)
2. **Change Metrics**: Do NOT fill (NaN = no valid lookback data)
3. **Z-Score**: Fill first 100-252 bars with `0.0` (neutral baseline)
4. **Spike Flag**: Compute from filled `oi_z` (no additional NaNs)

### 4. Feature Registry Entry

Add to `engine/features/registry.py`:

```python
# Raw OI
FeatureSpec(
    canonical="oi",
    dtype="float64",
    tier=3,  # Macro/Market tier
    required=False,
    aliases=["open_interest"],
    range_min=0.0,
    range_max=None,  # Unbounded (scales with market)
    description="Open Interest (USD notional) from OKX perpetual futures"
)

# OI Change (24H Absolute)
FeatureSpec(
    canonical="oi_change_24h",
    dtype="float64",
    tier=3,
    required=False,
    aliases=["oi_delta_24h"],
    range_min=None,
    range_max=None,
    description="Absolute change in OI vs 24 hours ago"
)

# OI Change (24H Percentage)
FeatureSpec(
    canonical="oi_change_pct_24h",
    dtype="float64",
    tier=3,
    required=False,
    aliases=["oi_pct_24h"],
    range_min=None,
    range_max=None,
    description="Percentage change in OI vs 24 hours ago"
)

# OI Z-Score
FeatureSpec(
    canonical="oi_z",
    dtype="float64",
    tier=3,
    required=False,
    aliases=["oi_zscore"],
    range_min=None,
    range_max=None,
    description="OI z-score (252H rolling window)"
)

# OI Spike Flag
FeatureSpec(
    canonical="oi_spike",
    dtype="int8",
    tier=3,
    required=False,
    aliases=["oi_2sigma"],
    range_min=0,
    range_max=1,
    description="Binary flag: |oi_z| > 2.0 (2-sigma event)"
)
```

### 5. Backfill Script Design

#### Script: `bin/fix_oi_change_pipeline.py`

**Status**: Already exists and is production-ready

#### Architecture
1. **Phase 1**: Fetch OI data from OKX API (with caching)
2. **Phase 2**: Calculate derived metrics (change, z-score, spike)
3. **Phase 3**: Validate against known events (Terra, FTX)
4. **Phase 4**: Patch MTF store (merge by timestamp)

#### CLI Usage
```bash
# Full fix (fetch + calculate + validate)
python3 bin/fix_oi_change_pipeline.py

# Use existing cache (skip fetch)
python3 bin/fix_oi_change_pipeline.py --skip-fetch

# Dry run (validation only)
python3 bin/fix_oi_change_pipeline.py --dry-run

# Custom date range
python3 bin/fix_oi_change_pipeline.py \
    --start-date 2022-01-01 \
    --end-date 2023-12-31
```

#### Pseudocode
```python
def main():
    # Phase 1: Fetch OI data
    if not args.skip_fetch:
        oi_df = fetch_okx_historical_oi(
            start_date='2022-01-01',
            end_date='2024-12-31',
            cache_path='data/cache/okx_oi_2022_2024.parquet'
        )
    else:
        oi_df = pd.read_parquet(args.cache_path)

    # Phase 2: Calculate derived metrics
    oi_df = calculate_oi_metrics(oi_df, window=252)
    # Adds: oi_change_24h, oi_change_pct_24h, oi_z, oi_spike

    # Phase 3: Validate against known events
    validation = validate_oi_metrics(oi_df)
    # Checks Terra collapse (May 2022) and FTX collapse (Nov 2022)

    # Phase 4: Patch MTF store
    output_path = patch_mtf_store(
        mtf_path='data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet',
        oi_df=oi_df,
        dry_run=args.dry_run
    )

    return 0 if validation['validation_passed'] else 1
```

### 6. Validation Strategy

#### Known Event Validation

| Event | Date Range | Expected Signal | Threshold |
|-------|------------|-----------------|-----------|
| Terra Collapse | May 9-12, 2022 | `oi_change_pct_24h < -15%` | CRITICAL |
| FTX Collapse | Nov 8-10, 2022 | `oi_change_pct_24h < -20%` | CRITICAL |
| Normal Periods | All other dates | `-5% < oi_change_pct_24h < +5%` | 85% of data |

#### Validation Checks
```python
def validate_oi_metrics(df):
    checks = {}

    # 1. Terra collapse (May 9-12, 2022)
    terra_df = df[(df['timestamp'] >= '2022-05-09') &
                  (df['timestamp'] <= '2022-05-12')]
    checks['terra_detected'] = terra_df['oi_change_pct_24h'].min() < -15.0

    # 2. FTX collapse (Nov 8-10, 2022)
    ftx_df = df[(df['timestamp'] >= '2022-11-08') &
                (df['timestamp'] <= '2022-11-10')]
    checks['ftx_detected'] = ftx_df['oi_change_pct_24h'].min() < -20.0

    # 3. Normal range (85% of data in ±5%)
    normal_pct = df['oi_change_pct_24h'].between(-5.0, 5.0).mean() * 100
    checks['normal_range_ok'] = normal_pct > 85.0

    return checks
```

### 7. S5 Archetype Integration

#### Current S5 Logic
```python
# S5 requires oi_change_pct_24h for "long squeeze" detection
if 'oi_change_pct_24h' not in ctx or pd.isna(ctx['oi_change_pct_24h']):
    return 0.0  # Graceful degradation (no signal)

# Detect rapid OI decline (liquidation cascade proxy)
if ctx['oi_change_pct_24h'] < -10.0:  # >10% drop in 24H
    cascade_score += 0.3
```

#### Post-Backfill Behavior
- **Before**: `oi_change_pct_24h` missing → S5 returns `0.0` (no matches)
- **After**: `oi_change_pct_24h` populated → S5 detects Terra/FTX events
- **Expected Impact**: S5 match count increases from 0 to 5-10 events (2022-2024)

---

## Option B: Synthetic OI Proxy (CONTINGENCY - NOT NEEDED)

**Status**: NOT REQUIRED (raw OI data is available)

This section is kept for documentation completeness but should NOT be implemented.

### Proxy Metrics (If Raw OI Unavailable)

If OKX API were unavailable, we would construct a synthetic OI score using:

1. **Volume Z-Score Changes** (sudden volume spikes = OI events)
2. **Funding Rate Acceleration** (rapid funding changes correlate with OI)
3. **Price-Volume Divergence** (price drops + high volume = forced liquidations)
4. **Wick Size** (large wicks = liquidation cascades)

### Synthetic OI Score Formula
```python
# NOT IMPLEMENTED - for reference only
oi_proxy = (
    0.40 * volume_z_change_24h +      # Volume spike
    0.30 * abs(funding_rate_change) + # Funding acceleration
    0.20 * price_vol_divergence +     # Divergence
    0.10 * wick_ratio                 # Liquidation proxy
)
```

---

## 8. Execution Plan

### Pre-Flight Checklist
- [ ] Verify OKX API access (no auth required for public endpoint)
- [ ] Backup MTF store before patching
- [ ] Check internet connectivity (API requires network access)

### Backfill Steps
```bash
# 1. Test API connectivity
curl "https://www.okx.com/api/v5/public/open-interest-history?instId=BTC-USDT-SWAP&period=1H&limit=10"

# 2. Dry run (fetch + validate, no write)
python3 bin/fix_oi_change_pipeline.py --dry-run

# 3. Review validation output
#    - Check Terra collapse detection
#    - Check FTX collapse detection
#    - Verify normal range (85%+ data in ±5%)

# 4. Full backfill (write to store)
python3 bin/fix_oi_change_pipeline.py

# 5. Validate patched store
python3 bin/validate_feature_store.py \
    --mtf-store data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
    --check oi,oi_change_24h,oi_change_pct_24h,oi_z
```

### Performance Expectations
- **Fetch Time**: ~5-10 minutes (with rate limiting)
- **Calculation Time**: ~1 minute (vectorized operations)
- **Total Time**: ~10-15 minutes
- **Cache Size**: ~5 MB (parquet cache)

---

## 9. Success Criteria

### Functional Requirements
- ✅ `oi`, `oi_change_24h`, `oi_change_pct_24h`, `oi_z` columns added to MTF store
- ✅ Terra collapse detected (May 2022: `oi_change_pct_24h < -15%`)
- ✅ FTX collapse detected (Nov 2022: `oi_change_pct_24h < -20%`)
- ✅ 85%+ of data in normal range (`-5% to +5%`)

### Quality Requirements
- ✅ First 24 bars have NaN for change metrics (expected)
- ✅ First 100-252 bars have `0.0` for `oi_z` (filled)
- ✅ No unexpected NaNs after row 252

### Integration Requirements
- ✅ S5 archetype produces non-zero matches (Terra/FTX events)
- ✅ S5 graceful degradation still works (no crashes on NaN)

---

## 10. Known Issues & Mitigations

### Issue 1: OKX API Rate Limits
**Mitigation**: 200ms sleep between requests (enforced in script)

### Issue 2: API Downtime
**Mitigation**: Cache fetched data to parquet, use `--skip-fetch` flag

### Issue 3: First 24 Bars Have NaN
**Mitigation**: Expected behavior (no valid lookback), S5 handles this gracefully

---

## 11. References

- **OKX API Docs**: https://www.okx.com/docs-v5/en/#rest-api-public-data-get-open-interest
- **Backfill Script**: `bin/fix_oi_change_pipeline.py`
- **S5 Archetype**: `engine/archetypes/bear_patterns_phase1.py` (S5_LONG_SQUEEZE)
- **Feature Registry**: `engine/features/registry.py`
- **Known Events**: `docs/BEAR_MARKET_ANALYSIS_2022.md`
