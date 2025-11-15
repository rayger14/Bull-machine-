# OI_CHANGE Pipeline Failure Diagnosis

**Date**: 2025-11-13
**Severity**: CRITICAL - Blocks S5 (Long Squeeze), S4 (Distribution), S1 (Liquidity Vacuum)
**Status**: Root cause identified, fix ready for implementation

---

## Executive Summary

The OI (Open Interest) feature pipeline has **2 critical failures**:

1. **OI raw data missing for 2022-2023** (only 2024 data available)
2. **Derived features never calculated** (oi_change_24h, oi_change_pct_24h, oi_z are all NaN)

This completely blocks bear archetype implementation that relies on liquidation cascade detection.

---

## Root Cause Analysis

### Issue 1: OI Data Coverage Gap (2022-2023)

**Current State:**
```
MTF Store: BTC_1H_2022-01-01_to_2024-12-31.parquet (26,236 rows)

OI Coverage by Year:
- 2022: 0 / 8,741 (0.0%)  ❌
- 2023: 0 / 8,734 (0.0%)  ❌
- 2024: 8,761 / 8,761 (100%)  ✅
```

**Root Cause:**
- Macro features file (`BTC_macro_features.parquet`) only contains data from 2024-01-05 onwards
- OI data was added to macro pipeline AFTER the 2022-2023 backtest window
- When MTF stores were merged, 2022-2023 rows got OI column but with NaN values

**Evidence:**
```python
# Macro features file
Shape: (15,550, 19)
Date range: 2024-01-05 to 2025-10-14
OI stats: mean=82.4B, std=22.7B (valid data)
```

### Issue 2: Derived OI Features Never Calculated

**Current State:**
```
oi_change_24h:     0 / 26,236 non-null (ALL NaN)  ❌
oi_change_pct_24h: 0 / 26,236 non-null (ALL NaN)  ❌
oi_z:              0 / 26,236 non-null (ALL NaN)  ❌
```

**Root Cause:**
- Columns were created (exist in schema) but never populated
- Calculation logic exists in `bin/patch_derivatives_columns.py` (lines 55-66):
  ```python
  df['oi_change_24h'] = df['oi'].diff(24)
  df['oi_change_pct_24h'] = df['oi'].pct_change(24) * 100
  df['oi_z'] = (df['oi'] - rolling_mean) / rolling_std
  ```
- BUT: This logic operates on derivatives parquet, not MTF store
- MTF merge process copied OI column but never ran derivative calculations

---

## Impact Assessment

### Blocked Bear Patterns

| Pattern | Required OI Features | Impact | Severity |
|---------|---------------------|--------|----------|
| S5: Long Squeeze | oi_change_pct, oi_z | Cannot detect OI spikes → no liquidation cascade detection | CRITICAL |
| S4: Distribution Climax | oi_z (optional) | Reduced confidence in volume analysis | HIGH |
| S1: Liquidity Vacuum | (indirect via funding correlation) | Degraded multi-factor scoring | MEDIUM |

### Known Event Detection Failures

Without OI data, the following 2022 events are **undetectable**:

1. **Terra/Luna Collapse (May 2022)**
   - Expected: oi_change_24h < -15% (massive long liquidations)
   - Actual: NaN (cannot detect)

2. **FTX Collapse (November 2022)**
   - Expected: oi_change_24h < -20% (exchange insolvency cascade)
   - Actual: NaN (cannot detect)

3. **June 2022 Capitulation**
   - Expected: oi_change_24h < -10% (bottom formation)
   - Actual: NaN (cannot detect)

---

## Pipeline Architecture Investigation

### Current Data Flow (Broken)

```
1. Raw OI Source (Macro Pipeline)
   └─> data/macro/BTC_macro_features.parquet
       └─> 'oi' column (2024+ only)

2. MTF Feature Store Build
   └─> Merges macro features into base MTF
       └─> Copies 'oi' column (NaN for 2022-2023)
       └─> Creates oi_change_24h, oi_change_pct_24h, oi_z columns (empty shells)

3. Derived Calculation (NEVER RUN)
   └─> bin/patch_derivatives_columns.py has logic
       └─> BUT: operates on derivatives file, not MTF store
       └─> Calculation never triggered for MTF
```

### Why Derived Features Are Missing

**Design flaw:** Two separate calculation paths that never merged:

**Path A (Derivatives File):**
- Script: `bin/patch_derivatives_columns.py`
- Input: `data/derivatives/BTC_derivatives_2022_2024.parquet`
- Output: Derivatives file with calculated OI metrics
- Status: May have been run, but file doesn't exist (no derivatives parquet found)

**Path B (MTF Store):**
- Script: Feature builder (unknown location)
- Input: Base MTF + macro features
- Output: MTF store with merged columns
- Status: Ran successfully but only copied raw OI, never calculated derivatives

**Gap:** Path A calculations never flowed into Path B's MTF store.

---

## Historical Data Availability

### OI Data Sources (2022-2024)

Based on investigation, we have **2 potential OI sources**:

#### Option A: OKX Historical OI API
- **Endpoint**: `GET /api/v5/public/open-interest-history`
- **Coverage**: 2020-01-01 onwards (confirmed available)
- **Granularity**: 1H (perfect match for MTF)
- **Data Quality**: High (official exchange data)
- **Script**: `bin/backfill_missing_macro_features.py` (lines 98-179)

**Status**: Script exists and works, but may not have been run for 2022-2023.

#### Option B: Coinglass API
- **Endpoint**: Funding + OI data
- **Coverage**: 2020+ (varies by metric)
- **Scripts**:
  - `bin/fetch_coinglass_funding.py`
  - `bin/fetch_coinglass_funding_v2.py`
- **Status**: Funding data available, OI needs verification

**Recommendation**: Use Option A (OKX) as primary, Option B (Coinglass) as fallback.

---

## Fix Strategy

### Phase 1: Backfill OI Raw Data (2022-2023)

**Goal**: Populate `oi` column for 2022-2023 rows in MTF store

**Approach**:
1. Fetch OKX historical OI (2022-01-01 to 2023-12-31)
2. Align timestamps with MTF store
3. Merge into existing `oi` column (replace NaN)

**Script**: `bin/fix_oi_change_pipeline.py` (Phase 1)

### Phase 2: Calculate Derived OI Metrics

**Goal**: Populate oi_change_24h, oi_change_pct_24h, oi_z for ALL rows

**Calculations**:
```python
# After Phase 1, full OI column is available
oi_change_24h = df['oi'].diff(24)  # Absolute change
oi_change_pct_24h = df['oi'].pct_change(24) * 100  # Percentage change
oi_z = (df['oi'] - rolling_mean_252h) / rolling_std_252h  # Z-score
```

**Script**: `bin/fix_oi_change_pipeline.py` (Phase 2)

### Phase 3: Validation Against Known Events

**Goal**: Verify fix correctness using historical bear market events

**Test Cases**:
1. **Terra Collapse (2022-05-09 to 2022-05-12)**
   - Expected: oi_change_pct < -15%
   - Timeframe: 4-day cascade

2. **FTX Collapse (2022-11-08 to 2022-11-10)**
   - Expected: oi_change_pct < -20%
   - Timeframe: 2-day panic

3. **Normal Range (any non-event period)**
   - Expected: -5% < oi_change_pct < +5%
   - Timeframe: 90% of data

**Script**: `bin/fix_oi_change_pipeline.py` (Phase 3 validation)

---

## Technical Specifications

### OI Metric Formulas

#### 1. oi_change_24h (Absolute Change)
```python
oi_change_24h = df['oi'].diff(24)
```
- **Unit**: Contracts (or USD notional)
- **Interpretation**: Raw change in open interest
- **Use case**: Magnitude of position changes

#### 2. oi_change_pct_24h (Percentage Change)
```python
oi_change_pct_24h = df['oi'].pct_change(24) * 100
```
- **Unit**: Percentage (%)
- **Interpretation**: Relative change (normalized)
- **Use case**: Cross-timeframe comparison, thresholds

#### 3. oi_z (Z-Score)
```python
rolling_mean = df['oi'].rolling(window=252, min_periods=100).mean()
rolling_std = df['oi'].rolling(window=252, min_periods=100).std()
oi_z = (df['oi'] - rolling_mean) / rolling_std
```
- **Window**: 252 hours (~10.5 days)
- **Unit**: Standard deviations
- **Interpretation**:
  - z > 2.0: High OI (crowded positioning)
  - z < -2.0: Low OI (thin market)
- **Use case**: Regime detection, outlier flagging

### Expected Value Ranges (Empirical)

Based on 2024 data (8,761 samples):

```
oi (raw):
  Mean: 82.4B USD
  Range: 38.7B - 126.1B USD

oi_change_pct_24h (expected):
  Mean: ~0.0% (mean-reverting)
  Std: ~3-5% (normal volatility)
  Extremes: -25% to +20% (events)

oi_z (expected):
  Mean: 0.0 (by definition)
  Std: 1.0 (by definition)
  Extremes: -3.5 to +3.5 (3-sigma events)
```

---

## Files Affected

### Read (Input)
1. `/data/macro/BTC_macro_features.parquet` - OI source for 2024+
2. `/data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet` - Target MTF store
3. OKX API - Historical OI for 2022-2023 (fetched on-demand)

### Write (Output)
1. `/data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet` - Patched MTF store
2. `/data/cache/okx_oi_2022_2023.parquet` - Cached OI fetch (optional)

### Script Locations
- Fix script: `/bin/fix_oi_change_pipeline.py` (to be created)
- Reference logic: `/bin/patch_derivatives_columns.py` (lines 52-66)
- Backfill reference: `/bin/backfill_missing_macro_features.py` (lines 98-210)

---

## Success Criteria

After fix implementation, validate:

### Quantitative Checks
✅ `oi` column: 26,236 / 26,236 non-null (100% coverage)
✅ `oi_change_pct_24h` mean ≠ 0, std > 0 (no longer all-zero)
✅ `oi_z` range: [-4, +4] (reasonable z-score bounds)
✅ Terra collapse (May 2022): oi_change_pct < -15% detected
✅ FTX collapse (Nov 2022): oi_change_pct < -20% detected

### Qualitative Checks
✅ S5 pattern logic can access oi_change_pct without KeyError
✅ Bear archetype backtests run without OI-related failures
✅ Liquidation cascade events (2022) are detectable

---

## Next Steps

1. **Create fix script**: `bin/fix_oi_change_pipeline.py`
2. **Fetch 2022-2023 OI**: Run OKX API backfill
3. **Calculate derivatives**: Apply formulas to full OI column
4. **Validate**: Run Terra/FTX event checks
5. **Export patched MTF**: Replace existing file
6. **Test S5**: Run bear archetype validation

**Estimated Time**:
- Scripting: 2 hours
- OI fetch: 30 minutes (API rate limits)
- Calculation + validation: 1 hour
- **Total: 3.5 hours**

---

## Related Issues

- **liquidity_score missing**: Separate issue (runtime-only feature)
- **Funding rate gaps**: Partially resolved (funding column exists, needs validation)
- **Regime classifier OI dependency**: Will be fixed by this pipeline repair

---

## References

### Code
- OI calculation logic: `bin/patch_derivatives_columns.py:52-66`
- OI backfill example: `bin/backfill_missing_macro_features.py:98-210`
- Bear pattern OI usage: `engine/archetypes/logic.py:1178` (S5 pattern)

### Data
- Macro features: `data/macro/BTC_macro_features.parquet`
- MTF store: `data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet`

### APIs
- OKX OI History: https://www.okx.com/docs-v5/en/#order-book-trading-market-data-get-open-interest-history
- Coinglass: https://coinglass.com/pro/futures/openInterest (requires auth)
