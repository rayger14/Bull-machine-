# Macro Features Pipeline Specification

**Status**: Phase 1 - Critical Data Blocker
**Priority**: MEDIUM (Blocks regime routing, S2 bear archetype)
**Data Availability**: ✅ CSV files exist in `data/` directory
**Recommendation**: Resample existing daily data to 1H with forward fill

---

## 1. Feature Definition

### Required Macro Features

| Feature | Source | Current File | Tier | Description |
|---------|--------|--------------|------|-------------|
| `dxy_z` | DXY (US Dollar Index) | `data/DXY_1H.csv` | 3 | Dollar strength z-score |
| `vix_z` | VIX (Equity Volatility) | `data/VIX_1H.csv` | 3 | Equity fear z-score |
| `move_z` | MOVE (Bond Volatility) | `data/MOVE_1H.csv` | 3 | Bond fear z-score |
| `rates_2y_z` | US 2-Year Yield | `data/US2Y_1H.csv` | 3 | Short-term rates z-score |
| `rates_10y_z` | US 10-Year Yield | `data/US10Y_1H.csv` | 3 | Long-term rates z-score |

### Additional Macro Features (Lower Priority)

| Feature | Source | Current File | Tier | Description |
|---------|--------|--------------|------|-------------|
| `gold_z` | Gold Spot Price | `data/GOLD_1H.csv` | 3 | Safe haven z-score |
| `wti_z` | WTI Crude Oil | `data/WTI_1H.csv` | 3 | Energy market z-score |
| `eurusd_z` | EUR/USD FX Rate | `data/EURUSD_1H.csv` | 3 | Euro strength z-score |
| `spy_z` | SPY (S&P 500 ETF) | `data/SPY_1H.csv` | 3 | Equity market z-score |

### Derived Macro Features

| Feature | Formula | Description |
|---------|---------|-------------|
| `yield_curve_slope` | `rates_10y - rates_2y` | Yield curve steepness (inversion detection) |
| `risk_appetite` | `(spy_z - vix_z) / 2` | Risk-on/risk-off composite |
| `macro_stress` | `(vix_z + move_z) / 2` | Cross-asset stress indicator |

---

## 2. Data Sources & Coverage

### Existing Data Files

All macro data exists as **1H CSVs** in `data/` directory:

```bash
data/
├── DXY_1H.csv      # US Dollar Index (hourly)
├── VIX_1H.csv      # Equity Volatility Index (hourly)
├── MOVE_1H.csv     # Bond Volatility Index (hourly)
├── US2Y_1H.csv     # 2-Year Treasury Yield (hourly)
├── US10Y_1H.csv    # 10-Year Treasury Yield (hourly)
├── GOLD_1H.csv     # Gold Spot Price (hourly)
├── WTI_1H.csv      # WTI Crude Oil (hourly)
├── EURUSD_1H.csv   # EUR/USD FX Rate (hourly)
└── SPY_1H.csv      # S&P 500 ETF (hourly)
```

### Data Format
All CSV files have the same structure:
```csv
time,open,high,low,close,Upper,Lower,Crossunder,Crossover,...
1752566400,97.956,97.995,97.937,97.961,...
```

### Historical Coverage
- **Date Range**: Varies by asset (TradingView data)
- **Critical Period**: 2022-2024 (must cover bear market)
- **Granularity**: 1H bars (already matches MTF store)

---

## 3. Z-Score Calculation

### Formula
```python
z_score = (value - rolling_mean) / rolling_std
```

### Window Selection

| Window Type | Bars | Period | Use Case |
|-------------|------|--------|----------|
| **Short-term** | 90 | 90H (~3.75 days) | Intraday regime shifts |
| **Medium-term** | 252 | 252H (~10.5 days) | **RECOMMENDED** (default) |
| **Long-term** | 720 | 720H (~30 days) | Macro trend detection |
| **Full-sample** | All | Entire history | Reference benchmark |

**Recommendation**: Use **252H rolling window** (10.5 days) for consistency with OI z-score.

### Implementation
```python
def calculate_macro_zscore(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Calculate z-scores for macro features.

    Args:
        df: DataFrame with 'close' column (macro asset price)
        window: Rolling window for z-score (default 252H)

    Returns:
        DataFrame with additional z-score column
    """
    # Calculate rolling statistics
    rolling_mean = df['close'].rolling(window=window, min_periods=100).mean()
    rolling_std = df['close'].rolling(window=window, min_periods=100).std()

    # Compute z-score
    df['z'] = (df['close'] - rolling_mean) / rolling_std

    # Fill initial NaNs with 0.0 (neutral)
    df['z'] = df['z'].fillna(0.0)

    return df
```

### NaN Handling
- **First 100 bars**: Fill with `0.0` (neutral baseline)
- **Bars 100-252**: Partial rolling (computed with `min_periods=100`)
- **After bar 252**: Full rolling window (no NaNs)

---

## 4. Resampling Strategy

### Current Situation
- **Existing Files**: Already 1H granularity
- **Action Required**: **No resampling needed** (data already matches MTF store)

### If Daily Data Were Available (for reference)
```python
# NOT NEEDED - data is already 1H
def resample_daily_to_1h(df_daily: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily macro data to 1H (forward fill).

    Args:
        df_daily: DataFrame with daily bars

    Returns:
        DataFrame with 1H bars (forward filled)
    """
    # Create hourly index
    hourly_index = pd.date_range(
        start=df_daily.index[0],
        end=df_daily.index[-1],
        freq='1H'
    )

    # Reindex and forward fill
    df_1h = df_daily.reindex(hourly_index, method='ffill')

    return df_1h
```

---

## 5. Feature Registry Entry

Add to `engine/features/registry.py`:

```python
# DXY Z-Score
FeatureSpec(
    canonical="dxy_z",
    dtype="float64",
    tier=3,
    required=False,
    aliases=["dollar_strength_z"],
    range_min=None,
    range_max=None,
    description="US Dollar Index z-score (252H rolling)"
)

# VIX Z-Score
FeatureSpec(
    canonical="vix_z",
    dtype="float64",
    tier=3,
    required=False,
    aliases=["equity_vol_z"],
    range_min=None,
    range_max=None,
    description="VIX equity volatility z-score (252H rolling)"
)

# MOVE Z-Score
FeatureSpec(
    canonical="move_z",
    dtype="float64",
    tier=3,
    required=False,
    aliases=["bond_vol_z"],
    range_min=None,
    range_max=None,
    description="MOVE bond volatility z-score (252H rolling)"
)

# 2Y Yield Z-Score
FeatureSpec(
    canonical="rates_2y_z",
    dtype="float64",
    tier=3,
    required=False,
    aliases=["us2y_z", "short_rates_z"],
    range_min=None,
    range_max=None,
    description="US 2-Year Treasury yield z-score (252H rolling)"
)

# 10Y Yield Z-Score
FeatureSpec(
    canonical="rates_10y_z",
    dtype="float64",
    tier=3,
    required=False,
    aliases=["us10y_z", "long_rates_z"],
    range_min=None,
    range_max=None,
    description="US 10-Year Treasury yield z-score (252H rolling)"
)

# Yield Curve Slope (Derived)
FeatureSpec(
    canonical="yield_curve_slope",
    dtype="float64",
    tier=3,
    required=False,
    aliases=["curve_slope", "yield_spread"],
    range_min=None,
    range_max=None,
    description="10Y - 2Y yield spread (inversion = negative)"
)

# Risk Appetite (Derived)
FeatureSpec(
    canonical="risk_appetite",
    dtype="float64",
    tier=3,
    required=False,
    aliases=["risk_on_off"],
    range_min=None,
    range_max=None,
    description="(SPY_z - VIX_z) / 2 composite risk metric"
)

# Macro Stress (Derived)
FeatureSpec(
    canonical="macro_stress",
    dtype="float64",
    tier=3,
    required=False,
    aliases=["cross_asset_stress"],
    range_min=None,
    range_max=None,
    description="(VIX_z + MOVE_z) / 2 cross-asset stress indicator"
)
```

---

## 6. Backfill Script Design

### Script: `bin/backfill_macro_features.py`

**Status**: Needs to be created (or adapt `bin/backfill_missing_macro_features.py`)

#### Architecture
1. **Load Macro CSVs**: Read all 1H macro data files
2. **Calculate Z-Scores**: Apply rolling z-score (252H window)
3. **Derive Features**: Compute yield curve slope, risk appetite, macro stress
4. **Align Timestamps**: Merge with MTF store by timestamp (outer join)
5. **Fill Gaps**: Forward fill (macro data persists across hours)
6. **Validate Coverage**: Check for NaNs, verify 2022-2024 coverage
7. **Patch MTF Store**: Add macro columns and write back

#### CLI Usage
```bash
# Full backfill (all macro features)
python3 bin/backfill_macro_features.py

# Dry run (validation only)
python3 bin/backfill_macro_features.py --dry-run

# Custom MTF store
python3 bin/backfill_macro_features.py \
    --mtf-store data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet

# Custom z-score window
python3 bin/backfill_macro_features.py --window 720  # 30-day window
```

#### Pseudocode
```python
def main():
    # 1. Load macro CSV files
    macro_files = {
        'dxy': 'data/DXY_1H.csv',
        'vix': 'data/VIX_1H.csv',
        'move': 'data/MOVE_1H.csv',
        'us2y': 'data/US2Y_1H.csv',
        'us10y': 'data/US10Y_1H.csv',
        'gold': 'data/GOLD_1H.csv',
        'wti': 'data/WTI_1H.csv',
        'eurusd': 'data/EURUSD_1H.csv',
        'spy': 'data/SPY_1H.csv',
    }

    macro_dfs = {}
    for name, path in macro_files.items():
        df = pd.read_csv(path)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
        df.set_index('time', inplace=True)
        macro_dfs[name] = df[['close']]

    # 2. Calculate z-scores
    for name, df in macro_dfs.items():
        df[f'{name}_z'] = calculate_macro_zscore(df, window=args.window)

    # 3. Derive features
    derived = pd.DataFrame(index=macro_dfs['us2y'].index)

    # Yield curve slope
    derived['yield_curve_slope'] = (
        macro_dfs['us10y']['close'] - macro_dfs['us2y']['close']
    )

    # Risk appetite
    derived['risk_appetite'] = (
        macro_dfs['spy']['spy_z'] - macro_dfs['vix']['vix_z']
    ) / 2.0

    # Macro stress
    derived['macro_stress'] = (
        macro_dfs['vix']['vix_z'] + macro_dfs['move']['move_z']
    ) / 2.0

    # 4. Merge all macro features
    macro_merged = pd.concat([
        macro_dfs['dxy'][['dxy_z']],
        macro_dfs['vix'][['vix_z']],
        macro_dfs['move'][['move_z']],
        macro_dfs['us2y'][['rates_2y_z']],
        macro_dfs['us10y'][['rates_10y_z']],
        derived
    ], axis=1)

    # 5. Load MTF store
    mtf_df = pd.read_parquet(args.mtf_store)

    # 6. Merge macro features (left join to preserve MTF rows)
    mtf_df = mtf_df.join(macro_merged, how='left')

    # 7. Forward fill gaps (macro data persists)
    macro_cols = ['dxy_z', 'vix_z', 'move_z', 'rates_2y_z', 'rates_10y_z',
                  'yield_curve_slope', 'risk_appetite', 'macro_stress']
    for col in macro_cols:
        mtf_df[col] = mtf_df[col].ffill().fillna(0.0)

    # 8. Validate coverage
    validate_macro_coverage(mtf_df, macro_cols)

    # 9. Write patched store
    if not args.dry_run:
        mtf_df.to_parquet(args.mtf_store)
        print(f"✅ Patched {len(mtf_df)} rows with {len(macro_cols)} macro features")

    return 0
```

---

## 7. Validation Strategy

### Coverage Validation

```python
def validate_macro_coverage(df: pd.DataFrame, macro_cols: list) -> Dict[str, Any]:
    """Validate macro feature coverage in MTF store"""
    results = {}

    # Check overall coverage
    for col in macro_cols:
        non_null = df[col].notna().sum()
        coverage_pct = (non_null / len(df)) * 100
        results[f'{col}_coverage'] = coverage_pct

        if coverage_pct < 95.0:
            print(f"⚠️ Low coverage for {col}: {coverage_pct:.1f}%")
        else:
            print(f"✅ {col}: {coverage_pct:.1f}% coverage")

    # Check critical periods (2022 bear market)
    bear_period = df[(df.index >= '2022-01-01') & (df.index <= '2022-12-31')]
    for col in macro_cols:
        bear_coverage = (bear_period[col].notna().sum() / len(bear_period)) * 100
        results[f'{col}_2022_coverage'] = bear_coverage

        if bear_coverage < 90.0:
            print(f"⚠️ Low 2022 coverage for {col}: {bear_coverage:.1f}%")

    return results
```

### Distribution Validation

```python
def validate_macro_distributions(df: pd.DataFrame) -> Dict[str, Any]:
    """Check z-score distributions (should be ~N(0, 1))"""
    results = {}

    z_cols = ['dxy_z', 'vix_z', 'move_z', 'rates_2y_z', 'rates_10y_z']

    for col in z_cols:
        mean = df[col].mean()
        std = df[col].std()

        # Z-scores should have mean ~0, std ~1
        mean_ok = abs(mean) < 0.5  # Relaxed (allows drift)
        std_ok = 0.7 < std < 1.3   # Relaxed (allows variation)

        results[f'{col}_mean'] = mean
        results[f'{col}_std'] = std
        results[f'{col}_valid'] = mean_ok and std_ok

        status = "✅" if (mean_ok and std_ok) else "⚠️"
        print(f"{status} {col}: mean={mean:.3f}, std={std:.3f}")

    return results
```

### Known Event Validation

```python
def validate_macro_events(df: pd.DataFrame) -> Dict[str, bool]:
    """Check macro spikes during known crisis events"""
    results = {}

    # Terra collapse (May 9-12, 2022) - expect VIX spike
    terra_period = df[(df.index >= '2022-05-09') & (df.index <= '2022-05-12')]
    vix_spike_terra = (terra_period['vix_z'] > 1.5).any()
    results['vix_spike_terra'] = vix_spike_terra
    print(f"{'✅' if vix_spike_terra else '⚠️'} VIX spike during Terra collapse: {vix_spike_terra}")

    # FTX collapse (Nov 8-10, 2022) - expect VIX + MOVE spike
    ftx_period = df[(df.index >= '2022-11-08') & (df.index <= '2022-11-10')]
    stress_spike_ftx = (ftx_period['macro_stress'] > 1.5).any()
    results['stress_spike_ftx'] = stress_spike_ftx
    print(f"{'✅' if stress_spike_ftx else '⚠️'} Macro stress spike during FTX collapse: {stress_spike_ftx}")

    # SVB collapse (Mar 10-13, 2023) - expect bond vol (MOVE) spike
    svb_period = df[(df.index >= '2023-03-10') & (df.index <= '2023-03-13')]
    move_spike_svb = (svb_period['move_z'] > 2.0).any()
    results['move_spike_svb'] = move_spike_svb
    print(f"{'✅' if move_spike_svb else '⚠️'} MOVE spike during SVB collapse: {move_spike_svb}")

    return results
```

---

## 8. Execution Plan

### Pre-Flight Checklist
- [ ] Verify all macro CSV files exist in `data/` directory
- [ ] Check CSV format (time, close columns present)
- [ ] Backup MTF store before patching
- [ ] Verify 2022-2024 coverage in CSV files

### Backfill Steps
```bash
# 1. Validate CSV files
ls -lh data/DXY_1H.csv data/VIX_1H.csv data/MOVE_1H.csv \
         data/US2Y_1H.csv data/US10Y_1H.csv

# 2. Check CSV headers
head -2 data/DXY_1H.csv
head -2 data/VIX_1H.csv

# 3. Dry run (validation only)
python3 bin/backfill_macro_features.py --dry-run

# 4. Review validation output
#    - Check coverage percentages (expect >95%)
#    - Verify z-score distributions (mean ~0, std ~1)
#    - Confirm crisis event spikes (Terra, FTX, SVB)

# 5. Full backfill (write to store)
python3 bin/backfill_macro_features.py

# 6. Validate patched store
python3 bin/validate_feature_store.py \
    --mtf-store data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \
    --check dxy_z,vix_z,move_z,rates_2y_z,rates_10y_z
```

### Performance Expectations
- **CSV Load Time**: ~1 minute (9 files)
- **Z-Score Calculation**: ~30 seconds (vectorized)
- **Merge Time**: ~30 seconds (timestamp alignment)
- **Total Time**: ~3-5 minutes
- **Memory**: ~200 MB (all CSVs + MTF store in RAM)

---

## 9. Success Criteria

### Functional Requirements
- ✅ All 5 core macro z-scores added: `dxy_z`, `vix_z`, `move_z`, `rates_2y_z`, `rates_10y_z`
- ✅ All 3 derived features added: `yield_curve_slope`, `risk_appetite`, `macro_stress`
- ✅ Coverage >95% for all features (forward fill applied)

### Quality Requirements
- ✅ Z-score distributions: mean ~0 (±0.5), std ~1 (±0.3)
- ✅ Crisis events detected:
  - VIX spike during Terra collapse (May 2022)
  - Macro stress spike during FTX collapse (Nov 2022)
  - MOVE spike during SVB collapse (Mar 2023)

### Integration Requirements
- ✅ Regime routing uses macro features (no NaN errors)
- ✅ S2 (Failed Rally) archetype uses `dxy_z` (no graceful degradation)

---

## 10. Known Issues & Mitigations

### Issue 1: CSV Timestamp Format Varies
**Mitigation**: Auto-detect timestamp format (Unix seconds vs ISO 8601):
```python
# Try Unix timestamp first
try:
    df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)
except:
    # Fallback to ISO format
    df['time'] = pd.to_datetime(df['time'], utc=True)
```

### Issue 2: Missing Macro Data During Weekends
**Mitigation**: Forward fill (macro values persist across trading gaps)

### Issue 3: Z-Score Drift Over Long Periods
**Mitigation**: Use rolling window (252H) instead of full-sample (avoids regime shift bias)

---

## 11. References

- **Existing Script**: `bin/backfill_missing_macro_features.py` (partial implementation)
- **Data Files**: `data/DXY_1H.csv`, `data/VIX_1H.csv`, etc.
- **Feature Registry**: `engine/features/registry.py`
- **Regime Classifier**: `engine/context/regime_classifier.py` (uses macro features)
- **S2 Archetype**: `engine/strategies/archetypes/bear/failed_rally_runtime.py` (uses `dxy_z`)
