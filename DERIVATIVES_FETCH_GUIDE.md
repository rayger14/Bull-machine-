# Derivatives Data Fetch Guide

## Overview

This guide explains how to fetch and add derivatives data (funding, OI, long/short, liquidations) to the BTC 2022-2023 feature store.

## Current Status ✅

**Feature Store**: `data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet`
- **Shape**: 17,475 bars × 88 features
- **Derivatives added**: `funding_Z` (partial coverage: 44.6%)
- **Missing**: OI, long/short ratio, liquidations for 2022-2023

## Step 1: Get Your CoinGlass API Key

1. Sign up at [CoinGlass](https://www.coinglass.com/)
2. Get your API key from the dashboard
3. Set it as an environment variable:

```bash
export COINGLASS_API_KEY="your_api_key_here"
```

## Step 2: Fetch All Derivatives Data

Run the comprehensive fetcher script:

```bash
python3 bin/fetch_derivatives_2022_2023.py
```

This will fetch:
- **Funding rates** (8-hourly, resampled to 1H)
- **Open Interest** (hourly, aggregated across exchanges)
- **Long/Short ratio** (hourly, Binance)
- **Liquidations** (hourly, long/short/total USD)

**Expected time**: ~15-30 minutes (rate-limited to 25 req/min)

**Output files** (saved to `data/derivatives/`):
```
BTC_funding_2022_2023.csv
BTC_oi_2022_2023.csv
BTC_ls_ratio_2022_2023.csv
BTC_liquidations_2022_2023.csv
```

## Step 3: Patch Feature Store

Add all derivatives to the feature store:

```bash
python3 bin/patch_derivatives_full.py \
  --input data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet \
  --funding data/derivatives/BTC_funding_2022_2023.csv \
  --oi data/derivatives/BTC_oi_2022_2023.csv \
  --ls-ratio data/derivatives/BTC_ls_ratio_2022_2023.csv \
  --liquidations data/derivatives/BTC_liquidations_2022_2023.csv
```

This will add **~16 new columns**:

### Funding (2 columns)
- `funding_rate`: 8-hour funding rate (%)
- `funding_Z`: 252-hour rolling z-score

### Open Interest (4 columns)
- `oi`: Total OI in USD
- `oi_change_24h`: 24-hour OI change (USD)
- `oi_change_pct_24h`: 24-hour OI change (%)
- `oi_Z`: 252-hour rolling z-score

### Long/Short Ratio (4 columns)
- `ls_ratio`: Long/short account ratio
- `long_pct`: % accounts long
- `short_pct`: % accounts short
- `ls_ratio_Z`: 252-hour rolling z-score

### Liquidations (8 columns)
- `liq_long_usd`: Hourly long liquidations (USD)
- `liq_short_usd`: Hourly short liquidations (USD)
- `liq_total_usd`: Total hourly liquidations (USD)
- `liq_long_24h`: 24-hour rolling sum (long)
- `liq_short_24h`: 24-hour rolling sum (short)
- `liq_total_24h`: 24-hour rolling sum (total)
- `liq_long_Z`: 168-hour rolling z-score (long)
- `liq_short_Z`: 168-hour rolling z-score (short)

## Final Feature Store

After patching, you'll have:
- **Shape**: 17,475 bars × **104 features** (was 88)
- **Complete derivatives coverage** for 2022-2023 (subject to CoinGlass data availability)

## Rate Limits (Hobbyist Plan)

- **Limit**: 30 requests/minute
- **Script uses**: 25 requests/minute (safe margin)
- **Exponential backoff**: Automatic retry on rate limit errors
- **Expected requests**: ~50-100 total (depending on data gaps)

## Data Availability Notes

CoinGlass Hobbyist plan may have historical data limits:
- **Funding**: Usually available back to 2023-02-09 (confirmed)
- **OI**: Usually available back to 2022 (varies by exchange)
- **Long/Short**: Usually available back to 2022 (varies by exchange)
- **Liquidations**: Usually available back to 2022 (varies by exchange)

If you hit data gaps for 2022, you may need to:
1. Upgrade to a higher CoinGlass plan
2. Export data manually from TradingView
3. Accept NaN/zero values for missing periods

## Troubleshooting

### "Too Many Requests" error
The script has exponential backoff built in. Just wait - it will retry automatically.

### "No data available" for 2022
This is expected for some metrics on Hobbyist plan. The script will continue and fetch what's available.

### Feature store validation fails
Run the schema validator to check coverage:
```bash
python3 bin/validate_feature_store_v10.py \
  --input data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet
```

## Next Steps

After patching derivatives:
1. Validate feature store with `bin/validate_feature_store_v10.py`
2. Run Router v10 backtests with regime detection
3. Compare results with/without derivatives context

## Quick Start (Full Command)

```bash
# Set API key
export COINGLASS_API_KEY="your_key_here"

# Fetch all data (~20 mins)
python3 bin/fetch_derivatives_2022_2023.py

# Patch feature store
python3 bin/patch_derivatives_full.py \
  --input data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet \
  --funding data/derivatives/BTC_funding_2022_2023.csv \
  --oi data/derivatives/BTC_oi_2022_2023.csv \
  --ls-ratio data/derivatives/BTC_ls_ratio_2022_2023.csv \
  --liquidations data/derivatives/BTC_liquidations_2022_2023.csv

# Validate
python3 bin/validate_feature_store_v10.py \
  --input data/features_mtf/BTC_1H_2022-01-01_to_2023-12-31.parquet
```

## Summary

✅ **Engine purity maintained**: No changes to detection logic
✅ **Builder-only patches**: All data merged at feature store level
✅ **Backward compatible**: Existing code works without derivatives
✅ **Optional features**: Router can use derivatives when available
✅ **No rebuild required**: ~5 min patch vs ~5 hour rebuild
