# Derivatives Data Backfill Method (May 2026)

## Why this exists

Historical OI / funding / long-short ratio / taker imbalance data is **not available** from any exchange's standard public REST API beyond ~30 days. We learned this the hard way after scoping a "2-4 hour Binance backfill" that turned out to be impossible via the obvious route. This document captures the workaround so the same dead-end is avoided in the future.

## The actual access problem

### What does NOT work

| Source | Why it fails |
|--------|--------------|
| Binance REST API (`fapi.binance.com/futures/data/*`) | **HTTP 451 — geo-blocked** from both this Mac AND the Oracle Cloud production server (US-based) |
| OKX REST API (`okx.com/api/v5/rubik/stat/*`) | Returns "Illegal time range" for queries beyond ~30 days. Only the last 30 days of OI/LS/taker data is exposed. Funding rate has wider history. |
| Bybit REST API (`api.bybit.com/v5/market/open-interest`) | Returned HTML (likely geo-blocked from this location) |
| Paid subscriptions tried previously | Did not have per-exchange historical OI; mostly price-feed aggregators |

**Net result of 2-3 hours of probing**: there is no free historical OI data accessible from public REST APIs for our location. This is a structural problem, not a "we didn't subscribe to the right thing" problem.

### What DOES work — `data.binance.vision`

Binance maintains a **public CDN/S3 archive** at `https://data.binance.vision/` that is **separate from the geo-blocked API**. It is accessible from both this Mac and the Oracle Cloud server. It hosts daily and monthly CSV archives of historical market data going back years.

This is the canonical fallback for any future "we need historical Binance data" need.

## What data was backfilled (May 2026)

| Column | Source path | Granularity | Coverage |
|--------|-------------|-------------|----------|
| `oi_value` | `daily/metrics/BTCUSDT/BTCUSDT-metrics-YYYY-MM-DD.zip` (`sum_open_interest_value`) | 5m → resampled to 1h | 2020-09-01 → 2024-12-31 |
| `oi_change_4h`, `oi_change_24h` | derived from `oi_value` | 1h | 2020-09-01 → 2024-12-31 |
| `oi_price_divergence` | derived (OI vs price 4h) | 1h | 2020-09-01 → 2024-12-31 |
| `binance_funding_rate` | `monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-YYYY-MM.zip` (`last_funding_rate`) | 8h → ffill 1h | 2020-01-01 → 2024-12-31 |
| `funding_oi_divergence` | derived (-1/0/+1) | 1h | 2020-09-01 → 2024-12-31 |
| `ls_ratio_extreme` | derived z-score of `count_long_short_ratio` | 1h | 2020-09-01 → 2024-12-31 |
| `taker_imbalance` | derived from `sum_taker_long_short_vol_ratio` | 1h | 2020-09-01 → 2024-12-31 |

Total: **8 new columns**, 62% non-null coverage (NaN for pre-2020-09 dates).

Parquet went from 287 → 295 cols. Backup preserved at `data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet.bak_pre_oi_backfill`.

## How to access Binance Vision

### URL patterns

```
https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/BTCUSDT-metrics-YYYY-MM-DD.zip
https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/BTCUSDT-fundingRate-YYYY-MM.zip
https://data.binance.vision/data/futures/um/daily/klines/BTCUSDT/1h/BTCUSDT-1h-YYYY-MM-DD.zip
https://data.binance.vision/data/futures/um/monthly/klines/BTCUSDT/1h/BTCUSDT-1h-YYYY-MM.zip
https://data.binance.vision/data/futures/um/daily/aggTrades/BTCUSDT/BTCUSDT-aggTrades-YYYY-MM-DD.zip
https://data.binance.vision/data/futures/um/daily/liquidationSnapshot/BTCUSDT/BTCUSDT-liquidationSnapshot-YYYY-MM-DD.zip
```

Replace `BTCUSDT` with any other symbol (ETHUSDT, SOLUSDT, etc.). Replace `um` (USDT-margined) with `cm` (coin-margined) for inverse perps.

### Schema of the metrics CSV

```
create_time,symbol,sum_open_interest,sum_open_interest_value,
count_toptrader_long_short_ratio,sum_toptrader_long_short_ratio,
count_long_short_ratio,sum_taker_long_short_vol_ratio
```

5-minute timestamps. `sum_open_interest_value` is in USD. `sum_taker_long_short_vol_ratio` is buy_vol / sell_vol (so a value > 1 means more aggressive buying).

### Schema of the funding rate CSV

```
calc_time,funding_interval_hours,last_funding_rate
```

`calc_time` is milliseconds epoch. `funding_interval_hours` is typically 8. `last_funding_rate` is the realized rate at that funding time.

## Running the backfill

Script: `scripts/data/backfill_binance_vision_derivatives.py`

```bash
# Full backfill (idempotent — caches zips locally)
python3 scripts/data/backfill_binance_vision_derivatives.py

# Use cached files only (skip downloads)
python3 scripts/data/backfill_binance_vision_derivatives.py --skip-download

# Dry run (compute features but don't write parquet)
python3 scripts/data/backfill_binance_vision_derivatives.py --dry-run

# Date range override
python3 scripts/data/backfill_binance_vision_derivatives.py --start 2023-01-01 --end 2024-12-31
```

**Cache location**: `data/cache/binance_vision/` (gitignored). ~20MB after a full pull.

**Rerun cost**: ~3 min from scratch, <30s from cache.

## Data validation

Sanity checks against known stress events (run by the backfill script automatically):

| Event | Expected | Observed |
|-------|----------|----------|
| May 2021 crash | Large OI decline (liquidations) | OI −49.5% in 24h ✓ |
| LUNA May 2022 | Forced unwinding | OI −20.9% ✓ |
| FTX collapse Nov 2022 | Long capitulation | OI −22.7%, LS_z +3.9 ✓ |
| Aug 2024 yen carry | Sharp drop | OI −29.5% in 24h ✓ |
| 2020 → 2024 OI growth | Institutional adoption | $0.6B → $5.4B ✓ |

If any future rerun shows these checks failing, the data pipeline has degraded.

## Adding new symbols or new data types

Same script structure works for ETHUSDT, SOLUSDT, etc. — change `SYMBOL` in `scripts/data/backfill_binance_vision_derivatives.py`. Other Vision data types (klines with `taker_buy_volume`, aggTrades, liquidationSnapshot) follow the same URL pattern and can be added by extending `download_*` functions.

## Limitations of this dataset

- **Pre-2020-09 is NaN** — BTC perpetual futures launched on Binance in 2019, and the metrics archive only goes back to Sep 2020. Cannot backfill earlier without a different source.
- **Binance-specific** — the live production engine uses OKX for OI (per heartbeat logs). The Binance historical data may differ in absolute level from OKX. **For relative changes (oi_change_4h, oi_change_24h, z-scores), this exchange mismatch is small enough to ignore.** For absolute-level features (oi_value), be aware they're Binance-only.
- **5-minute → 1-hour resampling** uses `.last()` for level quantities (OI, OI value) and `.mean()` for ratios. This loses some intra-hour detail but matches the parquet's 1-hour structure.
- **No spot-vs-futures basis data** — only futures.

## Pre-existing data conflicts

The parquet has `funding_rate` (71% coverage) and `funding_Z` (71% coverage) computed from some other historical funding source (probably the v12 feature build). The new `binance_funding_rate` column is the Binance-specific raw value. **They should agree closely** (Binance is the dominant perp market) but are kept as separate columns to avoid breaking anything that depends on the existing `funding_rate`.

The parquet also has `oi` (constant 1.0, basically a placeholder). The new `oi_value` is the real USD OI level.

## Future improvements

- **Live-engine accumulation** — wire the live runner to also save OI/funding/taker snapshots to `data/whale_history.parquet` daily. Eventually we can verify the Binance Vision historical data matches what OKX live records, and extend coverage forward continuously.
- **Multi-exchange aggregation** — if any one exchange's data has gaps or is suspect, average across Binance + OKX + Bybit Vision archives (Bybit also has a public archive at `https://public.bybit.com/`, untested here).
- **Liquidations data** — `daily/liquidationSnapshot/` has historical liquidation events. Untapped for now but could enable a "liquidation cascade" archetype.

## File index

- Script: `scripts/data/backfill_binance_vision_derivatives.py`
- Output parquet: `data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet`
- Cache: `data/cache/binance_vision/` (gitignored)
- Backup: `data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet.bak_pre_oi_backfill`
- Commit: see `quant/three-fix-followups` branch (commit `71c7936`)
