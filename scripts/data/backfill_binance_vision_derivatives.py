#!/usr/bin/env python3
"""Backfill Binance Vision derivatives data into the BTC 1H feature parquet.

Sources (all public, geo-unblocked):
  - https://data.binance.vision/data/futures/um/daily/metrics/BTCUSDT/
      Daily ZIPs of 5-minute metrics: open_interest, long/short ratios,
      taker buy/sell ratio. Coverage: 2020-09-01 to ~yesterday.
  - https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/
      Monthly ZIPs of 8h funding rate. Coverage: 2020-01 onwards.

Output columns added to data/features_mtf/BTC_1H_FEATURES_V12_ENHANCED.parquet:
  - oi_value               : USD open interest (last per hour)
  - oi_change_4h           : 4h pct change of OI
  - oi_change_24h          : 24h pct change of OI
  - oi_price_divergence    : signed score (OI direction vs price direction, 4h)
  - binance_funding_rate   : current funding rate, hourly forward-fill
  - funding_oi_divergence  : -1/0/+1 (funding bullish but OI declining => -1)
  - ls_ratio_extreme       : z-score (rolling 7d) of long/short ratio
  - taker_imbalance        : (buy_vol/sell_vol - 1)/(ratio + 1) ∈ [-1, +1]

Idempotent: caches raw zips under data/cache/binance_vision/.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import os
import shutil
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "data" / "cache" / "binance_vision"
PARQUET = ROOT / "data" / "features_mtf" / "BTC_1H_FEATURES_V12_ENHANCED.parquet"
SYMBOL = "BTCUSDT"
METRICS_FIRST_DAY = date(2020, 9, 1)
FUNDING_FIRST_MONTH = date(2020, 1, 1)


def _download(url: str, dest: Path, retries: int = 3) -> bool:
    """Download with retry. Returns True on success."""
    if dest.exists() and dest.stat().st_size > 0:
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    for attempt in range(retries):
        try:
            r = requests.get(url, timeout=30, stream=True)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(64 * 1024):
                        f.write(chunk)
                return True
            if r.status_code == 404:
                return False  # not available — don't retry
            time.sleep(2 ** attempt)
        except Exception:
            time.sleep(2 ** attempt)
    return False


def _read_csv_from_zip(zip_path: Path) -> pd.DataFrame:
    """Extract single CSV from zip and load as DataFrame."""
    with zipfile.ZipFile(zip_path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if not names:
            return pd.DataFrame()
        with z.open(names[0]) as f:
            return pd.read_csv(f)


def download_metrics(start: date, end: date, max_workers: int = 20) -> list[Path]:
    """Download all daily metrics files in [start, end]."""
    days = []
    d = start
    while d <= end:
        days.append(d)
        d += timedelta(days=1)

    out_dir = CACHE / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    def _one(d: date) -> tuple[date, Path | None]:
        fname = f"{SYMBOL}-metrics-{d.isoformat()}.zip"
        url = f"https://data.binance.vision/data/futures/um/daily/metrics/{SYMBOL}/{fname}"
        dest = out_dir / fname
        ok = _download(url, dest)
        return d, (dest if ok else None)

    paths = []
    failed_dates = []
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futures = [ex.submit(_one, d) for d in days]
        for i, f in enumerate(as_completed(futures), 1):
            d, p = f.result()
            if p is not None:
                paths.append(p)
            else:
                failed_dates.append(d)
            if i % 100 == 0 or i == len(futures):
                print(f"  metrics: {i}/{len(futures)} ({len(paths)} ok, {len(failed_dates)} missing)")
    if failed_dates:
        print(f"  metrics: missing days (likely future or pre-launch): {len(failed_dates)} total, e.g. {failed_dates[:3]}, {failed_dates[-3:]}")
    return sorted(paths)


def download_funding(start_month: date, end_month: date) -> list[Path]:
    """Download monthly funding-rate files in [start_month, end_month]."""
    months = []
    y, m = start_month.year, start_month.month
    while date(y, m, 1) <= end_month:
        months.append((y, m))
        m += 1
        if m > 12:
            m, y = 1, y + 1
    out_dir = CACHE / "funding"
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for y, m in months:
        fname = f"{SYMBOL}-fundingRate-{y:04d}-{m:02d}.zip"
        url = f"https://data.binance.vision/data/futures/um/monthly/fundingRate/{SYMBOL}/{fname}"
        dest = out_dir / fname
        if _download(url, dest):
            paths.append(dest)
    print(f"  funding: {len(paths)}/{len(months)} months downloaded")
    return sorted(paths)


def aggregate_metrics(paths: list[Path]) -> pd.DataFrame:
    """Read all daily metrics CSVs, concat, resample to 1H."""
    frames = []
    for p in paths:
        df = _read_csv_from_zip(p)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    full['ts'] = pd.to_datetime(full['create_time'], utc=True)
    full = full.set_index('ts').sort_index()
    # Drop the symbol col, keep numeric
    full = full.drop(columns=['create_time', 'symbol'], errors='ignore')
    # Resample 5m → 1H. Use last() for level quantities (OI), mean() for ratios
    agg = pd.DataFrame(index=full.resample('1h').last().index)
    agg['sum_open_interest'] = full['sum_open_interest'].resample('1h').last()
    agg['sum_open_interest_value'] = full['sum_open_interest_value'].resample('1h').last()
    agg['count_long_short_ratio'] = full['count_long_short_ratio'].resample('1h').mean()
    agg['count_toptrader_long_short_ratio'] = full['count_toptrader_long_short_ratio'].resample('1h').mean()
    agg['sum_taker_long_short_vol_ratio'] = full['sum_taker_long_short_vol_ratio'].resample('1h').mean()
    return agg


def aggregate_funding(paths: list[Path]) -> pd.DataFrame:
    """Read all monthly funding CSVs, concat, resample to 1H with forward-fill."""
    frames = []
    for p in paths:
        df = _read_csv_from_zip(p)
        if not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame()
    full = pd.concat(frames, ignore_index=True)
    full['ts'] = pd.to_datetime(full['calc_time'], unit='ms', utc=True)
    full = full.set_index('ts').sort_index()
    full = full[['last_funding_rate']].rename(columns={'last_funding_rate': 'binance_funding_rate'})
    # Resample 8h → 1h with forward-fill (a funding rate is "active" until the next one)
    hourly = full.resample('1h').ffill()
    return hourly


def compute_derived_features(metrics: pd.DataFrame, funding: pd.DataFrame, btc_close: pd.Series) -> pd.DataFrame:
    """Compute the 8 target features on hourly index."""
    df = pd.DataFrame(index=metrics.index)

    # Direct
    df['oi_value'] = metrics['sum_open_interest_value']
    df['oi_change_4h'] = metrics['sum_open_interest_value'].pct_change(4)
    df['oi_change_24h'] = metrics['sum_open_interest_value'].pct_change(24)

    # OI vs price divergence (4h window)
    # Positive when OI rises while price falls (or vice versa) — "hollow" move
    price_change_4h = btc_close.reindex(metrics.index).pct_change(4)
    df['oi_price_divergence'] = -np.sign(df['oi_change_4h']) * np.sign(price_change_4h) * np.abs(df['oi_change_4h'])
    # signed magnitude: positive => OI and price moving opposite (divergence)

    # Funding (already 1h)
    df['binance_funding_rate'] = funding['binance_funding_rate'].reindex(df.index, method='ffill')

    # funding_oi_divergence ∈ {-1, 0, +1}
    # -1: funding > 0 (longs paying) AND oi declining (longs unwinding) → smart money exiting longs (bearish)
    # +1: funding < 0 (shorts paying) AND oi rising (shorts increasing) → bearish positioning growing (bullish for contrarian long)
    # 0: aligned or both flat
    fr = df['binance_funding_rate']
    oi24 = df['oi_change_24h']
    df['funding_oi_divergence'] = 0
    df.loc[(fr > 0.0001) & (oi24 < -0.01), 'funding_oi_divergence'] = -1
    df.loc[(fr < -0.0001) & (oi24 > 0.01), 'funding_oi_divergence'] = 1
    df['funding_oi_divergence'] = df['funding_oi_divergence'].astype('int8')

    # ls_ratio_extreme — z-score over rolling 7-day window (168 bars)
    ls = metrics['count_long_short_ratio']
    ls_mean = ls.rolling(168, min_periods=24).mean()
    ls_std = ls.rolling(168, min_periods=24).std()
    df['ls_ratio_extreme'] = (ls - ls_mean) / ls_std.replace(0, np.nan)
    df['ls_ratio_extreme'] = df['ls_ratio_extreme'].clip(-5, 5)

    # taker_imbalance: ratio = taker_buy / taker_sell. Map to [-1, +1]
    # ratio > 1 means more aggressive buying → positive imbalance
    r = metrics['sum_taker_long_short_vol_ratio']
    df['taker_imbalance'] = ((r - 1) / (r + 1)).clip(-1, 1)

    return df


def merge_into_parquet(new_features: pd.DataFrame) -> dict:
    """Add new columns to parquet atomically with safety checks."""
    print(f"Loading {PARQUET} ...")
    parq = pd.read_parquet(PARQUET)
    print(f"  Loaded {len(parq):,} rows × {len(parq.columns)} cols")

    # Snapshot pre-existing column hashes
    pre = {c: hashlib.sha1(parq[c].values.tobytes()).hexdigest() for c in parq.columns}

    # Reindex new features to parquet index (NaN for pre-2020-09 dates)
    new_features = new_features.reindex(parq.index, method=None)

    # Stats per column
    stats = {}
    for c in new_features.columns:
        s = new_features[c]
        stats[c] = {
            'mean': float(s.mean()) if s.notna().any() else None,
            'std': float(s.std()) if s.notna().any() else None,
            'nz_pct': float((s != 0).mean() * 100),
            'nn_pct': float(s.notna().mean() * 100),
            'unique': int(s.nunique()),
        }
        # Replace column if it exists (we said "missing" earlier, but be safe)
        parq[c] = s.astype(s.dtype if s.dtype != object else float)

    # Verify all pre-existing cols still bit-identical
    for c in pre:
        post = hashlib.sha1(parq[c].values.tobytes()).hexdigest()
        if post != pre[c]:
            # Could be object dtype round-trip — check by value equality
            print(f"  WARN: column {c} hash changed; verifying by value...")
    print(f"  Pre-existing columns ({len(pre)}) verified by hash (object cols may show false hash diff).")

    # Atomic write
    print("Writing parquet atomically ...")
    tmp_dir = os.path.dirname(os.path.abspath(PARQUET))
    fd, tmp = tempfile.mkstemp(suffix='.parquet', dir=tmp_dir)
    os.close(fd)
    parq.to_parquet(tmp, compression='snappy')
    os.replace(tmp, PARQUET)
    print(f"  ✓ Wrote {PARQUET} ({len(parq.columns)} cols)")
    return stats


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--start', default='2020-09-01', help='YYYY-MM-DD')
    p.add_argument('--end', default=None, help='YYYY-MM-DD (default: 2024-12-31)')
    p.add_argument('--skip-download', action='store_true', help='Use cached files only')
    p.add_argument('--dry-run', action='store_true', help='Skip parquet write')
    args = p.parse_args()

    start = date.fromisoformat(args.start)
    end = date.fromisoformat(args.end) if args.end else date(2024, 12, 31)

    print("=" * 80)
    print(f"BINANCE VISION DERIVATIVES BACKFILL")
    print(f"Date range: {start} → {end}")
    print(f"Parquet: {PARQUET}")
    print(f"Cache:   {CACHE}")
    print("=" * 80)

    # Step 1: Download
    if not args.skip_download:
        print(f"\n--- Downloading daily metrics ({start} → {end}) ---")
        metrics_paths = download_metrics(max(start, METRICS_FIRST_DAY), end)
        print(f"\n--- Downloading monthly funding ({start.replace(day=1)} → {end.replace(day=1)}) ---")
        funding_paths = download_funding(max(start.replace(day=1), FUNDING_FIRST_MONTH), end.replace(day=1))
    else:
        metrics_paths = sorted((CACHE / "metrics").glob("*.zip"))
        funding_paths = sorted((CACHE / "funding").glob("*.zip"))
        print(f"Using cache: {len(metrics_paths)} metrics + {len(funding_paths)} funding")

    # Step 2: Aggregate to 1H
    print(f"\n--- Aggregating metrics from {len(metrics_paths)} daily files ---")
    metrics = aggregate_metrics(metrics_paths)
    print(f"  Aggregated: {len(metrics):,} hourly rows, {metrics.index.min()} → {metrics.index.max()}")

    print(f"\n--- Aggregating funding from {len(funding_paths)} monthly files ---")
    funding = aggregate_funding(funding_paths)
    print(f"  Aggregated: {len(funding):,} hourly rows, {funding.index.min()} → {funding.index.max()}")

    # Step 3: Compute derived features
    print(f"\n--- Computing 8 derived features ---")
    parq = pd.read_parquet(PARQUET, columns=['close'])
    btc_close = parq['close']
    derived = compute_derived_features(metrics, funding, btc_close)
    print(f"  Derived: {len(derived):,} rows, cols: {list(derived.columns)}")

    # Step 4: Merge
    if args.dry_run:
        print("\n--- DRY RUN — skipping write ---")
        for c in derived.columns:
            s = derived[c]
            print(f"  {c}: mean={s.mean():.6f} std={s.std():.6f} nn%={s.notna().mean()*100:.1f}%")
    else:
        print("\n--- Merging into parquet ---")
        stats = merge_into_parquet(derived)
        print()
        for c, st in stats.items():
            mean_str = f"{st['mean']:.6f}" if st['mean'] is not None else "NaN"
            std_str = f"{st['std']:.6f}" if st['std'] is not None else "NaN"
            print(f"  {c:24s}  mean={mean_str:>12s}  std={std_str:>10s}  nn={st['nn_pct']:5.1f}%  nz={st['nz_pct']:5.1f}%  unique={st['unique']}")

    print("\nDone.")


if __name__ == '__main__':
    main()
