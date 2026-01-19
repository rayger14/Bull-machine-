#!/usr/bin/env python3
"""
Wyckoff Cache Builder - Precompute Wyckoff phases for reuse

Runs the full Wyckoff engine ONCE on 1D and 4H data, saves results to cache.
The MTF feature store builder then joins these cached results instead of
re-running inference on every bar.

This ensures:
- Real Wyckoff (not SMA fallbacks)
- Fast builds (computed once, reused many times)
- Causal integrity (1H bars only see completed HTF bars)

Usage:
    python3 bin/build_wyckoff_cache.py --asset BTC --start 2024-01-01 --end 2024-12-31
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import argparse
from datetime import datetime

from engine.io.tradingview_loader import load_tv
from engine.wyckoff.wyckoff_engine import detect_wyckoff_phase
from engine.context.loader import load_macro_data

def filter_rth_only(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """Filter to RTH hours for equities (09:30-16:00 ET)"""
    if asset not in ['SPY', 'TSLA']:
        return df  # Crypto = 24/7, no filter

    # Convert to ET
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert('America/New_York')

    # Filter 09:30-16:00
    mask = (df_et.index.time >= pd.Timestamp('09:30').time()) & \
           (df_et.index.time < pd.Timestamp('16:00').time())

    # Convert back to UTC
    df_filtered = df_et[mask].copy()
    df_filtered.index = df_filtered.index.tz_convert('UTC')

    return df_filtered


def build_wyckoff_cache_1d(asset: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Build Wyckoff cache for 1D timeframe.

    Returns DataFrame with columns:
        time, wyck_phase, wyck_conf, wyck_bias, wyck_direction,
        crt_active, hps_score, is_warmup
    """
    print("\n" + "=" * 80)
    print(f"Building 1D Wyckoff Cache: {asset}")
    print("=" * 80)

    # Load 1D OHLCV (with extra warmup)
    df_1d_raw = load_tv(f"{asset}_1D")
    df_1d = filter_rth_only(df_1d_raw, asset)

    # Add 300-day warmup before start_date
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')
    warmup_start = start_ts - pd.Timedelta(days=300)

    print(f"   Loading from {warmup_start.date()} (with 300-day warmup)")
    df_1d = df_1d[(df_1d.index >= warmup_start) & (df_1d.index <= end_ts)].copy()
    df_1d.columns = [c.lower() for c in df_1d.columns]

    print(f"   Loaded {len(df_1d)} daily bars")

    # Run Wyckoff detector with expanding windows
    config = {}
    macro_data = load_macro_data()

    wyckoff_results = []
    warmup_bars = 100  # Minimum bars before confident phases

    for i, timestamp in enumerate(df_1d.index):
        if i % 30 == 0:
            print(f"   Processing 1D bar {i+1}/{len(df_1d)}...")

        # Use ALL history up to this timestamp (expanding window)
        historical_window = df_1d[df_1d.index <= timestamp]

        # Get macro context for this timestamp
        usdt_stag_strength = 0.5  # Default (can enhance with real macro later)

        if len(historical_window) >= 50:
            wyck_dict = detect_wyckoff_phase(historical_window, config, usdt_stag_strength)

            phase = wyck_dict.get('phase', 'transition')
            confidence = wyck_dict.get('confidence', 0.0)
            crt_active = wyck_dict.get('crt_active', False)
            hps_score = wyck_dict.get('hps_score', 0.0)

            # Mark warmup period (low confidence until enough history)
            is_warmup = len(historical_window) < warmup_bars
            if is_warmup:
                confidence = min(confidence, 0.3)  # Cap confidence during warmup

            # Determine bias from phase
            if phase in ['accumulation', 'spring', 'B', 'markup']:
                bias = 'long'
                direction = 1
            elif phase in ['distribution', 'upthrust', 'markdown']:
                bias = 'short'
                direction = -1
            else:
                bias = 'neutral'
                direction = 0
        else:
            # Insufficient data
            phase = 'transition'
            confidence = 0.0
            bias = 'neutral'
            direction = 0
            crt_active = False
            hps_score = 0.0
            is_warmup = True

        wyckoff_results.append({
            'time': timestamp,
            'wyck_phase': phase,
            'wyck_conf': confidence,
            'wyck_bias': bias,
            'wyck_direction': direction,
            'crt_active': crt_active,
            'hps_score': hps_score,
            'is_warmup': is_warmup
        })

    df_wyck = pd.DataFrame(wyckoff_results).set_index('time')

    # Filter to requested output range (exclude warmup period from save)
    df_output = df_wyck[(df_wyck.index >= start_ts) & (df_wyck.index <= end_ts)].copy()

    print(f"\n   ✅ Wyckoff 1D cache complete:")
    print(f"      Total bars: {len(df_output)}")
    print(f"      Confidence range: [{df_output['wyck_conf'].min():.2f}, {df_output['wyck_conf'].max():.2f}]")
    print(f"      Unique phases: {df_output['wyck_phase'].unique().tolist()}")
    print(f"      Warmup bars: {df_output['is_warmup'].sum()}")

    return df_output


def build_wyckoff_cache_4h(asset: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Build Wyckoff cache for 4H timeframe.

    Returns DataFrame with columns:
        time, wyck_phase, wyck_conf, wyck_bias, wyck_direction,
        crt_active, hps_score, is_warmup
    """
    print("\n" + "=" * 80)
    print(f"Building 4H Wyckoff Cache: {asset}")
    print("=" * 80)

    # Load 4H OHLCV (with extra warmup)
    df_4h_raw = load_tv(f"{asset}_4H")
    df_4h = filter_rth_only(df_4h_raw, asset)

    # Add 300-day warmup (= ~1800 4H bars)
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')
    warmup_start = start_ts - pd.Timedelta(days=300)

    print(f"   Loading from {warmup_start.date()} (with 300-day warmup)")
    df_4h = df_4h[(df_4h.index >= warmup_start) & (df_4h.index <= end_ts)].copy()
    df_4h.columns = [c.lower() for c in df_4h.columns]

    print(f"   Loaded {len(df_4h)} 4H bars")

    # Run Wyckoff detector with expanding windows
    config = {}

    wyckoff_results = []
    warmup_bars = 400  # ~66 days of 4H bars before confident phases

    for i, timestamp in enumerate(df_4h.index):
        if i % 200 == 0:
            print(f"   Processing 4H bar {i+1}/{len(df_4h)}...")

        # Use ALL history up to this timestamp
        historical_window = df_4h[df_4h.index <= timestamp]

        usdt_stag_strength = 0.5  # Default

        if len(historical_window) >= 50:
            wyck_dict = detect_wyckoff_phase(historical_window, config, usdt_stag_strength)

            phase = wyck_dict.get('phase', 'transition')
            confidence = wyck_dict.get('confidence', 0.0)
            crt_active = wyck_dict.get('crt_active', False)
            hps_score = wyck_dict.get('hps_score', 0.0)

            # Mark warmup period
            is_warmup = len(historical_window) < warmup_bars
            if is_warmup:
                confidence = min(confidence, 0.3)

            # Determine bias from phase
            if phase in ['accumulation', 'spring', 'B', 'markup']:
                bias = 'long'
                direction = 1
            elif phase in ['distribution', 'upthrust', 'markdown']:
                bias = 'short'
                direction = -1
            else:
                bias = 'neutral'
                direction = 0
        else:
            phase = 'transition'
            confidence = 0.0
            bias = 'neutral'
            direction = 0
            crt_active = False
            hps_score = 0.0
            is_warmup = True

        wyckoff_results.append({
            'time': timestamp,
            'wyck_phase': phase,
            'wyck_conf': confidence,
            'wyck_bias': bias,
            'wyck_direction': direction,
            'crt_active': crt_active,
            'hps_score': hps_score,
            'is_warmup': is_warmup
        })

    df_wyck = pd.DataFrame(wyckoff_results).set_index('time')

    # Filter to requested output range
    df_output = df_wyck[(df_wyck.index >= start_ts) & (df_wyck.index <= end_ts)].copy()

    print(f"\n   ✅ Wyckoff 4H cache complete:")
    print(f"      Total bars: {len(df_output)}")
    print(f"      Confidence range: [{df_output['wyck_conf'].min():.2f}, {df_output['wyck_conf'].max():.2f}]")
    print(f"      Unique phases: {df_output['wyck_phase'].unique().tolist()}")
    print(f"      Warmup bars: {df_output['is_warmup'].sum()}")

    return df_output


def main():
    parser = argparse.ArgumentParser(description='Build Wyckoff cache for MTF feature stores')
    parser.add_argument('--asset', required=True, help='Asset symbol (BTC, ETH, SPY, TSLA)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--tfs', default='1D,4H', help='Timeframes to build (comma-separated)')

    args = parser.parse_args()

    # Create output directory
    cache_dir = Path('data/wyckoff_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Wyckoff Cache Builder v1.0")
    print("=" * 80)
    print(f"Asset: {args.asset}")
    print(f"Period: {args.start} → {args.end}")
    print(f"Timeframes: {args.tfs}")
    print(f"Output: {cache_dir}")
    print("=" * 80)

    timeframes = args.tfs.split(',')

    # Build 1D cache
    if '1D' in timeframes:
        df_1d = build_wyckoff_cache_1d(args.asset, args.start, args.end)

        # Save to parquet
        output_path = cache_dir / f'{args.asset}_1D_{args.start}_to_{args.end}.parquet'
        df_1d.to_parquet(output_path)
        print(f"\n   💾 Saved: {output_path}")
        print(f"      Size: {output_path.stat().st_size / 1024:.1f} KB")

    # Build 4H cache
    if '4H' in timeframes:
        df_4h = build_wyckoff_cache_4h(args.asset, args.start, args.end)

        # Save to parquet
        output_path = cache_dir / f'{args.asset}_4H_{args.start}_to_{args.end}.parquet'
        df_4h.to_parquet(output_path)
        print(f"\n   💾 Saved: {output_path}")
        print(f"      Size: {output_path.stat().st_size / 1024:.1f} KB")

    print("\n" + "=" * 80)
    print("✅ Wyckoff Cache Build Complete!")
    print("=" * 80)
    print("\nNext steps:")
    print("  1. Run MTF feature store builder with --wyckoff-cache data/wyckoff_cache/")
    print("  2. Verify tf1d_wyck_* and tf4h_wyck_* columns have real values")
    print("  3. Run optimizer and confirm non-zero trades")
    print("=" * 80)


if __name__ == '__main__':
    main()
