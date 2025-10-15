#!/usr/bin/env python3
"""
Causal Feature Store Builder

Pre-computes all features using ONLY past data (no future leak).
This creates a reusable cache that both the optimizer and live engine
can consume, ensuring perfect alignment.

Features computed:
- Domain scores (Wyckoff, SMC, HOB, Momentum, Temporal)
- Technical indicators (ATR, ADX, RSI, SMA)
- Macro veto flags (VIX regime, crisis fuse, etc.)
- MTF alignment flags
- Macro exit triggers

All features are computed causally with proper lookback windows.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import argparse

from engine.io.tradingview_loader import load_tv
from engine.context.loader import load_macro_data, fetch_macro_snapshot
from engine.context.macro_engine import analyze_macro, create_default_macro_config
from engine.fusion.domain_fusion import analyze_fusion


def calculate_atr(df: pd.DataFrame, period: int = 20) -> pd.Series:
    """Calculate ATR causally (past-only)"""
    high = df['high']
    low = df['low']
    close = df['close']
    prev_close = close.shift(1)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    return tr.ewm(span=period, adjust=False).mean()


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ADX causally"""
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)

    # Directional Movement
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low

    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed indicators
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(span=period, adjust=False).mean() / atr

    # ADX
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di + 1e-10)
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate RSI causally"""
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / (loss + 1e-10)
    return 100 - (100 / (1 + rs))


def build_feature_store(asset: str, start_date: str, end_date: str, sample_every: int = 4):
    """
    Build complete feature store with causal computation

    Args:
        asset: Asset to build features for (BTC, ETH, SOL)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        sample_every: Sample domain scores every N bars for speed (default 4)
    """
    print(f"🏗️  Building Feature Store: {asset}")
    print(f"📅 Period: {start_date} → {end_date}")
    print(f"⚡ Sampling: Every {sample_every} bars for domain scores")
    print("=" * 70)

    # Load raw data
    print("\n📊 Loading OHLCV data...")
    df_1h = load_tv(f"{asset}_1H")
    df_4h = load_tv(f"{asset}_4H")
    df_1d = load_tv(f"{asset}_1D")

    # Filter to date range
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')
    df_1h = df_1h[(df_1h.index >= start_ts) & (df_1h.index <= end_ts)].copy()
    df_4h = df_4h[(df_4h.index >= start_ts) & (df_4h.index <= end_ts)].copy()
    df_1d = df_1d[(df_1d.index >= start_ts) & (df_1d.index <= end_ts)].copy()

    # Standardize columns
    for df in [df_1h, df_4h, df_1d]:
        df.columns = [c.lower() for c in df.columns]

    print(f"   {len(df_1h)} 1H bars, {len(df_4h)} 4H bars, {len(df_1d)} 1D bars")

    # Load macro data
    print("\n📈 Loading macro data...")
    macro_data = load_macro_data()
    macro_config = create_default_macro_config()

    # RELAXED: Raise macro veto threshold from 0.85 to 0.90 (less blocking)
    macro_config['macro_veto_threshold'] = 0.90
    print(f"   Macro veto threshold: {macro_config['macro_veto_threshold']} (relaxed)")

    # Build feature dataframe on 1H timeframe (most granular)
    print("\n🔧 Computing technical indicators...")
    features = pd.DataFrame(index=df_1h.index)

    # OHLCV
    features['open'] = df_1h['open']
    features['high'] = df_1h['high']
    features['low'] = df_1h['low']
    features['close'] = df_1h['close']
    features['volume'] = df_1h['volume']

    # Technical indicators
    features['atr_20'] = calculate_atr(df_1h, 20)
    features['atr_14'] = calculate_atr(df_1h, 14)
    features['adx_14'] = calculate_adx(df_1h, 14)
    features['rsi_14'] = calculate_rsi(df_1h, 14)

    # SMAs
    for period in [20, 50, 100]:
        features[f'sma_{period}'] = df_1h['close'].rolling(period).mean()

    print("   ✅ Technical indicators computed")

    # Compute macro veto and exit flags
    print("\n🌍 Computing macro veto flags...")
    macro_veto_flags = []
    macro_exit_flags = []

    for idx, timestamp in enumerate(df_1h.index):
        if idx % 100 == 0:
            print(f"   Processing macro bar {idx}/{len(df_1h)}...")
        ts_naive = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') else timestamp
        snapshot = fetch_macro_snapshot(macro_data, ts_naive)
        macro_result = analyze_macro(snapshot, macro_config)

        # Veto flag (blocks new entries)
        veto = macro_result['veto_strength'] >= macro_config['macro_veto_threshold']
        macro_veto_flags.append(veto)

        # Exit flag (forces position close)
        # Stricter threshold for exits
        crisis_exit = macro_result['veto_strength'] >= 0.90
        macro_exit_flags.append(crisis_exit)

    features['macro_veto'] = macro_veto_flags
    features['macro_exit_flag'] = macro_exit_flags

    print("   ✅ Macro flags computed")

    # Compute domain scores (sampled for speed)
    print(f"\n🔮 Computing domain scores (sampling every {sample_every} bars)...")

    wyckoff_scores = np.full(len(df_1h), 0.5)
    smc_scores = np.full(len(df_1h), 0.5)
    hob_scores = np.full(len(df_1h), 0.5)
    momentum_scores = np.full(len(df_1h), 0.5)
    temporal_scores = np.full(len(df_1h), 0.5)  # Placeholder for Gann/LPPLS

    # Map 1H bars to nearest 4H bar for domain score alignment
    sampled_count = 0
    for i in range(len(df_1h)):
        if i % 100 == 0:
            print(f"   Processing domain bar {i}/{len(df_1h)}...")
        current_time_1h = df_1h.index[i]

        # Only compute every Nth bar
        if i % sample_every != 0 and i != len(df_1h) - 1:
            # Forward-fill from previous
            if i > 0:
                wyckoff_scores[i] = wyckoff_scores[i-1]
                smc_scores[i] = smc_scores[i-1]
                hob_scores[i] = hob_scores[i-1]
                momentum_scores[i] = momentum_scores[i-1]
                temporal_scores[i] = temporal_scores[i-1]
            continue

        # Get windows (causal - only past data)
        window_1h = df_1h.iloc[:i+1].tail(200)
        window_4h = df_4h[df_4h.index <= current_time_1h].tail(100)
        window_1d = df_1d[df_1d.index <= current_time_1h].tail(50)

        # Need minimum data
        if len(window_1h) < 50 or len(window_4h) < 14 or len(window_1d) < 20:
            continue

        try:
            # Run fusion analysis
            fusion_result = analyze_fusion(
                window_1h, window_4h, window_1d,
                config={'fusion': {
                    'weights': {'wyckoff': 0.30, 'smc': 0.15, 'liquidity': 0.25, 'momentum': 0.30}
                }}
            )

            wyckoff_scores[i] = fusion_result.wyckoff_score
            smc_scores[i] = fusion_result.smc_score
            hob_scores[i] = fusion_result.hob_score
            momentum_scores[i] = fusion_result.momentum_score
            # temporal_scores[i] = fusion_result.temporal_score  # Add when available

            sampled_count += 1

        except Exception as e:
            # Keep neutral scores on error
            pass

    features['wyckoff'] = wyckoff_scores
    features['smc'] = smc_scores
    features['hob'] = hob_scores
    features['momentum'] = momentum_scores
    features['temporal'] = temporal_scores

    print(f"   ✅ Domain scores computed ({sampled_count} actual fusion calls)")

    # Compute MTF alignment (RELAXED: 2-of-3 instead of strict 3-of-3)
    print("\n🎯 Computing MTF alignment flags (2-of-3 rule)...")
    mtf_aligned = np.zeros(len(df_1h), dtype=bool)

    for i in range(len(df_1h)):
        current_time = df_1h.index[i]

        # Get current 4H and 1D bar
        nearest_4h = df_4h[df_4h.index <= current_time]
        nearest_1d = df_1d[df_1d.index <= current_time]

        if len(nearest_4h) > 50 and len(nearest_1d) > 20:
            # Check each timeframe alignment independently
            h1_aligned = df_1h['close'].iloc[i] > features[f'sma_20'].iloc[i]
            h4_aligned = nearest_4h['close'].iloc[-1] > nearest_4h['close'].rolling(20).mean().iloc[-1]
            d1_aligned = nearest_1d['close'].iloc[-1] > nearest_1d['close'].rolling(20).mean().iloc[-1]

            # RELAXED: At least 2 of 3 timeframes aligned (was 3 of 3)
            aligned_count = sum([h1_aligned, h4_aligned, d1_aligned])
            if aligned_count >= 2:
                mtf_aligned[i] = True

    features['mtf_align'] = mtf_aligned

    print("   ✅ MTF alignment computed (2-of-3 rule)")

    # Drop initial NaN rows (from rolling windows)
    features = features.dropna()

    # Save to parquet
    output_dir = Path('data/features/v18')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{asset}_1H.parquet'

    features.to_parquet(output_path)

    print(f"\n💾 Feature store saved:")
    print(f"   {output_path}")
    print(f"   {len(features)} bars × {len(features.columns)} features")
    print(f"   Size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("\n✅ Feature store build complete!")

    return features


def main():
    parser = argparse.ArgumentParser(description='Build causal feature store')
    parser.add_argument('--asset', default='BTC', help='Asset to process (BTC, ETH, SOL)')
    parser.add_argument('--start', default='2024-01-01', help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default='2025-10-01', help='End date (YYYY-MM-DD)')
    parser.add_argument('--sample', type=int, default=4, help='Sample domain scores every N bars')

    args = parser.parse_args()

    build_feature_store(args.asset, args.start, args.end, args.sample)


if __name__ == '__main__':
    main()
