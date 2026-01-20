#!/usr/bin/env python3
"""
Fast Domain Feature Backfill - Vectorized Implementation

Adds missing SMC, Wyckoff PTI, Temporal, and HOB features using fast vectorized operations.
Uses proxy implementations and sampling for speed.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def compute_smc_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute SMC features using fast vectorized operations.
    Uses proxy implementations based on price/volume patterns.
    """
    logger.info("Computing SMC features (vectorized)...")

    # SMC Score - composite of trend strength, volume, and structure
    # Based on ADX, volume ratio, and price position in range
    if 'adx_14' in df.columns:
        trend_component = (df['adx_14'] / 100.0).clip(0, 1)
    else:
        trend_component = pd.Series(0.5, index=df.index)

    # Volume component
    vol_ma = df['volume'].rolling(20).mean()
    vol_ratio = (df['volume'] / vol_ma).clip(0, 3) / 3.0
    vol_component = vol_ratio.fillna(0.5)

    # Price structure component (position in 20-bar range)
    rolling_high = df['high'].rolling(20).max()
    rolling_low = df['low'].rolling(20).min()
    range_position = ((df['close'] - rolling_low) / (rolling_high - rolling_low + 1e-9)).clip(0, 1)
    structure_component = range_position.fillna(0.5)

    # Composite SMC score
    df['smc_score'] = (
        trend_component * 0.4 +
        vol_component * 0.3 +
        structure_component * 0.3
    ).fillna(0.5)

    # BOS Detection - Price breaks above recent high with volume
    recent_high = df['high'].rolling(20).max().shift(1)
    breaks_high = df['close'] > recent_high
    strong_volume = df['volume'] > df['volume'].rolling(20).mean() * 1.5
    df['smc_bos'] = breaks_high & strong_volume

    # CHOCH - Change of Character (trend reversal)
    # Detect when BOS contradicts recent trend
    price_change = df['close'].diff(5)
    trend_changed = (price_change * price_change.shift(5)) < 0
    df['smc_choch'] = df['smc_bos'] & trend_changed

    # Liquidity Sweep - Price spikes beyond recent extremes then reverses
    # Lower wick > 60% of range AND close in upper 70%
    body_mid = (df['open'] + df['close']) / 2
    lower_wick = body_mid - df['low']
    bar_range = df['high'] - df['low']
    lower_wick_pct = lower_wick / (bar_range + 1e-9)
    close_position = (df['close'] - df['low']) / (bar_range + 1e-9)

    sweep_down = (lower_wick_pct > 0.6) & (close_position > 0.7)

    upper_wick = df['high'] - body_mid
    upper_wick_pct = upper_wick / (bar_range + 1e-9)
    close_position_inv = (df['high'] - df['close']) / (bar_range + 1e-9)

    sweep_up = (upper_wick_pct > 0.6) & (close_position_inv > 0.7)

    df['smc_liquidity_sweep'] = sweep_down | sweep_up

    # Supply/Demand Zones - Based on order blocks
    # Demand zone = bullish OB = up candle followed by strong up move
    up_candle = df['close'] > df['open']
    strong_up_move = (df['close'].shift(-1) > df['high']) & (df['volume'].shift(-1) > vol_ma.shift(-1) * 1.3)
    df['smc_demand_zone'] = up_candle & strong_up_move

    # Supply zone = bearish OB = down candle followed by strong down move
    down_candle = df['close'] < df['open']
    strong_down_move = (df['close'].shift(-1) < df['low']) & (df['volume'].shift(-1) > vol_ma.shift(-1) * 1.3)
    df['smc_supply_zone'] = down_candle & strong_down_move

    # HOB (Higher Order Book) features = same as SMC zones
    df['hob_demand_zone'] = df['smc_demand_zone']
    df['hob_supply_zone'] = df['smc_supply_zone']

    # HOB Imbalance = net demand/supply pressure
    demand_rolling = df['smc_demand_zone'].rolling(20).sum()
    supply_rolling = df['smc_supply_zone'].rolling(20).sum()
    df['hob_imbalance'] = ((demand_rolling - supply_rolling) / 20.0).clip(-1, 1).fillna(0)

    logger.info(f"  ✅ SMC features computed (vectorized)")
    logger.info(f"    - smc_score range: [{df['smc_score'].min():.3f}, {df['smc_score'].max():.3f}]")
    logger.info(f"    - smc_bos events: {df['smc_bos'].sum()}")
    logger.info(f"    - smc_liquidity_sweep events: {df['smc_liquidity_sweep'].sum()}")
    logger.info(f"    - smc_demand_zone events: {df['smc_demand_zone'].sum()}")
    logger.info(f"    - smc_supply_zone events: {df['smc_supply_zone'].sum()}")

    return df


def compute_wyckoff_pti_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Wyckoff PTI features using existing features.
    """
    logger.info("Computing Wyckoff PTI features (vectorized)...")

    # Check if PTI features exist
    has_pti = 'tf1h_pti_score' in df.columns

    if not has_pti:
        logger.warning("  ⚠️  PTI features not found - computing basic proxy")
        # PTI proxy: high at extremes with reversal
        rolling_high = df['high'].rolling(50).max()
        rolling_low = df['low'].rolling(50).min()
        at_high = (df['high'] >= rolling_high * 0.998).astype(float)
        at_low = (df['low'] <= rolling_low * 1.002).astype(float)
        reversal = ((df['close'] - df['open']) * (df['close'].shift(-1) - df['open'].shift(-1))) < 0
        df['tf1h_pti_score'] = ((at_high + at_low) * reversal.astype(float)).clip(0, 1)

    # Wyckoff trap events
    has_spring_a = 'wyckoff_spring_a' in df.columns
    has_spring_b = 'wyckoff_spring_b' in df.columns
    has_ut = 'wyckoff_ut' in df.columns
    has_utad = 'wyckoff_utad' in df.columns

    if has_spring_a or has_spring_b or has_ut or has_utad:
        trap_events = (
            df.get('wyckoff_spring_a', False) |
            df.get('wyckoff_spring_b', False) |
            df.get('wyckoff_ut', False) |
            df.get('wyckoff_utad', False)
        )
    else:
        # Create proxy trap events from price action
        trap_events = df['smc_liquidity_sweep']  # Use liquidity sweeps as proxy

    # PTI Confluence - high PTI score + Wyckoff trap event
    high_pti = df['tf1h_pti_score'] > 0.6
    df['wyckoff_pti_confluence'] = trap_events & high_pti

    # PTI Score - composite
    df['wyckoff_pti_score'] = (
        df['tf1h_pti_score'] * 0.6 +
        trap_events.astype(float) * 0.4
    )

    # Preliminary Support (PS) - proxy using LPS or create from price action
    if 'wyckoff_lps' in df.columns:
        df['wyckoff_ps'] = df['wyckoff_lps']
    else:
        # PS proxy: price at lows with volume exhaustion
        rolling_low = df['low'].rolling(50).min()
        at_lows = df['low'] <= rolling_low * 1.005
        vol_declining = df['volume'] < df['volume'].rolling(10).mean() * 0.8
        df['wyckoff_ps'] = at_lows & vol_declining

    logger.info(f"  ✅ Wyckoff PTI features computed (vectorized)")
    logger.info(f"    - wyckoff_pti_confluence events: {df['wyckoff_pti_confluence'].sum()}")
    logger.info(f"    - wyckoff_pti_score range: [{df['wyckoff_pti_score'].min():.3f}, {df['wyckoff_pti_score'].max():.3f}]")
    logger.info(f"    - wyckoff_ps events: {df['wyckoff_ps'].sum()}")

    return df


def compute_temporal_features_vectorized(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute temporal confluence features using Fibonacci time zones.
    """
    logger.info("Computing temporal features (vectorized)...")

    # Find swing highs and lows
    window = 10
    swing_highs = df['high'].rolling(window*2+1, center=True).apply(
        lambda x: 1 if len(x) == window*2+1 and x.iloc[window] == x.max() else 0,
        raw=False
    ).fillna(0).astype(bool)

    swing_lows = df['low'].rolling(window*2+1, center=True).apply(
        lambda x: 1 if len(x) == window*2+1 and x.iloc[window] == x.min() else 0,
        raw=False
    ).fillna(0).astype(bool)

    # Calculate bars since last swing
    swing_high_idx = np.where(swing_highs)[0]
    swing_low_idx = np.where(swing_lows)[0]

    df['_bars_since_high'] = 999
    df['_bars_since_low'] = 999

    for i in range(len(df)):
        if len(swing_high_idx[swing_high_idx < i]) > 0:
            df.iloc[i, df.columns.get_loc('_bars_since_high')] = i - swing_high_idx[swing_high_idx < i][-1]
        if len(swing_low_idx[swing_low_idx < i]) > 0:
            df.iloc[i, df.columns.get_loc('_bars_since_low')] = i - swing_low_idx[swing_low_idx < i][-1]

    # Fibonacci time zones (21, 34, 55, 89 bars)
    fib_ratios = [21, 34, 55, 89]
    df['_fib_confluence_count'] = 0

    for fib in fib_ratios:
        near_fib_high = (df['_bars_since_high'] - fib).abs() < 3
        near_fib_low = (df['_bars_since_low'] - fib).abs() < 3
        df['_fib_confluence_count'] += (near_fib_high | near_fib_low).astype(int)

    # Temporal confluence when 2+ Fibonacci time zones align
    df['temporal_confluence'] = df['_fib_confluence_count'] >= 2

    # Support/resistance cluster strength
    df['temporal_support_cluster'] = (
        (df['_fib_confluence_count'] / 3.0) *
        (df['_bars_since_low'] < 100).astype(float)
    ).clip(0, 1)

    df['temporal_resistance_cluster'] = (
        (df['_fib_confluence_count'] / 3.0) *
        (df['_bars_since_high'] < 100).astype(float)
    ).clip(0, 1)

    # Clean up temporary columns
    df = df.drop(columns=['_bars_since_high', '_bars_since_low', '_fib_confluence_count'])

    logger.info(f"  ✅ Temporal features computed (vectorized)")
    logger.info(f"    - temporal_confluence events: {df['temporal_confluence'].sum()}")
    logger.info(f"    - temporal_support_cluster non-zero: {(df['temporal_support_cluster'] > 0).sum()}/{len(df)}")
    logger.info(f"    - temporal_resistance_cluster non-zero: {(df['temporal_resistance_cluster'] > 0).sum()}/{len(df)}")

    return df


def backfill_feature_store(parquet_path: str, output_path: str = None, backup: bool = True):
    """
    Backfill missing domain features using fast vectorized operations.
    """
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Feature store not found: {parquet_path}")

    logger.info("=" * 80)
    logger.info("DOMAIN FEATURE BACKFILL (FAST VECTORIZED)")
    logger.info("=" * 80)
    logger.info(f"Input: {parquet_path}")

    # Load existing feature store
    logger.info("\n📊 Loading feature store...")
    df = pd.read_parquet(parquet_path)

    logger.info(f"  Loaded {len(df)} rows × {len(df.columns)} columns")
    logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Create backup if requested
    if backup:
        backup_path = parquet_path.parent / f"{parquet_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        logger.info(f"\n💾 Creating backup: {backup_path.name}")
        df.to_parquet(backup_path)

    # Compute missing features
    logger.info("\n🔧 Computing missing domain features...")
    initial_cols = len(df.columns)

    # 1. SMC Features (vectorized)
    df = compute_smc_features_vectorized(df)

    # 2. Wyckoff PTI Features (vectorized)
    df = compute_wyckoff_pti_features_vectorized(df)

    # 3. Temporal Features (vectorized)
    df = compute_temporal_features_vectorized(df)

    # Save updated feature store
    output_path = Path(output_path) if output_path else parquet_path

    logger.info(f"\n💾 Saving updated feature store...")
    logger.info(f"  Output: {output_path}")
    df.to_parquet(output_path)

    # Verification
    logger.info("\n✅ BACKFILL COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Updated feature store: {output_path}")
    logger.info(f"Total columns: {len(df.columns)} (added {len(df.columns) - initial_cols} new)")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Feature summary
    logger.info("\nNEW FEATURES ADDED:")

    new_features = {
        'smc_score': 'float',
        'smc_bos': 'bool',
        'smc_choch': 'bool',
        'smc_liquidity_sweep': 'bool',
        'smc_supply_zone': 'bool',
        'smc_demand_zone': 'bool',
        'hob_demand_zone': 'bool',
        'hob_supply_zone': 'bool',
        'hob_imbalance': 'float',
        'wyckoff_pti_confluence': 'bool',
        'wyckoff_pti_score': 'float',
        'wyckoff_ps': 'bool',
        'temporal_confluence': 'bool',
        'temporal_support_cluster': 'float',
        'temporal_resistance_cluster': 'float'
    }

    for feature, ftype in new_features.items():
        if feature in df.columns:
            non_null_pct = (df[feature].notna().sum() / len(df)) * 100

            if ftype == 'bool':
                event_count = df[feature].sum()
                logger.info(f"  ✅ {feature:<35} {event_count:>5} events ({non_null_pct:.1f}% non-null)")
            else:
                non_zero = (df[feature] != 0).sum() if df[feature].dtype in [float, int] else len(df)
                mean_val = df[feature].mean() if df[feature].dtype in [float, int] else 'N/A'
                logger.info(f"  ✅ {feature:<35} {non_zero:>5}/{len(df)} non-zero (mean={mean_val:.3f})")

    logger.info("\n" + "=" * 80)
    logger.info("STATUS: Ready for re-testing domain wiring ✅")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Fast domain feature backfill (vectorized)')
    parser.add_argument('--input', required=True, help='Path to feature store parquet file')
    parser.add_argument('--output', help='Output path (default: overwrite input)')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup creation')

    args = parser.parse_args()

    try:
        backfill_feature_store(
            parquet_path=args.input,
            output_path=args.output,
            backup=not args.no_backup
        )
    except Exception as e:
        logger.error(f"❌ Backfill failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
