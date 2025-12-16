#!/usr/bin/env python3
"""
Quick Domain Feature Backfill - STUB Implementation

Adds missing domain features with realistic distributions for testing.
This is NOT production-ready computation, just enough to test the wiring.

For production use, run the full backfill_domain_features.py script.
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


def add_smc_stub_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add SMC features with realistic stub values."""
    logger.info("Adding SMC stub features...")

    # Create realistic distributions based on price action
    volatility = df['close'].pct_change().rolling(20).std().fillna(0.01)
    volume_spike = (df.get('volume', pd.Series(1, index=df.index)) /
                   df.get('volume', pd.Series(1, index=df.index)).rolling(20).mean().fillna(1))

    # SMC score (higher during volatile periods)
    df['smc_score'] = 0.3 + (volatility * 30).clip(0, 0.4)

    # BOS/CHOCH events (when volatility spikes)
    high_vol = volatility > volatility.rolling(100).quantile(0.8)
    df['smc_bos'] = high_vol & (np.random.rand(len(df)) > 0.95)  # ~5% of high vol periods
    df['smc_choch'] = df['smc_bos'] & (np.random.rand(len(df)) > 0.7)  # ~30% of BOS are CHOCH

    # Liquidity sweeps (at local extremes)
    local_high = df['high'] == df['high'].rolling(20, center=True).max()
    local_low = df['low'] == df['low'].rolling(20, center=True).min()
    df['smc_liquidity_sweep'] = (local_high | local_low) & (np.random.rand(len(df)) > 0.9)

    # Supply/demand zones (near swing points)
    swing_high = df['high'].rolling(10, center=True).apply(
        lambda x: 1 if len(x) >= 10 and x.iloc[5] == x.max() else 0, raw=False
    ).fillna(0).astype(bool)

    swing_low = df['low'].rolling(10, center=True).apply(
        lambda x: 1 if len(x) >= 10 and x.iloc[5] == x.min() else 0, raw=False
    ).fillna(0).astype(bool)

    df['smc_supply_zone'] = swing_high
    df['smc_demand_zone'] = swing_low

    # HOB zones (alias for SMC zones)
    df['hob_supply_zone'] = df['smc_supply_zone']
    df['hob_demand_zone'] = df['smc_demand_zone']
    df['hob_imbalance'] = (df['smc_demand_zone'].astype(float) -
                           df['smc_supply_zone'].astype(float))

    logger.info(f"  ✅ SMC features added")
    logger.info(f"    - smc_bos events: {df['smc_bos'].sum()}")
    logger.info(f"    - smc_choch events: {df['smc_choch'].sum()}")
    logger.info(f"    - smc_liquidity_sweep events: {df['smc_liquidity_sweep'].sum()}")
    logger.info(f"    - smc_demand_zone events: {df['smc_demand_zone'].sum()}")

    return df


def add_wyckoff_pti_stub_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add Wyckoff PTI features with realistic stub values."""
    logger.info("Adding Wyckoff PTI stub features...")

    # Check for existing Wyckoff features
    has_spring = 'wyckoff_spring_a' in df.columns
    has_lps = 'wyckoff_lps' in df.columns
    has_pti = 'tf1h_pti_score' in df.columns

    if has_pti:
        pti_score = df['tf1h_pti_score']
    else:
        # Create basic PTI proxy from volatility
        volatility = df['close'].pct_change().rolling(20).std().fillna(0.01)
        pti_score = (volatility * 50).clip(0, 1)
        df['tf1h_pti_score'] = pti_score

    # PTI confluence (high PTI at reversal points)
    if has_spring:
        spring_events = df['wyckoff_spring_a'] | df.get('wyckoff_spring_b', False)
        df['wyckoff_pti_confluence'] = spring_events & (pti_score > 0.6)
    else:
        # Use price extremes as proxy
        local_extreme = (
            (df['low'] == df['low'].rolling(50, center=True).min()) |
            (df['high'] == df['high'].rolling(50, center=True).max())
        )
        df['wyckoff_pti_confluence'] = local_extreme & (pti_score > 0.6)

    # PTI score (composite)
    df['wyckoff_pti_score'] = pti_score * 0.6 + df['wyckoff_pti_confluence'].astype(float) * 0.4

    # Preliminary Support (use LPS as proxy if available)
    if has_lps:
        df['wyckoff_ps'] = df['wyckoff_lps']
    else:
        # Use swing lows as proxy
        swing_low = df['low'].rolling(10, center=True).apply(
            lambda x: 1 if len(x) >= 10 and x.iloc[5] == x.min() else 0, raw=False
        ).fillna(0).astype(bool)
        df['wyckoff_ps'] = swing_low & (np.random.rand(len(df)) > 0.9)  # 10% of swing lows

    logger.info(f"  ✅ Wyckoff PTI features added")
    logger.info(f"    - wyckoff_pti_confluence events: {df['wyckoff_pti_confluence'].sum()}")
    logger.info(f"    - wyckoff_ps events: {df['wyckoff_ps'].sum()}")

    return df


def add_temporal_stub_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add temporal features with realistic stub values."""
    logger.info("Adding temporal stub features...")

    # Get swing points
    swing_high = df['high'].rolling(10, center=True).apply(
        lambda x: 1 if len(x) >= 10 and x.iloc[5] == x.max() else 0, raw=False
    ).fillna(0).astype(bool)

    swing_low = df['low'].rolling(10, center=True).apply(
        lambda x: 1 if len(x) >= 10 and x.iloc[5] == x.min() else 0, raw=False
    ).fillna(0).astype(bool)

    # Calculate bars since last swing
    bars_since_swing_high = np.full(len(df), 999)
    bars_since_swing_low = np.full(len(df), 999)

    last_high_idx = -1
    last_low_idx = -1

    for i in range(len(df)):
        if swing_high.iloc[i]:
            last_high_idx = i
        if swing_low.iloc[i]:
            last_low_idx = i

        if last_high_idx >= 0:
            bars_since_swing_high[i] = i - last_high_idx
        if last_low_idx >= 0:
            bars_since_swing_low[i] = i - last_low_idx

    # Fibonacci time zones (21, 34, 55, 89 bars)
    fib_ratios = [21, 34, 55, 89]

    # Check if current bar is near a Fibonacci time from last swing
    near_fib_high = np.zeros(len(df), dtype=bool)
    near_fib_low = np.zeros(len(df), dtype=bool)

    for fib in fib_ratios:
        near_fib_high |= (np.abs(bars_since_swing_high - fib) < 3)
        near_fib_low |= (np.abs(bars_since_swing_low - fib) < 3)

    # Temporal confluence when near multiple Fib ratios
    confluence_count_high = np.sum([np.abs(bars_since_swing_high - fib) < 3 for fib in fib_ratios], axis=0)
    confluence_count_low = np.sum([np.abs(bars_since_swing_low - fib) < 3 for fib in fib_ratios], axis=0)

    df['temporal_confluence'] = (confluence_count_high >= 2) | (confluence_count_low >= 2)

    # Support/resistance cluster strength
    df['temporal_support_cluster'] = np.where(
        bars_since_swing_low < 100,
        np.minimum(1.0, confluence_count_low / 3.0),
        0.0
    )

    df['temporal_resistance_cluster'] = np.where(
        bars_since_swing_high < 100,
        np.minimum(1.0, confluence_count_high / 3.0),
        0.0
    )

    logger.info(f"  ✅ Temporal features added")
    logger.info(f"    - temporal_confluence events: {df['temporal_confluence'].sum()}")
    logger.info(f"    - temporal_support_cluster non-zero: {(df['temporal_support_cluster'] > 0).sum()}")

    return df


def quick_backfill(parquet_path: str, output_path: str = None, backup: bool = True):
    """Quick backfill of domain features with stubs."""
    parquet_path = Path(parquet_path)

    if not parquet_path.exists():
        raise FileNotFoundError(f"Feature store not found: {parquet_path}")

    logger.info("=" * 80)
    logger.info("QUICK DOMAIN FEATURE BACKFILL (STUB VALUES)")
    logger.info("=" * 80)
    logger.info(f"Input: {parquet_path}")
    logger.info("⚠️  WARNING: This creates STUB features for testing only!")
    logger.info("⚠️  For production, use backfill_domain_features.py")

    # Load
    logger.info("\n📊 Loading feature store...")
    df = pd.read_parquet(parquet_path)
    logger.info(f"  Loaded {len(df):,} rows × {len(df.columns)} columns")
    logger.info(f"  Date range: {df.index.min()} to {df.index.max()}")

    # Backup
    if backup:
        backup_path = parquet_path.parent / f"{parquet_path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.parquet"
        logger.info(f"\n💾 Creating backup: {backup_path.name}")
        df.to_parquet(backup_path)

    # Add features
    logger.info("\n🔧 Adding stub domain features...")

    df = add_smc_stub_features(df)
    df = add_wyckoff_pti_stub_features(df)
    df = add_temporal_stub_features(df)

    # Save
    output_path = Path(output_path) if output_path else parquet_path
    logger.info(f"\n💾 Saving updated feature store...")
    logger.info(f"  Output: {output_path}")
    df.to_parquet(output_path)

    # Verify
    logger.info("\n✅ QUICK BACKFILL COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Updated feature store: {output_path}")
    logger.info(f"Total columns: {len(df.columns)}")
    logger.info(f"File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

    logger.info("\n" + "=" * 80)
    logger.info("STATUS: Ready for domain wiring tests ✅")
    logger.info("NOTE: These are STUB features - rerun with full backfill for production")
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description='Quick backfill with stub features')
    parser.add_argument('--input', required=True, help='Path to feature store parquet')
    parser.add_argument('--output', help='Output path (default: overwrite input)')
    parser.add_argument('--no-backup', action='store_true', help='Skip backup')

    args = parser.parse_args()

    try:
        quick_backfill(
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
