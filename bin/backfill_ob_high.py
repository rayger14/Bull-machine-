#!/usr/bin/env python3
"""
Backfill Order Block (ob_high/ob_low) Features

ROOT CAUSE: Original OB detector has low coverage (15.5% in Dec 2022) due to
fixed 2% displacement threshold that fails during low volatility periods.

FIX: Use adaptive ATR-based thresholds + swing high/low validation.

Expected Results:
- 2022 coverage: 59.0% → 95%+
- December 2022: 15.5% → 90%+
- All periods: Minimum 90% coverage (10% NaN at start due to lookback is OK)

Usage:
    python bin/backfill_ob_high.py --asset BTC --start 2022-01-01 --end 2024-12-31
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
import argparse
from tqdm import tqdm

from engine.smc.order_blocks_adaptive import AdaptiveOrderBlockDetector, OrderBlockType

def backfill_ob_features(asset: str, start_date: str, end_date: str, dry_run: bool = False):
    """
    Recalculate ob_high and ob_low features using adaptive detector.

    Args:
        asset: Asset symbol (BTC, ETH, etc.)
        start_date: Start date for backfill (YYYY-MM-DD)
        end_date: End date for backfill (YYYY-MM-DD)
        dry_run: If True, show results without saving
    """
    print("=" * 80)
    print(f"Order Block Feature Backfill - Adaptive Thresholds")
    print("=" * 80)
    print(f"Asset:    {asset}")
    print(f"Period:   {start_date} → {end_date}")
    print(f"Mode:     {'DRY RUN (no saves)' if dry_run else 'LIVE (will update files)'}")
    print("=" * 80)

    # Load existing feature store
    feature_path = Path(f'data/features_mtf/{asset}_1H_{start_date}_to_{end_date}.parquet')
    if not feature_path.exists():
        print(f"❌ Error: Feature store not found at {feature_path}")
        return

    print(f"\n📂 Loading feature store: {feature_path.name}")
    df = pd.read_parquet(feature_path)
    print(f"   Total bars: {len(df)}")

    # Show current coverage
    print(f"\n📊 Current OB Coverage:")
    print(f"   tf1h_ob_high: {df['tf1h_ob_high'].notna().sum()}/{len(df)} ({df['tf1h_ob_high'].notna().sum()/len(df)*100:.1f}%)")
    print(f"   tf1h_ob_low:  {df['tf1h_ob_low'].notna().sum()}/{len(df)} ({df['tf1h_ob_low'].notna().sum()/len(df)*100:.1f}%)")

    # Monthly breakdown (first year only)
    print(f"\n📅 Monthly Coverage (Current):")
    first_year = df.index[0].year
    for month in range(1, 13):
        df_month = df[(df.index.year == first_year) & (df.index.month == month)]
        if len(df_month) > 0:
            coverage = df_month['tf1h_ob_high'].notna().sum() / len(df_month) * 100
            print(f"      {month:2d}/{first_year}: {coverage:5.1f}%")

    # Initialize adaptive detector
    print(f"\n🔧 Initializing Adaptive OB Detector...")
    config = {
        'min_displacement_pct_floor': 0.005,  # 0.5% minimum
        'atr_multiplier': 1.0,  # 1.0 × ATR
        'min_volume_ratio': 1.2,  # Was 1.5
        'lookback_bars': 50,
        'min_reaction_bars': 3,
        'swing_lookback': 30
    }
    detector = AdaptiveOrderBlockDetector(config)

    # Pre-calculate ATR if not present
    if 'atr_14' not in df.columns:
        print(f"   Calculating ATR...")
        high = df['high']
        low = df['low']
        close = df['close'].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr_14'] = tr.rolling(14).mean()

    # Recalculate OB features for each bar
    print(f"\n🔄 Recalculating Order Block features...")
    print(f"   Using adaptive thresholds: min={config['min_displacement_pct_floor']*100:.1f}%, ATR×{config['atr_multiplier']}")

    # Initialize new columns
    new_ob_high = pd.Series(index=df.index, dtype=float)
    new_ob_low = pd.Series(index=df.index, dtype=float)
    new_bb_high = pd.Series(index=df.index, dtype=float)
    new_bb_low = pd.Series(index=df.index, dtype=float)

    # Process in rolling windows for efficiency
    WINDOW_SIZE = 200  # Lookback window for OB detection
    MIN_HISTORY = 50   # Minimum history before detecting

    for i in tqdm(range(len(df)), desc="Processing bars"):
        try:
            # Get causal window (only past data)
            start_idx = max(0, i - WINDOW_SIZE)
            window = df.iloc[start_idx:i+1]

            if len(window) < MIN_HISTORY:
                continue

            # Detect order blocks in window
            order_blocks = detector.detect_order_blocks(window)

            # Get current price
            current_price = df['close'].iloc[i]
            current_timestamp = df.index[i]

            # Find nearest bearish OB (resistance) for ob_high
            bearish_obs = [ob for ob in order_blocks if ob.ob_type == OrderBlockType.BEARISH]
            if bearish_obs:
                # Sort by proximity to current price (within ±5%)
                nearby_bearish = [ob for ob in bearish_obs
                                 if abs(ob.high - current_price) / current_price <= 0.05]
                if nearby_bearish:
                    # Pick closest above current price, or highest if none above
                    above_price = [ob for ob in nearby_bearish if ob.high >= current_price]
                    if above_price:
                        nearest = min(above_price, key=lambda ob: ob.high - current_price)
                    else:
                        nearest = max(nearby_bearish, key=lambda ob: ob.high)

                    new_ob_high.iloc[i] = nearest.high
                    new_bb_high.iloc[i] = nearest.high  # BB = OB for now

            # Find nearest bullish OB (support) for ob_low
            bullish_obs = [ob for ob in order_blocks if ob.ob_type == OrderBlockType.BULLISH]
            if bullish_obs:
                # Sort by proximity to current price (within ±5%)
                nearby_bullish = [ob for ob in bullish_obs
                                 if abs(ob.low - current_price) / current_price <= 0.05]
                if nearby_bullish:
                    # Pick closest below current price, or lowest if none below
                    below_price = [ob for ob in nearby_bullish if ob.low <= current_price]
                    if below_price:
                        nearest = max(below_price, key=lambda ob: current_price - ob.low)
                    else:
                        nearest = min(nearby_bullish, key=lambda ob: ob.low)

                    new_ob_low.iloc[i] = nearest.low
                    new_bb_low.iloc[i] = nearest.low  # BB = OB for now

        except Exception as e:
            # Continue on error (will leave as NaN)
            pass

    # Replace old columns with new ones
    print(f"\n✅ Recalculation Complete!")
    print(f"\n📊 NEW OB Coverage:")
    print(f"   tf1h_ob_high: {new_ob_high.notna().sum()}/{len(df)} ({new_ob_high.notna().sum()/len(df)*100:.1f}%)")
    print(f"   tf1h_ob_low:  {new_ob_low.notna().sum()}/{len(df)} ({new_ob_low.notna().sum()/len(df)*100:.1f}%)")

    # Monthly breakdown (first year only)
    print(f"\n📅 Monthly Coverage (NEW):")
    for month in range(1, 13):
        df_month_idx = (df.index.year == first_year) & (df.index.month == month)
        if df_month_idx.sum() > 0:
            coverage = new_ob_high[df_month_idx].notna().sum() / df_month_idx.sum() * 100
            print(f"      {month:2d}/{first_year}: {coverage:5.1f}%")

    # Show improvement
    old_coverage = df['tf1h_ob_high'].notna().sum() / len(df) * 100
    new_coverage = new_ob_high.notna().sum() / len(df) * 100
    improvement = new_coverage - old_coverage

    print(f"\n📈 Improvement:")
    print(f"   Before: {old_coverage:.1f}%")
    print(f"   After:  {new_coverage:.1f}%")
    print(f"   Delta:  {improvement:+.1f}%")

    if new_coverage < 90:
        print(f"\n⚠️  WARNING: Coverage still below 90% target!")
        print(f"   Check detector parameters or data quality")

    # Save updated feature store
    if not dry_run:
        print(f"\n💾 Updating feature store...")
        df['tf1h_ob_high'] = new_ob_high
        df['tf1h_ob_low'] = new_ob_low
        df['tf1h_bb_high'] = new_bb_high
        df['tf1h_bb_low'] = new_bb_low

        # Create backup
        backup_path = feature_path.with_suffix('.parquet.backup')
        print(f"   Creating backup: {backup_path.name}")
        df_original = pd.read_parquet(feature_path)
        df_original.to_parquet(backup_path)

        # Save updated file
        print(f"   Saving updated file: {feature_path.name}")
        df.to_parquet(feature_path, compression='snappy')

        print(f"\n✅ Feature store updated successfully!")
        print(f"   Original backed up to: {backup_path}")
    else:
        print(f"\n🔍 DRY RUN - No changes saved")
        print(f"   Remove --dry-run flag to apply changes")

    print("=" * 80)


def validate_coverage(asset: str, start_date: str, end_date: str):
    """
    Validate that OB coverage meets requirements.

    Requirements:
    - Overall coverage >= 95%
    - Per-month coverage >= 90%
    - No month with coverage < 50%
    """
    print("=" * 80)
    print(f"Validating OB Coverage Requirements")
    print("=" * 80)

    feature_path = Path(f'data/features_mtf/{asset}_1H_{start_date}_to_{end_date}.parquet')
    df = pd.read_parquet(feature_path)

    print(f"\n📊 Overall Coverage:")
    overall_coverage = df['tf1h_ob_high'].notna().sum() / len(df) * 100
    print(f"   tf1h_ob_high: {overall_coverage:.1f}%")

    if overall_coverage >= 95:
        print(f"   ✅ PASS: >= 95% requirement")
    else:
        print(f"   ❌ FAIL: < 95% requirement")

    print(f"\n📅 Monthly Coverage:")
    all_months_pass = True
    for year in df.index.year.unique():
        print(f"\n   {year}:")
        for month in range(1, 13):
            df_month = df[(df.index.year == year) & (df.index.month == month)]
            if len(df_month) > 0:
                coverage = df_month['tf1h_ob_high'].notna().sum() / len(df_month) * 100
                status = "✅" if coverage >= 90 else "⚠️ " if coverage >= 50 else "❌"
                print(f"      {month:2d}/{year}: {coverage:5.1f}% {status}")

                if coverage < 90:
                    all_months_pass = False

    print(f"\n📈 Validation Result:")
    if overall_coverage >= 95 and all_months_pass:
        print(f"   ✅ ALL CHECKS PASSED")
    else:
        print(f"   ❌ VALIDATION FAILED")
        if overall_coverage < 95:
            print(f"      - Overall coverage below 95%")
        if not all_months_pass:
            print(f"      - Some months below 90% coverage")

    print("=" * 80)


def main():
    parser = argparse.ArgumentParser(
        description='Backfill Order Block (ob_high/ob_low) features with adaptive thresholds'
    )
    parser.add_argument('--asset', required=True,
                       help='Asset to process (BTC, ETH, etc.)')
    parser.add_argument('--start', required=True,
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True,
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--dry-run', action='store_true',
                       help='Show results without saving changes')
    parser.add_argument('--validate-only', action='store_true',
                       help='Only validate coverage (no backfill)')

    args = parser.parse_args()

    if args.validate_only:
        validate_coverage(args.asset, args.start, args.end)
    else:
        backfill_ob_features(args.asset, args.start, args.end, dry_run=args.dry_run)

        # Auto-validate after backfill if not dry-run
        if not args.dry_run:
            print(f"\n")
            validate_coverage(args.asset, args.start, args.end)


if __name__ == '__main__':
    main()
