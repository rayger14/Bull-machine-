#!/usr/bin/env python3
"""
Pre-compute and cache features with regime labels for faster optimization.

This script saves ~10-15 seconds per Optuna trial by computing regime labels
and features once, then loading from cache during trials.

Usage:
    python3 bin/cache_features_with_regime.py --asset BTC --start 2022-01-01 --end 2024-12-31
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from bin.build_mtf_feature_store import build_mtf_feature_store
from engine.regime_detector import RegimeDetector
from engine.event_calendar import EventCalendar


def cache_features_with_regime(asset: str, start_date: str, end_date: str, output_dir: str = 'data/cached'):
    """
    Build feature store and add regime labels.

    Args:
        asset: Asset symbol (BTC, ETH)
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        output_dir: Output directory for cached data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"🚀 Building feature store for {asset} ({start_date} → {end_date})")
    print("="*80)

    # Step 1: Build feature store
    print("\n[1/3] Building multi-timeframe feature store...")
    df = build_mtf_feature_store(
        asset=asset,
        start_date=start_date,
        end_date=end_date
    )

    print(f"  ✓ Loaded {len(df)} bars")
    print(f"  ✓ {len(df.columns)} features")
    print(f"  ✓ Date range: {df.index.min()} → {df.index.max()}")

    # Step 2: Add regime labels
    print("\n[2/3] Computing regime labels...")
    regime_detector = RegimeDetector()

    # Get regime probabilities and labels
    regime_probs = regime_detector.predict_proba(df)  # Returns dict of arrays
    regime_labels = regime_detector.classify(df)       # Returns array of labels

    # Add to dataframe
    df['regime_label'] = regime_labels

    # Add probability columns
    for regime_name, probs in regime_probs.items():
        df[f'regime_prob_{regime_name.lower()}'] = probs

    print(f"  ✓ Added regime labels")
    print(f"  ✓ Regime distribution:")
    for regime, count in pd.Series(regime_labels).value_counts().items():
        pct = count / len(regime_labels) * 100
        print(f"    - {regime}: {count} bars ({pct:.1f}%)")

    # Step 3: Add event calendar
    print("\n[3/3] Adding event calendar flags...")
    event_calendar = EventCalendar()

    for idx, row in df.iterrows():
        timestamp = pd.Timestamp(idx)
        events = event_calendar.check_event(timestamp)

        # Add binary flags for each event type
        df.loc[idx, 'event_fomc'] = 'fomc' in events
        df.loc[idx, 'event_cpi'] = 'cpi' in events
        df.loc[idx, 'event_nfp'] = 'nfp' in events
        df.loc[idx, 'event_earnings'] = 'earnings' in events
        df.loc[idx, 'has_event'] = len(events) > 0

    print(f"  ✓ Added event flags")

    # Save cache
    cache_filename = f"{asset.lower()}_features_{start_date}_{end_date}_cached.parquet"
    cache_path = output_path / cache_filename

    print(f"\n💾 Saving cached features...")
    df.to_parquet(cache_path, compression='zstd')

    # Save metadata
    metadata = {
        'asset': asset,
        'start_date': start_date,
        'end_date': end_date,
        'n_bars': len(df),
        'n_features': len(df.columns),
        'regime_detector_model': 'regime_gmm_v3.1_fixed.pkl',
        'created_at': datetime.now().isoformat(),
        'columns': list(df.columns)
    }

    metadata_path = cache_path.with_suffix('.json')
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved to: {cache_path}")
    print(f"  ✓ Size: {cache_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  ✓ Metadata: {metadata_path}")

    print("\n" + "="*80)
    print("✅ Cache complete!")
    print(f"\nUsage in optimization scripts:")
    print(f"```python")
    print(f"df = pd.read_parquet('{cache_path}')")
    print(f"regime_labels = df['regime_label'].values")
    print(f"# No need to run RegimeDetector.classify() - already cached!")
    print(f"```")

    return df, cache_path


def load_cached_features(cache_path: str):
    """
    Load pre-computed features from cache.

    Args:
        cache_path: Path to cached parquet file

    Returns:
        DataFrame with features and regime labels
    """
    df = pd.read_parquet(cache_path)

    # Load metadata
    metadata_path = Path(cache_path).with_suffix('.json')
    if metadata_path.exists():
        import json
        with open(metadata_path) as f:
            metadata = json.load(f)
        print(f"📂 Loaded cache from {metadata['created_at']}")
        print(f"  • Asset: {metadata['asset']}")
        print(f"  • Period: {metadata['start_date']} → {metadata['end_date']}")
        print(f"  • Bars: {metadata['n_bars']:,}")

    return df


def main():
    parser = argparse.ArgumentParser(description='Cache features with regime labels for optimization')
    parser.add_argument('--asset', default='BTC', help='Asset symbol (default: BTC)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--output-dir', default='data/cached', help='Output directory')

    args = parser.parse_args()

    try:
        df, cache_path = cache_features_with_regime(
            asset=args.asset,
            start_date=args.start,
            end_date=args.end,
            output_dir=args.output_dir
        )

        return 0

    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main())
