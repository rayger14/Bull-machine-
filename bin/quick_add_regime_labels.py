#!/usr/bin/env python3
"""
Quick script to add regime labels to existing MTF feature store.
Use this when cache_features_with_regime.py is too slow.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
from engine.regime_detector import RegimeDetector
from datetime import datetime
import json

def add_regime_labels_to_mtf(mtf_path: str, output_path: str):
    """Load existing MTF file and add regime labels."""

    print(f"📂 Loading existing MTF data: {mtf_path}")
    df = pd.read_parquet(mtf_path)

    # Ensure timestamp is the index
    if 'timestamp' in df.columns and not isinstance(df.index, pd.DatetimeIndex):
        df = df.set_index('timestamp')
        df.index = pd.to_datetime(df.index)

    print(f"  ✓ Loaded {len(df)} bars with {len(df.columns)} features")
    print(f"  ✓ Index: {type(df.index).__name__}")

    print("\n🔮 Adding regime labels...")
    regime_detector = RegimeDetector()

    # Use classify_batch to add regime labels and confidence
    df = regime_detector.classify_batch(df)

    print(f"  ✓ Added regime labels")
    print(f"\n  Regime distribution:")
    for regime, count in df['regime_label'].value_counts().items():
        pct = count / len(df) * 100
        print(f"    - {regime}: {count} bars ({pct:.1f}%)")

    print(f"\n💾 Saving to: {output_path}")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, compression='zstd')

    # Save metadata
    metadata = {
        'source': mtf_path,
        'n_bars': len(df),
        'n_features': len(df.columns),
        'regime_detector_model': 'regime_gmm_v3.1_fixed.pkl',
        'created_at': datetime.now().isoformat(),
        'method': 'quick_add_regime_labels'
    }

    metadata_path = Path(output_path).with_suffix('.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"  ✓ Saved {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  ✓ Metadata: {metadata_path}")
    print("\n✅ Done!")

    return df


if __name__ == '__main__':
    mtf_file = 'data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'
    output_file = 'data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet'

    print("="*70)
    print("QUICK REGIME LABEL ADDITION")
    print("="*70)

    try:
        add_regime_labels_to_mtf(mtf_file, output_file)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
