#!/usr/bin/env python3
"""
Quick patch script to add Phase 4 features to existing feature store.
This avoids re-running the entire 45+ minute build process.
"""

import pandas as pd
import sys
from pathlib import Path

def add_phase4_features(parquet_path: str):
    """Add tf4h_fusion_score and volume_zscore to existing feature store."""

    print(f"\n📂 Loading existing feature store: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    print(f"   Current shape: {df.shape[0]} bars × {df.shape[1]} features")

    # Phase 4 Re-Entry: Add tf4h_fusion_score and volume_zscore
    print("\n🎯 Computing Phase 4 re-entry features...")

    # tf4h_fusion_score: Simple weighted average of tf4h Wyckoff/structure indicators
    # Use tf4h_internal_phase (the actual column name in the feature store)
    phase_score_map = {
        'accumulation': 0.7,
        'markup': 0.9,
        'distribution': -0.7,
        'markdown': -0.9,
        'transition': 0.0,  # Neutral
        'unknown': 0.0,
        'neutral': 0.0
    }

    # Convert tf4h_internal_phase to score
    if 'tf4h_internal_phase' in df.columns:
        df['tf4h_fusion_score'] = df['tf4h_internal_phase'].map(phase_score_map).fillna(0.0)
        print(f"   ✅ Mapped tf4h_internal_phase to scores")
    elif 'tf4h_wyckoff_phase' in df.columns:
        # Fallback for older feature stores
        df['tf4h_fusion_score'] = df['tf4h_wyckoff_phase'].map(phase_score_map).fillna(0.0)
        print(f"   ✅ Mapped tf4h_wyckoff_phase to scores")
    else:
        # Last resort: Use structure alignment as proxy
        df['tf4h_fusion_score'] = df.get('tf4h_structure_alignment', 0.0)
        print(f"   ⚠️  No 4H phase column found, using tf4h_structure_alignment")

    # Smooth with 4-bar rolling mean to reduce noise
    df['tf4h_fusion_score'] = df['tf4h_fusion_score'].rolling(4, min_periods=1).mean()

    # volume_zscore: Z-score of volume with 20-bar lookback
    df['volume_zscore'] = (
        (df['volume'] - df['volume'].rolling(20, min_periods=1).mean()) /
        df['volume'].rolling(20, min_periods=1).std()
    ).fillna(0.0)

    print(f"   ✅ Added tf4h_fusion_score (range: {df['tf4h_fusion_score'].min():.3f} to {df['tf4h_fusion_score'].max():.3f})")
    print(f"   ✅ Added volume_zscore (range: {df['volume_zscore'].min():.2f} to {df['volume_zscore'].max():.2f})")

    # Save back to parquet
    print(f"\n💾 Saving patched feature store...")
    df.to_parquet(parquet_path)
    print(f"   Final shape: {df.shape[0]} bars × {df.shape[1]} features")
    print(f"\n✅ Phase 4 features added successfully!")

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python3 patch_phase4_features.py <parquet_file>")
        print("Example: python3 patch_phase4_features.py data/features_mtf/BTC_1H_2024-07-01_to_2024-09-30.parquet")
        sys.exit(1)

    parquet_path = sys.argv[1]
    if not Path(parquet_path).exists():
        print(f"❌ Error: File not found: {parquet_path}")
        sys.exit(1)

    add_phase4_features(parquet_path)
