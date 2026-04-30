#!/usr/bin/env python3
"""
Validate PTI fix impact on archetype detection.

Compares archetype detection rates before and after PTI fix.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np


def analyze_pti_distribution(df: pd.DataFrame, name: str):
    """Analyze PTI feature distributions."""

    print(f"\n{'=' * 70}")
    print(f"{name} - PTI Distribution Analysis")
    print('=' * 70)

    pti_cols = ['tf1h_pti_score', 'tf1h_pti_confidence', 'tf1d_pti_score']

    for col in pti_cols:
        if col not in df.columns:
            print(f"\n{col}: MISSING")
            continue

        print(f"\n{col}:")
        print(f"  Unique values: {df[col].nunique():,}")
        print(f"  Mean: {df[col].mean():.4f}")
        print(f"  Std: {df[col].std():.4f}")
        print(f"  Min: {df[col].min():.4f}")
        print(f"  Max: {df[col].max():.4f}")

        # Show distribution
        bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
        counts = pd.cut(df[col], bins=bins, include_lowest=True).value_counts(sort=False)

        print(f"  Distribution:")
        for interval, count in counts.items():
            pct = 100 * count / len(df)
            print(f"    {interval}: {count:,} ({pct:.1f}%)")


def simulate_archetype_detection(df: pd.DataFrame, name: str):
    """
    Simulate archetype detection rates based on PTI thresholds.

    Spring A requirements:
    - fusion_score >= 0.4
    - pti_score > 0.5
    - volume_spike >= 0.3 (approximated by volume_zscore > 1.0)

    Wick Trap K requirements:
    - fusion_score >= 0.4
    - pti_confidence > 0.6
    - wick_ratio > 2.0
    """

    print(f"\n{'=' * 70}")
    print(f"{name} - Archetype Detection Simulation")
    print('=' * 70)

    # Check required columns
    required_cols = ['tf1h_pti_score', 'tf1h_pti_confidence', 'fusion_score']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"\n⚠️  Missing columns: {missing}")
        return

    # Get fusion score (or approximate it)
    if 'fusion_score' in df.columns:
        fusion = df['fusion_score']
    elif 'fusion_total' in df.columns:
        fusion = df['fusion_total']
    else:
        print("⚠️  No fusion score column found")
        return

    # Spring A detection
    print("\n1. Spring A (Archetype A):")
    print("   Requirements:")
    print("     - fusion_score >= 0.4")
    print("     - tf1h_pti_score > 0.5")
    print("     - volume_zscore > 1.0 (volume spike)")

    # Count bars meeting each threshold
    fusion_ok = (fusion >= 0.4).sum()
    pti_ok = (df['tf1h_pti_score'] > 0.5).sum()

    # Approximate volume spike
    if 'volume_zscore' in df.columns:
        vol_ok = (df['volume_zscore'] > 1.0).sum()
    elif 'volume_z' in df.columns:
        vol_ok = (df['volume_z'] > 1.0).sum()
    else:
        vol_ok = len(df)  # Assume all pass if no volume column

    # Combined
    spring_a_signals = (
        (fusion >= 0.4) &
        (df['tf1h_pti_score'] > 0.5) &
        ((df.get('volume_zscore', 2.0) > 1.0) | (df.get('volume_z', 2.0) > 1.0))
    ).sum()

    print(f"\n   Threshold Analysis:")
    print(f"     fusion >= 0.4: {fusion_ok:,} bars ({100*fusion_ok/len(df):.1f}%)")
    print(f"     pti_score > 0.5: {pti_ok:,} bars ({100*pti_ok/len(df):.1f}%)")
    print(f"     volume_spike: {vol_ok:,} bars ({100*vol_ok/len(df):.1f}%)")
    print(f"\n   ✅ Combined Spring A signals: {spring_a_signals:,} bars ({100*spring_a_signals/len(df):.1f}%)")

    # Wick Trap K detection
    print("\n2. Wick Trap K (Archetype K):")
    print("   Requirements:")
    print("     - fusion_score >= 0.4")
    print("     - tf1h_pti_confidence > 0.6")
    print("     - wick_ratio > 2.0")

    # Count bars meeting each threshold
    pti_conf_ok = (df['tf1h_pti_confidence'] > 0.6).sum()

    # Approximate wick ratio
    if 'wick_ratio' in df.columns:
        wick_ok = (df['wick_ratio'] > 2.0).sum()
    else:
        wick_ok = len(df)  # Assume all pass if no wick column

    # Combined
    wick_trap_k_signals = (
        (fusion >= 0.4) &
        (df['tf1h_pti_confidence'] > 0.6) &
        (df.get('wick_ratio', 3.0) > 2.0)
    ).sum()

    print(f"\n   Threshold Analysis:")
    print(f"     fusion >= 0.4: {fusion_ok:,} bars ({100*fusion_ok/len(df):.1f}%)")
    print(f"     pti_confidence > 0.6: {pti_conf_ok:,} bars ({100*pti_conf_ok/len(df):.1f}%)")
    print(f"     wick_ratio > 2.0: {wick_ok:,} bars ({100*wick_ok/len(df):.1f}%)")
    print(f"\n   ✅ Combined Wick Trap K signals: {wick_trap_k_signals:,} bars ({100*wick_trap_k_signals/len(df):.1f}%)")


def main():
    """Compare before and after PTI fix."""

    print("=" * 70)
    print("PTI Fix Impact Validation")
    print("=" * 70)

    # Load both versions
    before_path = Path('data/features_mtf/BTC_1H_FUSION_FIXED.parquet')
    after_path = Path('data/features_mtf/BTC_1H_PTI_FIXED_20260201.parquet')

    print(f"\nBefore: {before_path}")
    print(f"After:  {after_path}")

    if not before_path.exists():
        print(f"\n❌ Before file not found: {before_path}")
        return 1

    if not after_path.exists():
        print(f"\n❌ After file not found: {after_path}")
        return 1

    # Load data
    print("\nLoading data...")
    df_before = pd.read_parquet(before_path)
    df_after = pd.read_parquet(after_path)

    print(f"✅ Loaded {len(df_before):,} bars")

    # Analyze distributions
    analyze_pti_distribution(df_before, "BEFORE FIX")
    analyze_pti_distribution(df_after, "AFTER FIX")

    # Simulate archetype detection
    simulate_archetype_detection(df_before, "BEFORE FIX")
    simulate_archetype_detection(df_after, "AFTER FIX")

    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)

    # PTI scores
    pti_before = (df_before['tf1h_pti_score'] > 0.5).sum()
    pti_after = (df_after['tf1h_pti_score'] > 0.5).sum()

    print(f"\nBars with tf1h_pti_score > 0.5:")
    print(f"  Before: {pti_before:,} ({100*pti_before/len(df_before):.1f}%)")
    print(f"  After:  {pti_after:,} ({100*pti_after/len(df_after):.1f}%)")
    print(f"  Change: {pti_after - pti_before:+,} bars")

    # PTI confidence
    pti_conf_before = (df_before['tf1h_pti_confidence'] > 0.6).sum()
    pti_conf_after = (df_after['tf1h_pti_confidence'] > 0.6).sum()

    print(f"\nBars with tf1h_pti_confidence > 0.6:")
    print(f"  Before: {pti_conf_before:,} ({100*pti_conf_before/len(df_before):.1f}%)")
    print(f"  After:  {pti_conf_after:,} ({100*pti_conf_after/len(df_after):.1f}%)")
    print(f"  Change: {pti_conf_after - pti_conf_before:+,} bars")

    print("\n✅ Validation complete!")

    return 0


if __name__ == '__main__':
    sys.exit(main())
