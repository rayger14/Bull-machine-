#!/usr/bin/env python3
"""
Validate that MTF fusion fix unblocked archetypes G and M.

Quick diagnostic to verify:
1. MTF fusion scores are no longer frozen
2. Archetypes G and M can detect signals
3. Feature distributions look reasonable
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

def check_archetype_g_signals(df: pd.DataFrame) -> dict:
    """
    Check if Archetype G (Re-Accumulation) can now detect signals.

    Archetype G logic (simplified):
    - Requires high tf4h_fusion_score (was frozen at 0.1)
    - Wyckoff re-accumulation pattern
    - Market structure confirmation
    """
    # Check if tf4h_fusion_score has variance
    if df['tf4h_fusion_score'].nunique() < 100:
        return {
            'archetype': 'G',
            'status': 'BLOCKED',
            'reason': 'tf4h_fusion_score still frozen',
            'signal_count': 0
        }

    # Simplified G detection (actual logic in archetype_model.py)
    potential_signals = (
        (df['tf4h_fusion_score'] > 0.5) &  # High 4h fusion
        (df['fusion_wyckoff'] > 0.3) &     # Wyckoff signal present
        (df.get('k2_fusion_score', 0) > 0.2)  # Composite confirmation
    )

    signal_count = potential_signals.sum()

    return {
        'archetype': 'G',
        'status': 'UNBLOCKED' if signal_count > 0 else 'NO_SIGNALS',
        'signal_count': signal_count,
        'signal_pct': (signal_count / len(df)) * 100,
        'feature_stats': {
            'tf4h_fusion_score': {
                'unique': df['tf4h_fusion_score'].nunique(),
                'mean': df['tf4h_fusion_score'].mean(),
                'std': df['tf4h_fusion_score'].std()
            }
        }
    }

def check_archetype_m_signals(df: pd.DataFrame) -> dict:
    """
    Check if Archetype M (Ratio Coil) can now detect signals.

    Archetype M logic (simplified):
    - Requires stable tf1d_fusion_score (was frozen at 0.25)
    - Ratio coil pattern (price compression)
    - Multi-timeframe alignment
    """
    # Check if tf1d_fusion_score has variance
    if df['tf1d_fusion_score'].nunique() < 100:
        return {
            'archetype': 'M',
            'status': 'BLOCKED',
            'reason': 'tf1d_fusion_score still frozen',
            'signal_count': 0
        }

    # Simplified M detection
    potential_signals = (
        (df['tf1d_fusion_score'] > 0.3) &  # Daily fusion signal
        (df.get('k2_fusion_score', 0) > 0.15) &  # Composite
        (df['fusion_momentum'] < 0.4)  # Low momentum (coil)
    )

    signal_count = potential_signals.sum()

    return {
        'archetype': 'M',
        'status': 'UNBLOCKED' if signal_count > 0 else 'NO_SIGNALS',
        'signal_count': signal_count,
        'signal_pct': (signal_count / len(df)) * 100,
        'feature_stats': {
            'tf1d_fusion_score': {
                'unique': df['tf1d_fusion_score'].nunique(),
                'mean': df['tf1d_fusion_score'].mean(),
                'std': df['tf1d_fusion_score'].std()
            },
            'k2_fusion_score': {
                'unique': df.get('k2_fusion_score', pd.Series([0])).nunique(),
                'mean': df.get('k2_fusion_score', pd.Series([0])).mean(),
                'std': df.get('k2_fusion_score', pd.Series([0])).std()
            }
        }
    }

def plot_mtf_distributions(df: pd.DataFrame, output_dir: Path):
    """Plot MTF fusion score distributions to visualize fix."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('MTF Fusion Score Distributions (After Fix)', fontsize=16)

    # tf4h_fusion_score
    axes[0, 0].hist(df['tf4h_fusion_score'], bins=50, edgecolor='black', alpha=0.7)
    axes[0, 0].set_title(f'tf4h_fusion_score\n({df["tf4h_fusion_score"].nunique()} unique values)')
    axes[0, 0].set_xlabel('Score')
    axes[0, 0].set_ylabel('Frequency')
    axes[0, 0].axvline(df['tf4h_fusion_score'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 0].legend()

    # tf1d_fusion_score
    axes[0, 1].hist(df['tf1d_fusion_score'], bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[0, 1].set_title(f'tf1d_fusion_score\n({df["tf1d_fusion_score"].nunique()} unique values)')
    axes[0, 1].set_xlabel('Score')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].axvline(df['tf1d_fusion_score'].mean(), color='red', linestyle='--', label='Mean')
    axes[0, 1].legend()

    # k2_fusion_score
    if 'k2_fusion_score' in df.columns:
        axes[1, 0].hist(df['k2_fusion_score'], bins=50, edgecolor='black', alpha=0.7, color='green')
        axes[1, 0].set_title(f'k2_fusion_score\n({df["k2_fusion_score"].nunique()} unique values)')
        axes[1, 0].set_xlabel('Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].axvline(df['k2_fusion_score'].mean(), color='red', linestyle='--', label='Mean')
        axes[1, 0].legend()

    # Time series of tf4h_fusion_score (sample)
    sample = df.tail(1000)  # Last 1000 rows
    axes[1, 1].plot(sample.index, sample['tf4h_fusion_score'], linewidth=0.5, alpha=0.7)
    axes[1, 1].set_title('tf4h_fusion_score (Recent 1000 bars)')
    axes[1, 1].set_xlabel('Timestamp')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    output_path = output_dir / 'mtf_fusion_distributions.png'
    plt.savefig(output_path, dpi=150)
    print(f"   Saved distribution plot: {output_path}")

def main():
    print("=" * 80)
    print("VALIDATING MTF FUSION FIX")
    print("=" * 80)

    # Load fixed feature store
    input_path = Path("data/features_mtf/BTC_1H_MTF_FIXED_20260201.parquet")
    print(f"\n1. Loading fixed feature store: {input_path}")

    if not input_path.exists():
        print(f"   ERROR: File not found: {input_path}")
        print("   Run fix_frozen_mtf_fusion.py first")
        return

    df = pd.read_parquet(input_path)
    print(f"   Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   Date range: {df.index.min()} to {df.index.max()}")

    # Check MTF fusion scores
    print("\n2. Verifying MTF fusion scores are unfrozen:")
    print("-" * 80)

    mtf_features = ['tf4h_fusion_score', 'tf1d_fusion_score', 'k2_fusion_score']
    for feature in mtf_features:
        if feature in df.columns:
            unique = df[feature].nunique()
            status = "UNFROZEN" if unique >= 100 else "STILL_FROZEN"
            print(f"   {feature:25s} | {unique:6d} unique | [{status}]")
        else:
            print(f"   {feature:25s} | NOT FOUND")

    # Check Archetype G
    print("\n3. Checking Archetype G (Re-Accumulation):")
    print("-" * 80)

    g_results = check_archetype_g_signals(df)
    print(f"   Status: {g_results['status']}")
    print(f"   Potential signals: {g_results['signal_count']:,} ({g_results.get('signal_pct', 0):.2f}%)")

    if 'feature_stats' in g_results:
        print(f"   tf4h_fusion_score stats:")
        stats = g_results['feature_stats']['tf4h_fusion_score']
        print(f"      Unique values: {stats['unique']:,}")
        print(f"      Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")

    # Check Archetype M
    print("\n4. Checking Archetype M (Ratio Coil):")
    print("-" * 80)

    m_results = check_archetype_m_signals(df)
    print(f"   Status: {m_results['status']}")
    print(f"   Potential signals: {m_results['signal_count']:,} ({m_results.get('signal_pct', 0):.2f}%)")

    if 'feature_stats' in m_results:
        print(f"   tf1d_fusion_score stats:")
        stats = m_results['feature_stats']['tf1d_fusion_score']
        print(f"      Unique values: {stats['unique']:,}")
        print(f"      Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")

        print(f"   k2_fusion_score stats:")
        stats = m_results['feature_stats']['k2_fusion_score']
        print(f"      Unique values: {stats['unique']:,}")
        print(f"      Mean: {stats['mean']:.6f}, Std: {stats['std']:.6f}")

    # Generate plots
    print("\n5. Generating distribution plots:")
    print("-" * 80)

    output_dir = Path("data/features_mtf")
    try:
        plot_mtf_distributions(df, output_dir)
    except Exception as e:
        print(f"   Warning: Could not generate plots: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    all_unfrozen = all(
        df[f].nunique() >= 100 for f in mtf_features if f in df.columns
    )

    print(f"\nMTF Fusion Scores: {'✅ ALL UNFROZEN' if all_unfrozen else '❌ STILL FROZEN'}")
    print(f"Archetype G: {g_results['status']} ({g_results['signal_count']:,} signals)")
    print(f"Archetype M: {m_results['status']} ({m_results['signal_count']:,} signals)")

    if all_unfrozen and g_results['status'] != 'BLOCKED' and m_results['status'] != 'BLOCKED':
        print("\n✅ VALIDATION PASSED - Archetypes G and M unblocked")
    else:
        print("\n❌ VALIDATION FAILED - Check details above")

    print("=" * 80)

if __name__ == "__main__":
    main()
