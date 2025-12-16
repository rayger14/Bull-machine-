#!/usr/bin/env python3
"""
Add Derived Features to Feature Store

This script adds 4 missing archetype-required features that can be derived
from existing data in the feature store:
1. volatility_spike - ATR-based volatility spike detection
2. oversold - Regime-adaptive RSI oversold conditions
3. funding_reversal - Funding rate reversal detection
4. resilience - Price resilience/recovery metric

Usage:
    python bin/add_derived_features.py

Output:
    Updates the feature store parquet file with 4 new columns
"""

import pandas as pd
import numpy as np
from pathlib import Path
import argparse


def create_volatility_spike(df, window=20, threshold=1.5):
    """
    Detect volatility spikes using ATR

    Args:
        df: DataFrame with 'atr_14' column
        window: Rolling window for ATR mean (default: 20)
        threshold: Multiplier for spike detection (default: 1.5)

    Returns:
        Series: Binary indicator (1.0 = spike, 0.0 = normal)
    """
    atr_ma = df['atr_14'].rolling(window).mean()
    return (df['atr_14'] > atr_ma * threshold).astype(float)


def create_oversold(df, regime_adaptive=True):
    """
    Detect oversold conditions - can be regime-adaptive

    Args:
        df: DataFrame with 'rsi_14' and 'macro_regime' columns
        regime_adaptive: If True, use regime-specific thresholds

    Returns:
        Series: Binary indicator (1.0 = oversold, 0.0 = not oversold)
    """
    if regime_adaptive and 'macro_regime' in df.columns:
        # Regime-adaptive thresholds
        regime_thresholds = {
            'crisis': 25,      # More extreme threshold in crisis
            'risk_off': 30,    # Standard threshold in risk-off
            'neutral': 30,     # Standard threshold in neutral
            'risk_on': 35      # Higher threshold in risk-on (harder to be "oversold")
        }

        oversold = df.apply(
            lambda row: row['rsi_14'] < regime_thresholds.get(row['macro_regime'], 30),
            axis=1
        ).astype(float)
    else:
        # Simple fixed threshold
        oversold = (df['rsi_14'] < 30).astype(float)

    return oversold


def create_funding_reversal(df, extreme_threshold=-1.5, recovery_threshold=-1.0):
    """
    Detect funding rate reversals from extreme negative

    Identifies when funding rate recovers from extreme negative levels,
    which can signal capitulation and potential reversal.

    Args:
        df: DataFrame with 'funding_Z' column
        extreme_threshold: Z-score threshold for extreme negative (default: -1.5)
        recovery_threshold: Z-score threshold for recovery (default: -1.0)

    Returns:
        Series: Binary indicator (1.0 = reversal detected, 0.0 = no reversal)
    """
    if 'funding_Z' not in df.columns:
        print("WARNING: funding_Z not found, returning zeros")
        return pd.Series(0.0, index=df.index)

    reversal = (
        (df['funding_Z'].shift(1) < extreme_threshold) &
        (df['funding_Z'] > recovery_threshold)
    ).astype(float)

    return reversal


def create_resilience(df, method='intrabar'):
    """
    Measure price resilience/recovery

    Args:
        df: DataFrame with OHLC and 'atr_14' columns
        method: 'intrabar' or 'atr_normalized'

    Returns:
        Series: Resilience score (higher = more resilient)
    """
    if method == 'intrabar':
        # Intrabar recovery: how well price recovered from low
        # 0.0 = closed at low, 1.0 = closed at high
        bar_range = df['high'] - df['low']
        resilience = (df['close'] - df['low']) / bar_range.replace(0, np.nan)

    elif method == 'atr_normalized':
        # ATR-normalized bounce from recent low
        recent_low = df['low'].rolling(5).min()
        bounce = df['close'] - recent_low
        resilience = bounce / df['atr_14']

    else:
        raise ValueError(f"Unknown method: {method}")

    # Fill NaN with 0.5 (neutral resilience)
    return resilience.fillna(0.5)


def add_derived_features(df, verbose=True):
    """
    Add all 4 derived features to DataFrame

    Args:
        df: Input DataFrame with required base features
        verbose: Print progress information

    Returns:
        DataFrame: Input DataFrame with 4 new columns added
    """
    if verbose:
        print("\nAdding derived features...")
        print(f"Input shape: {df.shape}")

    # 1. Volatility Spike
    if verbose:
        print("\n1. Creating volatility_spike...")
    df['volatility_spike'] = create_volatility_spike(df)
    if verbose:
        spike_count = df['volatility_spike'].sum()
        spike_pct = (spike_count / len(df) * 100)
        print(f"   Detected {spike_count:,.0f} spikes ({spike_pct:.2f}% of data)")

    # 2. Oversold
    if verbose:
        print("\n2. Creating oversold (regime-adaptive)...")
    df['oversold'] = create_oversold(df, regime_adaptive=True)
    if verbose:
        oversold_count = df['oversold'].sum()
        oversold_pct = (oversold_count / len(df) * 100)
        print(f"   Detected {oversold_count:,.0f} oversold conditions ({oversold_pct:.2f}% of data)")
        if 'macro_regime' in df.columns:
            print("   Oversold by regime:")
            regime_oversold = df[df['oversold'] == 1.0]['macro_regime'].value_counts()
            for regime, count in regime_oversold.items():
                print(f"     - {regime}: {count:,}")

    # 3. Funding Reversal
    if verbose:
        print("\n3. Creating funding_reversal...")
    df['funding_reversal'] = create_funding_reversal(df)
    if verbose:
        reversal_count = df['funding_reversal'].sum()
        reversal_pct = (reversal_count / len(df) * 100)
        print(f"   Detected {reversal_count:,.0f} funding reversals ({reversal_pct:.2f}% of data)")

    # 4. Resilience
    if verbose:
        print("\n4. Creating resilience...")
    df['resilience'] = create_resilience(df, method='intrabar')
    if verbose:
        print(f"   Mean resilience: {df['resilience'].mean():.3f}")
        print(f"   Std resilience: {df['resilience'].std():.3f}")
        print(f"   Range: [{df['resilience'].min():.3f}, {df['resilience'].max():.3f}]")

    if verbose:
        print(f"\nOutput shape: {df.shape}")
        print(f"Added {df.shape[1] - 167} new columns")

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Add derived features to feature store"
    )
    parser.add_argument(
        '--input',
        type=str,
        default='data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet',
        help='Input feature store parquet file'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output parquet file (default: overwrite input)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Test feature creation without writing output'
    )

    args = parser.parse_args()

    # Resolve paths
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return 1

    output_path = Path(args.output) if args.output else input_path

    print("=" * 80)
    print("DERIVED FEATURES PIPELINE")
    print("=" * 80)
    print(f"\nInput:  {input_path}")
    print(f"Output: {output_path}")
    print(f"Mode:   {'DRY RUN' if args.dry_run else 'WRITE'}")

    # Load data
    print("\nLoading feature store...")
    df = pd.read_parquet(input_path)
    print(f"Loaded {len(df):,} rows, {len(df.columns):,} columns")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Check for required columns
    required_cols = ['close', 'high', 'low', 'open', 'atr_14', 'rsi_14', 'funding_Z', 'macro_regime']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"\nERROR: Missing required columns: {missing_cols}")
        return 1

    # Add derived features
    df_enriched = add_derived_features(df, verbose=True)

    # Verify new columns exist
    new_cols = ['volatility_spike', 'oversold', 'funding_reversal', 'resilience']
    for col in new_cols:
        if col not in df_enriched.columns:
            print(f"\nERROR: Failed to create column: {col}")
            return 1

    # Show sample
    print("\n" + "=" * 80)
    print("SAMPLE OUTPUT")
    print("=" * 80)
    sample_cols = ['close', 'rsi_14', 'atr_14'] + new_cols
    print(df_enriched[sample_cols].head(10).to_string())

    # Statistics
    print("\n" + "=" * 80)
    print("FEATURE STATISTICS")
    print("=" * 80)
    print(df_enriched[new_cols].describe().to_string())

    # Write output
    if args.dry_run:
        print("\n✓ DRY RUN COMPLETE - No files written")
        print(f"Would have written {len(df_enriched):,} rows to: {output_path}")
    else:
        print(f"\nWriting to: {output_path}")
        df_enriched.to_parquet(output_path)
        print(f"✓ Successfully wrote {len(df_enriched):,} rows, {len(df_enriched.columns):,} columns")

        # Verify written file
        verify_df = pd.read_parquet(output_path)
        print(f"✓ Verification: Read back {len(verify_df):,} rows, {len(verify_df.columns):,} columns")

    print("\n" + "=" * 80)
    print("COMPLETE")
    print("=" * 80)

    return 0


if __name__ == '__main__':
    exit(main())
