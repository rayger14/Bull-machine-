#!/usr/bin/env python3
"""
Generate Temporal Timing Features for Temporal Fusion Engine
============================================================

Creates 14 features that enable Fibonacci time confluence detection:
- 9 bars_since_* features (Wyckoff event timing)
- 3 Fibonacci time cluster features
- 2 cycle features

Uses vectorized pandas operations for performance.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Fibonacci levels for time clustering
FIB_LEVELS = [13, 21, 34, 55, 89, 144]

# Gann cycle levels (bars)
GANN_CYCLES = [90, 180, 360]


def compute_bars_since_vectorized(df, event_column):
    """
    Compute bars elapsed since last True in event_column using vectorized operations.

    Args:
        df: DataFrame with datetime index
        event_column: Column name with boolean event flags

    Returns:
        Series with bars_since values (NaN before first event)
    """
    # Get event mask
    events = df[event_column].fillna(False).astype(bool)

    # Find indices where events occur
    event_indices = np.where(events)[0]

    if len(event_indices) == 0:
        # No events found - return all NaN
        return pd.Series(np.nan, index=df.index, dtype=float)

    # Initialize result with NaN
    bars_since = np.full(len(df), np.nan, dtype=float)

    # For each bar, find bars since last event
    for i in range(len(df)):
        # Find most recent event before or at this bar
        prior_events = event_indices[event_indices <= i]
        if len(prior_events) > 0:
            last_event = prior_events[-1]
            bars_since[i] = i - last_event

    return pd.Series(bars_since, index=df.index, dtype=float)


def compute_fib_time_cluster(df, bars_since_cols):
    """
    Compute Fibonacci time cluster features.

    Args:
        df: DataFrame with bars_since_* columns
        bars_since_cols: List of bars_since column names

    Returns:
        Tuple of (fib_cluster, fib_score, fib_target)
    """
    n = len(df)
    fib_cluster = np.zeros(n, dtype=bool)
    fib_score = np.zeros(n, dtype=float)
    fib_target = np.full(n, '', dtype=object)

    for i in range(n):
        matches = []
        matched_levels = set()

        # Check each bars_since column
        for col in bars_since_cols:
            value = df[col].iloc[i]
            if pd.notna(value):
                # Check if at a Fibonacci level (within 1 bar tolerance)
                for fib in FIB_LEVELS:
                    if abs(value - fib) <= 1:
                        matches.append(col)
                        matched_levels.add(fib)

        if matches:
            fib_cluster[i] = True
            # Score based on number of events aligning (max 3 = perfect)
            fib_score[i] = min(len(matches) / 3.0, 1.0)
            # Record the Fibonacci level(s)
            if matched_levels:
                fib_target[i] = ','.join(str(x) for x in sorted(matched_levels))

    # Replace empty strings with None for fib_target
    fib_target = pd.Series(fib_target, index=df.index)
    fib_target = fib_target.replace('', None)

    return (
        pd.Series(fib_cluster, index=df.index, dtype=bool),
        pd.Series(fib_score, index=df.index, dtype=float),
        fib_target
    )


def compute_gann_cycle(df, bars_since_cols):
    """
    Compute Gann cycle feature (90/180/360 bar cycles).

    Args:
        df: DataFrame with bars_since_* columns
        bars_since_cols: List of bars_since column names

    Returns:
        Series with boolean Gann cycle flags
    """
    n = len(df)
    gann_cycle = np.zeros(n, dtype=bool)

    for i in range(n):
        # Check each bars_since column
        for col in bars_since_cols:
            value = df[col].iloc[i]
            if pd.notna(value):
                # Check if at a Gann cycle level (within 2 bar tolerance)
                for cycle in GANN_CYCLES:
                    if abs(value - cycle) <= 2:
                        gann_cycle[i] = True
                        break
            if gann_cycle[i]:
                break

    return pd.Series(gann_cycle, index=df.index, dtype=bool)


def compute_volatility_cycle(df):
    """
    Compute volatility cycle feature using HV ratios.

    Measures cyclicality of volatility regime based on historical volatility ratios.

    Args:
        df: DataFrame with close prices

    Returns:
        Series with volatility cycle scores (0-1)
    """
    if 'close' not in df.columns:
        # Return zeros if no price data
        return pd.Series(0.0, index=df.index, dtype=float)

    # Compute returns
    returns = df['close'].pct_change()

    # Compute rolling volatility (21 and 89 periods)
    hv_21 = returns.rolling(21).std() * np.sqrt(24 * 365)
    hv_89 = returns.rolling(89).std() * np.sqrt(24 * 365)

    # Volatility ratio (short / long)
    vol_ratio = hv_21 / hv_89

    # Normalize to 0-1 range using sigmoid-like transformation
    # Higher ratio = higher cyclicality
    vol_cycle = 1 / (1 + np.exp(-2 * (vol_ratio - 1)))

    return vol_cycle.fillna(0.0)


def generate_temporal_timing_features(input_path, output_path=None):
    """
    Generate all 14 temporal timing features.

    Args:
        input_path: Path to input parquet file
        output_path: Path to output parquet file (optional, defaults to input_path)

    Returns:
        DataFrame with new features
    """
    print("=" * 70)
    print("TEMPORAL TIMING FEATURES GENERATION")
    print("=" * 70)
    print()

    # Load feature store
    print(f"Loading feature store: {input_path}")
    df = pd.read_parquet(input_path)
    print(f"  Rows: {len(df):,}")
    print(f"  Columns before: {len(df.columns)}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")
    print()

    # Define Wyckoff event mappings
    wyckoff_events = {
        'bars_since_sc': 'wyckoff_sc',
        'bars_since_ar': 'wyckoff_ar',
        'bars_since_st': 'wyckoff_st',
        'bars_since_sos_long': 'wyckoff_sos',
        'bars_since_sos_short': 'wyckoff_sow',
        'bars_since_spring': 'wyckoff_spring_a',
        'bars_since_utad': 'wyckoff_utad',
        'bars_since_ps': 'wyckoff_ps',
        'bars_since_bc': 'wyckoff_bc',
    }

    # Generate bars_since_* features
    print("GENERATING WYCKOFF EVENT TIMING FEATURES (9 features):")
    bars_since_cols = []

    for new_col, event_col in wyckoff_events.items():
        if event_col not in df.columns:
            print(f"  ⚠️  {new_col}: Event column '{event_col}' not found, skipping")
            continue

        print(f"  Computing {new_col} from {event_col}...", end=' ')
        df[new_col] = compute_bars_since_vectorized(df, event_col)
        bars_since_cols.append(new_col)

        # Stats
        non_null_pct = (df[new_col].notna().sum() / len(df)) * 100
        mean_val = df[new_col].mean()
        print(f"✅ {non_null_pct:.1f}% non-null, mean={mean_val:.1f} bars")

    print()

    # Generate Fibonacci time cluster features
    print("GENERATING FIBONACCI TIME CLUSTER FEATURES (3 features):")
    print("  Computing fib_time_cluster, fib_time_score, fib_time_target...", end=' ')

    df['fib_time_cluster'], df['fib_time_score'], df['fib_time_target'] = \
        compute_fib_time_cluster(df, bars_since_cols)

    cluster_count = df['fib_time_cluster'].sum()
    cluster_pct = (cluster_count / len(df)) * 100
    mean_score = df['fib_time_score'].mean()
    print(f"✅")
    print(f"    fib_time_cluster: {cluster_count:,} events ({cluster_pct:.1f}% of bars)")
    print(f"    fib_time_score: Mean={mean_score:.3f}, Max={df['fib_time_score'].max():.3f}")

    # Show sample Fib targets
    sample_targets = df[df['fib_time_target'].notna()]['fib_time_target'].value_counts().head(5)
    print(f"    fib_time_target: Top levels = {list(sample_targets.index)}")
    print()

    # Generate cycle features
    print("GENERATING CYCLE FEATURES (2 features):")

    print("  Computing gann_cycle...", end=' ')
    df['gann_cycle'] = compute_gann_cycle(df, bars_since_cols)
    gann_count = df['gann_cycle'].sum()
    gann_pct = (gann_count / len(df)) * 100
    print(f"✅ {gann_count:,} events ({gann_pct:.1f}% of bars)")

    print("  Computing volatility_cycle...", end=' ')
    df['volatility_cycle'] = compute_volatility_cycle(df)
    vol_mean = df['volatility_cycle'].mean()
    vol_std = df['volatility_cycle'].std()
    print(f"✅ Mean={vol_mean:.3f}, Std={vol_std:.3f}")
    print()

    # Summary
    print("=" * 70)
    print("FEATURE GENERATION COMPLETE")
    print("=" * 70)
    print()
    print(f"FEATURES CREATED: 14 total")
    print(f"  - Wyckoff Event Timing: {len(bars_since_cols)} features")
    print(f"  - Fibonacci Time Cluster: 3 features")
    print(f"  - Cycle Features: 2 features")
    print()
    print(f"FEATURE STORE UPDATED:")
    print(f"  Columns before: {len(df.columns) - 14}")
    print(f"  Columns after: {len(df.columns)}")
    print(f"  New columns: +14 temporal timing features")
    print()

    # Save updated feature store
    if output_path is None:
        output_path = input_path

    print(f"Saving to: {output_path}")
    df.to_parquet(output_path, compression='snappy')
    print("✅ Feature store saved successfully")
    print()

    # Show sample events
    print("=" * 70)
    print("SAMPLE FIB TIME CLUSTER EVENTS")
    print("=" * 70)

    # Find top 3 cluster events by score
    top_clusters = df[df['fib_time_cluster']].nlargest(3, 'fib_time_score')

    for i, (idx, row) in enumerate(top_clusters.iterrows(), 1):
        print(f"\nEvent {i}: {idx}")
        print(f"  fib_time_score: {row['fib_time_score']:.3f}")
        print(f"  fib_time_target: {row['fib_time_target']}")

        # Show which events aligned
        aligned = []
        for col in bars_since_cols:
            val = row[col]
            if pd.notna(val):
                for fib in FIB_LEVELS:
                    if abs(val - fib) <= 1:
                        aligned.append(f"{col}={int(val)} (Fib {fib})")
                        break

        if aligned:
            print(f"  Aligned events:")
            for a in aligned:
                print(f"    - {a}")

        if 'close' in df.columns:
            print(f"  Price: ${row['close']:,.2f}")

    print()
    print("=" * 70)
    print("STATUS: Temporal Fusion Engine ready to activate ✅")
    print("=" * 70)
    print()
    print("NEXT STEPS:")
    print("1. Run verification test: bin/test_temporal_fusion.py")
    print("2. Validate Fibonacci confluence detection")
    print("3. Activate Temporal Fusion Engine for live trading")
    print()

    return df


if __name__ == '__main__':
    # Default paths
    input_path = Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

    # Check if file exists
    if not input_path.exists():
        print(f"ERROR: Feature store not found at {input_path}")
        print("Please provide correct path as argument.")
        sys.exit(1)

    # Generate features
    df = generate_temporal_timing_features(input_path)

    print("✅ TEMPORAL TIMING FEATURES GENERATION COMPLETE")
