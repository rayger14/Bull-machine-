#!/usr/bin/env python3
"""
Add S1 V2 features to existing feature store.

This script enriches the feature store with S1 V2 capitulation features without
rebuilding the entire feature set. It loads existing parquet files, adds the new
features, and saves them back.

S1 V2 Features Added:
- liquidity_drain_pct: Relative drain vs 7d average
- liquidity_velocity: Rate of change over 6 bars
- liquidity_persistence: Consecutive drain bars
- capitulation_depth: Drawdown from 30d high
- crisis_composite: Macro stress indicator
- volume_climax_last_3b: Max volume in last 3 bars
- wick_exhaustion_last_3b: Max wick rejection in last 3 bars
"""

import pandas as pd
import argparse
from pathlib import Path
from datetime import datetime
from engine.strategies.archetypes.bear.liquidity_vacuum_runtime import apply_liquidity_vacuum_enrichment

def add_s1_v2_features(input_path: str, output_path: str = None, backup: bool = True):
    """
    Add S1 V2 features to feature store parquet file.

    Args:
        input_path: Path to input parquet file
        output_path: Path to output file (default: overwrite input)
        backup: Create backup before overwriting (default: True)
    """
    input_file = Path(input_path)

    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    print(f"Loading feature store: {input_path}")
    df = pd.read_parquet(input_path)

    print(f"Loaded {len(df):,} rows")
    print(f"Columns before: {len(df.columns)}")
    print(f"Date range: {df.index.min()} to {df.index.max()}")

    # Check if S1 V2 features already exist
    v2_features = [
        'liquidity_drain_pct',
        'liquidity_velocity',
        'liquidity_persistence',
        'capitulation_depth',
        'crisis_composite',
        'volume_climax_last_3b',
        'wick_exhaustion_last_3b'
    ]

    existing_v2 = [f for f in v2_features if f in df.columns]
    if existing_v2:
        print(f"\nWARNING: {len(existing_v2)} S1 V2 features already exist:")
        for feat in existing_v2:
            print(f"  - {feat}")
        response = input("\nOverwrite existing features? (y/n): ")
        if response.lower() != 'y':
            print("Aborted.")
            return
        # Drop existing features
        df = df.drop(columns=existing_v2)

    print("\nApplying S1 V2 enrichment...")
    df_enriched = apply_liquidity_vacuum_enrichment(df)

    # Verify new features were added
    new_features = [f for f in v2_features if f in df_enriched.columns]
    print(f"\nAdded {len(new_features)} S1 V2 features:")
    for feat in new_features:
        non_null = df_enriched[feat].notna().sum()
        pct = (non_null / len(df_enriched)) * 100
        print(f"  - {feat}: {non_null:,}/{len(df_enriched):,} ({pct:.1f}%) non-null")

    print(f"\nColumns after: {len(df_enriched.columns)}")

    # Determine output path
    if output_path is None:
        output_path = input_path

    output_file = Path(output_path)

    # Create backup if overwriting
    if backup and output_file.exists() and str(output_file) == str(input_file):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup_path = output_file.parent / f"{output_file.stem}_backup_{timestamp}{output_file.suffix}"
        print(f"\nCreating backup: {backup_path}")
        df.to_parquet(backup_path)

    # Save enriched dataframe
    print(f"Saving enriched features to: {output_path}")
    df_enriched.to_parquet(output_path)

    # Get file sizes
    if backup and output_file.exists():
        original_size = input_file.stat().st_size / (1024 * 1024)
        new_size = output_file.stat().st_size / (1024 * 1024)
        print(f"\nFile size: {original_size:.2f}MB -> {new_size:.2f}MB ({new_size - original_size:+.2f}MB)")

    print("\n" + "="*80)
    print("S1 V2 FEATURE ENRICHMENT COMPLETE")
    print("="*80)
    print(f"Output: {output_path}")
    print(f"Features added: {', '.join(new_features)}")
    print("\nNext step: Run validation backtest with configs/s1_v2_quick_fix.json")


def main():
    parser = argparse.ArgumentParser(
        description='Add S1 V2 features to feature store',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Add features to main feature store (creates backup)
  python bin/add_s1_v2_features.py data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet

  # Add features and save to new file
  python bin/add_s1_v2_features.py \\
    data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \\
    data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31_with_s1v2.parquet

  # Add features without backup
  python bin/add_s1_v2_features.py \\
    data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet \\
    --no-backup
        """
    )

    parser.add_argument(
        'input',
        help='Path to input parquet file'
    )

    parser.add_argument(
        'output',
        nargs='?',
        default=None,
        help='Path to output parquet file (default: overwrite input)'
    )

    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='Skip backup when overwriting'
    )

    args = parser.parse_args()

    try:
        add_s1_v2_features(
            input_path=args.input,
            output_path=args.output,
            backup=not args.no_backup
        )
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == '__main__':
    exit(main())
