#!/usr/bin/env python3
"""
Feature Store Contract Validator - Pre-flight checks before backtest/Optuna.

This validator enforces contracts and fails fast with human-readable reports.
Run this BEFORE launching any backtest or optimization to catch data issues early.

Usage:
    # Validate existing feature store
    python3 bin/validate_feature_store.py data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet

    # Validate with specific tier
    python3 bin/validate_feature_store.py data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet --tier 3

    # Strict mode (warnings = errors)
    python3 bin/validate_feature_store.py data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet --strict

    # Show parameter bounds for optimization
    python3 bin/validate_feature_store.py data/cached/btc_features_2022-01-01_2024-12-31_cached.parquet --bounds
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import pandas as pd
from datetime import datetime

from engine.features.validate import FeatureValidator, validate_feature_store
from engine.features.registry import get_registry


def main():
    parser = argparse.ArgumentParser(
        description="Feature Store Contract Validator - Pre-flight checks before backtest/Optuna",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contract checks:
  • Required columns present
  • Data types match specification
  • Values within valid ranges
  • Index is DatetimeIndex (not RangeIndex!)
  • No duplicates in index
  • No nulls in required columns
  • OHLC consistency (high >= low, etc.)
  • Regime probabilities sum to ~1.0

Exit codes:
  0 = All checks passed
  1 = Validation failed (errors found)
  2 = File not found or load error

This validator saves hours by catching issues BEFORE optimization runs.
        """
    )

    parser.add_argument(
        'path',
        type=str,
        help='Path to feature store parquet file'
    )

    parser.add_argument(
        '--tier',
        type=int,
        default=None,
        help='Expected tier level (1, 2, or 3). Auto-detected if not specified.'
    )

    parser.add_argument(
        '--strict',
        action='store_true',
        help='Treat warnings as errors'
    )

    parser.add_argument(
        '--normalize',
        action='store_true',
        help='Normalize column names via registry before validation'
    )

    parser.add_argument(
        '--bounds',
        action='store_true',
        help='Compute and display parameter bounds for optimization'
    )

    parser.add_argument(
        '--features',
        type=str,
        default='tf4h_fusion_score,adx_14,rsi_14,atr_20,liquidity_score',
        help='Comma-separated features for parameter bounds (default: common optimization features)'
    )

    args = parser.parse_args()

    print("=" * 70)
    print("FEATURE STORE CONTRACT VALIDATION")
    print("=" * 70)
    print(f"File: {args.path}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Check if file exists
    path = Path(args.path)
    if not path.exists():
        print(f"❌ ERROR: File not found: {path}")
        return 2

    try:
        # Load dataframe
        print("▶ Loading dataframe...")
        df = pd.read_parquet(path)
        print(f"  ✓ Loaded {len(df)} rows, {len(df.columns)} columns")

        # Check index type
        print(f"  → Index type: {type(df.index).__name__}")
        if not isinstance(df.index, pd.DatetimeIndex):
            print(f"  ⚠️  Index is {type(df.index).__name__}, not DatetimeIndex")
            if 'timestamp' in df.columns:
                print(f"  → Converting timestamp column to DatetimeIndex...")
                df = df.set_index('timestamp')
                df.index = pd.to_datetime(df.index)
                print(f"  ✓ Converted to DatetimeIndex")

        # Auto-detect tier if not specified
        tier = args.tier
        if tier is None:
            if 'regime_label' in df.columns:
                tier = 3
            elif 'tf4h_fusion_score' in df.columns:
                tier = 2
            else:
                tier = 1
            print(f"  → Auto-detected tier: {tier}")

        # Get validator
        validator = FeatureValidator()

        # Normalize if requested
        if args.normalize:
            print("\n▶ Normalizing column names via registry...")
            df, _ = validator.normalize_and_validate(df, tier, strict=args.strict)
            print("  ✓ Normalized to canonical names")

        # Validate
        print("\n▶ Running contract validation...")
        result = validator.validate(df, tier, strict=args.strict)

        # Print result
        print()
        print(result)

        # Compute parameter bounds if requested
        if args.bounds:
            print()
            print("=" * 70)
            print("PARAMETER BOUNDS (data-derived)")
            print("=" * 70)

            features = [f.strip() for f in args.features.split(',')]
            features = [f for f in features if f in df.columns]

            if not features:
                print("⚠️  No specified features found in dataframe")
            else:
                bounds = validator.compute_parameter_bounds(df, features)

                print()
                print("Use these bounds in Optuna to prevent ranges that never trigger:")
                print()
                for feature, (min_val, max_val) in sorted(bounds.items()):
                    print(f"  {feature:30s} [{min_val:8.4f}, {max_val:8.4f}]")
                    print(f"    → suggest_float('{feature}', {min_val:.4f}, {max_val:.4f})")
                    print()

        # Print file info
        print()
        print("=" * 70)
        print("FILE INFO")
        print("=" * 70)
        print(f"Path: {path}")
        print(f"Size: {path.stat().st_size / 1024 / 1024:.1f} MB")
        print(f"Modified: {datetime.fromtimestamp(path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Date range: {df.index.min()} → {df.index.max()}")
        print(f"Memory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.1f} MB")

        # Print regime distribution if present
        if 'regime_label' in df.columns:
            print()
            print("Regime distribution:")
            for regime, count in df['regime_label'].value_counts().items():
                pct = count / len(df) * 100
                print(f"  {regime:12s} {count:6d} bars ({pct:5.1f}%)")

        print("=" * 70)

        # Return exit code
        if result.passed:
            print()
            print("✅ VALIDATION PASSED - Safe to use in backtest/Optuna")
            return 0
        else:
            print()
            print("❌ VALIDATION FAILED - Fix errors before using")
            return 1

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ VALIDATION ERROR")
        print("=" * 70)
        print(f"Error: {e}")
        print()

        import traceback
        traceback.print_exc()

        return 2


if __name__ == '__main__':
    sys.exit(main())
