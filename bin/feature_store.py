#!/usr/bin/env python3
"""
Feature Store CLI - Single entry point for building and managing feature stores.

Usage:
    # Build full feature store (all tiers)
    python3 bin/feature_store.py --asset BTC --start 2022-01-01 --end 2024-12-31

    # Build specific tiers only
    python3 bin/feature_store.py --asset BTC --start 2022-01-01 --end 2024-12-31 --tiers 1,2

    # Load and validate existing store
    python3 bin/feature_store.py --asset BTC --start 2022-01-01 --end 2024-12-31 --load-only

This replaces scattered builder scripts with a single façade.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
from datetime import datetime

from engine.features.builder import FeatureStoreBuilder, BuildSpec


def main():
    parser = argparse.ArgumentParser(
        description="Feature Store Builder - Single entry point for all feature operations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build full BTC feature store (all tiers)
  %(prog)s --asset BTC --start 2022-01-01 --end 2024-12-31

  # Build only tiers 1 and 2 (skip regime)
  %(prog)s --asset BTC --start 2022-01-01 --end 2024-12-31 --tiers 1,2

  # Build without validation (faster, not recommended)
  %(prog)s --asset BTC --start 2022-01-01 --end 2024-12-31 --no-validate

  # Load and validate existing store
  %(prog)s --asset BTC --start 2022-01-01 --end 2024-12-31 --load-only
        """
    )

    parser.add_argument(
        '--asset',
        type=str,
        required=True,
        help='Asset symbol (BTC, ETH, SPY, etc.)'
    )

    parser.add_argument(
        '--start',
        type=str,
        required=True,
        help='Start date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end',
        type=str,
        required=True,
        help='End date (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--tiers',
        type=str,
        default='1,2,3',
        help='Comma-separated tiers to build (default: 1,2,3)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default='data/feature_store',
        help='Output directory (default: data/feature_store)'
    )

    parser.add_argument(
        '--resolution',
        type=str,
        default='1H',
        help='Time resolution (default: 1H)'
    )

    parser.add_argument(
        '--no-validate',
        action='store_true',
        help='Skip validation (not recommended)'
    )

    parser.add_argument(
        '--no-normalize',
        action='store_true',
        help='Skip column name normalization'
    )

    parser.add_argument(
        '--load-only',
        action='store_true',
        help='Load and validate existing store (do not build)'
    )

    args = parser.parse_args()

    # Parse tiers
    tiers = [int(t.strip()) for t in args.tiers.split(',')]

    # Create builder
    builder = FeatureStoreBuilder()

    start_time = datetime.now()

    try:
        if args.load_only:
            # Load existing store
            print("=" * 70)
            print("LOAD & VALIDATE EXISTING FEATURE STORE")
            print("=" * 70)
            print()

            df = builder.load(
                args.asset,
                args.start,
                args.end,
                validate=not args.no_validate
            )

            print()
            print("=" * 70)
            print("✅ LOAD COMPLETE")
            print("=" * 70)
            print(f"Shape: {df.shape}")
            print(f"Columns: {len(df.columns)}")
            print(f"Date range: {df.index.min()} → {df.index.max()}")

        else:
            # Build new store
            spec = BuildSpec(
                asset=args.asset,
                start=args.start,
                end=args.end,
                tiers=tiers,
                resolution=args.resolution,
                output_dir=args.output,
                validate=not args.no_validate,
                normalize=not args.no_normalize
            )

            df, report = builder.build(spec)

            # Print summary
            print()
            print("📊 BUILD SUMMARY")
            print("-" * 70)
            print(f"Asset: {args.asset}")
            print(f"Period: {args.start} → {args.end}")
            print(f"Tiers: {tiers}")
            print(f"Final shape: {df.shape[0]} rows × {df.shape[1]} columns")

            if 'parameter_bounds' in report and report['parameter_bounds']:
                print()
                print("📐 Parameter Bounds (data-derived):")
                for feature, (min_val, max_val) in sorted(report['parameter_bounds'].items()):
                    print(f"  {feature:30s} [{min_val:8.4f}, {max_val:8.4f}]")

            if 'validation' in report:
                val = report['validation']
                if val['warnings']:
                    print()
                    print("⚠️  Warnings:")
                    for warning in val['warnings']:
                        print(f"  - {warning}")

        elapsed = (datetime.now() - start_time).total_seconds()
        print()
        print(f"⏱️  Elapsed time: {elapsed:.1f}s")

        return 0

    except Exception as e:
        print()
        print("=" * 70)
        print("❌ BUILD FAILED")
        print("=" * 70)
        print(f"Error: {e}")
        print()

        import traceback
        traceback.print_exc()

        return 1


if __name__ == '__main__':
    sys.exit(main())
