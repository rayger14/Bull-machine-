#!/usr/bin/env python3
"""
Feature Store Health Audit

Validates that a rebuilt feature store has:
1. Expected column count (schema parity)
2. Non-flat variance in critical fusion components
3. Low NaN rates in signal columns
4. Correct date span coverage
"""

import argparse
import pandas as pd
import numpy as np
import sys
from pathlib import Path


def audit_feature_store(asset: str, year: int,
                       require_columns: int = 80,
                       check_variance: list = None,
                       nan_threshold: float = 0.001):
    """
    Audit a feature store for health metrics.

    Args:
        asset: Asset symbol (BTC, ETH, SPY)
        year: Year to check (2024)
        require_columns: Minimum expected column count
        check_variance: List of columns requiring non-flat variance
        nan_threshold: Max allowed NaN rate (default 0.1%)

    Returns:
        bool: True if all checks pass
    """

    # Build expected path
    store_path = Path(f"data/features_mtf/{asset}_1H_{year}-01-01_to_{year}-12-31.parquet")

    if not store_path.exists():
        print(f"❌ FAIL: Feature store not found at {store_path}")
        return False

    print(f"\n{'='*60}")
    print(f"Feature Store Health Audit: {asset} {year}")
    print(f"{'='*60}\n")

    # Load store
    print(f"Loading {store_path.name}...")
    df = pd.read_parquet(store_path)

    passed = True

    # Check 1: Column count
    print(f"\n1️⃣  Schema Parity Check")
    print(f"   Columns: {len(df.columns)}")
    if len(df.columns) >= require_columns:
        print(f"   ✅ PASS: {len(df.columns)} >= {require_columns}")
    else:
        print(f"   ❌ FAIL: {len(df.columns)} < {require_columns} (missing columns)")
        passed = False

    # Check 2: Date span
    print(f"\n2️⃣  Date Span Check")
    if 'timestamp' in df.columns:
        start = pd.to_datetime(df['timestamp'].min())
        end = pd.to_datetime(df['timestamp'].max())
        days = (end - start).days
        print(f"   Range: {start.date()} to {end.date()} ({days} days)")
        if days >= 360:  # Allow for some missing days
            print(f"   ✅ PASS: Full-year coverage ({days} days)")
        else:
            print(f"   ❌ FAIL: Incomplete year ({days} days < 360)")
            passed = False
    else:
        print(f"   ⚠️  SKIP: No timestamp column")

    # Check 3: Variance in critical columns
    if check_variance:
        print(f"\n3️⃣  Variance Check (fusion components)")
        for col in check_variance:
            if col not in df.columns:
                print(f"   ❌ {col:40} MISSING")
                passed = False
            else:
                var = df[col].var()
                mean = df[col].mean()
                std = df[col].std()

                if var > 0.001:  # Non-trivial variance
                    print(f"   ✅ {col:40} var={var:.6f}, mean={mean:.3f}, std={std:.3f}")
                elif var > 0.0:
                    print(f"   ⚠️  {col:40} LOW VARIANCE: {var:.6f} (may be flat)")
                    passed = False
                else:
                    print(f"   ❌ {col:40} FLATLINED: var={var:.6f}")
                    passed = False

    # Check 4: NaN rates
    print(f"\n4️⃣  NaN Rate Check")
    critical_cols = check_variance if check_variance else []
    critical_cols = [c for c in critical_cols if c in df.columns]

    if critical_cols:
        for col in critical_cols:
            nan_rate = df[col].isna().mean()
            if nan_rate <= nan_threshold:
                print(f"   ✅ {col:40} NaN rate: {nan_rate:.4%}")
            else:
                print(f"   ❌ {col:40} NaN rate: {nan_rate:.4%} (> {nan_threshold:.4%})")
                passed = False
    else:
        print(f"   ⚠️  SKIP: No variance check columns specified")

    # Check 5: k2_fusion_score specific check
    print(f"\n5️⃣  K2 Fusion Score Check")
    if 'k2_fusion_score' in df.columns:
        k2_mean = df['k2_fusion_score'].mean()
        k2_std = df['k2_fusion_score'].std()
        k2_var = df['k2_fusion_score'].var()
        k2_min = df['k2_fusion_score'].min()
        k2_max = df['k2_fusion_score'].max()

        print(f"   Mean:     {k2_mean:.6f}")
        print(f"   Std:      {k2_std:.6f}")
        print(f"   Variance: {k2_var:.6f}")
        print(f"   Range:    {k2_min:.3f} to {k2_max:.3f}")

        if k2_var > 0.001:
            print(f"   ✅ PASS: Real variance (not flatlined)")
        elif k2_var > 0.0:
            print(f"   ⚠️  WARNING: Low variance {k2_var:.6f}")
        else:
            print(f"   ❌ FAIL: FLATLINED at constant {k2_mean:.3f}")
            passed = False
    else:
        print(f"   ⚠️  SKIP: k2_fusion_score not in store")

    # Summary
    print(f"\n{'='*60}")
    if passed:
        print(f"✅ AUDIT PASSED: Feature store is healthy")
    else:
        print(f"❌ AUDIT FAILED: Fix issues before using this store")
    print(f"{'='*60}\n")

    return passed


def main():
    parser = argparse.ArgumentParser(description="Audit feature store health")
    parser.add_argument("--asset", required=True, help="Asset symbol (BTC, ETH, SPY)")
    parser.add_argument("--year", type=int, default=2024, help="Year to check")
    parser.add_argument("--require-columns", type=int, default=80, help="Min column count")
    parser.add_argument("--check-variance", help="Comma-separated list of columns requiring variance")
    parser.add_argument("--nan-threshold", type=float, default=0.001, help="Max NaN rate (0.001 = 0.1%)")

    args = parser.parse_args()

    # Parse variance check columns
    variance_cols = []
    if args.check_variance:
        variance_cols = [c.strip() for c in args.check_variance.split(',')]

    # Run audit
    passed = audit_feature_store(
        asset=args.asset,
        year=args.year,
        require_columns=args.require_columns,
        check_variance=variance_cols,
        nan_threshold=args.nan_threshold
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
