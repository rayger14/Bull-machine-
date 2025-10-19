#!/usr/bin/env python3
"""
Verify feature store has no constant/empty columns

Usage:
    python3 scripts/verify_features.py data/features_mtf/BTC_1H_2024-07-01_to_2024-09-30.parquet
"""

import sys
import pandas as pd
from pathlib import Path

def verify_feature_store(path: str):
    """Check for constant/empty columns and show domain stats"""

    print("=" * 80)
    print(f"Feature Store Verification: {Path(path).name}")
    print("=" * 80)

    # Load parquet
    try:
        df = pd.read_parquet(path)
    except Exception as e:
        print(f"❌ Error loading parquet: {e}")
        return False

    print(f"\n✅ Loaded {len(df)} rows, {len(df.columns)} columns")
    print(f"   Date range: {df.index[0]} to {df.index[-1]}")

    # Find constant/empty columns
    flat_cols = []
    for col in df.columns:
        nunique = df[col].nunique(dropna=True)
        if nunique <= 1:
            flat_cols.append(col)

    if flat_cols:
        print(f"\n⚠️  Constant/Empty Columns ({len(flat_cols)}):")
        for col in flat_cols:
            print(f"   - {col}")
    else:
        print(f"\n✅ No constant/empty columns")

    # Domain-specific stats
    domains = {
        'wyckoff': ['tf1d_wyckoff', 'wyckoff'],
        'smc': ['smc', 'bos', 'choch', 'fvg'],
        'hob': ['hob', 'liquidity'],
        'boms': ['boms'],
        'pti': ['pti'],
        'fakeout': ['fakeout'],
        'frvp': ['frvp'],
        'macro': ['macro'],
        'momentum': ['momentum', 'adx', 'rsi']
    }

    print(f"\n{'=' * 80}")
    print("Domain-Specific Statistics")
    print("=" * 80)

    for domain_name, patterns in domains.items():
        # Find columns matching patterns
        domain_cols = []
        for pattern in patterns:
            domain_cols.extend([c for c in df.columns if pattern.lower() in c.lower()])

        if not domain_cols:
            continue

        print(f"\n{domain_name.upper()}:")

        # Show stats for each column
        for col in sorted(set(domain_cols)):
            nunique = df[col].nunique(dropna=True)
            dtype = df[col].dtype

            if dtype == 'object' or dtype == 'bool':
                # Categorical
                values = df[col].value_counts()
                print(f"  {col}: {nunique} unique values")
                if nunique <= 10:
                    for val, count in values.items():
                        print(f"    - {val}: {count} ({count/len(df)*100:.1f}%)")
            else:
                # Numeric
                stats = df[col].describe()
                print(f"  {col}:")
                print(f"    min={stats['min']:.3f}, max={stats['max']:.3f}, "
                      f"mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                      f"unique={nunique}")

    print(f"\n{'=' * 80}")
    print("Verification Complete")
    print("=" * 80)

    # Pass/Fail criteria
    critical_flat = [c for c in flat_cols if any(
        pattern in c.lower() for pattern in
        ['wyckoff', 'smc', 'hob', 'pti', 'frvp', 'macro', 'momentum']
    )]

    if critical_flat:
        print(f"\n❌ FAIL: {len(critical_flat)} critical domain columns are constant:")
        for col in critical_flat:
            print(f"   - {col}")
        return False
    else:
        print(f"\n✅ PASS: All critical domain columns are varying")
        if flat_cols:
            print(f"   Note: {len(flat_cols)} non-critical columns are constant (OK)")
        return True


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python3 scripts/verify_features.py <path_to_parquet>")
        sys.exit(1)

    path = sys.argv[1]

    if not Path(path).exists():
        print(f"❌ File not found: {path}")
        sys.exit(1)

    success = verify_feature_store(path)
    sys.exit(0 if success else 1)
