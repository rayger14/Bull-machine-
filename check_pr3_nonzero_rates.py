#!/usr/bin/env python3
"""
Quick checker for PR#3 non-zero rates.
Validates that BOMS calculation fixes increased non-zero rates.
"""
import pandas as pd
import sys

# Load rebuilt feature store
feature_store_path = 'data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet'
print(f"Loading feature store: {feature_store_path}")
df = pd.read_parquet(feature_store_path)

print(f"Total rows: {len(df):,}")
print()

# Check P0 columns
p0_columns = {
    'tf4h_boms_displacement': {'expected_min': 5.0, 'before': 0.0},
    'tf1d_boms_strength': {'expected_min': 5.0, 'before': 3.3},
}

print("=" * 60)
print("PR#3 Non-Zero Rate Validation")
print("=" * 60)

for col, config in p0_columns.items():
    if col in df.columns:
        non_zero_count = (df[col] != 0).sum()
        non_zero_pct = (non_zero_count / len(df)) * 100

        status = "✅ PASS" if non_zero_pct >= config['expected_min'] else "❌ FAIL"
        improvement = non_zero_pct - config['before']

        print(f"\n{status} {col}:")
        print(f"   Before:   {config['before']:.1f}% non-zero")
        print(f"   After:    {non_zero_pct:.1f}% non-zero ({non_zero_count:,} / {len(df):,})")
        print(f"   Change:   +{improvement:.1f} percentage points")
        print(f"   Target:   >{config['expected_min']:.1f}%")

        if non_zero_pct > 0:
            print(f"   Min:      {df[col].min():.4f}")
            print(f"   Max:      {df[col].max():.4f}")
            print(f"   Mean:     {df[col].mean():.4f}")
    else:
        print(f"\n❌ ERROR: Column '{col}' not found in feature store")

print()
print("=" * 60)
