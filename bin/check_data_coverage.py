#!/usr/bin/env python3
"""Check OI and funding data coverage."""

import pandas as pd
import sys

# Load features
df = pd.read_parquet("data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet")

print("="*80)
print("DATA COVERAGE ANALYSIS")
print("="*80)

# Check funding data
if 'funding_Z' in df.columns:
    funding_null = df['funding_Z'].isna().mean() * 100
    print(f"\nFunding Z-score null%: {funding_null:.1f}%")
else:
    print("\nWARNING: funding_Z column not found!")

# Check OI data
oi_cols = [col for col in df.columns if 'oi_' in col.lower()]
print(f"\nOI columns found: {len(oi_cols)}")
for col in oi_cols[:5]:  # Show first 5
    null_pct = df[col].isna().mean() * 100
    print(f"  {col}: {null_pct:.1f}% null")

# Check by period
print("\n" + "="*80)
print("BY PERIOD")
print("="*80)

periods = [
    ('2022', '2022-01-01', '2022-12-31'),
    ('2023', '2023-01-01', '2023-12-31'),
    ('2024', '2024-01-01', '2024-12-31'),
]

for period_name, start, end in periods:
    period_df = df[start:end]

    if 'funding_Z' in df.columns:
        funding_null = period_df['funding_Z'].isna().mean() * 100
    else:
        funding_null = 100.0

    # Check first OI column
    if oi_cols:
        oi_null = period_df[oi_cols[0]].isna().mean() * 100
    else:
        oi_null = 100.0

    print(f"\n{period_name}:")
    print(f"  Funding null: {funding_null:.1f}%")
    print(f"  OI null: {oi_null:.1f}%")
    print(f"  Rows: {len(period_df)}")

# PASS/FAIL criteria
print("\n" + "="*80)
print("PASS/FAIL ASSESSMENT")
print("="*80)

overall_funding_null = df['funding_Z'].isna().mean() * 100 if 'funding_Z' in df.columns else 100
overall_oi_null = df[oi_cols[0]].isna().mean() * 100 if oi_cols else 100

funding_pass = overall_funding_null < 20
oi_pass = overall_oi_null < 20

print(f"\nOverall funding null < 20%: {'PASS' if funding_pass else 'FAIL'} ({overall_funding_null:.1f}%)")
print(f"Overall OI null < 20%: {'PASS' if oi_pass else 'FAIL'} ({overall_oi_null:.1f}%)")

if funding_pass and oi_pass:
    print("\n✅ DATA COVERAGE: PASS")
    sys.exit(0)
else:
    print("\n❌ DATA COVERAGE: FAIL")
    sys.exit(1)
