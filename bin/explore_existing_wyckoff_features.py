#!/usr/bin/env python3
"""
Explore what Wyckoff features actually exist in the 2022 data.
"""
import pandas as pd
from pathlib import Path

def main():
    # Load data
    feature_file = Path('/Users/raymondghandchi/Bull-machine-/Bull-machine-/data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')
    df = pd.read_parquet(feature_file)
    df_2022 = df[(df.index >= '2022-01-01') & (df.index < '2023-01-01')]

    # Get all Wyckoff columns
    wyckoff_cols = sorted([c for c in df_2022.columns if 'wyckoff' in c.lower()])

    print("=" * 100)
    print("ALL WYCKOFF FEATURES IN 2022 DATA")
    print("=" * 100)
    print(f"\nTotal Wyckoff columns: {len(wyckoff_cols)}\n")

    for col in wyckoff_cols:
        col_data = df_2022[col]
        non_null_pct = (col_data.notna().sum() / len(col_data)) * 100
        is_numeric = pd.api.types.is_numeric_dtype(col_data)

        if is_numeric:
            valid = col_data.dropna()
            if len(valid) > 0:
                print(f"{col:<40} | {non_null_pct:>6.1f}% non-null | mean={valid.mean():>8.4f} | std={valid.std():>8.4f} | unique={valid.nunique()}")
            else:
                print(f"{col:<40} | {non_null_pct:>6.1f}% non-null | ALL NULL")
        else:
            valid = col_data.dropna()
            if len(valid) > 0:
                top_value = valid.value_counts().index[0] if len(valid.value_counts()) > 0 else "N/A"
                print(f"{col:<40} | {non_null_pct:>6.1f}% non-null | CATEGORICAL | top='{top_value}' | unique={valid.nunique()}")
            else:
                print(f"{col:<40} | {non_null_pct:>6.1f}% non-null | CATEGORICAL | ALL NULL")

    # Now check for SMC patterns
    print("\n" + "=" * 100)
    print("ALL SMC/HOB FEATURES IN 2022 DATA")
    print("=" * 100)

    smc_cols = sorted([c for c in df_2022.columns if any(x in c.lower() for x in ['smc', 'hob', 'supply', 'demand'])])
    if smc_cols:
        print(f"\nTotal SMC columns: {len(smc_cols)}\n")
        for col in smc_cols:
            col_data = df_2022[col]
            non_null_pct = (col_data.notna().sum() / len(col_data)) * 100
            is_numeric = pd.api.types.is_numeric_dtype(col_data)

            if is_numeric:
                valid = col_data.dropna()
                if len(valid) > 0:
                    print(f"{col:<40} | {non_null_pct:>6.1f}% non-null | mean={valid.mean():>8.4f} | std={valid.std():>8.4f}")
                else:
                    print(f"{col:<40} | {non_null_pct:>6.1f}% non-null | ALL NULL")
            else:
                print(f"{col:<40} | {non_null_pct:>6.1f}% non-null | CATEGORICAL")
    else:
        print("\n❌ NO SMC columns found")

    # Check for temporal patterns
    print("\n" + "=" * 100)
    print("ALL TEMPORAL FEATURES IN 2022 DATA")
    print("=" * 100)

    temporal_cols = sorted([c for c in df_2022.columns if 'temporal' in c.lower()])
    if temporal_cols:
        print(f"\nTotal Temporal columns: {len(temporal_cols)}\n")
        for col in temporal_cols:
            col_data = df_2022[col]
            non_null_pct = (col_data.notna().sum() / len(col_data)) * 100
            is_numeric = pd.api.types.is_numeric_dtype(col_data)

            if is_numeric:
                valid = col_data.dropna()
                if len(valid) > 0:
                    print(f"{col:<40} | {non_null_pct:>6.1f}% non-null | mean={valid.mean():>8.4f} | std={valid.std():>8.4f}")
                else:
                    print(f"{col:<40} | {non_null_pct:>6.1f}% non-null | ALL NULL")
            else:
                print(f"{col:<40} | {non_null_pct:>6.1f}% non-null | CATEGORICAL")
    else:
        print("\n❌ NO temporal columns found")

    # Sample some Wyckoff events to see if they're working
    print("\n" + "=" * 100)
    print("SAMPLE WYCKOFF EVENTS (showing rows where wyckoff_spring_a == 1)")
    print("=" * 100)

    if 'wyckoff_spring_a' in df_2022.columns:
        spring_events = df_2022[df_2022['wyckoff_spring_a'] == 1.0]
        if len(spring_events) > 0:
            print(f"\nFound {len(spring_events)} Spring A events in 2022:")
            print(spring_events.index[:10].tolist())
        else:
            print("\n❌ No Spring A events triggered in 2022")

    if 'wyckoff_sow' in df_2022.columns:
        sow_events = df_2022[df_2022['wyckoff_sow'] == 1.0]
        if len(sow_events) > 0:
            print(f"\nFound {len(sow_events)} Sign of Weakness events in 2022:")
            print(sow_events.index[:10].tolist())
        else:
            print("\n❌ No Sign of Weakness events triggered in 2022")

if __name__ == '__main__':
    main()
