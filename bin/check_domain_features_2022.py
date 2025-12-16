#!/usr/bin/env python3
"""
Check if Wyckoff, SMC, and Temporal domain features exist in 2022 feature store.
"""
import pandas as pd
import numpy as np
from pathlib import Path

# Feature lists to check
WYCKOFF_FEATURES = [
    'wyckoff_spring_a',
    'wyckoff_spring_b',
    'wyckoff_ps',
    'wyckoff_utad',
    'wyckoff_sow',
    'wyckoff_phase_abc',
    'wyckoff_pti_confluence',
    'wyckoff_pti_score'
]

SMC_FEATURES = [
    'smc_score',
    'smc_bos',
    'smc_liquidity_sweep',
    'smc_supply_zone',
    'hob_demand_zone',
    'hob_supply_zone'
]

TEMPORAL_FEATURES = [
    'temporal_confluence',
    'temporal_support_cluster',
    'wyckoff_pti_confluence'  # Also in Wyckoff list
]

def analyze_feature(df, feature_name):
    """Analyze a single feature's data quality."""
    if feature_name not in df.columns:
        return {
            'exists': False,
            'non_null_pct': 0,
            'mean': None,
            'std': None,
            'min': None,
            'max': None,
            'unique_values': 0,
            'status': '❌ MISSING'
        }

    col = df[feature_name]
    non_null_pct = (col.notna().sum() / len(col)) * 100

    # Calculate stats only on non-null values
    if non_null_pct > 0:
        valid_data = col.dropna()

        # Check if column is numeric or categorical
        is_numeric = pd.api.types.is_numeric_dtype(valid_data)

        if is_numeric:
            mean_val = valid_data.mean()
            std_val = valid_data.std()
            min_val = valid_data.min()
            max_val = valid_data.max()
        else:
            # For categorical, show most common values
            mean_val = f"Categorical: {valid_data.mode()[0] if len(valid_data.mode()) > 0 else 'N/A'}"
            std_val = None
            min_val = None
            max_val = None

        unique_vals = valid_data.nunique()

        # Determine status
        if non_null_pct < 1:
            status = '❌ ALMOST ALL NULL'
        elif unique_vals == 1:
            status = '❌ ALL SAME VALUE'
        elif non_null_pct < 5:
            status = '⚠️ VERY SPARSE (<5%)'
        elif non_null_pct < 20:
            status = '⚠️ SPARSE (<20%)'
        elif is_numeric and (std_val == 0 or (std_val < 0.01 and abs(mean_val) < 0.01)):
            status = '⚠️ NO VARIATION'
        else:
            status = '✅ USABLE'
    else:
        mean_val = std_val = min_val = max_val = None
        unique_vals = 0
        status = '❌ ALL NULL'

    return {
        'exists': True,
        'non_null_pct': non_null_pct,
        'mean': mean_val,
        'std': std_val,
        'min': min_val,
        'max': max_val,
        'unique_values': unique_vals,
        'status': status
    }

def print_feature_table(features_dict, title):
    """Print formatted table of feature analysis."""
    print(f"\n{title}")
    print("=" * 120)
    print(f"{'Feature':<30} {'Exists':<8} {'Non-Null %':<12} {'Mean':<20} {'Std':<12} {'Min':<12} {'Max':<12} {'Status':<25}")
    print("-" * 120)

    for feature, stats in features_dict.items():
        exists = 'YES' if stats['exists'] else 'NO'
        non_null = f"{stats['non_null_pct']:.1f}%" if stats['exists'] else 'N/A'

        # Handle both numeric and string mean values
        if stats['mean'] is None:
            mean = 'N/A'
        elif isinstance(stats['mean'], str):
            mean = stats['mean'][:18]  # Truncate long categorical labels
        else:
            mean = f"{stats['mean']:.4f}"

        std = f"{stats['std']:.4f}" if isinstance(stats['std'], (int, float)) else 'N/A'
        min_val = f"{stats['min']:.4f}" if isinstance(stats['min'], (int, float)) else 'N/A'
        max_val = f"{stats['max']:.4f}" if isinstance(stats['max'], (int, float)) else 'N/A'

        print(f"{feature:<30} {exists:<8} {non_null:<12} {mean:<20} {std:<12} {min_val:<12} {max_val:<12} {stats['status']:<25}")

def main():
    # Find the main feature store file
    data_path = Path('/Users/raymondghandchi/Bull-machine-/Bull-machine-/data/features_mtf')

    # Try main file first
    feature_file = data_path / 'BTC_1H_2022-01-01_to_2024-12-31.parquet'
    if not feature_file.exists():
        # Fallback to 2022-2023 file
        feature_file = data_path / 'BTC_1H_2022-01-01_to_2023-12-31.parquet'

    print("=" * 120)
    print("DOMAIN FEATURE AVAILABILITY REPORT (2022)")
    print("=" * 120)
    print(f"\nFEATURE STORE LOCATION:")
    print(f"{feature_file}")

    # Load data
    print(f"\nLoading data...")
    df = pd.read_parquet(feature_file)

    # Filter to 2022
    df_2022 = df[(df.index >= '2022-01-01') & (df.index < '2023-01-01')]

    print(f"\nDATA RANGE:")
    print(f"Full dataset: {df.index.min()} to {df.index.max()}")
    print(f"2022 subset: {df_2022.index.min()} to {df_2022.index.max()}")
    print(f"Total rows in 2022: {len(df_2022)}")
    print(f"Total columns: {len(df_2022.columns)}")

    # Show all columns with 'wyckoff' in name
    wyckoff_cols = [c for c in df_2022.columns if 'wyckoff' in c.lower()]
    smc_cols = [c for c in df_2022.columns if 'smc' in c.lower()]
    temporal_cols = [c for c in df_2022.columns if 'temporal' in c.lower()]

    print(f"\nCOLUMNS MATCHING DOMAIN PATTERNS:")
    print(f"Wyckoff columns found: {len(wyckoff_cols)}")
    if wyckoff_cols:
        print(f"  {wyckoff_cols[:10]}")  # Show first 10
    print(f"SMC columns found: {len(smc_cols)}")
    if smc_cols:
        print(f"  {smc_cols}")
    print(f"Temporal columns found: {len(temporal_cols)}")
    if temporal_cols:
        print(f"  {temporal_cols}")

    # Analyze Wyckoff features
    wyckoff_stats = {}
    for feature in WYCKOFF_FEATURES:
        wyckoff_stats[feature] = analyze_feature(df_2022, feature)

    print_feature_table(wyckoff_stats, "WYCKOFF FEATURES (2022)")

    # Analyze SMC features
    smc_stats = {}
    for feature in SMC_FEATURES:
        smc_stats[feature] = analyze_feature(df_2022, feature)

    print_feature_table(smc_stats, "SMC FEATURES (2022)")

    # Analyze Temporal features
    temporal_stats = {}
    for feature in TEMPORAL_FEATURES:
        temporal_stats[feature] = analyze_feature(df_2022, feature)

    print_feature_table(temporal_stats, "TEMPORAL FEATURES (2022)")

    # Summary statistics
    all_stats = {**wyckoff_stats, **smc_stats, **temporal_stats}

    total_features = len(all_stats)
    missing = sum(1 for s in all_stats.values() if not s['exists'])
    all_null = sum(1 for s in all_stats.values() if s['exists'] and s['non_null_pct'] < 1)
    sparse = sum(1 for s in all_stats.values() if s['exists'] and 1 <= s['non_null_pct'] < 20)
    usable = sum(1 for s in all_stats.values() if '✅' in s['status'])

    print("\n" + "=" * 120)
    print("SUMMARY STATISTICS")
    print("=" * 120)
    print(f"Total features checked: {total_features}")
    print(f"❌ Missing from schema: {missing}")
    print(f"❌ All NULL values: {all_null}")
    print(f"⚠️ Sparse data (<20%): {sparse}")
    print(f"✅ Usable features: {usable}")

    # Root cause analysis
    print("\n" + "=" * 120)
    print("ROOT CAUSE DIAGNOSIS")
    print("=" * 120)

    if missing > 0:
        print(f"\n⚠️ {missing} features are MISSING from the feature store:")
        for feat, stats in all_stats.items():
            if not stats['exists']:
                print(f"   - {feat}")
        print("\n   → These features were never computed or not included in this parquet file")
        print("   → Agent 2's wiring will have NO EFFECT if these columns don't exist")

    if all_null > 0:
        print(f"\n⚠️ {all_null} features exist but are ALL NULL:")
        for feat, stats in all_stats.items():
            if stats['exists'] and stats['non_null_pct'] < 1:
                print(f"   - {feat}")
        print("\n   → Features were added to schema but never computed with real values")
        print("   → Feature computation pipeline may have failed or been skipped")

    if sparse > 0:
        print(f"\n⚠️ {sparse} features are SPARSE (<20% non-null):")
        for feat, stats in all_stats.items():
            if stats['exists'] and 1 <= stats['non_null_pct'] < 20:
                print(f"   - {feat} ({stats['non_null_pct']:.1f}% non-null)")
        print("\n   → These features trigger rarely (may be intentional for rare events)")
        print("   → But may lack sufficient signal for meaningful contribution")

    if usable > 0:
        print(f"\n✅ {usable} features are USABLE with good data:")
        for feat, stats in all_stats.items():
            if '✅' in stats['status']:
                if isinstance(stats['std'], (int, float)):
                    print(f"   - {feat} ({stats['non_null_pct']:.1f}% non-null, std={stats['std']:.4f})")
                else:
                    print(f"   - {feat} ({stats['non_null_pct']:.1f}% non-null, categorical)")
        print("\n   → If these exist but have no effect, check:")
        print("      1. Boost magnitudes too small (0.01 won't overcome base score)")
        print("      2. Features uncorrelated with regime changes")
        print("      3. Feature values outside expected ranges")

    # Recommendation
    print("\n" + "=" * 120)
    print("RECOMMENDATION")
    print("=" * 120)

    if missing > total_features * 0.5:
        print("\n🔴 CRITICAL: >50% of domain features are MISSING from feature store")
        print("   ACTION: Investigate feature computation pipeline")
        print("   - Check bin/add_derived_features.py")
        print("   - Check engine/wyckoff/ modules")
        print("   - Features may need to be computed retroactively for 2022")
    elif all_null > 0:
        print("\n🟡 WARNING: Some features exist but have no data")
        print("   ACTION: Check feature computation logic")
        print("   - Features may have been added to schema but not implemented")
        print("   - Or computation may have failed silently")
    elif sparse > usable:
        print("\n🟡 WARNING: Most features are too sparse to be useful")
        print("   ACTION: Review feature definitions")
        print("   - Very rare features (<5% occurrence) may not provide enough signal")
        print("   - Consider aggregating or creating derived features")
    else:
        print("\n🟢 GOOD: Most domain features exist with usable data")
        print("   If Agent 2's wiring had no effect, investigate:")
        print("   1. Boost magnitude (try 0.1 or 0.2 instead of 0.01)")
        print("   2. Feature correlation with regime changes")
        print("   3. Integration bugs in threshold_policy.py")

if __name__ == '__main__':
    main()
