#!/usr/bin/env python3
"""
Systematic Feature Store Reality Check
=======================================
Load the actual feature store and verify what features REALLY exist.
No assumptions, no documentation - just the raw data.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Feature lists to check
WYCKOFF_FEATURES = [
    'wyckoff_ps',           # Preliminary Support
    'wyckoff_spring_a',     # Spring A
    'wyckoff_spring_b',     # Spring B
    'wyckoff_sc',           # Selling Climax
    'wyckoff_ar',           # Automatic Rally
    'wyckoff_st',           # Secondary Test
    'wyckoff_sos',          # Sign of Strength
    'wyckoff_sow',          # Sign of Weakness
    'wyckoff_lps',          # Last Point of Support
    'wyckoff_bc',           # Buying Climax
    'wyckoff_utad',         # Upthrust After Distribution
    'wyckoff_lpsy',         # Last Point of Supply
    'wyckoff_phase',        # Phase label
    'wyckoff_phase_abc',    # Phase ABC classification
    'wyckoff_accumulation', # Accumulation phase
    'wyckoff_distribution', # Distribution phase
    'wyckoff_markup',       # Markup phase
    'wyckoff_markdown',     # Markdown phase
    'wyckoff_pti_score',    # PTI score
    'wyckoff_pti_confluence', # PTI confluence
    'wyckoff_pti_trap_type', # PTI trap type
    'wyckoff_confidence',   # Event confidence
    'wyckoff_strength'      # Event strength
]

SMC_FEATURES = [
    'smc_score',            # Composite SMC score
    'smc_bos',              # Break of Structure
    'smc_choch',            # Change of Character
    'smc_liquidity_sweep',  # Liquidity sweep
    'tf1h_bos_bearish',     # 1H bearish BOS
    'tf1h_bos_bullish',     # 1H bullish BOS
    'tf4h_bos_bearish',     # 4H bearish BOS
    'tf4h_bos_bullish',     # 4H bullish BOS
    'smc_supply_zone',      # Supply zone
    'smc_demand_zone',      # Demand zone
    'smc_fvg_bear',         # Fair value gap bearish
    'smc_fvg_bull'          # Fair value gap bullish
]

HOB_FEATURES = [
    'hob_demand_zone',      # Higher Order Book demand
    'hob_supply_zone',      # Higher Order Book supply
    'hob_imbalance',        # Order book imbalance
    'hob_strength',         # Zone strength
    'hob_quality'           # Zone quality
]

TEMPORAL_FEATURES = [
    'temporal_confluence',
    'temporal_support_cluster',
    'temporal_resistance_cluster',
    'fib_time_cluster',
    'fib_time_score',
    'tf4h_fusion_score',
    'tf1h_fusion_score',
    'gann_cycle',
    'volatility_cycle'
]

def analyze_feature(df, feature_name):
    """Analyze a single feature to determine its status"""
    if feature_name not in df.columns:
        return 'MISSING', None

    col = df[feature_name]

    # Calculate statistics
    total_rows = len(df)
    non_null_count = col.notna().sum()
    non_null_pct = (non_null_count / total_rows) * 100
    unique_vals = col.nunique()

    # Determine data type
    dtype = str(col.dtype)

    # Sample values (non-null)
    sample = col.dropna().head(5).tolist() if non_null_count > 0 else []

    # Check if it's real data
    if non_null_pct < 0.1:  # Less than 0.1% coverage
        return 'EMPTY', {'coverage': non_null_pct, 'unique': unique_vals, 'dtype': dtype}
    elif unique_vals == 1:
        return 'CONSTANT', {'coverage': non_null_pct, 'unique': unique_vals, 'value': sample[0] if sample else None, 'dtype': dtype}
    else:
        return 'EXISTS', {
            'coverage': non_null_pct,
            'unique': unique_vals,
            'dtype': dtype,
            'sample': sample,
            'min': col.min() if dtype in ['float64', 'int64'] else None,
            'max': col.max() if dtype in ['float64', 'int64'] else None
        }

def print_category_results(category_name, features, results):
    """Print results for a feature category"""
    exists = []
    empty = []
    constant = []
    missing = []

    for feature in features:
        status, info = results[feature]
        if status == 'EXISTS':
            exists.append((feature, info))
        elif status == 'EMPTY':
            empty.append((feature, info))
        elif status == 'CONSTANT':
            constant.append((feature, info))
        else:  # MISSING
            missing.append(feature)

    total = len(features)
    print(f"\n{'━' * 80}")
    print(f"{category_name} ({total} checked):")
    print(f"{'━' * 80}\n")

    if exists:
        print(f"✅ EXISTS (real data): {len(exists)}/{total}")
        for feature, info in exists:
            coverage = info['coverage']
            unique = info['unique']
            dtype = info['dtype']
            print(f"  {feature:30s} {coverage:6.1f}% coverage, {unique:6d} unique, dtype={dtype}")
            if 'sample' in info and info['sample']:
                sample_str = str(info['sample'][:3])
                if len(sample_str) > 60:
                    sample_str = sample_str[:57] + '...'
                print(f"    Sample: {sample_str}")
        print()

    if empty:
        print(f"⚠️  EXISTS (but empty/null): {len(empty)}/{total}")
        for feature, info in empty:
            print(f"  {feature:30s} {info['coverage']:.2f}% coverage (effectively empty)")
        print()

    if constant:
        print(f"⚠️  EXISTS (but constant): {len(constant)}/{total}")
        for feature, info in constant:
            print(f"  {feature:30s} All values = {info.get('value')}")
        print()

    if missing:
        print(f"❌ MISSING (not in store): {len(missing)}/{total}")
        for feature in missing:
            print(f"  {feature}")
        print()

    return {
        'exists': len(exists),
        'empty': len(empty),
        'constant': len(constant),
        'missing': len(missing)
    }

def main():
    # Load feature store
    feature_store_path = Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet')

    print("=" * 80)
    print("COMPLETE FEATURE STORE AUDIT")
    print("=" * 80)
    print()

    if not feature_store_path.exists():
        print(f"❌ ERROR: Feature store not found at {feature_store_path}")
        return

    print(f"Loading: {feature_store_path}")
    df = pd.read_parquet(feature_store_path)

    print(f"\nFEATURE STORE:")
    print(f"  File: {feature_store_path}")
    print(f"  Columns: {len(df.columns)}")
    print(f"  Rows: {len(df):,}")
    if hasattr(df.index, 'min') and hasattr(df.index, 'max'):
        print(f"  Date Range: {df.index.min()} to {df.index.max()}")
    print(f"  File Size: {feature_store_path.stat().st_size / (1024*1024):.1f} MB")

    # Analyze all features
    all_features = {
        'Wyckoff': WYCKOFF_FEATURES,
        'SMC': SMC_FEATURES,
        'HOB': HOB_FEATURES,
        'Temporal/Fusion': TEMPORAL_FEATURES
    }

    all_results = {}
    summary = {
        'exists': 0,
        'empty': 0,
        'constant': 0,
        'missing': 0
    }

    for category, features in all_features.items():
        results = {}
        for feature in features:
            status, info = analyze_feature(df, feature)
            results[feature] = (status, info)
            all_results[feature] = (status, info)

        cat_summary = print_category_results(category, features, results)
        for key in summary:
            summary[key] += cat_summary[key]

    # Print overall summary
    total = sum(summary.values())
    print(f"\n{'━' * 80}")
    print("OVERALL SUMMARY:")
    print(f"{'━' * 80}\n")
    print(f"Total Features Checked: {total}")
    print(f"✅ Exist with real data: {summary['exists']} ({summary['exists']/total*100:.1f}%)")
    print(f"⚠️  Exist but empty: {summary['empty']} ({summary['empty']/total*100:.1f}%)")
    print(f"⚠️  Exist but constant: {summary['constant']} ({summary['constant']/total*100:.1f}%)")
    print(f"❌ Truly missing: {summary['missing']} ({summary['missing']/total*100:.1f}%)")

    # Generate list of features that need implementation
    needs_generation = []
    for feature, (status, info) in all_results.items():
        if status in ['MISSING', 'EMPTY', 'CONSTANT']:
            needs_generation.append((feature, status))

    if needs_generation:
        print(f"\n{'━' * 80}")
        print("FEATURES REQUIRING GENERATION/IMPLEMENTATION:")
        print(f"{'━' * 80}\n")
        for feature, status in needs_generation:
            print(f"  {feature:40s} [{status}]")
        print(f"\nTotal requiring work: {len(needs_generation)}/{total}")

    print(f"\n{'━' * 80}")
    print("STATUS: Reality check complete")
    print(f"{'━' * 80}\n")

    # Also show a sample of what columns ARE in the feature store
    print("\nACTUAL COLUMNS IN FEATURE STORE (sample of first 50):")
    print("=" * 80)
    for i, col in enumerate(sorted(df.columns)[:50]):
        print(f"  {col}")
    if len(df.columns) > 50:
        print(f"\n  ... and {len(df.columns) - 50} more columns")

if __name__ == '__main__':
    main()
