#!/usr/bin/env python3
"""
Trial Data Consolidation Script

Merges all Optuna/optimization trial CSVs into single training dataset for
the config optimizer (Phase 2 Meta-Optimizer).

Usage:
    python3 bin/consolidate_trials.py \
        --asset BTC \
        --output reports/ml/config_training_data.csv
"""

import pandas as pd
import argparse
from pathlib import Path
import hashlib
import json


def load_trial_csv(path: Path) -> pd.DataFrame:
    """Load trial CSV and add source metadata"""
    print(f"  Loading {path}")
    df = pd.read_csv(path)
    df['source_file'] = path.name
    df['source_dir'] = path.parent.name
    print(f"    Rows: {len(df)}")
    return df


def compute_row_hash(row: pd.Series, key_cols: list) -> str:
    """Compute hash of key columns to detect duplicates"""
    # Sort columns for consistent hashing
    key_vals = '|'.join([str(row[col]) for col in sorted(key_cols)])
    return hashlib.md5(key_vals.encode()).hexdigest()


def deduplicate_trials(df: pd.DataFrame, config_cols: list) -> pd.DataFrame:
    """Remove duplicate trials based on config parameters"""
    print(f"\n  Deduplicating based on {len(config_cols)} config columns...")

    # Compute hash of config columns
    df['config_hash'] = df.apply(lambda row: compute_row_hash(row, config_cols), axis=1)

    # Count duplicates
    duplicates = df['config_hash'].duplicated()
    n_dupes = duplicates.sum()

    if n_dupes > 0:
        print(f"    Found {n_dupes} duplicate configs")
        # Keep first occurrence (usually earliest trial)
        df = df[~duplicates].copy()
    else:
        print(f"    No duplicates found")

    # Drop hash column
    df = df.drop(columns=['config_hash'])

    return df


def validate_schema(df: pd.DataFrame, required_cols: list) -> bool:
    """Validate that required columns are present"""
    missing = set(required_cols) - set(df.columns)
    if missing:
        print(f"  WARNING: Missing columns: {missing}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Consolidate optimization trial data')
    parser.add_argument('--asset', type=str, default='BTC', help='Asset to consolidate trials for')
    parser.add_argument('--output', type=str, required=True, help='Output CSV path')
    parser.add_argument('--min-rows', type=int, default=50, help='Minimum rows required')

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"TRIAL DATA CONSOLIDATION - {args.asset}")
    print(f"{'='*60}\n")

    # Define trial data sources
    trial_sources = [
        'reports/optuna_btc_frontier_v7/BTC_all_trials.csv',
        'reports/year_opt_v1/BTC_all_trials.csv',
        'reports/archetype_optimization_v2/BTC_all_trials.csv',
        'reports/optuna_multifold_v6/BTC_all_trials.csv'
    ]

    # Required columns for config optimizer
    required_metrics = ['year_pf', 'year_wr', 'year_dd', 'year_trades', 'year_pnl']

    # Config parameters (features for meta-optimizer)
    config_params = [
        'final_fusion_floor', 'neutralize_fusion_drop', 'neutralize_min_bars',
        'neutralize_pti_margin', 'min_liquidity',
        'w_wyckoff', 'w_liquidity', 'w_momentum',
        'size_min', 'size_max',
        'B_fusion', 'C_fusion', 'H_fusion', 'K_fusion', 'L_fusion',
        'trail_atr_mult', 'max_bars', 'range_stop_factor', 'trend_stop_factor'
    ]

    required_cols = required_metrics + config_params

    # Load all trial CSVs
    print("Loading trial data...")
    dfs = []
    for source in trial_sources:
        path = Path(source)
        if path.exists():
            df = load_trial_csv(path)

            # Validate schema
            if validate_schema(df, required_cols):
                dfs.append(df)
            else:
                print(f"    SKIPPED: Missing required columns")
        else:
            print(f"  SKIPPED: {source} not found")

    if not dfs:
        print("\nERROR: No valid trial data found")
        return

    # Concatenate all dataframes
    print(f"\nConcatenating {len(dfs)} dataframes...")
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Total rows before deduplication: {len(combined)}")

    # Deduplicate based on config parameters
    combined = deduplicate_trials(combined, config_params)
    print(f"  Total rows after deduplication: {len(combined)}")

    # Check minimum rows requirement
    if len(combined) < args.min_rows:
        print(f"\nWARNING: Only {len(combined)} rows (minimum: {args.min_rows})")
        print("  Proceeding anyway, but model may be undertrained")

    # Sort by year_pf descending (best trials first)
    combined = combined.sort_values('year_pf', ascending=False).reset_index(drop=True)

    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save consolidated data
    combined.to_csv(output_path, index=False)

    print(f"\n{'='*60}")
    print("CONSOLIDATION COMPLETE")
    print(f"{'='*60}")
    print(f"Total trials: {len(combined)}")
    print(f"Unique configs: {len(combined)}")
    print(f"Config features: {len(config_params)}")
    print(f"Target metrics: {required_metrics}")
    print(f"\nPerformance range:")
    print(f"  PF: {combined['year_pf'].min():.2f} - {combined['year_pf'].max():.2f}")
    print(f"  WR: {combined['year_wr'].min()*100:.1f}% - {combined['year_wr'].max()*100:.1f}%")
    print(f"  Trades: {int(combined['year_trades'].min())} - {int(combined['year_trades'].max())}")
    print(f"\nSaved to: {output_path}")

    # Save metadata
    metadata = {
        'asset': args.asset,
        'n_trials': int(len(combined)),
        'n_config_features': len(config_params),
        'config_features': config_params,
        'target_metrics': required_metrics,
        'sources': trial_sources,
        'pf_range': [float(combined['year_pf'].min()), float(combined['year_pf'].max())],
        'wr_range': [float(combined['year_wr'].min()), float(combined['year_wr'].max())],
        'sample_per_feature_ratio': len(combined) / len(config_params)
    }

    metadata_path = output_path.parent / f"{args.asset}_consolidation_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: {metadata_path}")
    print(f"\nSample/feature ratio: {metadata['sample_per_feature_ratio']:.1f}x")
    if metadata['sample_per_feature_ratio'] < 5:
        print("  ⚠️  WARNING: Low sample/feature ratio. Consider feature selection.")
    elif metadata['sample_per_feature_ratio'] >= 10:
        print("  ✓ Good sample/feature ratio for training")


if __name__ == '__main__':
    main()
