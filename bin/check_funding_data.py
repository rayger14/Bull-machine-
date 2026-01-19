#!/usr/bin/env python3
"""
STEP 5a: Check Funding Data Coverage

Verifies funding rate data is available and complete for S4/S5 archetypes.

Usage:
    python bin/check_funding_data.py
    python bin/check_funding_data.py --data data/features_1h.parquet
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))


def check_funding_coverage(df: pd.DataFrame, periods: dict) -> dict:
    """
    Check funding data coverage across periods.

    Args:
        df: Feature dataframe
        periods: Dict of period_name -> (start, end)

    Returns:
        Coverage statistics
    """
    results = {}

    funding_cols = [col for col in df.columns if 'funding' in col.lower()]

    if not funding_cols:
        return {
            'error': 'No funding columns found',
            'available_columns': list(df.columns)
        }

    print(f"\nFunding columns found: {funding_cols}")

    for period_name, (start, end) in periods.items():
        period_df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)]

        if len(period_df) == 0:
            results[period_name] = {'error': 'No data in period'}
            continue

        period_results = {}

        for col in funding_cols:
            null_pct = 100 * period_df[col].isna().sum() / len(period_df)
            period_results[col] = {
                'null_pct': null_pct,
                'mean': period_df[col].mean(),
                'std': period_df[col].std(),
                'count': period_df[col].notna().sum()
            }

        results[period_name] = period_results

    return results


def main():
    parser = argparse.ArgumentParser(description="Check funding data coverage")
    parser.add_argument(
        '--data',
        type=str,
        default='data/features_1h.parquet',
        help='Path to feature data'
    )

    args = parser.parse_args()

    data_path = Path(args.data)

    if not data_path.exists():
        print(f"Error: Data file not found: {data_path}")
        return 1

    print(f"Loading data from {data_path}...")
    df = pd.read_parquet(data_path)

    # Ensure timestamp column
    if 'timestamp' not in df.columns:
        if df.index.name == 'timestamp':
            df = df.reset_index()
        else:
            print("Error: No timestamp column found")
            return 1

    # Define test periods
    periods = {
        '2022': ('2022-01-01', '2022-12-31'),
        '2023': ('2023-01-01', '2023-12-31'),
        '2024': ('2024-01-01', '2024-12-31')
    }

    results = check_funding_coverage(df, periods)

    # Print results
    print("\n" + "="*60)
    print("FUNDING DATA COVERAGE REPORT")
    print("="*60)

    max_null_pct = 0

    for period_name, period_results in results.items():
        print(f"\n{period_name}:")

        if 'error' in period_results:
            print(f"  Error: {period_results['error']}")
            continue

        for col, stats in period_results.items():
            null_pct = stats['null_pct']
            max_null_pct = max(max_null_pct, null_pct)

            status = "✓" if null_pct < 20 else "✗"
            color = "\033[0;32m" if null_pct < 20 else "\033[0;31m"
            reset = "\033[0m"

            print(f"  {col}:")
            print(f"    Null: {color}{null_pct:.1f}%{reset} {status}")
            print(f"    Mean: {stats['mean']:.6f}")
            print(f"    Count: {stats['count']}")

    # Summary
    print("\n" + "="*60)

    if max_null_pct < 20:
        print("\033[0;32m✓ PASS\033[0m: Funding data coverage acceptable")
        print(f"Max null: {max_null_pct:.1f}% < 20%")
        return 0
    else:
        print("\033[0;31m✗ FAIL\033[0m: Funding data incomplete")
        print(f"Max null: {max_null_pct:.1f}% ≥ 20%")
        print("\nTo fix:")
        print("  python bin/fix_oi_change_pipeline.py")
        return 1


if __name__ == '__main__':
    exit(main())
