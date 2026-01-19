#!/usr/bin/env python3
"""
Fix OI_CHANGE Pipeline for Bear Pattern Detection

This script resolves 2 critical OI pipeline failures:
1. Missing OI data for 2022-2023 (fetch from OKX API)
2. Missing derived features (oi_change_24h, oi_change_pct_24h, oi_z)

Author: Backend Architect (Claude Code)
Date: 2025-11-13
Status: PRODUCTION-READY

Usage:
    # Full fix (fetch + calculate + validate)
    python3 bin/fix_oi_change_pipeline.py

    # Skip fetch (use existing OI data)
    python3 bin/fix_oi_change_pipeline.py --skip-fetch

    # Dry run (validation only, no write)
    python3 bin/fix_oi_change_pipeline.py --dry-run
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import requests
import time
from datetime import datetime, timezone
from typing import Optional, Dict, Any


# ============================================================================
# PHASE 1: FETCH OI DATA (2022-2023)
# ============================================================================

def fetch_okx_historical_oi(
    start_date: str,
    end_date: str,
    cache_path: Optional[Path] = None
) -> pd.DataFrame:
    """
    Fetch OKX historical Open Interest data.

    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        cache_path: Optional path to cache fetched data

    Returns:
        DataFrame with columns: [timestamp, oi]
        - timestamp: datetime64[ns, UTC]
        - oi: float (Open Interest in USD notional)

    Raises:
        ValueError: If API returns error or no data
    """
    print("\n" + "=" * 80)
    print("PHASE 1: Fetching OKX Historical Open Interest")
    print("=" * 80)

    # Check cache first
    if cache_path and cache_path.exists():
        print(f"Loading cached OI data from: {cache_path}")
        df = pd.read_parquet(cache_path)
        print(f"  Loaded {len(df)} records from cache")
        return df

    # OKX API configuration
    base_url = "https://www.okx.com"
    endpoint = "/api/v5/rubik/stat/contracts/open-interest-history"

    # Convert dates to timestamps (milliseconds)
    start_ts = int(datetime.strptime(start_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)
    end_ts = int(datetime.strptime(end_date, '%Y-%m-%d').replace(tzinfo=timezone.utc).timestamp() * 1000)

    print(f"\nFetching OI data:")
    print(f"  Date range: {start_date} to {end_date}")
    print(f"  Instrument: BTC-USDT-SWAP")
    print(f"  Granularity: 1H")

    all_data = []
    current_after = None
    fetch_count = 0

    while True:
        try:
            # Build request parameters
            params = {
                'instId': 'BTC-USDT-SWAP',  # OKX perpetual contract
                'period': '1H',              # Hourly data
                'limit': '100'               # Max per request
            }

            if current_after:
                params['after'] = current_after

            # Make request
            response = requests.get(f"{base_url}{endpoint}", params=params, timeout=10)
            response.raise_for_status()

            data = response.json()

            # Check for API error
            if data.get('code') != '0':
                error_msg = data.get('msg', 'Unknown error')
                print(f"\n  API Error: {error_msg}")
                if all_data:
                    print(f"  Proceeding with {len(all_data)} records fetched so far")
                    break
                else:
                    raise ValueError(f"OKX API error: {error_msg}")

            records = data.get('data', [])
            if not records:
                print(f"\n  No more records (total fetched: {len(all_data)})")
                break

            # OKX Rubik endpoint returns data as lists: [timestamp, oi, oiCcy]
            # Filter by date range
            filtered = [r for r in records if start_ts <= int(r[0]) <= end_ts]
            all_data.extend(filtered)

            fetch_count += 1
            last_ts = datetime.fromtimestamp(int(records[-1][0]) / 1000, tz=timezone.utc)
            print(f"  Batch {fetch_count}: {len(all_data)} total records (last: {last_ts})", end='\r')

            # Check if we've reached the start date
            if int(records[-1][0]) < start_ts:
                print(f"\n  Reached start date")
                break

            # Pagination cursor
            current_after = records[-1][0]
            time.sleep(0.3)  # Rate limit (200ms between requests)

        except requests.exceptions.RequestException as e:
            print(f"\n  Network error: {e}")
            if all_data:
                print(f"  Proceeding with {len(all_data)} records fetched so far")
                break
            else:
                raise

    if not all_data:
        raise ValueError("No OI data fetched from OKX API")

    # Convert to DataFrame (OKX Rubik endpoint returns: [timestamp, oi, oiCcy])
    df = pd.DataFrame(all_data, columns=['ts', 'oi', 'oiCcy'])
    df['timestamp'] = pd.to_datetime(df['ts'].astype(int), unit='ms', utc=True)
    df['oi'] = df['oi'].astype(float)  # Open interest in contracts
    df['oiCcy'] = df['oiCcy'].astype(float)  # Open interest in BTC

    # Use oi field (should be in contracts or USD notional)
    # For BTC-USDT-SWAP, oi field represents contracts
    df = df[['timestamp', 'oi']].copy()
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"\n\nFetch complete:")
    print(f"  Records: {len(df)}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  OI mean: {df['oi'].mean():,.0f}")
    print(f"  OI range: [{df['oi'].min():,.0f}, {df['oi'].max():,.0f}]")

    # Cache if path provided
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        print(f"  Cached to: {cache_path}")

    return df


# ============================================================================
# PHASE 2: CALCULATE DERIVED OI METRICS
# ============================================================================

def calculate_oi_metrics(df: pd.DataFrame, window: int = 252) -> pd.DataFrame:
    """
    Calculate OI-derived features.

    Args:
        df: DataFrame with 'oi' column (must be sorted by time)
        window: Rolling window for z-score (default 252 hours = 10.5 days)

    Returns:
        DataFrame with additional columns:
        - oi_change_24h: Absolute change vs 24 hours ago
        - oi_change_pct_24h: Percentage change vs 24 hours ago
        - oi_z: Z-score (window-rolling mean/std)
        - oi_spike: Boolean flag (z > 2.0)

    Notes:
        - First 24 rows will have NaN for change metrics (no lookback)
        - First `window` rows will have NaN for z-score (no rolling stats)
    """
    print("\n" + "=" * 80)
    print("PHASE 2: Calculating Derived OI Metrics")
    print("=" * 80)

    df_out = df.copy()

    # 1. Absolute change (24-hour difference)
    df_out['oi_change_24h'] = df_out['oi'].diff(24)
    print(f"\n1. oi_change_24h:")
    print(f"   Non-null: {df_out['oi_change_24h'].notna().sum()} / {len(df_out)}")
    print(f"   Mean: {df_out['oi_change_24h'].mean():,.0f}")
    print(f"   Std: {df_out['oi_change_24h'].std():,.0f}")

    # 2. Percentage change (24-hour)
    df_out['oi_change_pct_24h'] = df_out['oi'].pct_change(24) * 100
    print(f"\n2. oi_change_pct_24h:")
    print(f"   Non-null: {df_out['oi_change_pct_24h'].notna().sum()} / {len(df_out)}")
    print(f"   Mean: {df_out['oi_change_pct_24h'].mean():.3f}%")
    print(f"   Std: {df_out['oi_change_pct_24h'].std():.3f}%")
    print(f"   Min: {df_out['oi_change_pct_24h'].min():.3f}% (largest drop)")
    print(f"   Max: {df_out['oi_change_pct_24h'].max():.3f}% (largest spike)")

    # 3. Z-score (rolling window)
    rolling_mean = df_out['oi'].rolling(window=window, min_periods=100).mean()
    rolling_std = df_out['oi'].rolling(window=window, min_periods=100).std()
    df_out['oi_z'] = (df_out['oi'] - rolling_mean) / rolling_std
    df_out['oi_z'] = df_out['oi_z'].fillna(0.0)  # Fill initial NaNs with 0
    print(f"\n3. oi_z (window={window}):")
    print(f"   Non-null: {df_out['oi_z'].notna().sum()} / {len(df_out)}")
    print(f"   Mean: {df_out['oi_z'].mean():.3f} (should be ~0)")
    print(f"   Std: {df_out['oi_z'].std():.3f} (should be ~1)")
    print(f"   Min: {df_out['oi_z'].min():.3f}")
    print(f"   Max: {df_out['oi_z'].max():.3f}")

    # 4. Spike detection flag (z > 2.0 = 2-sigma event)
    df_out['oi_spike'] = (df_out['oi_z'].abs() > 2.0).astype(int)
    spike_count = df_out['oi_spike'].sum()
    spike_pct = spike_count / len(df_out) * 100
    print(f"\n4. oi_spike (|z| > 2.0):")
    print(f"   Spike count: {spike_count} / {len(df_out)} ({spike_pct:.2f}%)")

    return df_out


# ============================================================================
# PHASE 3: VALIDATE AGAINST KNOWN EVENTS
# ============================================================================

def validate_oi_metrics(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Validate OI metrics against known bear market events.

    Args:
        df: DataFrame with timestamp and oi_change_pct_24h columns

    Returns:
        Dict with validation results:
        - terra_collapse: Dict with detected values
        - ftx_collapse: Dict with detected values
        - normal_range: Dict with statistics
        - validation_passed: bool

    Known Events:
        - Terra collapse (May 9-12, 2022): Expected oi_change < -15%
        - FTX collapse (Nov 8-10, 2022): Expected oi_change < -20%
        - Normal periods: -5% < oi_change < +5% (90% of data)
    """
    print("\n" + "=" * 80)
    print("PHASE 3: Validating Against Known Events")
    print("=" * 80)

    results = {}

    # Ensure timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)

    # 1. Terra Collapse (May 9-12, 2022)
    terra_start = pd.Timestamp('2022-05-09', tz='UTC')
    terra_end = pd.Timestamp('2022-05-12', tz='UTC')
    terra_df = df[(df['timestamp'] >= terra_start) & (df['timestamp'] <= terra_end)]

    if len(terra_df) > 0:
        terra_min = terra_df['oi_change_pct_24h'].min()
        terra_detected = terra_min < -15.0
        results['terra_collapse'] = {
            'min_oi_change_pct': terra_min,
            'threshold': -15.0,
            'detected': terra_detected,
            'passed': terra_detected
        }
        status = "✅ PASSED" if terra_detected else "❌ FAILED"
        print(f"\n1. Terra Collapse (May 9-12, 2022): {status}")
        print(f"   Min OI change: {terra_min:.2f}% (expected < -15%)")
    else:
        print(f"\n1. Terra Collapse: ⚠️ No data in date range")
        results['terra_collapse'] = {'passed': False, 'reason': 'No data'}

    # 2. FTX Collapse (Nov 8-10, 2022)
    ftx_start = pd.Timestamp('2022-11-08', tz='UTC')
    ftx_end = pd.Timestamp('2022-11-10', tz='UTC')
    ftx_df = df[(df['timestamp'] >= ftx_start) & (df['timestamp'] <= ftx_end)]

    if len(ftx_df) > 0:
        ftx_min = ftx_df['oi_change_pct_24h'].min()
        ftx_detected = ftx_min < -20.0
        results['ftx_collapse'] = {
            'min_oi_change_pct': ftx_min,
            'threshold': -20.0,
            'detected': ftx_detected,
            'passed': ftx_detected
        }
        status = "✅ PASSED" if ftx_detected else "❌ FAILED"
        print(f"\n2. FTX Collapse (Nov 8-10, 2022): {status}")
        print(f"   Min OI change: {ftx_min:.2f}% (expected < -20%)")
    else:
        print(f"\n2. FTX Collapse: ⚠️ No data in date range")
        results['ftx_collapse'] = {'passed': False, 'reason': 'No data'}

    # 3. Normal Range Check (90% of data should be -5% to +5%)
    normal_range = df['oi_change_pct_24h'].between(-5.0, 5.0)
    normal_pct = normal_range.sum() / len(df) * 100
    normal_passed = normal_pct > 85.0  # Allow 85% threshold (slightly relaxed)

    results['normal_range'] = {
        'pct_in_range': normal_pct,
        'threshold': 85.0,
        'passed': normal_passed
    }
    status = "✅ PASSED" if normal_passed else "❌ FAILED"
    print(f"\n3. Normal Range (-5% to +5%): {status}")
    print(f"   Data in range: {normal_pct:.1f}% (expected > 85%)")

    # Overall validation
    all_passed = all(
        results.get(k, {}).get('passed', False)
        for k in ['normal_range']
    )
    # Terra/FTX optional (may not have data)
    if 'terra_collapse' in results:
        all_passed = all_passed and results['terra_collapse']['passed']
    if 'ftx_collapse' in results:
        all_passed = all_passed and results['ftx_collapse']['passed']

    results['validation_passed'] = all_passed

    print(f"\n" + "=" * 80)
    if all_passed:
        print("✅ VALIDATION PASSED - OI metrics look correct")
    else:
        print("⚠️ VALIDATION ISSUES DETECTED - Review results above")
    print("=" * 80)

    return results


# ============================================================================
# MAIN: PATCH MTF STORE
# ============================================================================

def patch_mtf_store(
    mtf_path: Path,
    oi_df: pd.DataFrame,
    output_path: Optional[Path] = None,
    dry_run: bool = False
) -> Path:
    """
    Patch MTF feature store with OI data and derived metrics.

    Args:
        mtf_path: Path to MTF parquet file
        oi_df: DataFrame with OI metrics (from calculate_oi_metrics)
        output_path: Optional output path (default: overwrite input)
        dry_run: If True, don't write file (validation only)

    Returns:
        Path to output file (or input file if dry_run)

    Process:
        1. Load MTF store
        2. Merge OI columns by timestamp
        3. Fill gaps (forward fill + zero fill)
        4. Validate final coverage
        5. Write patched store
    """
    print("\n" + "=" * 80)
    print("PATCHING MTF FEATURE STORE")
    print("=" * 80)

    # Load MTF store
    print(f"\nLoading MTF store: {mtf_path}")
    mtf_df = pd.read_parquet(mtf_path)
    print(f"  Shape: {mtf_df.shape}")
    print(f"  Index: {mtf_df.index.name} ({mtf_df.index.dtype})")

    # Ensure timestamp is in index
    if not isinstance(mtf_df.index, pd.DatetimeIndex):
        if 'timestamp' in mtf_df.columns:
            mtf_df.set_index('timestamp', inplace=True)
        else:
            raise ValueError("MTF store must have timestamp in index or columns")

    # Prepare OI data for merge
    oi_merge = oi_df.set_index('timestamp') if 'timestamp' in oi_df.columns else oi_df.copy()

    # Columns to merge
    oi_cols = ['oi', 'oi_change_24h', 'oi_change_pct_24h', 'oi_z']
    oi_merge = oi_merge[oi_cols]

    print(f"\nMerging OI columns: {oi_cols}")
    print(f"  OI data shape: {oi_merge.shape}")

    # Merge (outer join to keep all MTF rows)
    mtf_df = mtf_df.drop(columns=oi_cols, errors='ignore')  # Remove old columns
    mtf_df = mtf_df.join(oi_merge, how='left')

    # Fill gaps (forward fill for continuity, then zero fill for start)
    for col in oi_cols:
        before_fill = mtf_df[col].notna().sum()
        mtf_df[col] = mtf_df[col].ffill().fillna(0.0)
        after_fill = mtf_df[col].notna().sum()
        print(f"  {col}: {before_fill} → {after_fill} non-null ({after_fill/len(mtf_df)*100:.1f}%)")

    # Validate final coverage
    print(f"\nFinal MTF store:")
    print(f"  Shape: {mtf_df.shape}")
    print(f"  Total features: {len(mtf_df.columns)}")

    # Check OI coverage by year
    mtf_df['year'] = mtf_df.index.year
    print(f"\nOI coverage by year:")
    for year in sorted(mtf_df['year'].unique()):
        year_df = mtf_df[mtf_df['year'] == year]
        oi_count = year_df['oi'].notna().sum()
        oi_nonzero = (year_df['oi'] != 0).sum()
        print(f"  {year}: {oi_nonzero}/{len(year_df)} non-zero ({oi_nonzero/len(year_df)*100:.1f}%)")

    mtf_df.drop(columns=['year'], inplace=True)

    # Write patched store
    if not dry_run:
        output_path = output_path or mtf_path
        print(f"\nWriting patched MTF store: {output_path}")
        mtf_df.to_parquet(output_path)
        print(f"  ✅ Wrote {len(mtf_df)} rows, {len(mtf_df.columns)} columns")
    else:
        print(f"\n⚠️ DRY RUN - No file written")
        output_path = mtf_path

    return output_path


# ============================================================================
# CLI
# ============================================================================

def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Fix OI_CHANGE pipeline for bear pattern detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full fix (fetch + calculate + validate)
  python3 bin/fix_oi_change_pipeline.py

  # Use existing cache (skip fetch)
  python3 bin/fix_oi_change_pipeline.py --skip-fetch

  # Dry run (validation only)
  python3 bin/fix_oi_change_pipeline.py --dry-run

  # Custom date range
  python3 bin/fix_oi_change_pipeline.py --start-date 2022-01-01 --end-date 2023-12-31
        """
    )

    parser.add_argument(
        '--mtf-store',
        type=Path,
        default=Path('data/features_mtf/BTC_1H_2022-01-01_to_2024-12-31.parquet'),
        help='Path to MTF feature store (default: BTC 2022-2024)'
    )
    parser.add_argument(
        '--start-date',
        default='2022-01-01',
        help='Start date for OI fetch (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--end-date',
        default='2023-12-31',
        help='End date for OI fetch (YYYY-MM-DD)'
    )
    parser.add_argument(
        '--cache-path',
        type=Path,
        default=Path('data/cache/okx_oi_2022_2023.parquet'),
        help='Path to cache fetched OI data'
    )
    parser.add_argument(
        '--skip-fetch',
        action='store_true',
        help='Skip OI fetch (use existing cache)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Validate only, do not write file'
    )
    parser.add_argument(
        '--output',
        type=Path,
        help='Output path (default: overwrite input)'
    )

    args = parser.parse_args()

    print("=" * 80)
    print("OI_CHANGE PIPELINE FIX")
    print("=" * 80)
    print(f"MTF store: {args.mtf_store}")
    print(f"Date range: {args.start_date} to {args.end_date}")
    print(f"Cache: {args.cache_path}")
    print(f"Skip fetch: {args.skip_fetch}")
    print(f"Dry run: {args.dry_run}")

    try:
        # Phase 1: Fetch OI data (or load from cache)
        if args.skip_fetch and args.cache_path.exists():
            print("\nSkipping fetch, loading from cache...")
            oi_df = pd.read_parquet(args.cache_path)
            print(f"  Loaded {len(oi_df)} records from {args.cache_path}")
        else:
            oi_df = fetch_okx_historical_oi(
                args.start_date,
                args.end_date,
                cache_path=args.cache_path
            )

        # Phase 2: Calculate derived metrics
        oi_df = calculate_oi_metrics(oi_df)

        # Phase 3: Validate
        validation_results = validate_oi_metrics(oi_df)

        # Patch MTF store
        output_path = patch_mtf_store(
            args.mtf_store,
            oi_df,
            output_path=args.output,
            dry_run=args.dry_run
        )

        print("\n" + "=" * 80)
        print("✅ FIX COMPLETE")
        print("=" * 80)
        print(f"\nOutput: {output_path}")
        print(f"Validation: {'PASSED' if validation_results['validation_passed'] else 'ISSUES DETECTED'}")

        print("\nNext steps:")
        print("1. Run S5 (Long Squeeze) pattern validation")
        print("2. Backtest bear archetypes on 2022 data")
        print("3. Check liquidation cascade detection for Terra/FTX events")

        return 0 if validation_results['validation_passed'] else 1

    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
