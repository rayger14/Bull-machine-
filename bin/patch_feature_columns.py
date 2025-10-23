#!/usr/bin/env python3
"""
Production-ready feature store column patcher.

Patches specific columns in existing feature stores without full rebuild.
Includes health checks, atomic writes, and JSON output for CI validation.

Usage:
    python bin/patch_feature_columns.py \
        --asset BTC --tf 1H --start 2024-01-01 --end 2024-12-31 \
        --cols tf4h_boms_displacement,tf1d_boms_strength

Author: Bull Machine v2.0 - PR#1 Infrastructure & Safety
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.structure.boms_detector import detect_boms


# ============================================================================
# Section 1: Core Loader/Writer
# ============================================================================

def load_feature_store(asset: str, tf: str, start: str, end: str) -> Tuple[pd.DataFrame, Path]:
    """
    Load existing feature store parquet file.

    Args:
        asset: Asset symbol (BTC, ETH, etc.)
        tf: Timeframe (1H, 4H, 1D)
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)

    Returns:
        (DataFrame, Path to file)

    Raises:
        FileNotFoundError: If feature store doesn't exist
    """
    path = Path(f"data/features_mtf/{asset}_{tf}_{start}_to_{end}.parquet")

    if not path.exists():
        raise FileNotFoundError(f"Feature store not found: {path}")

    logging.info(f"Loading feature store: {path}")
    df = pd.read_parquet(path)
    logging.info(f"  Loaded {len(df)} rows × {len(df.columns)} columns")

    return df, path


def atomic_save(df: pd.DataFrame, path: Path, cols_patched: List[str]) -> None:
    """
    Atomically save patched feature store with backup.

    Uses tmp file → os.replace pattern for safe writes.

    Args:
        df: Patched DataFrame
        path: Target parquet path
        cols_patched: List of columns that were patched
    """
    tmp_path = path.with_suffix('.parquet.tmp')
    backup_path = path.with_suffix('.parquet.backup')

    logging.info("Saving patched feature store...")

    # Add metadata about patch
    df.attrs['__patched_cols'] = ','.join(cols_patched)
    df.attrs['__patched_at'] = datetime.now().isoformat()

    # Write to temp file
    df.to_parquet(tmp_path)
    logging.info(f"  Wrote tmp: {tmp_path}")

    # Backup original (if exists)
    if path.exists():
        if backup_path.exists():
            backup_path.unlink()
        path.rename(backup_path)
        logging.info(f"  Backed up: {backup_path}")

    # Atomic replace
    tmp_path.rename(path)
    logging.info(f"  ✓ Saved: {path}")


# ============================================================================
# Section 2: Column Registry + Calculator Stubs
# ============================================================================

COLUMN_REGISTRY = {
    'tf4h_boms_displacement': {
        'description': 'BOMS displacement on 4H timeframe (absolute price)',
        'expected_nonzero_min': 0.05,  # Expect > 5% non-zero
        'calculator': 'patch_boms_displacement'
    },
    'tf1d_boms_strength': {
        'description': 'BOMS strength on 1D timeframe (normalized 0-1)',
        'expected_nonzero_min': 0.05,
        'calculator': 'patch_boms_strength'
    },
    'tf4h_fusion_score': {
        'description': 'Fusion score from 4H structure indicators (0-1)',
        'expected_nonzero_min': 0.15,
        'calculator': 'patch_tf4h_fusion'
    },
}


def resample_to_timeframe(df_1h: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """
    Resample 1H OHLCV to higher timeframe.

    Args:
        df_1h: 1H OHLCV DataFrame with DatetimeIndex
        timeframe: Target timeframe ('4H', '1D')

    Returns:
        Resampled DataFrame
    """
    agg_dict = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }

    return df_1h.resample(timeframe).agg(agg_dict).dropna()


def patch_boms_displacement(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute tf4h_boms_displacement with fixed calculation.

    BOMS displacement is calculated in ABSOLUTE price terms (not percentages)
    for use in archetype system threshold comparisons with ATR multiples.

    Args:
        df: 1H feature store DataFrame

    Returns:
        DataFrame with patched 'tf4h_boms_displacement' column
    """
    logging.info("Patching tf4h_boms_displacement...")

    # Ensure column exists
    if 'tf4h_boms_displacement' not in df.columns:
        df['tf4h_boms_displacement'] = 0.0

    # Resample to 4H for BOMS detection
    df_4h = resample_to_timeframe(df[['open', 'high', 'low', 'close', 'volume']], '4H')

    for idx in range(len(df)):
        if idx % 500 == 0:
            logging.info(f"  Processing bar {idx}/{len(df)}...")

        timestamp = df.index[idx]
        window_4h = df_4h[df_4h.index <= timestamp].tail(100)

        if len(window_4h) >= 30:
            boms_4h = detect_boms(window_4h, timeframe='4H')
            df.iloc[idx, df.columns.get_loc('tf4h_boms_displacement')] = boms_4h.displacement

    non_zero_count = (df['tf4h_boms_displacement'] > 0).sum()
    non_zero_pct = non_zero_count / len(df) * 100
    logging.info(f"  ✓ Non-zero: {non_zero_count}/{len(df)} ({non_zero_pct:.1f}%)")

    return df


def patch_boms_strength(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute tf1d_boms_strength with proper normalization.

    BOMS strength is displacement normalized to [0, 1] range:
        strength = min(displacement / (2.0 × ATR_1D), 1.0)

    Args:
        df: 1H feature store DataFrame

    Returns:
        DataFrame with patched 'tf1d_boms_strength' column
    """
    logging.info("Patching tf1d_boms_strength...")

    # Ensure column exists
    if 'tf1d_boms_strength' not in df.columns:
        df['tf1d_boms_strength'] = 0.0

    # Resample to 1D for BOMS detection
    df_1d = resample_to_timeframe(df[['open', 'high', 'low', 'close', 'volume']], '1D')

    for idx in range(len(df)):
        if idx % 500 == 0:
            logging.info(f"  Processing bar {idx}/{len(df)}...")

        timestamp = df.index[idx]
        window_1d = df_1d[df_1d.index <= timestamp].tail(100)

        if len(window_1d) >= 30:
            boms_1d = detect_boms(window_1d, timeframe='1D')

            # Calculate ATR for normalization
            atr_1d = window_1d['close'].pct_change().abs().rolling(14).mean().iloc[-1] * window_1d['close'].iloc[-1]

            if atr_1d > 0 and boms_1d.displacement > 0:
                strength = min(boms_1d.displacement / (2.0 * atr_1d), 1.0)
                df.iloc[idx, df.columns.get_loc('tf1d_boms_strength')] = strength

    non_zero_count = (df['tf1d_boms_strength'] > 0).sum()
    non_zero_pct = non_zero_count / len(df) * 100
    logging.info(f"  ✓ Non-zero: {non_zero_count}/{len(df)} ({non_zero_pct:.1f}%)")

    return df


def patch_tf4h_fusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute tf4h_fusion_score from available 4H structure indicators.

    Fusion score = weighted sum of:
        - structure_alignment (30%)
        - squiggle_entry_window (20%)
        - squiggle_confidence (20%)
        - choch_flag (30%)

    Args:
        df: 1H feature store DataFrame

    Returns:
        DataFrame with patched 'tf4h_fusion_score' column
    """
    logging.info("Patching tf4h_fusion_score...")

    # Calculate from available 4H features
    tf4h_fusion = pd.Series(0.0, index=df.index)

    if 'tf4h_structure_alignment' in df.columns:
        tf4h_fusion += df['tf4h_structure_alignment'].astype(float) * 0.30

    if 'tf4h_squiggle_entry_window' in df.columns:
        tf4h_fusion += df['tf4h_squiggle_entry_window'].astype(float) * 0.20

        if 'tf4h_squiggle_confidence' in df.columns:
            tf4h_fusion += df['tf4h_squiggle_confidence'] * 0.20

    if 'tf4h_choch_flag' in df.columns:
        tf4h_fusion += df['tf4h_choch_flag'].astype(float) * 0.30

    df['tf4h_fusion_score'] = tf4h_fusion.clip(upper=1.0)

    non_zero_count = (df['tf4h_fusion_score'] > 0).sum()
    non_zero_pct = non_zero_count / len(df) * 100
    logging.info(f"  ✓ Non-zero: {non_zero_count}/{len(df)} ({non_zero_pct:.1f}%)")

    return df


# Map column names to calculator functions
COLUMN_CALCULATORS = {
    'tf4h_boms_displacement': patch_boms_displacement,
    'tf1d_boms_strength': patch_boms_strength,
    'tf4h_fusion_score': patch_tf4h_fusion,
}


# ============================================================================
# Section 3: Health-Check Summary (JSON)
# ============================================================================

def compute_health_metrics(df: pd.DataFrame, cols: List[str]) -> Dict:
    """
    Compute health metrics for patched columns.

    Returns JSON-serializable dict with:
        - Non-zero counts and percentages
        - Basic statistics (min, max, mean, p50, p75, p95)
        - Health check pass/fail status

    Args:
        df: Feature store DataFrame
        cols: List of patched columns

    Returns:
        Dict with health metrics
    """
    health = {
        'timestamp': datetime.now().isoformat(),
        'total_rows': len(df),
        'columns_patched': cols,
        'metrics': {},
        'health_checks': {}
    }

    for col in cols:
        if col not in df.columns:
            health['metrics'][col] = {'error': 'Column not found'}
            health['health_checks'][col] = 'FAIL'
            continue

        series = df[col]
        non_zero = (series != 0).sum()
        non_zero_pct = non_zero / len(df) * 100

        # Compute statistics (only on non-zero values for better insight)
        non_zero_vals = series[series != 0]

        metrics = {
            'non_zero_count': int(non_zero),
            'non_zero_pct': round(non_zero_pct, 2),
            'min': float(series.min()) if len(series) > 0 else 0.0,
            'max': float(series.max()) if len(series) > 0 else 0.0,
            'mean': float(series.mean()) if len(series) > 0 else 0.0,
            'p50': float(series.quantile(0.50)) if len(series) > 0 else 0.0,
            'p75': float(series.quantile(0.75)) if len(series) > 0 else 0.0,
            'p95': float(series.quantile(0.95)) if len(series) > 0 else 0.0,
        }

        if len(non_zero_vals) > 0:
            metrics['non_zero_mean'] = float(non_zero_vals.mean())
            metrics['non_zero_p50'] = float(non_zero_vals.quantile(0.50))
            metrics['non_zero_p95'] = float(non_zero_vals.quantile(0.95))

        health['metrics'][col] = metrics

        # Health check: Compare against expected minimum
        if col in COLUMN_REGISTRY:
            expected_min = COLUMN_REGISTRY[col]['expected_nonzero_min']
            passed = (non_zero_pct / 100) >= expected_min
            health['health_checks'][col] = 'PASS' if passed else 'FAIL'
        else:
            health['health_checks'][col] = 'UNKNOWN'

    return health


# ============================================================================
# Section 4: CLI Wrapper
# ============================================================================

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Patch specific columns in feature store',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Patch BOMS displacement and strength for BTC 2024
  python bin/patch_feature_columns.py \\
      --asset BTC --tf 1H --start 2024-01-01 --end 2024-12-31 \\
      --cols tf4h_boms_displacement,tf1d_boms_strength

  # Patch all P0 columns
  python bin/patch_feature_columns.py \\
      --asset BTC --tf 1H --start 2024-01-01 --end 2024-12-31 \\
      --cols tf4h_boms_displacement,tf1d_boms_strength,tf4h_fusion_score

  # Output health JSON only (no patching)
  python bin/patch_feature_columns.py \\
      --asset BTC --tf 1H --start 2024-01-01 --end 2024-12-31 \\
      --health-only
        """
    )

    parser.add_argument('--asset', required=True, help='Asset symbol (BTC, ETH, etc.)')
    parser.add_argument('--tf', required=True, help='Timeframe (1H, 4H, 1D)')
    parser.add_argument('--start', required=True, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', required=True, help='End date (YYYY-MM-DD)')
    parser.add_argument('--cols', help='Comma-separated column names to patch')
    parser.add_argument('--health-only', action='store_true',
                        help='Only compute and output health metrics (no patching)')
    parser.add_argument('--json-output', help='Path to save health JSON (default: stdout)')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')

    return parser.parse_args()


# ============================================================================
# Section 5: Main Runner + Logging
# ============================================================================

def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logging.info("=" * 70)
    logging.info("Feature Store Column Patcher - PR#1 Infrastructure & Safety")
    logging.info("=" * 70)

    # Load feature store
    try:
        df, path = load_feature_store(args.asset, args.tf, args.start, args.end)
    except FileNotFoundError as e:
        logging.error(str(e))
        sys.exit(1)

    # Health-only mode: compute metrics and exit
    if args.health_only:
        logging.info("Health-only mode: Computing metrics for all columns in registry...")
        all_cols = list(COLUMN_REGISTRY.keys())
        health = compute_health_metrics(df, all_cols)

        if args.json_output:
            with open(args.json_output, 'w') as f:
                json.dump(health, f, indent=2)
            logging.info(f"Health JSON written to: {args.json_output}")
        else:
            print(json.dumps(health, indent=2))

        sys.exit(0)

    # Patch mode: validate columns and patch
    if not args.cols:
        logging.error("--cols required for patch mode (or use --health-only)")
        sys.exit(1)

    cols_to_patch = [c.strip() for c in args.cols.split(',')]

    logging.info(f"Patching columns: {cols_to_patch}")

    # Validate columns
    unknown_cols = [c for c in cols_to_patch if c not in COLUMN_CALCULATORS]
    if unknown_cols:
        logging.error(f"Unknown columns (no calculator): {unknown_cols}")
        logging.error(f"Available: {list(COLUMN_CALCULATORS.keys())}")
        sys.exit(1)

    # Apply patches
    for col in cols_to_patch:
        calculator_func = COLUMN_CALCULATORS[col]
        df = calculator_func(df)

    # Compute health metrics
    health = compute_health_metrics(df, cols_to_patch)

    # Output health JSON
    if args.json_output:
        with open(args.json_output, 'w') as f:
            json.dump(health, f, indent=2)
        logging.info(f"Health JSON written to: {args.json_output}")
    else:
        logging.info("\nHealth Metrics:")
        print(json.dumps(health, indent=2))

    # Check health status
    failed_checks = [col for col, status in health['health_checks'].items() if status == 'FAIL']
    if failed_checks:
        logging.warning(f"Health checks FAILED for: {failed_checks}")
        logging.warning("Columns patched but did not meet expected non-zero thresholds")
    else:
        logging.info("✓ All health checks PASSED")

    # Save with atomic replace
    atomic_save(df, path, cols_to_patch)

    logging.info("=" * 70)
    logging.info("✅ Patch complete!")
    logging.info("=" * 70)


if __name__ == '__main__':
    main()
