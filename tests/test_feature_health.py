#!/usr/bin/env python3
"""
Feature Health Tests

Validates feature store health after building.
Tests expected non-zero rates and value ranges for P0 columns.

P0 Columns (Critical for Archetype System):
1. tf4h_boms_displacement - BOMS displacement on 4H (absolute price)
2. tf1d_boms_strength - BOMS strength on 1D (normalized 0-1)
3. tf4h_fusion_score - Fusion score from 4H indicators (0-1)

Usage:
    # Run all health tests
    pytest tests/test_feature_health.py -v

    # Test specific feature store
    pytest tests/test_feature_health.py::test_btc_2024_health -v

    # Generate health report
    python tests/test_feature_health.py --asset BTC --year 2024

Architecture:
- Uses thresholds from bin/patch_feature_columns.py COLUMN_REGISTRY
- Fails tests if non-zero rates below expected minimums
- Checks value ranges to prevent unit mismatches
- Generates JSON health reports for CI
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pytest
import json
from typing import Dict, Any
from dataclasses import dataclass
import argparse


# ============================================================================
# Health Check Configuration
# ============================================================================

# P0 column health thresholds (from patch tool COLUMN_REGISTRY)
HEALTH_THRESHOLDS = {
    'tf4h_boms_displacement': {
        'expected_nonzero_min': 0.05,  # > 5% non-zero
        'expected_range': (0.0, 20000.0),  # Absolute price units (e.g., BTC 50-5000)
        'description': 'BOMS displacement on 4H (absolute price)'
    },
    'tf1d_boms_strength': {
        'expected_nonzero_min': 0.05,  # > 5% non-zero
        'expected_range': (0.0, 1.0),  # Normalized strength [0, 1]
        'description': 'BOMS strength on 1D (normalized 0-1)'
    },
    'tf4h_fusion_score': {
        'expected_nonzero_min': 0.15,  # > 15% non-zero
        'expected_range': (0.0, 1.0),  # Fusion score [0, 1]
        'description': 'Fusion score from 4H structure indicators'
    },
}


# ============================================================================
# Health Check Functions
# ============================================================================

@dataclass
class FeatureHealth:
    """Health check result for a single feature."""
    column: str
    total_rows: int
    non_zero_count: int
    non_zero_pct: float
    min_val: float
    max_val: float
    mean_val: float
    p50: float
    p75: float
    p95: float
    health_status: str  # 'PASS', 'FAIL', 'WARN'
    issues: list


def check_feature_health(df: pd.DataFrame, column: str) -> FeatureHealth:
    """
    Check health of a single feature column.

    Args:
        df: Feature store DataFrame
        column: Column name to check

    Returns:
        FeatureHealth dataclass with stats and status

    Example:
        >>> health = check_feature_health(df, 'tf4h_boms_displacement')
        >>> assert health.health_status == 'PASS'
        >>> assert health.non_zero_pct > 5.0
    """
    if column not in df.columns:
        return FeatureHealth(
            column=column,
            total_rows=len(df),
            non_zero_count=0,
            non_zero_pct=0.0,
            min_val=0.0,
            max_val=0.0,
            mean_val=0.0,
            p50=0.0,
            p75=0.0,
            p95=0.0,
            health_status='FAIL',
            issues=[f"Column '{column}' not found in DataFrame"]
        )

    values = df[column].values
    total_rows = len(values)

    # Non-zero stats
    non_zero_mask = values != 0
    non_zero_count = np.sum(non_zero_mask)
    non_zero_pct = (non_zero_count / total_rows) * 100 if total_rows > 0 else 0.0

    # Value stats
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    mean_val = float(np.mean(values))
    p50 = float(np.percentile(values, 50))
    p75 = float(np.percentile(values, 75))
    p95 = float(np.percentile(values, 95))

    # Health checks
    issues = []
    if column in HEALTH_THRESHOLDS:
        config = HEALTH_THRESHOLDS[column]

        # Check non-zero rate
        expected_nonzero_min = config['expected_nonzero_min'] * 100
        if non_zero_pct < expected_nonzero_min:
            issues.append(f"Non-zero rate {non_zero_pct:.1f}% < expected {expected_nonzero_min:.1f}%")

        # Check value range
        expected_min, expected_max = config['expected_range']
        if min_val < expected_min or max_val > expected_max:
            issues.append(f"Value range [{min_val:.2f}, {max_val:.2f}] outside expected [{expected_min}, {expected_max}]")

    # Determine status
    if len(issues) == 0:
        health_status = 'PASS'
    elif non_zero_pct == 0.0:
        health_status = 'FAIL'  # Complete failure
    else:
        health_status = 'WARN'  # Partial issue

    return FeatureHealth(
        column=column,
        total_rows=total_rows,
        non_zero_count=int(non_zero_count),
        non_zero_pct=non_zero_pct,
        min_val=min_val,
        max_val=max_val,
        mean_val=mean_val,
        p50=p50,
        p75=p75,
        p95=p95,
        health_status=health_status,
        issues=issues
    )


def generate_health_report(df: pd.DataFrame, columns: list = None) -> Dict[str, Any]:
    """
    Generate health report for feature store.

    Args:
        df: Feature store DataFrame
        columns: List of columns to check (defaults to P0 columns)

    Returns:
        Dict with health report (suitable for JSON export)

    Example:
        >>> report = generate_health_report(df)
        >>> print(json.dumps(report, indent=2))
    """
    if columns is None:
        columns = list(HEALTH_THRESHOLDS.keys())

    results = {}
    for col in columns:
        health = check_feature_health(df, col)
        results[col] = {
            'total_rows': health.total_rows,
            'non_zero_count': health.non_zero_count,
            'non_zero_pct': health.non_zero_pct,
            'min': health.min_val,
            'max': health.max_val,
            'mean': health.mean_val,
            'p50': health.p50,
            'p75': health.p75,
            'p95': health.p95,
            'health_status': health.health_status,
            'issues': health.issues
        }

    # Overall status
    statuses = [r['health_status'] for r in results.values()]
    if all(s == 'PASS' for s in statuses):
        overall_status = 'PASS'
    elif any(s == 'FAIL' for s in statuses):
        overall_status = 'FAIL'
    else:
        overall_status = 'WARN'

    return {
        'overall_status': overall_status,
        'total_rows': len(df),
        'columns_checked': columns,
        'results': results
    }


# ============================================================================
# Pytest Tests
# ============================================================================

@pytest.fixture
def btc_2024_feature_store():
    """Load BTC 2024 feature store."""
    path = Path('data/features_mtf/BTC_1H_2024-01-01_to_2024-12-31.parquet')
    if not path.exists():
        pytest.skip(f"Feature store not found: {path}")
    return pd.read_parquet(path)


def test_tf4h_boms_displacement_health(btc_2024_feature_store):
    """Test tf4h_boms_displacement meets health thresholds."""
    df = btc_2024_feature_store
    health = check_feature_health(df, 'tf4h_boms_displacement')

    assert health.health_status != 'FAIL', f"tf4h_boms_displacement health check failed: {health.issues}"
    assert health.non_zero_pct >= 5.0, f"Non-zero rate {health.non_zero_pct:.1f}% < 5.0%"
    assert 0.0 <= health.max_val <= 20000.0, f"Value range check failed: max={health.max_val}"


def test_tf1d_boms_strength_health(btc_2024_feature_store):
    """Test tf1d_boms_strength meets health thresholds."""
    df = btc_2024_feature_store
    health = check_feature_health(df, 'tf1d_boms_strength')

    assert health.health_status != 'FAIL', f"tf1d_boms_strength health check failed: {health.issues}"
    assert health.non_zero_pct >= 5.0, f"Non-zero rate {health.non_zero_pct:.1f}% < 5.0%"
    assert 0.0 <= health.max_val <= 1.0, f"Value range check failed: max={health.max_val}"


def test_tf4h_fusion_score_health(btc_2024_feature_store):
    """Test tf4h_fusion_score meets health thresholds."""
    df = btc_2024_feature_store
    health = check_feature_health(df, 'tf4h_fusion_score')

    assert health.health_status != 'FAIL', f"tf4h_fusion_score health check failed: {health.issues}"
    assert health.non_zero_pct >= 15.0, f"Non-zero rate {health.non_zero_pct:.1f}% < 15.0%"
    assert 0.0 <= health.max_val <= 1.0, f"Value range check failed: max={health.max_val}"


def test_all_p0_columns_present(btc_2024_feature_store):
    """Test all P0 columns exist in feature store."""
    df = btc_2024_feature_store
    missing_cols = [col for col in HEALTH_THRESHOLDS.keys() if col not in df.columns]

    assert len(missing_cols) == 0, f"Missing P0 columns: {missing_cols}"


def test_no_nan_in_p0_columns(btc_2024_feature_store):
    """Test P0 columns have no NaN values."""
    df = btc_2024_feature_store

    for col in HEALTH_THRESHOLDS.keys():
        if col in df.columns:
            nan_count = df[col].isna().sum()
            assert nan_count == 0, f"Column '{col}' has {nan_count} NaN values"


def test_no_inf_in_p0_columns(btc_2024_feature_store):
    """Test P0 columns have no infinite values."""
    df = btc_2024_feature_store

    for col in HEALTH_THRESHOLDS.keys():
        if col in df.columns:
            inf_count = np.isinf(df[col].values).sum()
            assert inf_count == 0, f"Column '{col}' has {inf_count} infinite values"


# ============================================================================
# CLI for Health Reports
# ============================================================================

def main():
    """Generate health report from command line."""
    parser = argparse.ArgumentParser(description='Generate feature store health report')
    parser.add_argument('--asset', type=str, required=True, help='Asset symbol (BTC, ETH, SPY, etc.)')
    parser.add_argument('--year', type=int, required=True, help='Year (e.g., 2024)')
    parser.add_argument('--output', type=str, help='Output JSON path (optional)')

    args = parser.parse_args()

    # Load feature store
    path = Path(f'data/features_mtf/{args.asset}_1H_{args.year}-01-01_to_{args.year}-12-31.parquet')
    if not path.exists():
        print(f"ERROR: Feature store not found: {path}")
        sys.exit(1)

    print(f"Loading feature store: {path}")
    df = pd.read_parquet(path)
    print(f"Loaded {len(df)} rows × {len(df.columns)} columns")

    # Generate health report
    print("\nGenerating health report...")
    report = generate_health_report(df)

    # Print summary
    print(f"\n{'='*60}")
    print(f"Health Report: {args.asset} {args.year}")
    print(f"{'='*60}")
    print(f"Overall Status: {report['overall_status']}")
    print(f"Total Rows: {report['total_rows']:,}")
    print(f"Columns Checked: {len(report['columns_checked'])}")
    print()

    for col, result in report['results'].items():
        status_icon = "✅" if result['health_status'] == 'PASS' else "⚠️" if result['health_status'] == 'WARN' else "❌"
        print(f"{status_icon} {col}:")
        print(f"   Non-zero: {result['non_zero_count']:,} / {result['total_rows']:,} ({result['non_zero_pct']:.1f}%)")
        print(f"   Range: [{result['min']:.4f}, {result['max']:.4f}]")
        print(f"   Mean: {result['mean']:.4f}, P50: {result['p50']:.4f}, P95: {result['p95']:.4f}")
        if result['issues']:
            print(f"   Issues: {'; '.join(result['issues'])}")
        print()

    # Save JSON if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"Health report saved to: {output_path}")

    # Exit with status code
    if report['overall_status'] == 'FAIL':
        sys.exit(1)
    elif report['overall_status'] == 'WARN':
        sys.exit(0)  # Warnings don't fail CI
    else:
        sys.exit(0)


if __name__ == '__main__':
    main()
