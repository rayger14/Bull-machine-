#!/usr/bin/env python3
"""
Audit Archetype Pipeline - Comprehensive Feature Coverage Check

Verifies that archetypes have access to all required domain features.
Reports:
1. Feature coverage % by domain (Wyckoff, SMC, Temporal, etc.)
2. Domain engine enablement status
3. Missing features that prevent full brain activation
4. Config parameter status (default vs optimized)

Expected Result After Fixes:
- Feature coverage: 95-100% (up from ~20%)
- All 6 domain engines: ENABLED
- Missing features: 0-5 (temporal only, if not implemented)

Usage:
    python bin/audit_archetype_pipeline.py
    python bin/audit_archetype_pipeline.py --s1
    python bin/audit_archetype_pipeline.py --verbose
"""

import json
from pathlib import Path
from typing import Dict, List
import pandas as pd
import sys

# Add engine to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.features.feature_mapper import FeatureMapper


def load_feature_store() -> pd.DataFrame:
    """Load feature store data."""
    data_paths = [
        'data/features_2022_with_regimes.parquet',
        'data/feature_store.parquet',
    ]

    for path in data_paths:
        if Path(path).exists():
            return pd.read_parquet(path)

    raise FileNotFoundError(f"No feature store found. Tried: {data_paths}")


def audit_config_engines(config_path: Path) -> Dict:
    """Audit domain engine enablement in config."""
    with open(config_path) as f:
        cfg = json.load(f)

    feature_flags = cfg.get('feature_flags', {})

    engines = {
        'wyckoff': feature_flags.get('enable_wyckoff', False),
        'smc': feature_flags.get('enable_smc', False),
        'temporal': feature_flags.get('enable_temporal', False),
        'hob': feature_flags.get('enable_hob', False),
        'fusion': feature_flags.get('enable_fusion', False),
        'macro': feature_flags.get('enable_macro', False),
    }

    enabled_count = sum(engines.values())
    total_count = len(engines)

    return {
        'engines': engines,
        'enabled_count': enabled_count,
        'total_count': total_count,
        'enabled_pct': enabled_count / total_count * 100 if total_count > 0 else 0
    }


def audit_config_calibration(config_path: Path, archetype_key: str) -> Dict:
    """Check if config has calibration metadata."""
    with open(config_path) as f:
        cfg = json.load(f)

    archetype_cfg = cfg.get('archetypes', {}).get('thresholds', {}).get(archetype_key, {})
    metadata = archetype_cfg.get('_calibration_metadata', {})

    is_calibrated = bool(metadata and 'best_pf' in metadata)

    return {
        'is_calibrated': is_calibrated,
        'metadata': metadata
    }


def print_domain_coverage(coverage: Dict, verbose: bool = False):
    """Print domain coverage in a formatted table."""
    print("\nDOMAIN FEATURE COVERAGE")
    print("=" * 70)
    print(f"{'Domain':<20} {'Available':<12} {'Total':<8} {'Coverage':<10} {'Status'}")
    print("-" * 70)

    for domain, stats in sorted(coverage.items()):
        avail = stats['available']
        total = stats['total']
        pct = stats['coverage_pct']

        # Determine status
        if pct >= 90:
            status = "FULL"
        elif pct >= 75:
            status = "PARTIAL"
        elif pct > 0:
            status = "LOW"
        else:
            status = "NONE"

        print(f"{domain:<20} {avail:<12} {total:<8} {pct:>6.1f}%    {status}")

        if verbose and stats['missing']:
            print(f"  Missing: {', '.join(stats['missing'])}")

    print("=" * 70)

    # Summary
    total_features = sum(s['total'] for s in coverage.values())
    total_available = sum(s['available'] for s in coverage.values())
    overall_pct = total_available / total_features * 100 if total_features > 0 else 0

    print(f"\nOverall Coverage: {total_available}/{total_features} ({overall_pct:.1f}%)")

    if overall_pct >= 95:
        print("   EXCELLENT - Full brain activation")
    elif overall_pct >= 75:
        print("   PARTIAL - Some domains limited")
    else:
        print("   CRITICAL - Major feature gaps detected")


def print_engine_status(engine_audit: Dict):
    """Print domain engine enablement status."""
    print("\nDOMAIN ENGINE STATUS")
    print("=" * 70)

    for engine, enabled in sorted(engine_audit['engines'].items()):
        status = "ENABLED" if enabled else "DISABLED"
        print(f"  {engine.upper():<15} {status}")

    print("-" * 70)
    enabled = engine_audit['enabled_count']
    total = engine_audit['total_count']
    pct = engine_audit['enabled_pct']

    print(f"  Total: {enabled}/{total} engines enabled ({pct:.0f}%)")

    if pct >= 100:
        print("  All engines active - full brain mode")
    elif pct >= 50:
        print("  Partial activation - some domains offline")
    else:
        print("  CRITICAL - most engines disabled (running at <50% capacity)")


def print_calibration_status(calib_audit: Dict):
    """Print calibration status."""
    print("\nCALIBRATION STATUS")
    print("=" * 70)

    if calib_audit['is_calibrated']:
        meta = calib_audit['metadata']
        print(f"  Status: CALIBRATED")
        print(f"  Study: {meta.get('study_name', 'Unknown')}")
        print(f"  Trial: {meta.get('trial_id', 'Unknown')}")
        print(f"  Best PF: {meta.get('best_pf', 0):.4f}")
        print(f"  Applied: {meta.get('applied_date', 'Unknown')[:10]}")
    else:
        print(f"  Status: NOT CALIBRATED")
        print(f"  Using default/placeholder parameters")
        print(f"  Run: bin/apply_optimized_calibrations.py to apply Optuna results")


def audit_archetype(archetype: str, config_path: Path, archetype_key: str,
                   feature_store: pd.DataFrame, verbose: bool = False):
    """Audit a single archetype."""
    print("\n" + "=" * 70)
    print(f"ARCHETYPE: {archetype.upper()}")
    print(f"Config: {config_path}")
    print("=" * 70)

    # Check config exists
    if not config_path.exists():
        print(f"Config not found: {config_path}")
        return False

    # 1. Feature coverage audit
    mapper = FeatureMapper()
    coverage = mapper.audit_feature_coverage(feature_store)
    print_domain_coverage(coverage, verbose)

    # 2. Engine status audit
    engine_audit = audit_config_engines(config_path)
    print_engine_status(engine_audit)

    # 3. Calibration status audit
    calib_audit = audit_config_calibration(config_path, archetype_key)
    print_calibration_status(calib_audit)

    # Overall assessment
    print("\nOVERALL ASSESSMENT")
    print("=" * 70)

    total_features = sum(s['total'] for s in coverage.values())
    total_available = sum(s['available'] for s in coverage.values())
    coverage_pct = total_available / total_features * 100 if total_features > 0 else 0

    engine_pct = engine_audit['enabled_pct']
    is_calibrated = calib_audit['is_calibrated']

    issues = []
    if coverage_pct < 95:
        issues.append(f"Feature coverage low ({coverage_pct:.0f}%)")
    if engine_pct < 100:
        issues.append(f"Some engines disabled ({engine_pct:.0f}%)")
    if not is_calibrated:
        issues.append("Not calibrated (using defaults)")

    if not issues:
        print("READY FOR PRODUCTION")
        print("  - Full feature coverage")
        print("  - All engines enabled")
        print("  - Calibrated parameters")
        return True
    else:
        print("NEEDS ATTENTION")
        for issue in issues:
            print(f"  - {issue}")

        print("\nRECOMMENDED ACTIONS:")
        if coverage_pct < 95:
            print("  1. Verify feature store is up to date")
            print("  2. Check for missing feature pipelines")
        if engine_pct < 100:
            print("  1. Run: bin/enable_domain_engines.py --" + archetype.lower())
        if not is_calibrated:
            print("  1. Run: bin/apply_optimized_calibrations.py --" + archetype.lower())

        return False


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Audit archetype feature coverage and configuration",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument('--s1', action='store_true', help='Audit S1 only')
    parser.add_argument('--s4', action='store_true', help='Audit S4 only')
    parser.add_argument('--s5', action='store_true', help='Audit S5 only')
    parser.add_argument('--all', action='store_true', help='Audit all archetypes')
    parser.add_argument('--verbose', '-v', action='store_true', help='Show missing feature details')

    args = parser.parse_args()

    # Load feature store
    try:
        print("Loading feature store...")
        feature_store = load_feature_store()
        print(f"Loaded {len(feature_store)} rows, {len(feature_store.columns)} columns")
    except Exception as e:
        print(f"Error loading feature store: {e}")
        sys.exit(1)

    # Define archetypes
    archetype_specs = {
        's1': (Path('configs/s1_v2_production.json'), 'liquidity_vacuum'),
        's4': (Path('configs/s4_optimized_oos_test.json'), 'funding_divergence'),
        's5': (Path('configs/system_s5_production.json'), 'long_squeeze'),
    }

    # Determine which to audit
    if args.all:
        to_audit = ['s1', 's4', 's5']
    elif args.s1 or args.s4 or args.s5:
        to_audit = []
        if args.s1:
            to_audit.append('s1')
        if args.s4:
            to_audit.append('s4')
        if args.s5:
            to_audit.append('s5')
    else:
        # Default: audit all
        to_audit = ['s1', 's4', 's5']

    # Audit each
    results = {}
    for archetype in to_audit:
        config_path, archetype_key = archetype_specs[archetype]
        success = audit_archetype(
            archetype,
            config_path,
            archetype_key,
            feature_store,
            verbose=args.verbose
        )
        results[archetype] = success

    # Final summary
    print("\n" + "=" * 70)
    print("AUDIT SUMMARY")
    print("=" * 70)

    for archetype, success in results.items():
        status = "READY" if success else "NEEDS WORK"
        print(f"  {archetype.upper()}: {status}")

    all_ready = all(results.values())
    if all_ready:
        print("\nAll archetypes ready for production")
        sys.exit(0)
    else:
        print("\nSome archetypes need fixes")
        sys.exit(1)


if __name__ == "__main__":
    main()
