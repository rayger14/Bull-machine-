#!/usr/bin/env python3
"""
Validate Archetype Coverage - Check Feature Requirements
=========================================================

Validates that all 16 archetypes have their required features in the feature store.

This script:
1. Loads archetype registry (archetype_registry.yaml)
2. Checks which features each archetype requires
3. Validates feature store has all required features
4. Reports missing features blocking each archetype

Usage:
    python bin/validate_archetype_coverage.py
    python bin/validate_archetype_coverage.py --features data/features_mtf/BTC_1H_FULL_BACKFILLED_2018-01-01_to_2024-12-31.parquet

Author: Claude Code (Backend Architect)
Date: 2026-01-21
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import yaml
import argparse
import logging
from typing import Dict, List, Set

logging.basicConfig(
    level=logging.INFO,
    format='%(levelname)s: %(message)s'
)
logger = logging.getLogger(__name__)


def load_archetype_registry(registry_path: str = 'archetype_registry.yaml') -> Dict:
    """Load archetype registry from YAML."""
    with open(registry_path, 'r') as f:
        return yaml.safe_load(f)


def extract_required_features(archetype: Dict) -> Set[str]:
    """Extract all required features (critical + recommended) from archetype spec."""
    required = set()

    features = archetype.get('requires_features', {})

    # Critical features (must have)
    if 'critical' in features:
        required.update(features['critical'])

    # Recommended features (should have)
    if 'recommended' in features:
        required.update(features['recommended'])

    return required


def map_feature_name(feature_name: str, available_columns: List[str]) -> str:
    """
    Map canonical feature name to actual column name in feature store.

    Handles common variations:
    - adx_14 → adx
    - bos_detected → tf1h_bos_bullish
    - pti_score_1h → tf1h_pti_trap_type
    """
    # Direct match
    if feature_name in available_columns:
        return feature_name

    # Common mappings
    mappings = {
        'adx_14': 'adx',
        'atr_20': 'atr_14',
        'bos_detected': 'tf1h_bos_bullish',
        'choch_detected': 'tf4h_choch_flag',
        'pti_score_1h': 'tf1h_pti_trap_type',
        'pti_score_1d': 'tf1d_pti_score',
        'wyckoff_phase': 'wyckoff_phase_abc',
        'order_block_proximity': 'tf1h_ob_high',  # Proxy
        'boms_strength': 'tf1d_boms_strength',
        'wick_lower_ratio': 'lower_wick',
        'wick_upper_ratio': 'upper_wick',
        'volume_zscore': 'volume_z',
        'oi_change': 'tf1d_boms_displacement',  # Proxy
        'funding_rate': 'funding_Z',  # Proxy
        'liquidity_drain_pct': 'liquidity_score',  # Proxy (inverse)
        'displacement_detected': 'tf4h_boms_displacement',
        'atr_percentile': 'atr_pct',
        'fvg_reclaim': 'tf1h_fvg_present',  # Proxy
        'wick_anomaly_score': 'lower_wick',  # Proxy
        'wyckoff_m1_signal': 'wyckoff_spring_a',  # Proxy
        'wyckoff_m2_signal': 'wyckoff_utad',  # Proxy
    }

    if feature_name in mappings:
        mapped = mappings[feature_name]
        if mapped in available_columns:
            return mapped

    # Not found
    return None


def validate_archetype_coverage(
    registry: Dict,
    feature_store_columns: List[str]
) -> Dict:
    """
    Validate that all archetypes have their required features.

    Returns:
        Dict with validation results:
        - total_archetypes: int
        - archetypes_ready: int
        - archetypes_blocked: int
        - blocking_features: Dict[archetype_slug, List[missing_features]]
    """
    results = {
        'total_archetypes': 0,
        'archetypes_ready': 0,
        'archetypes_blocked': 0,
        'blocking_features': {},
        'archetype_status': {}
    }

    # Process each archetype
    for archetype in registry.get('archetypes', []):
        slug = archetype['slug']
        maturity = archetype.get('maturity', 'unknown')

        # Skip stub/deprecated archetypes
        if maturity in ['stub', 'deprecated']:
            continue

        results['total_archetypes'] += 1

        # Get required features
        required_features = extract_required_features(archetype)

        # Check which features are missing
        missing_features = []
        available_features = []

        for feature in required_features:
            mapped_feature = map_feature_name(feature, feature_store_columns)

            if mapped_feature is None:
                missing_features.append(feature)
            else:
                available_features.append(f"{feature} → {mapped_feature}")

        # Update results
        if len(missing_features) == 0:
            results['archetypes_ready'] += 1
            results['archetype_status'][slug] = 'READY'
        else:
            results['archetypes_blocked'] += 1
            results['blocking_features'][slug] = missing_features
            results['archetype_status'][slug] = 'BLOCKED'

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Validate archetype coverage against feature store'
    )
    parser.add_argument(
        '--features',
        default='data/features_mtf/BTC_1H_FULL_2018-01-01_to_2024-12-31.parquet',
        help='Path to feature store parquet file'
    )
    parser.add_argument(
        '--registry',
        default='archetype_registry.yaml',
        help='Path to archetype registry YAML'
    )

    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("ARCHETYPE COVERAGE VALIDATION")
    logger.info("=" * 80)
    logger.info("")

    # Load archetype registry
    logger.info(f"Loading archetype registry: {args.registry}")
    registry = load_archetype_registry(args.registry)
    logger.info(f"✅ Loaded {len(registry.get('archetypes', []))} archetypes")
    logger.info("")

    # Load feature store
    logger.info(f"Loading feature store: {args.features}")
    df = pd.read_parquet(args.features)
    logger.info(f"✅ Loaded {len(df):,} bars, {len(df.columns)} columns")
    logger.info("")

    # Validate coverage
    logger.info("Validating archetype coverage...")
    results = validate_archetype_coverage(registry, df.columns.tolist())
    logger.info("")

    # Report results
    logger.info("=" * 80)
    logger.info("VALIDATION RESULTS")
    logger.info("=" * 80)
    logger.info(f"Total archetypes:    {results['total_archetypes']}")
    logger.info(f"Archetypes ready:    {results['archetypes_ready']} ✅")
    logger.info(f"Archetypes blocked:  {results['archetypes_blocked']} ❌")
    logger.info("")

    # Report status by archetype
    logger.info("Archetype Status:")
    logger.info("-" * 80)

    for archetype in registry.get('archetypes', []):
        slug = archetype['slug']
        maturity = archetype.get('maturity', 'unknown')

        if maturity in ['stub', 'deprecated']:
            continue

        status = results['archetype_status'].get(slug, 'UNKNOWN')
        status_icon = "✅" if status == 'READY' else "❌"

        logger.info(f"  {status_icon} {slug:30s} {status}")

    logger.info("")

    # Report blocking features
    if results['archetypes_blocked'] > 0:
        logger.info("=" * 80)
        logger.info("BLOCKING FEATURES (Missing Features by Archetype)")
        logger.info("=" * 80)

        for slug, missing_features in results['blocking_features'].items():
            logger.info(f"\n{slug}:")
            for feature in missing_features:
                logger.info(f"  - {feature}")

        logger.info("")
        logger.info("=" * 80)
        logger.info("RECOMMENDATION")
        logger.info("=" * 80)
        logger.info("Run feature backfilling to restore missing features:")
        logger.info("  python bin/backfill_historical_features.py --phase all --validate")
        logger.info("")

    else:
        logger.info("=" * 80)
        logger.info("✅ ALL ARCHETYPES READY - Feature coverage 100%")
        logger.info("=" * 80)
        logger.info("")
        logger.info("Next steps:")
        logger.info("  1. Run archetype smoke test: python bin/smoke_test_all_archetypes.py")
        logger.info("  2. Validate signal counts (expect 100-200 trades)")
        logger.info("  3. Run baseline backtest")
        logger.info("")

    return 0 if results['archetypes_blocked'] == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
