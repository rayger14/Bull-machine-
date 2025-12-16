#!/usr/bin/env python3
"""
STEP 2: Validate Feature Name Mapping

Ensures that feature names expected by configs exactly match
what exists in the feature store.

Common mapping issues:
- funding_z vs funding_Z
- volume_climax_3b vs volume_climax_last_3b
- wick_exhaustion_3b vs wick_exhaustion_last_3b
- btc_d vs BTC.D
- order_block_bull vs is_bullish_ob

Usage:
    python bin/verify_feature_mapping.py
    python bin/verify_feature_mapping.py --config configs/s4_optimized_oos_test.json
"""

import argparse
import pandas as pd
from pathlib import Path
import sys
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from engine.features.feature_mapper import FeatureMapper


COMMON_MAPPINGS = {
    'funding_z': 'funding_Z',
    'volume_climax_3b': 'volume_climax_last_3b',
    'wick_exhaustion_3b': 'wick_exhaustion_last_3b',
    'btc_d': 'BTC.D',
    'usdt_d': 'USDT.D',
    'order_block_bull': 'is_bullish_ob',
    'order_block_bear': 'is_bearish_ob',
    'fvg_bull': 'fvg_bullish',
    'fvg_bear': 'fvg_bearish'
}


def load_feature_store_columns(data_path: Path) -> set:
    """Load actual column names from feature store."""
    if not data_path.exists():
        print(f"Warning: Feature store not found: {data_path}")
        return set()

    df = pd.read_parquet(data_path)
    return set(df.columns)


def extract_required_features(config: dict) -> set:
    """Extract feature names required by config."""
    features = set()

    # Check various config sections that might reference features
    if 'features' in config:
        if isinstance(config['features'], list):
            features.update(config['features'])
        elif isinstance(config['features'], dict):
            for domain_features in config['features'].values():
                if isinstance(domain_features, list):
                    features.update(domain_features)

    # Check threshold policies
    if 'threshold_policy' in config:
        policy = config['threshold_policy']
        if 'features' in policy:
            features.update(policy['features'])

    # Check archetype-specific sections
    for key in config:
        if isinstance(config[key], dict) and 'features' in config[key]:
            if isinstance(config[key]['features'], list):
                features.update(config[key]['features'])

    return features


def verify_mappings(
    required_features: set,
    available_features: set,
    mapper: FeatureMapper
) -> tuple:
    """
    Verify feature mappings.

    Returns:
        (valid_mappings, missing_features)
    """
    valid = {}
    missing = []

    for feature in required_features:
        # Try direct access
        if feature in available_features:
            valid[feature] = feature
            continue

        # Try mapper
        try:
            mapped = mapper.get_canonical_name(feature)
            if mapped in available_features:
                valid[feature] = mapped
                continue
        except KeyError:
            pass

        # Try common mappings
        if feature in COMMON_MAPPINGS:
            mapped = COMMON_MAPPINGS[feature]
            if mapped in available_features:
                valid[feature] = mapped
                continue

        # Feature not found
        missing.append(feature)

    return valid, missing


def main():
    parser = argparse.ArgumentParser(
        description="Verify feature name mappings"
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Config file to verify'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/features_1h.parquet',
        help='Feature store path'
    )

    args = parser.parse_args()

    # Load feature store
    data_path = Path(args.data)
    available_features = load_feature_store_columns(data_path)

    print(f"\nFeature store: {data_path}")
    print(f"Available features: {len(available_features)}")

    # Load configs to check
    configs_to_check = []

    if args.config:
        configs_to_check.append(Path(args.config))
    else:
        # Check standard archetype configs
        configs_to_check = [
            Path('configs/s1_v2_production.json'),
            Path('configs/s4_optimized_oos_test.json'),
            Path('configs/s5_production.json')
        ]

    # Initialize mapper
    mapper = FeatureMapper()

    all_valid = True
    total_valid = 0
    total_missing = 0

    for config_path in configs_to_check:
        if not config_path.exists():
            print(f"\nWarning: Config not found: {config_path}")
            continue

        print(f"\n{'='*60}")
        print(f"Checking: {config_path.name}")
        print('='*60)

        # Load config
        with open(config_path) as f:
            config = json.load(f)

        # Extract required features
        required = extract_required_features(config)

        if not required:
            print("No features found in config")
            continue

        print(f"\nRequired features: {len(required)}")

        # Verify mappings
        valid, missing = verify_mappings(required, available_features, mapper)

        # Print valid mappings
        if valid:
            print(f"\n✓ Valid mappings ({len(valid)}):")
            for config_name, store_name in sorted(valid.items()):
                if config_name != store_name:
                    print(f"  {config_name:<30} → {store_name}")
                else:
                    print(f"  {config_name:<30} ✓ (direct match)")

        # Print missing
        if missing:
            print(f"\n✗ Missing features ({len(missing)}):")
            for feature in sorted(missing):
                print(f"  {feature}")
            all_valid = False

        total_valid += len(valid)
        total_missing += len(missing)

    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\nTotal valid mappings:   {total_valid}")
    print(f"Total missing features: {total_missing}")

    if all_valid and total_missing == 0:
        print("\n\033[0;32m✓ PASS\033[0m: All feature mappings verified")
        return 0
    else:
        print("\n\033[0;31m✗ FAIL\033[0m: Missing feature mappings")
        print("\nTo fix:")
        print("  1. Update engine/features/feature_mapper.py with missing mappings")
        print("  2. Or rename features in configs to match feature store")
        print("  3. Or regenerate features with correct names")
        return 1


if __name__ == '__main__':
    exit(main())
