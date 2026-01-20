#!/usr/bin/env python3
"""
PR-A: Utility to copy legacy static thresholds to archetype_overrides format

This tool extracts thresholds from a legacy static config and generates
archetype_overrides that can be added to an adaptive config to force
exact parity with the legacy code path.

Usage:
    python3 bin/copy_legacy_thresholds.py \
        --legacy-config configs/baseline_btc_bull_pf20.json \
        --output configs/archetype_overrides_pf20.json

The output can then be merged into the adaptive config:
    "archetype_overrides": {
        "order_block_retest": {
            "static": {"fusion": -0.02}  # Force delta to match legacy
        }
    }
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import argparse
import logging
from typing import Dict, Any

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


# Archetype name mapping (internal names → config keys)
ARCHETYPE_MAP = {
    'trap_reversal': 'A',
    'order_block_retest': 'B',
    'wick_trap': 'C',
    'failed_continuation': 'D',
    'volume_exhaustion': 'E',
    'exhaustion_reversal': 'F',
    'liquidity_sweep': 'G',
    'momentum_continuation': 'H',
    'trap_within_trend': 'K',
    'retest_cluster': 'L',
    'confluence_breakout': 'M'
}


def extract_legacy_thresholds(config_path: str) -> Dict[str, Dict[str, float]]:
    """
    Extract archetype thresholds from a legacy static config.

    Returns:
        Dict mapping archetype_name -> {param: value}
        Example: {'order_block_retest': {'fusion': 0.359, 'liquidity': 0.14}}
    """
    with open(config_path) as f:
        config = json.load(f)

    arch_config = config.get('archetypes', {})
    thresholds_cfg = arch_config.get('thresholds', {})

    legacy_thresholds = {}

    for arch_name, cfg_key in ARCHETYPE_MAP.items():
        if cfg_key in thresholds_cfg:
            legacy_thresholds[arch_name] = thresholds_cfg[cfg_key]

    # Also extract global min_liquidity
    if 'min_liquidity' in thresholds_cfg:
        legacy_thresholds['_global_min_liquidity'] = thresholds_cfg['min_liquidity']

    logger.info(f"Extracted {len(legacy_thresholds)} archetype threshold configs from {config_path}")
    return legacy_thresholds


def convert_to_overrides(
    legacy_thresholds: Dict[str, Dict[str, float]],
    base_thresholds: Dict[str, Dict[str, float]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    """
    Convert legacy absolute thresholds to delta overrides.

    For parity testing, we want:
        final_threshold = base_threshold + override_delta

    So:
        override_delta = legacy_threshold - base_threshold

    Args:
        legacy_thresholds: Thresholds from legacy config
        base_thresholds: Base thresholds from adaptive config

    Returns:
        archetype_overrides format:
        {
            "order_block_retest": {
                "static": {"fusion": +0.02, "liquidity": -0.01}
            }
        }
    """
    overrides = {}

    for arch_name, legacy_vals in legacy_thresholds.items():
        if arch_name.startswith('_'):
            continue  # Skip meta keys

        base_vals = base_thresholds.get(arch_name, {})
        deltas = {}

        for param, legacy_val in legacy_vals.items():
            base_val = base_vals.get(param)
            if base_val is not None:
                delta = legacy_val - base_val
                if abs(delta) > 1e-6:  # Only include non-zero deltas
                    deltas[param] = round(delta, 6)

        if deltas:
            overrides[arch_name] = {
                'static': deltas
            }

    return overrides


def main():
    parser = argparse.ArgumentParser(description='PR-A: Copy legacy thresholds to override format')
    parser.add_argument('--legacy-config', required=True,
                        help='Path to legacy static config (e.g., baseline_btc_bull_pf20.json)')
    parser.add_argument('--base-config', required=True,
                        help='Path to base adaptive config (e.g., btc_v8_adaptive.json)')
    parser.add_argument('--output', required=True,
                        help='Output path for archetype_overrides JSON')
    parser.add_argument('--merge', action='store_true',
                        help='Merge overrides directly into base config')

    args = parser.parse_args()

    logger.info("="*80)
    logger.info("PR-A: Legacy Threshold → Override Converter")
    logger.info("="*80)

    # Step 1: Extract legacy thresholds
    logger.info(f"\n1. Reading legacy config: {args.legacy_config}")
    legacy_thresholds = extract_legacy_thresholds(args.legacy_config)

    # Step 2: Extract base thresholds from adaptive config
    logger.info(f"\n2. Reading base config: {args.base_config}")
    with open(args.base_config) as f:
        base_config = json.load(f)

    base_arch_thresholds = base_config.get('archetypes', {}).get('thresholds', {})

    # Convert config keys to archetype names
    base_thresholds = {}
    for arch_name, cfg_key in ARCHETYPE_MAP.items():
        if cfg_key in base_arch_thresholds:
            base_thresholds[arch_name] = base_arch_thresholds[cfg_key]

    # Step 3: Compute deltas
    logger.info(f"\n3. Computing delta overrides...")
    overrides = convert_to_overrides(legacy_thresholds, base_thresholds)

    logger.info(f"   Generated {len(overrides)} archetype overrides")

    # Step 4: Output
    if args.merge:
        # Merge into base config
        base_config['archetype_overrides'] = base_config.get('archetype_overrides', {})

        for arch_name, override in overrides.items():
            base_config['archetype_overrides'][arch_name] = override

        output_path = args.base_config
        logger.info(f"\n4. Merging overrides into {output_path}")

        with open(output_path, 'w') as f:
            json.dump(base_config, f, indent=2)

        logger.info(f"   ✅ Updated {output_path}")

    else:
        # Write standalone overrides file
        logger.info(f"\n4. Writing overrides to {args.output}")

        output = {
            "_note": "Generated by copy_legacy_thresholds.py for parity testing",
            "_legacy_source": args.legacy_config,
            "_base_source": args.base_config,
            "archetype_overrides": overrides
        }

        with open(args.output, 'w') as f:
            json.dump(output, f, indent=2)

        logger.info(f"   ✅ Created {args.output}")
        logger.info(f"\n   To use: Copy 'archetype_overrides' section into your adaptive config")

    # Step 5: Summary
    logger.info("\n" + "="*80)
    logger.info("Summary:")
    logger.info("="*80)

    for arch_name, override in sorted(overrides.items()):
        deltas = override.get('static', {})
        if deltas:
            logger.info(f"  {arch_name}:")
            for param, delta in sorted(deltas.items()):
                sign = '+' if delta >= 0 else ''
                logger.info(f"    {param}: {sign}{delta}")

    logger.info("="*80)


if __name__ == '__main__':
    main()
