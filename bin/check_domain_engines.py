#!/usr/bin/env python3
"""
STEP 3: Confirm Domain Engines Are ON

Verifies all 6 domain engines are enabled in archetype configs:
- Wyckoff
- SMC
- Temporal
- HOB (Higher Order Beliefs)
- Fusion
- Macro

Usage:
    python bin/check_domain_engines.py --s1 --s4 --s5
    python bin/check_domain_engines.py --config configs/custom.json
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Tuple

# Domain engines to check
REQUIRED_ENGINES = [
    'wyckoff',
    'smc',
    'temporal',
    'hob',
    'fusion',
    'macro'
]

def check_config_engines(config_path: Path) -> Tuple[List[str], List[str]]:
    """
    Check which domain engines are enabled in a config.

    Returns:
        (enabled_engines, disabled_engines)
    """
    if not config_path.exists():
        return [], REQUIRED_ENGINES.copy()

    with open(config_path) as f:
        config = json.load(f)

    enabled = []
    disabled = []

    # Check feature_flags or domain_engines section
    feature_flags = config.get('feature_flags', {})
    domain_engines = config.get('domain_engines', {})

    for engine in REQUIRED_ENGINES:
        # Check multiple possible config patterns
        engine_enabled = False

        # Pattern 1: feature_flags.use_<engine>
        if feature_flags.get(f'use_{engine}', False):
            engine_enabled = True

        # Pattern 2: domain_engines.<engine>.enabled
        if domain_engines.get(engine, {}).get('enabled', False):
            engine_enabled = True

        # Pattern 3: <engine>_enabled at top level
        if config.get(f'{engine}_enabled', False):
            engine_enabled = True

        # Pattern 4: engines list
        if engine in config.get('engines', []):
            engine_enabled = True

        if engine_enabled:
            enabled.append(engine)
        else:
            disabled.append(engine)

    return enabled, disabled


def print_engine_status(config_name: str, enabled: List[str], disabled: List[str]):
    """Print engine status for a config."""
    print(f"\n{config_name}:")

    for engine in REQUIRED_ENGINES:
        status = "✓ ENABLED" if engine in enabled else "✗ DISABLED"
        color = "\033[0;32m" if engine in enabled else "\033[0;31m"
        reset = "\033[0m"

        engine_display = engine.upper().ljust(10)
        print(f"  {engine_display}: {color}{status}{reset}")

    return len(enabled), len(disabled)


def main():
    parser = argparse.ArgumentParser(
        description="Check domain engine status in archetype configs"
    )
    parser.add_argument('--s1', action='store_true', help='Check S1 config')
    parser.add_argument('--s4', action='store_true', help='Check S4 config')
    parser.add_argument('--s5', action='store_true', help='Check S5 config')
    parser.add_argument('--config', type=str, help='Check specific config file')

    args = parser.parse_args()

    configs_to_check = []

    # Build list of configs to check
    if args.s1:
        configs_to_check.append(
            ('S1 (Liquidity Vacuum)', Path('configs/s1_v2_production.json'))
        )

    if args.s4:
        configs_to_check.append(
            ('S4 (Funding Divergence)', Path('configs/s4_optimized_oos_test.json'))
        )

    if args.s5:
        configs_to_check.append(
            ('S5 (Failed Rally)', Path('configs/s5_production.json'))
        )

    if args.config:
        config_path = Path(args.config)
        configs_to_check.append(
            (f'Custom ({config_path.name})', config_path)
        )

    if not configs_to_check:
        print("No configs specified. Use --s1, --s4, --s5, or --config")
        return 1

    # Check all configs
    total_enabled = 0
    total_disabled = 0

    for config_name, config_path in configs_to_check:
        enabled, disabled = check_config_engines(config_path)
        enabled_count, disabled_count = print_engine_status(
            config_name, enabled, disabled
        )
        total_enabled += enabled_count
        total_disabled += disabled_count

    # Summary
    expected_total = len(configs_to_check) * len(REQUIRED_ENGINES)

    print(f"\n{'='*50}")
    print(f"Overall: {total_enabled}/{expected_total} engines enabled")

    if total_disabled == 0:
        print("\033[0;32m✓ PASS\033[0m: All domain engines enabled")
        return 0
    else:
        print(f"\033[0;31m✗ FAIL\033[0m: {total_disabled} engines disabled")
        print("\nTo fix:")
        print("  python bin/enable_domain_engines.py --all")
        return 1


if __name__ == '__main__':
    exit(main())
