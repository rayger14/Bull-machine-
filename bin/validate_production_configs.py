#!/usr/bin/env python3
"""
Validate Production Archetype Configurations

This script validates all 16 production archetype configs for:
1. JSON syntax correctness
2. Required field presence
3. Threshold sanity checks
4. Feature flag consistency
5. Regime routing completeness

Usage:
    python bin/validate_production_configs.py
    python bin/validate_production_configs.py --verbose
"""

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Archetype metadata
ARCHETYPES = {
    'A': {'name': 'Spring', 'regime': 'bull', 'direction': 'long'},
    'B': {'name': 'Order Block Retest', 'regime': 'bull', 'direction': 'long'},
    'C': {'name': 'Wick Trap', 'regime': 'bull', 'direction': 'long'},
    'D': {'name': 'Failed Continuation', 'regime': 'bull', 'direction': 'long'},
    'E': {'name': 'Volume Exhaustion', 'regime': 'bull', 'direction': 'long'},
    'F': {'name': 'Exhaustion Reversal', 'regime': 'bull', 'direction': 'long'},
    'G': {'name': 'Liquidity Sweep', 'regime': 'bull', 'direction': 'long'},
    'H': {'name': 'Momentum Continuation', 'regime': 'bull', 'direction': 'long'},
    'K': {'name': 'Trap Within Trend', 'regime': 'bull', 'direction': 'long'},
    'L': {'name': 'Retest Cluster', 'regime': 'bull', 'direction': 'long'},
    'M': {'name': 'Confluence Breakout', 'regime': 'bull', 'direction': 'long'},
    'S1': {'name': 'Liquidity Vacuum', 'regime': 'bear', 'direction': 'long'},
    'S3': {'name': 'Whipsaw', 'regime': 'neutral', 'direction': 'neutral'},
    'S4': {'name': 'Funding Divergence', 'regime': 'bear', 'direction': 'long'},
    'S5': {'name': 'Long Squeeze', 'regime': 'bear', 'direction': 'short'},
    'S8': {'name': 'Volume Fade Chop', 'regime': 'neutral', 'direction': 'neutral'},
}

REQUIRED_TOP_LEVEL = [
    'version', 'profile', 'description', 'regime_classifier',
    'fusion', 'risk', 'archetypes', 'feature_flags'
]

REQUIRED_ENGINES = [
    'enable_wyckoff', 'enable_smc', 'enable_temporal',
    'enable_hob', 'enable_fusion', 'enable_macro'
]

REGIME_TYPES = ['risk_on', 'neutral', 'risk_off', 'crisis']


def validate_json_syntax(config_path: Path) -> Tuple[bool, str, Dict]:
    """Validate JSON syntax and load config."""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        return True, "Valid JSON", config
    except json.JSONDecodeError as e:
        return False, f"JSON syntax error: {e}", {}
    except Exception as e:
        return False, f"Error reading file: {e}", {}


def validate_structure(config: Dict) -> List[str]:
    """Validate config has required top-level fields."""
    errors = []

    for field in REQUIRED_TOP_LEVEL:
        if field not in config:
            errors.append(f"Missing required field: {field}")

    return errors


def validate_feature_flags(config: Dict) -> List[str]:
    """Validate all 6 domain engines are enabled."""
    errors = []

    if 'feature_flags' not in config:
        return ["Missing feature_flags section"]

    flags = config['feature_flags']

    for engine in REQUIRED_ENGINES:
        if engine not in flags:
            errors.append(f"Missing feature flag: {engine}")
        elif not flags[engine]:
            errors.append(f"Engine not enabled: {engine}")

    return errors


def validate_archetype_config(config: Dict, code: str) -> List[str]:
    """Validate archetype-specific configuration."""
    errors = []

    if 'archetypes' not in config:
        return ["Missing archetypes section"]

    arch = config['archetypes']

    # Check archetype is enabled
    enable_key = f'enable_{code}'
    if enable_key not in arch:
        errors.append(f"Missing {enable_key}")
    elif not arch[enable_key]:
        errors.append(f"Archetype {code} not enabled")

    # Check thresholds section exists
    if 'thresholds' not in arch:
        errors.append("Missing thresholds section")
        return errors

    # Find threshold config (various naming conventions)
    threshold_configs = arch['thresholds']
    threshold_key = None

    # Common threshold key patterns
    possible_keys = [
        code.lower(),
        ARCHETYPES[code]['name'].lower().replace(' ', '_'),
        'spring' if code == 'A' else None,
        'order_block_retest' if code == 'B' else None,
        'wick_trap' if code == 'C' else None,
        'failed_continuation' if code == 'D' else None,
        'volume_exhaustion' if code == 'E' else None,
        'exhaustion_reversal' if code == 'F' else None,
        'liquidity_sweep' if code == 'G' else None,
        'momentum_continuation' if code == 'H' else None,
        'trap_within_trend' if code == 'K' else None,
        'retest_cluster' if code == 'L' else None,
        'confluence_breakout' if code == 'M' else None,
        'liquidity_vacuum' if code == 'S1' else None,
        'whipsaw' if code == 'S3' else None,
        'funding_divergence' if code == 'S4' else None,
        'long_squeeze' if code == 'S5' else None,
        'volume_fade_chop' if code == 'S8' else None,
    ]

    for key in possible_keys:
        if key and key in threshold_configs:
            threshold_key = key
            break

    if not threshold_key:
        errors.append(f"No threshold config found for {code}")
        return errors

    thresholds = threshold_configs[threshold_key]

    # Validate threshold values
    if 'fusion_threshold' in thresholds:
        ft = thresholds['fusion_threshold']
        if ft < 0.2 or ft > 1.0:
            errors.append(f"fusion_threshold {ft} out of range [0.2, 1.0]")

    if 'archetype_weight' in thresholds:
        aw = thresholds['archetype_weight']
        if aw < 0.5 or aw > 3.0:
            errors.append(f"archetype_weight {aw} out of range [0.5, 3.0]")

    if 'direction' in thresholds:
        direction = thresholds['direction']
        expected = ARCHETYPES[code]['direction']
        if direction != expected:
            errors.append(f"Direction mismatch: got {direction}, expected {expected}")

    return errors


def validate_regime_routing(config: Dict) -> List[str]:
    """Validate regime routing configuration."""
    errors = []

    if 'archetypes' not in config or 'routing' not in config['archetypes']:
        return ["Missing routing section"]

    routing = config['archetypes']['routing']

    for regime in REGIME_TYPES:
        if regime not in routing:
            errors.append(f"Missing regime: {regime}")
        else:
            if 'weights' not in routing[regime]:
                errors.append(f"Missing weights for regime: {regime}")
            if 'final_gate_delta' not in routing[regime]:
                errors.append(f"Missing final_gate_delta for regime: {regime}")

    return errors


def validate_risk_config(config: Dict) -> List[str]:
    """Validate risk management configuration."""
    errors = []

    if 'risk' not in config:
        return ["Missing risk section"]

    risk = config['risk']

    required_risk = ['base_risk_pct', 'max_position_size_pct', 'max_portfolio_risk_pct']
    for field in required_risk:
        if field not in risk:
            errors.append(f"Missing risk parameter: {field}")

    # Sanity checks
    if 'base_risk_pct' in risk:
        if risk['base_risk_pct'] < 0.005 or risk['base_risk_pct'] > 0.05:
            errors.append(f"base_risk_pct {risk['base_risk_pct']} out of safe range [0.005, 0.05]")

    return errors


def validate_config_file(config_path: Path, verbose: bool = False) -> Tuple[bool, List[str]]:
    """Validate a single config file."""
    all_errors = []

    # Parse archetype code from filename
    filename = config_path.stem
    parts = filename.split('_')
    if len(parts) < 2:
        return False, ["Cannot parse archetype code from filename"]

    code = parts[1].upper()
    if code not in ARCHETYPES:
        return False, [f"Unknown archetype code: {code}"]

    if verbose:
        print(f"\nValidating {code} ({ARCHETYPES[code]['name']})...")

    # 1. JSON syntax
    valid, msg, config = validate_json_syntax(config_path)
    if not valid:
        all_errors.append(msg)
        return False, all_errors

    # 2. Structure
    errors = validate_structure(config)
    all_errors.extend(errors)

    # 3. Feature flags
    errors = validate_feature_flags(config)
    all_errors.extend(errors)

    # 4. Archetype config
    errors = validate_archetype_config(config, code)
    all_errors.extend(errors)

    # 5. Regime routing
    errors = validate_regime_routing(config)
    all_errors.extend(errors)

    # 6. Risk config
    errors = validate_risk_config(config)
    all_errors.extend(errors)

    return len(all_errors) == 0, all_errors


def main():
    """Main validation routine."""
    import argparse

    parser = argparse.ArgumentParser(description='Validate production archetype configs')
    parser.add_argument('--verbose', action='store_true', help='Verbose output')
    args = parser.parse_args()

    # Find all production configs
    config_dir = Path(__file__).parent.parent / 'configs' / 'archetypes' / 'production'
    config_files = sorted(config_dir.glob('archetype_*.json'))

    if not config_files:
        print(f"ERROR: No config files found in {config_dir}")
        sys.exit(1)

    print(f"Found {len(config_files)} production configs")
    print("=" * 70)

    results = {}
    total_errors = 0

    for config_path in config_files:
        valid, errors = validate_config_file(config_path, verbose=args.verbose)

        if valid:
            status = "PASS"
            symbol = "✓"
        else:
            status = "FAIL"
            symbol = "✗"
            total_errors += len(errors)

        results[config_path.name] = (status, errors)

        if args.verbose or not valid:
            print(f"{symbol} {config_path.name}: {status}")
            if errors:
                for error in errors:
                    print(f"    - {error}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    passed = sum(1 for status, _ in results.values() if status == "PASS")
    failed = len(results) - passed

    print(f"Total configs: {len(results)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    print(f"Total errors: {total_errors}")

    if failed == 0:
        print("\n✓ All production configs are valid!")
        sys.exit(0)
    else:
        print(f"\n✗ {failed} configs failed validation")
        print("\nFailed configs:")
        for name, (status, errors) in results.items():
            if status == "FAIL":
                print(f"  - {name}")
        sys.exit(1)


if __name__ == '__main__':
    main()
