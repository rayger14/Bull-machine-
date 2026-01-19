#!/usr/bin/env python3
"""
Validate MVP Config Fix

This script validates that the critical fix to mvp configs was applied correctly.
It checks for the presence of adaptive_fusion and regime_override parameters.

Usage:
    python3 configs/mvp/validate_mvp_configs.py
"""

import json
import sys
from pathlib import Path

def validate_config(config_path, expected_regime_year, expected_regime):
    """Validate a single config file."""
    print(f"\n{'='*70}")
    print(f"Validating: {config_path.name}")
    print('='*70)

    errors = []
    warnings = []

    try:
        with open(config_path) as f:
            config = json.load(f)
    except json.JSONDecodeError as e:
        errors.append(f"Invalid JSON: {e}")
        return errors, warnings

    # Check adaptive_fusion
    if not config.get('adaptive_fusion'):
        errors.append("Missing or disabled 'adaptive_fusion' parameter")
    else:
        print("✓ adaptive_fusion: True")

    # Check regime_classifier exists
    if 'regime_classifier' not in config:
        errors.append("Missing 'regime_classifier' section")
    else:
        print("✓ regime_classifier: EXISTS")

        # Check regime_override
        regime_classifier = config['regime_classifier']
        if 'regime_override' not in regime_classifier:
            errors.append("Missing 'regime_override' in regime_classifier")
        else:
            regime_override = regime_classifier['regime_override']
            if str(expected_regime_year) not in regime_override:
                errors.append(f"Missing regime override for year {expected_regime_year}")
            else:
                actual_regime = regime_override[str(expected_regime_year)]
                if actual_regime != expected_regime:
                    errors.append(f"Expected regime '{expected_regime}' for {expected_regime_year}, got '{actual_regime}'")
                else:
                    print(f"✓ regime_override[{expected_regime_year}]: {actual_regime}")

    # Check archetypes section
    if 'archetypes' not in config:
        errors.append("Missing 'archetypes' section")
    else:
        archetypes = config['archetypes']

        # Check bear archetypes enabled
        s2_enabled = archetypes.get('enable_S2', False)
        s5_enabled = archetypes.get('enable_S5', False)

        print(f"  S2 (failed_rally): {s2_enabled}")
        print(f"  S5 (long_squeeze): {s5_enabled}")

        if not (s2_enabled and s5_enabled):
            warnings.append("Bear archetypes S2 or S5 not enabled")

        # Check routing exists
        if 'routing' not in archetypes:
            errors.append("Missing 'routing' section in archetypes")
        else:
            routing = archetypes['routing']
            if expected_regime not in routing:
                errors.append(f"Missing '{expected_regime}' regime in routing")
            else:
                weights = routing[expected_regime].get('weights', {})
                print(f"✓ routing[{expected_regime}] weights:")
                for arch, weight in weights.items():
                    print(f"    {arch}: {weight}")

    # Check fusion thresholds
    if 'fusion' in config:
        fusion = config['fusion']
        threshold = fusion.get('entry_threshold_confidence')
        print(f"  Fusion entry threshold: {threshold}")

    # Check ML filter
    if 'ml_filter' in config:
        ml_filter = config['ml_filter']
        if ml_filter.get('enabled'):
            threshold = ml_filter.get('threshold')
            print(f"  ML filter threshold: {threshold}")

    return errors, warnings


def main():
    """Main validation routine."""
    print("\nMVP CONFIG VALIDATION")
    print("=" * 70)
    print("Checking for critical fix: adaptive_fusion + regime_override")
    print()

    base_path = Path(__file__).parent

    # Validate bull market config
    bull_config = base_path / "mvp_bull_market_v1.json"
    bull_errors, bull_warnings = validate_config(bull_config, 2024, "risk_on")

    # Validate bear market config
    bear_config = base_path / "mvp_bear_market_v1.json"
    bear_errors, bear_warnings = validate_config(bear_config, 2022, "risk_off")

    # Print summary
    print(f"\n{'='*70}")
    print("VALIDATION SUMMARY")
    print('='*70)

    all_errors = bull_errors + bear_errors
    all_warnings = bull_warnings + bear_warnings

    if all_errors:
        print("\n❌ ERRORS FOUND:")
        for error in all_errors:
            print(f"  - {error}")
        print("\nFix required before running backtests!")
        return 1

    if all_warnings:
        print("\n⚠️  WARNINGS:")
        for warning in all_warnings:
            print(f"  - {warning}")

    print("\n✅ VALIDATION PASSED")
    print("\nConfigs are ready for backtesting.")
    print("\nNext steps:")
    print("  1. Run bull market backtest (2024):")
    print("     python3 bin/backtest_knowledge_v2.py \\")
    print("       --asset BTC --start 2024-01-01 --end 2024-12-31 \\")
    print("       --config configs/mvp/mvp_bull_market_v1.json")
    print()
    print("  2. Run bear market backtest (2022):")
    print("     python3 bin/backtest_knowledge_v2.py \\")
    print("       --asset BTC --start 2022-01-01 --end 2022-12-31 \\")
    print("       --config configs/mvp/mvp_bear_market_v1.json")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
