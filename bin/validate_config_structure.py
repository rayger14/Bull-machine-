#!/usr/bin/env python3
"""
Validate Bull Machine Config Structure

Checks that config files have the correct structure for ThresholdPolicy to read
archetype thresholds. Use this before running backtests to catch structural issues.

Usage:
    python bin/validate_config_structure.py configs/optimized/my_config.json
    python bin/validate_config_structure.py configs/optimized/*.json  # Check all
"""

import json
import sys
from pathlib import Path


def validate_config_structure(config_path: Path) -> tuple[bool, list[str]]:
    """
    Validate config structure for ThresholdPolicy compatibility.

    Returns:
        (is_valid, issues) - tuple of boolean and list of issue strings
    """
    issues = []

    try:
        with open(config_path) as f:
            cfg = json.load(f)
    except json.JSONDecodeError as e:
        return False, [f"Invalid JSON: {e}"]
    except Exception as e:
        return False, [f"Failed to read file: {e}"]

    # Check for old incorrect structure
    if 'archetype_config' in cfg:
        issues.append("❌ Found 'archetype_config' top-level key (deprecated/wrong)")
        issues.append("   Should be 'archetypes' instead")

    # Check for correct structure
    if 'archetypes' not in cfg:
        issues.append("❌ Missing 'archetypes' top-level key")
        return False, issues

    archetypes = cfg['archetypes']

    # Check for thresholds subdirectory
    if 'thresholds' not in archetypes:
        issues.append("❌ Missing 'archetypes.thresholds' subdirectory")
        return False, issues

    thresholds = archetypes['thresholds']

    # Check threshold content
    if not isinstance(thresholds, dict):
        issues.append("❌ 'archetypes.thresholds' must be a dict")
        return False, issues

    if not thresholds:
        issues.append("⚠️  'archetypes.thresholds' is empty (no archetype configs)")

    # Check required metadata
    required_fields = ['use_archetypes', 'max_trades_per_day']
    for field in required_fields:
        if field not in archetypes:
            issues.append(f"⚠️  Missing recommended field 'archetypes.{field}'")

    # Validate individual archetype configs
    valid_archetypes = 0
    empty_archetypes = 0

    for arch_name, arch_config in thresholds.items():
        if not isinstance(arch_config, dict):
            issues.append(f"⚠️  Archetype '{arch_name}' config is not a dict")
            continue

        # Filter out comment keys
        params = {k: v for k, v in arch_config.items() if not k.startswith('_')}

        if not params:
            empty_archetypes += 1
        else:
            valid_archetypes += 1

    # Report archetype stats
    if valid_archetypes == 0 and empty_archetypes > 0:
        issues.append(f"❌ All {empty_archetypes} archetypes are empty (no parameters)")
    elif empty_archetypes > 0:
        issues.append(f"⚠️  {empty_archetypes} archetypes are empty, {valid_archetypes} have parameters")

    # Success if no critical issues
    is_valid = all('❌' not in issue for issue in issues)

    return is_valid, issues


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    config_paths = sys.argv[1:]

    print("="*70)
    print("CONFIG STRUCTURE VALIDATION")
    print("="*70)

    results = {}

    for config_path_str in config_paths:
        config_path = Path(config_path_str)

        if not config_path.exists():
            print(f"\n❌ {config_path.name}")
            print(f"   File not found: {config_path}")
            results[config_path.name] = False
            continue

        is_valid, issues = validate_config_structure(config_path)
        results[config_path.name] = is_valid

        # Print result
        status = "✓ VALID" if is_valid else "✗ INVALID"
        print(f"\n{status}: {config_path.name}")

        if issues:
            for issue in issues:
                print(f"  {issue}")
        else:
            # Load to show stats
            with open(config_path) as f:
                cfg = json.load(f)
            thresholds = cfg.get('archetypes', {}).get('thresholds', {})
            valid_count = sum(
                1 for v in thresholds.values()
                if isinstance(v, dict) and any(not k.startswith('_') for k in v.keys())
            )
            print(f"  ✓ Structure correct")
            print(f"  ✓ {valid_count} archetypes configured")

    # Summary
    print("\n" + "="*70)
    total = len(results)
    valid = sum(results.values())
    invalid = total - valid

    print(f"SUMMARY: {valid}/{total} configs valid")

    if invalid > 0:
        print(f"\n⚠️  {invalid} config(s) need fixing")
        print("\nCommon fixes:")
        print("  1. Rename 'archetype_config' → 'archetypes'")
        print("  2. Ensure structure: config['archetypes']['thresholds'][arch_name]")
        print("  3. Add 'use_archetypes' and 'max_trades_per_day' to 'archetypes'")
        print("\nSee: THRESHOLD_POLICY_CONFIG_FIX.md")
        sys.exit(1)
    else:
        print("✓ All configs valid!")
        sys.exit(0)


if __name__ == '__main__':
    main()
