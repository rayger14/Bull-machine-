#!/usr/bin/env python3
"""
Validate the optimized R-ladder implementation across all config files.
"""

import json
from pathlib import Path
from typing import List, Dict

CONFIG_FILES = [
    "configs/optimized/optimized_all_archetypes.json",
    "configs/optimized/optimized_long_squeeze.json",
    "configs/optimized/optimized_order_block_retest.json",
    "configs/optimized/optimized_wick_trap.json",
    "configs/nautilus_production_full_soul.json"
]

EXPECTED_R_LADDER = [
    {"r_multiple": 1.0, "exit_pct": 0.25},
    {"r_multiple": 2.0, "exit_pct": 0.25},
    {"r_multiple": 3.0, "exit_pct": 0.25},
    {"r_multiple": 4.0, "exit_pct": 0.25}
]

def validate_config(filepath: Path) -> Dict:
    """Validate a single config file."""
    with open(filepath, 'r') as f:
        config = json.load(f)

    results = {
        'file': filepath.name,
        'valid': True,
        'archetypes': {},
        'errors': []
    }

    # Check exit_logic exists
    if 'exit_logic' not in config:
        results['valid'] = False
        results['errors'].append("Missing exit_logic section")
        return results

    # Check archetypes
    archetypes = config['exit_logic'].get('archetypes', {})
    for archetype_name, archetype_config in archetypes.items():
        targets = archetype_config.get('profit_targets', [])

        # Validate structure
        if targets != EXPECTED_R_LADDER:
            results['valid'] = False
            results['errors'].append(f"{archetype_name}: R-ladder mismatch")

        # Validate sum
        total = sum(t['exit_pct'] for t in targets)
        if abs(total - 1.0) > 0.001:
            results['valid'] = False
            results['errors'].append(f"{archetype_name}: exit_pcts sum to {total}, not 1.0")

        # Validate order
        r_multiples = [t['r_multiple'] for t in targets]
        if r_multiples != sorted(r_multiples):
            results['valid'] = False
            results['errors'].append(f"{archetype_name}: R-multiples not ascending")

        results['archetypes'][archetype_name] = {
            'targets': len(targets),
            'total_exit': total,
            'r_range': f"{targets[0]['r_multiple']}R - {targets[-1]['r_multiple']}R"
        }

    return results

def main():
    """Validate all config files."""
    print("=" * 80)
    print("OPTIMIZED R-LADDER VALIDATION")
    print("=" * 80)

    print("\nExpected R-Ladder:")
    for i, target in enumerate(EXPECTED_R_LADDER, 1):
        cumulative = sum(t['exit_pct'] for t in EXPECTED_R_LADDER[:i])
        print(f"  {target['exit_pct']*100:5.1f}% at {target['r_multiple']}R (cumulative: {cumulative*100:5.1f}%)")

    print("\n" + "=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)

    base_dir = Path(__file__).parent.parent
    all_valid = True

    for config_file in CONFIG_FILES:
        filepath = base_dir / config_file
        if not filepath.exists():
            print(f"\n❌ {config_file}: FILE NOT FOUND")
            all_valid = False
            continue

        results = validate_config(filepath)

        if results['valid']:
            print(f"\n✓ {results['file']}")
            print(f"  Archetypes validated: {len(results['archetypes'])}")

            # Show a sample archetype
            if results['archetypes']:
                sample = list(results['archetypes'].items())[0]
                print(f"  Example ({sample[0]}): {sample[1]['r_range']}, Total: {sample[1]['total_exit']*100:.0f}%")
        else:
            print(f"\n❌ {results['file']}")
            for error in results['errors']:
                print(f"  ERROR: {error}")
            all_valid = False

    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    if all_valid:
        print("\n✅ ALL VALIDATIONS PASSED")
        print("\nKey improvements:")
        print("  • Removed premature exits at 0.5R")
        print("  • Extended ladder to 4R (was 1.5-2.0R max)")
        print("  • Keeps 50% position after 2R for extended moves")
        print("  • Allows capturing 3R-5R+ winners")
        print("\nExpected impact:")
        print("  • Higher average R-multiple per trade")
        print("  • Better alignment with 20-30% win rate systems")
        print("  • Improved profitability on winning trades")
    else:
        print("\n❌ VALIDATION FAILED")
        print("Review errors above and fix configuration files")

    print("\n" + "=" * 80)

    return 0 if all_valid else 1

if __name__ == "__main__":
    exit(main())
