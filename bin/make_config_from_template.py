#!/usr/bin/env python3
"""
Config Builder - Apply deltas to template_v10.json

Usage:
    python3 bin/make_config_from_template.py \\
        --template configs/v10_bases/template_v10.json \\
        --delta configs/deltas/bull_base.delta.json \\
        --out configs/v10_bases/btc_bull_v10_base.json
"""

import argparse
import json
from pathlib import Path
from copy import deepcopy


def apply_delta(base: dict, delta: dict) -> dict:
    """
    Recursively apply delta to base config.
    Delta values override base values at all levels.
    """
    result = deepcopy(base)

    for key, value in delta.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            # Recursive merge for nested dicts
            result[key] = apply_delta(result[key], value)
        else:
            # Direct override
            result[key] = deepcopy(value)

    return result


def normalize_fusion_weights(config: dict) -> dict:
    """Ensure fusion weights sum to 1.0."""
    if 'fusion' in config and 'weights' in config['fusion']:
        weights = config['fusion']['weights']
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-6:
            print(f"⚠️  Normalizing fusion weights (sum={total:.4f} → 1.0)")
            for k in weights:
                weights[k] /= total
    return config


def validate_config(config: dict) -> bool:
    """Basic validation checks."""
    errors = []

    # Check fusion weights
    if 'fusion' in config and 'weights' in config['fusion']:
        weights = config['fusion']['weights']
        total = sum(weights.values())
        if abs(total - 1.0) > 1e-3:
            errors.append(f"fusion.weights sum to {total:.3f}, not 1.0")

    # Check required sections
    required = ['fusion', 'ml_filter', 'runtime', 'entries', 'exits',
                'risk', 'mtf', 'archetypes', 'decision_gates', 'pnl_tracker']
    for section in required:
        if section not in config:
            errors.append(f"Missing required section: {section}")

    # Check numeric ranges
    if 'fusion' in config:
        if 'entry_threshold_confidence' in config['fusion']:
            val = config['fusion']['entry_threshold_confidence']
            if not (0 <= val <= 1):
                errors.append(f"fusion.entry_threshold_confidence={val} out of [0,1]")

    if 'pnl_tracker' in config and 'exits' in config['pnl_tracker']:
        exits = config['pnl_tracker']['exits']
        if 'max_bars_in_trade' in exits:
            if exits['max_bars_in_trade'] < 1:
                errors.append("pnl_tracker.exits.max_bars_in_trade must be >= 1")

    if errors:
        print("❌ Config validation errors:")
        for e in errors:
            print(f"   - {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Build config from template + delta")
    parser.add_argument('--template', required=True, help="Template config path (e.g., template_v10.json)")
    parser.add_argument('--delta', required=True, help="Delta JSON with overrides")
    parser.add_argument('--out', required=True, help="Output config path")
    parser.add_argument('--validate-only', action='store_true', help="Only validate, don't write")
    args = parser.parse_args()

    # Load template
    template_path = Path(args.template)
    if not template_path.exists():
        print(f"❌ Template not found: {template_path}")
        return 1

    with open(template_path, 'r') as f:
        base_config = json.load(f)

    print(f"✅ Loaded template: {template_path}")

    # Load delta
    delta_path = Path(args.delta)
    if not delta_path.exists():
        print(f"❌ Delta not found: {delta_path}")
        return 1

    with open(delta_path, 'r') as f:
        delta = json.load(f)

    print(f"✅ Loaded delta: {delta_path}")

    # Apply delta
    config = apply_delta(base_config, delta)

    # Normalize weights
    config = normalize_fusion_weights(config)

    # Validate
    if not validate_config(config):
        return 1

    print("✅ Config validation passed")

    if args.validate_only:
        print("   (validate-only mode, not writing output)")
        return 0

    # Write output
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"✅ Wrote config: {out_path}")

    return 0


if __name__ == '__main__':
    exit(main())
