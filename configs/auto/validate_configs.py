#!/usr/bin/env python3
"""
Validate auto-generated regime configs.

Checks that all configs have:
- Valid JSON structure
- Required fields
- Regime override set correctly
- Threshold metadata present
"""

import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent


def validate_config(config_path: Path) -> tuple[bool, list]:
    """Validate a single config file."""
    errors = []
    
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
    except Exception as e:
        return False, [f"Failed to load JSON: {e}"]
    
    # Check required top-level fields
    required_fields = ['version', 'description', 'regime_classifier', 'archetypes']
    for field in required_fields:
        if field not in config:
            errors.append(f"Missing required field: {field}")
    
    # Check regime override exists
    if 'regime_classifier' in config:
        if 'regime_override' not in config['regime_classifier']:
            errors.append("Missing regime_override in regime_classifier")
    
    # Check archetype thresholds exist
    if 'archetypes' in config:
        if 'thresholds' not in config['archetypes']:
            errors.append("Missing thresholds in archetypes")
        else:
            # Check min_liquidity is set
            if 'min_liquidity' not in config['archetypes']['thresholds']:
                errors.append("Missing min_liquidity threshold")
    
    # Check metadata exists
    if '_threshold_metadata' not in config:
        errors.append("Missing _threshold_metadata")
    else:
        metadata_fields = ['regime_id', 'regime_name', 'sample_size', 'fusion_threshold']
        for field in metadata_fields:
            if field not in config['_threshold_metadata']:
                errors.append(f"Missing metadata field: {field}")
    
    return len(errors) == 0, errors


def main():
    """Validate all generated configs."""
    print("=" * 80)
    print("REGIME CONFIG VALIDATION")
    print("=" * 80)
    
    config_dir = PROJECT_ROOT / "configs/auto"
    
    # Find all regime configs
    regime_configs = sorted(config_dir.glob("config_regime_*.json"))
    
    if not regime_configs:
        print("ERROR: No regime configs found!")
        return 1
    
    print(f"\nFound {len(regime_configs)} regime configs\n")
    
    all_valid = True
    
    for config_path in regime_configs:
        print(f"Validating {config_path.name}...")
        valid, errors = validate_config(config_path)
        
        if valid:
            print("  ✓ Valid")
        else:
            print("  ✗ Invalid")
            for error in errors:
                print(f"    - {error}")
            all_valid = False
    
    print("\n" + "=" * 80)
    if all_valid:
        print("ALL CONFIGS VALID")
        print("=" * 80)
        return 0
    else:
        print("VALIDATION FAILED")
        print("=" * 80)
        return 1


if __name__ == "__main__":
    sys.exit(main())
