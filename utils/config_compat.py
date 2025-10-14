"""
Config Compatibility Layer for Bull Machine v1.8.6

Handles naming mismatches between old configs and current engine without
requiring full refactors. Keeps frozen production configs working.

Example mappings:
- 'hob' → 'liquidity' (old name for HOB/liquidity engine)
- 'smc' → 'smc' (no change, but example for future)
- 'momentum_score' → 'momentum' (if needed)
"""

from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


def apply_key_aliases(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply backward-compatible key aliases to config

    Remaps deprecated/renamed keys in weights, features, signals blocks
    without breaking unknown keys or changing the original structure.

    Args:
        cfg: Configuration dictionary loaded from JSON

    Returns:
        Modified config with aliases applied
    """
    # Define aliases: old_name → new_name
    aliases = {
        'hob': 'liquidity',              # HOB was renamed to liquidity in some versions
        'temporal': 'temporal',          # No-op, but kept for clarity
        # Add more as needed without breaking existing configs
    }

    sections_to_remap = ['weights', 'features', 'signals', 'domain_weights']

    for section in sections_to_remap:
        # Check if section exists in root
        if section in cfg and isinstance(cfg[section], dict):
            _remap_section(cfg[section], aliases, section)

        # Check nested fusion.weights (common location)
        if 'fusion' in cfg and section in cfg['fusion'] and isinstance(cfg['fusion'][section], dict):
            _remap_section(cfg['fusion'][section], aliases, f"fusion.{section}")

    return cfg


def _remap_section(section_dict: Dict[str, Any], aliases: Dict[str, str], section_name: str):
    """Helper to remap keys in a specific section"""
    remapped = []

    for old_key, new_key in aliases.items():
        if old_key in section_dict and new_key not in section_dict:
            section_dict[new_key] = section_dict.pop(old_key)
            remapped.append(f"{old_key}→{new_key}")
            logger.info(f"Config compat: Remapped {section_name}.{old_key} → {new_key}")

    if remapped:
        logger.info(f"Applied {len(remapped)} alias(es) in {section_name}: {', '.join(remapped)}")


def validate_fusion_weights(cfg: Dict[str, Any]) -> bool:
    """
    Validate fusion weights sum to ~1.0 and all are positive

    Args:
        cfg: Configuration dictionary

    Returns:
        True if valid, raises ValueError if invalid
    """
    if 'fusion' not in cfg or 'weights' not in cfg['fusion']:
        logger.warning("No fusion.weights found in config")
        return True  # Allow configs without fusion section

    weights = cfg['fusion']['weights']

    # Check all weights are positive
    for key, val in weights.items():
        if not isinstance(val, (int, float)) or val < 0:
            raise ValueError(f"Invalid weight {key}={val}, must be non-negative number")

    # Check sum is close to 1.0
    total = sum(weights.values())
    if abs(total - 1.0) > 0.01:
        logger.warning(f"Fusion weights sum to {total:.3f}, not 1.0. Normalizing...")
        # Auto-normalize
        for key in weights:
            weights[key] = weights[key] / total
        logger.info(f"Normalized weights: {weights}")

    return True


def normalize_config_for_hybrid(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply all compatibility transforms needed for hybrid_runner

    This is the main entry point - call this after loading config JSON.

    Args:
        cfg: Raw config dictionary

    Returns:
        Normalized config ready for hybrid_runner
    """
    # Apply key aliases
    cfg = apply_key_aliases(cfg)

    # Validate fusion weights
    validate_fusion_weights(cfg)

    return cfg


# Example usage
if __name__ == '__main__':
    import json
    import sys

    logging.basicConfig(level=logging.INFO)

    if len(sys.argv) < 2:
        print("Usage: python utils/config_compat.py <config_path>")
        sys.exit(1)

    config_path = sys.argv[1]

    with open(config_path) as f:
        cfg = json.load(f)

    print(f"\nOriginal config fusion.weights:")
    print(json.dumps(cfg.get('fusion', {}).get('weights', {}), indent=2))

    cfg = normalize_config_for_hybrid(cfg)

    print(f"\nNormalized config fusion.weights:")
    print(json.dumps(cfg.get('fusion', {}).get('weights', {}), indent=2))

    print("\n✅ Config compatibility check passed!")
