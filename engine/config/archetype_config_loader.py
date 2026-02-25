#!/usr/bin/env python3
"""
Archetype Configuration Loader

Loads archetype configs from YAML files for isolation and independent optimization.
Supports both YAML and legacy JSON configs for backward compatibility.

Design:
- Each archetype has its own YAML file in configs/archetypes/
- Configs include per-archetype fusion weights for independent optimization
- Loader validates schema and provides defaults
- Backward compatible with existing JSON configs

Usage:
    from engine.config.archetype_config_loader import load_archetype_configs

    configs = load_archetype_configs("configs/archetypes/")
    spring_config = configs["spring"]
    fusion_weights = spring_config["fusion_weights"]
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Any

import yaml

logger = logging.getLogger(__name__)

# Default fusion weights (balanced across all domains)
DEFAULT_FUSION_WEIGHTS = {
    "wyckoff": 0.25,
    "liquidity": 0.25,
    "momentum": 0.25,
    "smc": 0.25
}

# Default exit logic
DEFAULT_EXIT_LOGIC = {
    "max_hold_hours": 168,
    "scale_out_enabled": True,
    "scale_out_levels": [1.0, 2.0, 3.0],
    "scale_out_pcts": [0.3, 0.4, 0.3],
    "trailing_start_r": 1.0,
    "trailing_atr_mult": 2.0
}

# Default position sizing
DEFAULT_POSITION_SIZING = {
    "risk_per_trade_pct": 0.02,
    "max_position_size_pct": 0.1,
    "atr_stop_mult": 2.5
}

# Default regime preferences (neutral - no bias)
DEFAULT_REGIME_PREFERENCES = {
    "risk_on": 1.0,
    "neutral": 1.0,
    "risk_off": 1.0,
    "crisis": 1.0
}

# Required fields for archetype config schema
REQUIRED_FIELDS = ["name", "display_name", "direction", "thresholds"]

# Optional fields with defaults
OPTIONAL_FIELDS = {
    "fusion_weights": DEFAULT_FUSION_WEIGHTS,
    "exit_logic": DEFAULT_EXIT_LOGIC,
    "position_sizing": DEFAULT_POSITION_SIZING,
    "regime_preferences": DEFAULT_REGIME_PREFERENCES,
    "allowed_regimes": [],
    "aliases": [],
    "description": "",
    "priority": 100,
    "enabled": True,
    "hard_gates": [],  # Declarative pattern gates evaluated BEFORE fusion computation
    "gate_mode": "hard",  # "hard" (block) or "soft" (penalize fusion score)
}


def load_archetype_config(file_path: str) -> Dict[str, Any]:
    """
    Load a single archetype config from YAML file.

    Args:
        file_path: Path to YAML config file

    Returns:
        Validated archetype config dict

    Raises:
        ValueError: If config is invalid
        FileNotFoundError: If file doesn't exist
    """
    logger.info(f"Loading archetype config: {file_path}")

    # Load YAML
    with open(file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Validate required fields
    for field in REQUIRED_FIELDS:
        if field not in config:
            raise ValueError(f"Missing required field '{field}' in {file_path}")

    # Add optional fields with defaults
    for field, default in OPTIONAL_FIELDS.items():
        if field not in config:
            config[field] = default
            logger.debug(f"Using default {field} for {config['name']}")

    # Validate fusion weights sum to ~1.0
    fusion_weights = config["fusion_weights"]
    weight_sum = sum(fusion_weights.values())
    if not (0.95 <= weight_sum <= 1.05):
        logger.warning(
            f"Fusion weights for {config['name']} sum to {weight_sum:.3f}, "
            f"expected ~1.0. Normalizing..."
        )
        # Normalize weights
        for key in fusion_weights:
            fusion_weights[key] /= weight_sum

    # Validate direction
    if config["direction"] not in ["long", "short", "neutral"]:
        raise ValueError(
            f"Invalid direction '{config['direction']}' in {file_path}. "
            f"Must be 'long', 'short', or 'neutral'"
        )

    # Validate regime preferences
    for regime in config["regime_preferences"]:
        if regime not in ["risk_on", "neutral", "risk_off", "crisis"]:
            raise ValueError(
                f"Invalid regime '{regime}' in {file_path}. "
                f"Must be one of: risk_on, neutral, risk_off, crisis"
            )

    logger.info(
        f"Loaded {config['name']} ({config['display_name']}) - "
        f"Fusion: W={fusion_weights['wyckoff']:.2f} "
        f"L={fusion_weights['liquidity']:.2f} "
        f"M={fusion_weights['momentum']:.2f} "
        f"S={fusion_weights['smc']:.2f}"
    )

    return config


def load_archetype_configs(
    config_dir: str = "configs/archetypes/",
    enabled_only: bool = False
) -> Dict[str, Dict[str, Any]]:
    """
    Load all archetype configs from YAML directory.

    Args:
        config_dir: Directory containing archetype YAML files
        enabled_only: If True, only load configs with enabled=True

    Returns:
        Dict mapping archetype_name -> config

    Example:
        >>> configs = load_archetype_configs()
        >>> spring = configs["spring"]
        >>> spring["fusion_weights"]
        {'wyckoff': 0.6, 'liquidity': 0.2, 'momentum': 0.1, 'smc': 0.1}
    """
    config_path = Path(config_dir)

    if not config_path.exists():
        raise FileNotFoundError(f"Archetype config directory not found: {config_dir}")

    configs = {}

    # Load all YAML files
    yaml_files = list(config_path.glob("*.yaml")) + list(config_path.glob("*.yml"))

    logger.info(f"Loading {len(yaml_files)} archetype configs from {config_dir}")

    for yaml_file in yaml_files:
        try:
            config = load_archetype_config(str(yaml_file))

            # Skip if disabled (if enabled_only=True)
            if enabled_only and not config.get("enabled", True):
                logger.info(f"Skipping disabled archetype: {config['name']}")
                continue

            configs[config["name"]] = config

        except Exception as e:
            logger.error(f"Failed to load {yaml_file}: {e}")
            # Continue loading other configs

    logger.info(f"Successfully loaded {len(configs)} archetype configs")

    return configs


def get_archetype_fusion_weights(
    archetype_name: str,
    configs: Optional[Dict[str, Dict[str, Any]]] = None,
    config_dir: str = "configs/archetypes/"
) -> Dict[str, float]:
    """
    Get fusion weights for a specific archetype.

    Args:
        archetype_name: Name of archetype (e.g., "spring")
        configs: Pre-loaded configs dict (optional, will load if not provided)
        config_dir: Directory to load configs from (if configs not provided)

    Returns:
        Dict of fusion weights: {wyckoff, liquidity, momentum, smc}

    Example:
        >>> get_archetype_fusion_weights("spring")
        {'wyckoff': 0.6, 'liquidity': 0.2, 'momentum': 0.1, 'smc': 0.1}
    """
    if configs is None:
        configs = load_archetype_configs(config_dir)

    if archetype_name not in configs:
        logger.warning(
            f"Archetype '{archetype_name}' not found in configs. "
            f"Using default fusion weights."
        )
        return DEFAULT_FUSION_WEIGHTS.copy()

    return configs[archetype_name]["fusion_weights"]


def get_enabled_archetypes(
    configs: Optional[Dict[str, Dict[str, Any]]] = None,
    config_dir: str = "configs/archetypes/"
) -> List[str]:
    """
    Get list of enabled archetype names.

    Args:
        configs: Pre-loaded configs dict (optional)
        config_dir: Directory to load configs from

    Returns:
        List of enabled archetype names
    """
    if configs is None:
        configs = load_archetype_configs(config_dir)

    enabled = [
        name for name, config in configs.items()
        if config.get("enabled", True)
    ]

    logger.info(f"Found {len(enabled)} enabled archetypes")
    return enabled


def merge_with_base_config(
    archetype_configs: Dict[str, Dict[str, Any]],
    base_config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Merge archetype configs with base config for backward compatibility.

    This creates a combined config in the legacy format that can be used
    with existing code while preserving the new per-archetype fusion weights.

    Args:
        archetype_configs: Dict of archetype configs from YAML
        base_config: Base config dict (from JSON)

    Returns:
        Merged config dict
    """
    merged = base_config.copy()

    # Extract archetype thresholds
    thresholds = {}
    for name, config in archetype_configs.items():
        thresholds[name] = config["thresholds"]

    # Update base config
    if "archetypes" not in merged:
        merged["archetypes"] = {}

    merged["archetypes"]["thresholds"] = thresholds

    # Store fusion weights separately for archetype-specific fusion
    merged["archetype_fusion_weights"] = {
        name: config["fusion_weights"]
        for name, config in archetype_configs.items()
    }

    # Store exit logic
    merged["archetype_exit_logic"] = {
        name: config["exit_logic"]
        for name, config in archetype_configs.items()
    }

    # Store regime preferences
    merged["archetype_regime_preferences"] = {
        name: config["regime_preferences"]
        for name, config in archetype_configs.items()
    }

    logger.info(
        f"Merged {len(archetype_configs)} archetype configs with base config"
    )

    return merged


def validate_archetype_config(config: Dict[str, Any]) -> bool:
    """
    Validate archetype config schema.

    Args:
        config: Archetype config dict

    Returns:
        True if valid, False otherwise
    """
    # Check required fields
    for field in REQUIRED_FIELDS:
        if field not in config:
            logger.error(f"Missing required field: {field}")
            return False

    # Check fusion weights
    if "fusion_weights" in config:
        required_weights = {"wyckoff", "liquidity", "momentum", "smc"}
        if set(config["fusion_weights"].keys()) != required_weights:
            logger.error(
                f"Invalid fusion_weights. Expected {required_weights}, "
                f"got {set(config['fusion_weights'].keys())}"
            )
            return False

    # Check direction
    if config["direction"] not in ["long", "short", "neutral"]:
        logger.error(f"Invalid direction: {config['direction']}")
        return False

    return True


# Convenience function for CLI usage
def main():
    """CLI tool to validate archetype configs."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Load and validate archetype configs"
    )
    parser.add_argument(
        "--dir",
        default="configs/archetypes/",
        help="Directory containing archetype YAML files"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    try:
        configs = load_archetype_configs(args.dir)
        print(f"\nSuccessfully loaded {len(configs)} archetype configs:")

        for name, config in sorted(configs.items()):
            weights = config["fusion_weights"]
            print(
                f"  {name:20s} | "
                f"W={weights['wyckoff']:.2f} "
                f"L={weights['liquidity']:.2f} "
                f"M={weights['momentum']:.2f} "
                f"S={weights['smc']:.2f} | "
                f"{config['direction']:7s} | "
                f"{'ENABLED' if config['enabled'] else 'DISABLED'}"
            )

        print(f"\nAll configs valid ✓")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
