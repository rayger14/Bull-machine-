"""
Bull Machine v1.5.0 - Asset Profile Layer (APL) Configuration Loader
Implements deep merge of base configs with asset-specific overrides.
"""

import json
import os
from copy import deepcopy
from typing import Dict, Any, Optional


def _load(path: str) -> Dict[str, Any]:
    """Load JSON configuration from file."""
    with open(path, "r") as f:
        return json.load(f)


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries, with override taking precedence."""
    result = deepcopy(base)

    for key, value in override.items():
        if isinstance(value, dict) and key in result and isinstance(result[key], dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result


def load_config(asset: Optional[str] = None, version: str = "v150") -> Dict[str, Any]:
    """
    Load configuration with Asset Profile Layer support.

    Args:
        asset: Asset symbol (e.g., 'ETH', 'BTC'). If provided and asset profile exists,
               it will be merged with base config.
        version: Config version to load (default: 'v150')

    Returns:
        Merged configuration dictionary

    Examples:
        >>> config = load_config()  # Base config only
        >>> eth_config = load_config("ETH")  # Base + ETH overrides
    """
    # Load base configuration
    base_path = os.path.join("configs", version, "base.json")
    base_config = _load(base_path)

    # If asset profiles disabled or no asset specified, return base
    if not base_config.get("use_asset_profiles", False) or not asset:
        return base_config

    # Check for asset-specific profile
    asset_path = os.path.join("configs", version, "assets", f"{asset}.json")
    if os.path.exists(asset_path):
        asset_config = _load(asset_path)
        return _deep_merge(base_config, asset_config)

    # Asset profile not found, return base config
    return base_config


def get_feature_flags(config: Dict[str, Any]) -> Dict[str, bool]:
    """Extract feature flags from configuration."""
    return config.get("features", {})


def is_feature_enabled(config: Dict[str, Any], feature_name: str) -> bool:
    """Check if a specific feature flag is enabled."""
    return config.get("features", {}).get(feature_name, False)


def get_acceptance_gates(config: Dict[str, Any]) -> Dict[str, Any]:
    """Get acceptance gate thresholds for validation."""
    return config.get("acceptance_gates", {})


def validate_config(config: Dict[str, Any]) -> bool:
    """Validate configuration structure and required fields."""
    required_fields = ["entry_threshold", "quality_floors", "atr", "features"]

    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    # Validate quality floors
    required_floors = ["wyckoff", "liquidity", "structure", "momentum", "volume", "context", "mtf"]
    quality_floors = config["quality_floors"]

    for floor in required_floors:
        if floor not in quality_floors:
            raise ValueError(f"Missing quality floor: {floor}")
        if not 0 <= quality_floors[floor] <= 1:
            raise ValueError(f"Quality floor {floor} must be between 0 and 1")

    # Validate feature flags
    if not isinstance(config["features"], dict):
        raise ValueError("Features must be a dictionary")

    return True


if __name__ == "__main__":
    # Quick test
    print("Testing config loader...")

    base = load_config()
    print(f"Base entry threshold: {base['entry_threshold']}")

    eth = load_config("ETH")
    print(f"ETH entry threshold: {eth['entry_threshold']}")

    print(f"Feature flags: {get_feature_flags(base)}")
    print(f"MTF DL2 enabled: {is_feature_enabled(base, 'mtf_dl2')}")