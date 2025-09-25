"""
Bull Machine v1.5.0 - Configuration Loader
Handles loading and merging configuration files with asset profiles
"""

import json
import os
import copy
from typing import Dict, Any, Optional


def load_config(asset: Optional[str] = None, version: str = "v150") -> Dict[str, Any]:
    """
    Load configuration with optional asset profile override.

    Args:
        asset: Asset name for profile-specific config (e.g., "ETH")
        version: Config version directory (default: "v150")

    Returns:
        Merged configuration dictionary
    """
    # Base configuration
    base_config = {
        "entry_threshold": 0.45,
        "quality_floors": {
            "wyckoff": 0.37,
            "liquidity": 0.32,
            "structure": 0.35,
            "momentum": 0.40,
            "volume": 0.30,
            "context": 0.35,
            "mtf": 0.40
        },
        "layer_weights": {
            "wyckoff": 0.30,
            "liquidity": 0.25,
            "structure": 0.15,
            "momentum": 0.15,
            "volume": 0.15,
            "context": 0.05,
            "mtf": 0.10
        },
        "atr": {
            "window": 14,
            "mult": {
                "trend": 2.0,
                "range": 1.5
            }
        },
        "features": {
            "mtf_dl2": False,
            "six_candle_leg": False,
            "orderflow_lca": False,
            "negative_vip": False,
            "live_data": False
        },
        "use_asset_profiles": True,
        "profile_name": "base"
    }

    # Try to load from file if it exists
    try:
        base_path = os.path.join("configs", version, "base.json")
        if os.path.exists(base_path):
            with open(base_path, 'r') as f:
                base_config = json.load(f)
    except Exception:
        pass  # Use default config

    # Return base config if no asset profile requested
    if not asset or not base_config.get("use_asset_profiles", False):
        return base_config

    # Try to load asset-specific overrides
    try:
        asset_path = os.path.join("configs", version, "assets", f"{asset}.json")
        if os.path.exists(asset_path):
            with open(asset_path, 'r') as f:
                asset_config = json.load(f)
            # Deep merge asset config into base config
            merged_config = _deep_merge(base_config, asset_config)
            merged_config["profile_name"] = asset
            return merged_config
    except Exception:
        pass  # Use base config if asset config fails to load

    return base_config


def _deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    Args:
        base: Base dictionary
        override: Override dictionary

    Returns:
        Merged dictionary (new copy)
    """
    result = copy.deepcopy(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            result[key] = _deep_merge(result[key], value)
        else:
            # Override the value
            result[key] = copy.deepcopy(value)

    return result


def is_feature_enabled(config: Dict[str, Any], feature: str) -> bool:
    """
    Check if a feature flag is enabled in the configuration.

    Args:
        config: Configuration dictionary
        feature: Feature name

    Returns:
        True if feature is enabled, False otherwise
    """
    features = config.get("features", {})
    return features.get(feature, False)


def get_feature_flags(config: Dict[str, Any]) -> Dict[str, bool]:
    """
    Get all feature flags from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of feature flags
    """
    return config.get("features", {})


def validate_config(config: Dict[str, Any]) -> bool:
    """
    Validate configuration dictionary.

    Args:
        config: Configuration dictionary to validate

    Returns:
        True if valid, raises ValueError if invalid
    """
    # Required fields
    required_fields = ["entry_threshold", "quality_floors", "features"]
    for field in required_fields:
        if field not in config:
            raise ValueError(f"Missing required config field: {field}")

    # Validate quality floors
    quality_floors = config["quality_floors"]
    required_floors = ["wyckoff", "liquidity", "structure", "momentum", "volume", "context", "mtf"]

    for floor in required_floors:
        if floor not in quality_floors:
            raise ValueError(f"Missing quality floor: {floor}")

        floor_value = quality_floors[floor]
        if not isinstance(floor_value, (int, float)) or floor_value < 0 or floor_value > 1:
            raise ValueError(f"Quality floor {floor} must be between 0 and 1, got {floor_value}")

    # Validate features
    features = config["features"]
    if not isinstance(features, dict):
        raise ValueError("Features must be a dictionary")

    return True


def get_acceptance_gates(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get acceptance gates from configuration.

    Args:
        config: Configuration dictionary

    Returns:
        Dictionary of acceptance gates
    """
    return config.get("acceptance_gates", {})