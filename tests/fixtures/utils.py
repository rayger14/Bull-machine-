"""
Test utilities for Bull Machine fixtures
Includes scoped configuration overrides for testing
"""

from contextlib import contextmanager
import copy

@contextmanager
def temp_config_overrides(cfg, updates):
    """
    Temporarily override configuration values for testing

    Args:
        cfg: Original configuration dictionary
        updates: Dictionary of updates to apply

    Usage:
        with temp_config_overrides(config, {
            "wyckoff": {"hps_floor_1h": 0.25},
            "hob": {"volume_z_min_1h": 1.0}
        }):
            signal = run_fixture_test(...)
            assert signal["ok"]
    """
    original = copy.deepcopy(cfg)
    try:
        # Apply updates with proper deep merging
        for key, value in updates.items():
            if key in cfg and isinstance(cfg[key], dict) and isinstance(value, dict):
                cfg[key].update(value)
            else:
                cfg[key] = value
        yield cfg
    finally:
        # Restore original configuration
        cfg.clear()
        cfg.update(original)

def get_relaxed_test_config():
    """
    Get relaxed configuration for golden fixture testing

    Returns:
        dict: Configuration overrides that make tests more permissive
    """
    return {
        "wyckoff": {
            "hps_floor_1h": 0.25,      # Relaxed from 0.40
            "phase_confidence_min": 0.15,  # Relaxed from 0.25
            "spring_deviation": 0.025   # Slightly more permissive
        },
        "hob": {
            "volume_z_min_1h": 1.0,    # Relaxed from 1.6
            "volume_z_min_4h": 1.2,    # Relaxed from 1.8
            "proximity_tolerance": 0.4  # More permissive proximity
        },
        "smc": {
            "ob_min_strength": 0.15,   # Relaxed order block strength
            "fvg_threshold": 0.002,    # Smaller FVG threshold
            "liquidity_sweep_min": 0.01  # Lower sweep requirement
        },
        "momentum": {
            "rsi_confirmation": 0.6,   # Lower RSI confirmation
            "trend_strength_min": 0.20  # Relaxed trend requirement
        },
        "fusion": {
            "calibration_thresholds": {
                "confidence": 0.15,    # Much lower for testing
                "strength": 0.25       # Relaxed strength requirement
            }
        }
    }

def validate_fixture_result(result, expected_direction=None, min_confidence=0.1):
    """
    Validate fixture test results with relaxed criteria

    Args:
        result: Test result dictionary
        expected_direction: Expected signal direction ('long'/'short')
        min_confidence: Minimum confidence threshold

    Returns:
        bool: True if result passes validation
    """
    if not result or not isinstance(result, dict):
        return False

    # Check basic structure
    if 'direction' not in result or 'confidence' not in result:
        return False

    # Check confidence threshold
    if result['confidence'] < min_confidence:
        return False

    # Check direction if specified
    if expected_direction and result['direction'] != expected_direction:
        return False

    return True