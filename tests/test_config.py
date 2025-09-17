import pytest
import json
import os

def test_config_loads_successfully():
    """Test that v1.2.1 config loads without errors"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'bull_machine', 'config', 'config_v1_2_1.json')

    assert os.path.exists(config_path), f"Config file not found at {config_path}"

    with open(config_path, 'r') as f:
        config = json.load(f)

    assert config is not None
    assert config.get('version') == '1.2.1'

def test_config_weights_valid():
    """Test that fusion weights are properly configured"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'bull_machine', 'config', 'config_v1_2_1.json')

    with open(config_path, 'r') as f:
        config = json.load(f)

    weights = config['fusion']['weights']

    # Check all expected modules are present
    expected_modules = ['wyckoff', 'liquidity', 'structure', 'momentum', 'volume', 'context']
    for module in expected_modules:
        assert module in weights, f"Missing weight for {module}"

    # Check weights sum to 1.0
    total_weight = sum(weights.values())
    assert abs(total_weight - 1.0) < 0.01, f"Weights sum to {total_weight}, should be 1.0"

    # Check individual weights are reasonable
    for module, weight in weights.items():
        assert 0.0 <= weight <= 1.0, f"{module} weight {weight} out of range [0,1]"

def test_config_thresholds_achievable():
    """Test that configured thresholds are achievable"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'bull_machine', 'config', 'config_v1_2_1.json')

    with open(config_path, 'r') as f:
        config = json.load(f)

    fusion_cfg = config['fusion']

    # Enter threshold should be achievable with perfect scores
    enter_threshold = fusion_cfg['enter_threshold']
    assert 0.0 <= enter_threshold <= 1.0, f"Enter threshold {enter_threshold} out of range"
    assert enter_threshold <= 0.80, f"Enter threshold {enter_threshold} may be too high"

    # Trend alignment threshold should be reasonable
    trend_threshold = fusion_cfg['trend_alignment_threshold']
    assert 0.0 <= trend_threshold <= 1.0, f"Trend threshold {trend_threshold} out of range"
    assert trend_threshold <= 0.85, f"Trend threshold {trend_threshold} may be too high"

    # Volatility shock sigma should allow normal market movement
    vol_sigma = fusion_cfg['volatility_shock_sigma']
    assert vol_sigma >= 2.0, f"Volatility sigma {vol_sigma} too low, will block normal moves"
    assert vol_sigma <= 6.0, f"Volatility sigma {vol_sigma} too high, won't block real shocks"

def test_config_features_enabled():
    """Test that core features are properly enabled"""
    config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                              'bull_machine', 'config', 'config_v1_2_1.json')

    with open(config_path, 'r') as f:
        config = json.load(f)

    features = config['features']

    # Core features should be enabled
    assert features['wyckoff'] is True
    assert features['advanced_liquidity'] is True
    assert features['advanced_fusion'] is True
    assert features['veto_system'] is True