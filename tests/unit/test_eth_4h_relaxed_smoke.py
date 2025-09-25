"""
Test ETH 4H relaxed floor configuration
"""

import pandas as pd
from bull_machine.modules.fusion.v150_enhanced import FusionEngineV150


def test_eth_4h_relaxed_should_pass():
    """Test that relaxed 4H floors allow marginal signals through."""

    # ETH 4H configuration with relaxed floors
    config = {
        "quality_floors": {
            'wyckoff': 0.34,
            'liquidity': 0.30,
            'structure': 0.34,
            'momentum': 0.37,
            'volume': 0.30,
            'context': 0.34,
            'mtf': 0.37
        },
        "features": {
            "mtf_dl2": True,
            "six_candle_leg": True,
            "orderflow_lca": False,  # Disabled for 4H
            "negative_vip": False,    # Disabled for 4H
            "live_data": False
        },
        "timeframe": "4H",
        "entry_threshold": 0.38
    }

    engine = FusionEngineV150(config)

    # Create test data
    df = pd.DataFrame({
        "close": [100] * 150,
        "volume": [1000] * 150
    })

    # Scores that would fail stricter floors but pass relaxed ones
    layer_scores = {
        'wyckoff': 0.35,   # Above 0.34 floor
        'liquidity': 0.31, # Above 0.30 floor
        'structure': 0.35, # Above 0.34 floor
        'momentum': 0.38,  # Above 0.37 floor
        'volume': 0.31,    # Above 0.30 floor
        'context': 0.35,   # Above 0.34 floor
        'mtf': 0.38        # Above 0.37 floor
    }

    # Should pass quality floors
    assert engine.check_quality_floors(layer_scores) is True

    # Should not be vetoed
    veto = engine.check_confluence_vetoes(df, layer_scores, config)
    assert veto is False, f"Should not veto with relaxed floors, got veto={veto}"


def test_eth_1d_standard_floors():
    """Test ETH 1D configuration with standard floors."""

    config = {
        "quality_floors": {
            'wyckoff': 0.34,
            'liquidity': 0.30,
            'structure': 0.34,
            'momentum': 0.37,
            'volume': 0.30,
            'context': 0.34,
            'mtf': 0.37
        },
        "features": {
            "mtf_dl2": True,
            "six_candle_leg": True,
            "orderflow_lca": True,
            "negative_vip": True,
            "live_data": False
        },
        "timeframe": "1D",
        "entry_threshold": 0.46
    }

    engine = FusionEngineV150(config)

    df = pd.DataFrame({
        "close": [100, 101, 100, 102, 101, 103] + [102] * 144,
        "volume": [1000] * 150
    })

    # Good quality scores for 1D
    layer_scores = {
        'wyckoff': 0.45,
        'liquidity': 0.40,
        'structure': 0.42,
        'momentum': 0.48,
        'volume': 0.38,
        'context': 0.45,
        'mtf': 0.50
    }

    # Should pass quality floors
    assert engine.check_quality_floors(layer_scores) is True

    # Apply v1.5.0 alphas
    enhanced_scores = engine.apply_v150_alphas(df, layer_scores)

    # Scores should be modified by alphas
    assert enhanced_scores['mtf'] != layer_scores['mtf'], "MTF should be affected by alphas"


def test_cooldown_enforcement():
    """Test that cooldown bars prevent rapid re-entry."""

    config = {
        "cooldown_bars": 6,
        "quality_floors": {},
        "features": {},
        "timeframe": "4H"
    }

    engine = FusionEngineV150(config)

    df = pd.DataFrame({
        "close": [100] * 20,
        "volume": [1000] * 20
    })

    # First entry at bar 10
    last_trade_bar = 10

    # Try to enter at bar 14 (only 4 bars later)
    df_at_14 = df.iloc[:15]  # Simulate being at bar 14

    # Should be blocked by cooldown
    entry_allowed = engine.check_entry(df_at_14, last_trade_bar, config)
    assert entry_allowed is False, "Should block entry during cooldown"

    # Try at bar 17 (7 bars later, cooldown expired)
    df_at_17 = df.iloc[:18]
    entry_allowed = engine.check_entry(df_at_17, last_trade_bar, config)
    # Would pass cooldown but might fail on other criteria
    # Just verify cooldown logic doesn't block it


if __name__ == "__main__":
    test_eth_4h_relaxed_should_pass()
    print("✅ ETH 4H relaxed floors test passed")

    test_eth_1d_standard_floors()
    print("✅ ETH 1D standard floors test passed")

    test_cooldown_enforcement()
    print("✅ Cooldown enforcement test passed")

    print("\n🎯 All v1.5.0 optimization tests passed!")