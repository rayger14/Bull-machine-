"""Smoke tests for v1.4.2 fusion compatibility"""

import pytest
from bull_machine.modules.fusion.advanced import AdvancedFusionEngine
from bull_machine.core.types import WyckoffResult


def test_compat_legacy_keys():
    """Ensure AdvancedFusionEngine maintains backward compatibility"""
    config = {
        "features": {"trend_filter": True},
        "fusion": {"enter_threshold": 0.5, "weights": {"wyckoff": 1.0}},
        "mode": {"enter_threshold": 0.5},
    }

    fusion = AdvancedFusionEngine(config)

    # Test basic initialization
    assert hasattr(fusion, "fuse")
    assert hasattr(fusion, "fuse_with_mtf")

    # Test that fuse method works
    wy = WyckoffResult(
        regime="trending", phase="D", bias="long", phase_confidence=0.8, trend_confidence=0.8, range=None
    )
    modules = {"wyckoff": wy, "liquidity": {"score": 0.8, "pressure": "bullish"}, "series": None}

    res = fusion.fuse(modules)
    # Should return a Signal object or None (not crash)
    assert res is None or hasattr(res, "confidence")


def test_enhanced_fusion_basic():
    """Test enhanced fusion engine basic functionality"""
    from bull_machine.modules.fusion.enhanced import EnhancedFusionEngineV1_4

    config = {
        "mode": {"enter_threshold": 0.42},
        "weights": {"wyckoff": 0.3, "liquidity": 0.25},
        "quality_floors": {"wyckoff": 0.55, "liquidity": 0.50},
    }

    engine = EnhancedFusionEngineV1_4(config)
    assert hasattr(engine, "fuse_with_mtf")
    assert engine.enter_threshold == 0.42
