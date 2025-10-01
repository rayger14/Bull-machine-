"""Smoke tests for v1.4.2 fusion compatibility"""

import sys
import os
# Add parent directory to path when running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

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


if __name__ == "__main__":
    # Run tests when executed directly
    print("Running fusion compatibility tests...")
    try:
        test_compat_legacy_keys()
        print("✅ test_compat_legacy_keys passed")
    except AssertionError as e:
        print(f"❌ test_compat_legacy_keys failed: {e}")
        exit(1)

    try:
        test_enhanced_fusion_basic()
        print("✅ test_enhanced_fusion_basic passed")
    except AssertionError as e:
        print(f"❌ test_enhanced_fusion_basic failed: {e}")
        exit(1)

    print("✅ All fusion compatibility tests passed!")
