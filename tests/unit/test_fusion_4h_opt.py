"""
Unit tests for v1.5.0 optimized fusion engine - 4H focus
"""

import pandas as pd
import pytest
from bull_machine.core.fusion_enhanced import FusionEngineV150


class TestFusion4HOptimized:
    """Test 4H optimized fusion engine with relaxed floors."""

    def test_confluence_vetoes_4h_relaxed(self):
        """Test that relaxed floors allow more trades through."""
        config = {
            "quality_floors": {
                'wyckoff': 0.32,
                'liquidity': 0.28,
                'structure': 0.32,
                'momentum': 0.35,
                'volume': 0.28,
                'context': 0.32,
                'mtf': 0.35
            },
            "features": {
                "mtf_dl2": False,
                "six_candle_leg": False,
                "orderflow_lca": False,
                "negative_vip": False,
                "live_data": False
            },
            "entry_threshold": 0.40,
            "timeframe": "4H"
        }

        engine = FusionEngineV150(config)

        # Layer scores that would fail with strict floors but pass with relaxed ones
        layer_scores = {
            'wyckoff': 0.33,      # Above relaxed 0.32 (was failing 0.37)
            'liquidity': 0.29,    # Above relaxed 0.28 (was failing 0.32)
            'structure': 0.33,    # Above relaxed 0.32 (was failing 0.35)
            'momentum': 0.36,     # Above relaxed 0.35 (was failing 0.40)
            'volume': 0.29,       # Above relaxed 0.28 (was failing 0.30)
            'context': 0.33,      # Above relaxed 0.32 (was failing 0.35)
            'mtf': 0.36           # Above relaxed 0.35 (was failing 0.40)
        }

        # Should pass quality floors with relaxed settings
        quality_passed = engine.check_quality_floors(layer_scores)
        assert quality_passed is True

    def test_4h_selective_features_disabled(self):
        """Test that 4H config has LCA and VIP disabled."""
        config = {
            "features": {
                "mtf_dl2": True,
                "six_candle_leg": True,
                "orderflow_lca": False,  # Disabled for 4H
                "negative_vip": False,   # Disabled for 4H
                "live_data": False
            },
            "quality_floors": {
                'wyckoff': 0.32, 'liquidity': 0.28, 'structure': 0.32,
                'momentum': 0.35, 'volume': 0.28, 'context': 0.32, 'mtf': 0.35
            },
            "entry_threshold": 0.40,
            "timeframe": "4H"
        }

        engine = FusionEngineV150(config)

        # Mock DataFrame
        df = pd.DataFrame({
            'close': [100, 101, 102, 101, 100, 101] * 20,
            'volume': [1000] * 120
        })

        layer_scores = {
            'wyckoff': 0.45, 'liquidity': 0.40, 'structure': 0.42,
            'momentum': 0.48, 'volume': 0.38, 'context': 0.45, 'mtf': 0.50
        }

        # Apply v1.5.0 alphas
        enhanced_scores = engine.apply_v150_alphas(df, layer_scores)

        # LCA and VIP should not affect scores since they're disabled
        assert enhanced_scores['structure'] == layer_scores['structure']  # No LCA impact
        assert enhanced_scores['volume'] == layer_scores['volume']        # No VIP impact

        # MTF features should still apply
        assert enhanced_scores['mtf'] != layer_scores['mtf']  # MTF should be affected

    def test_reduced_alpha_impact(self):
        """Test that alpha impacts are reduced in optimized version."""
        config = {
            "features": {
                "mtf_dl2": True,
                "six_candle_leg": True,
                "orderflow_lca": True,
                "negative_vip": True,
                "live_data": False
            },
            "quality_floors": {
                'wyckoff': 0.32, 'liquidity': 0.28, 'structure': 0.32,
                'momentum': 0.35, 'volume': 0.28, 'context': 0.32, 'mtf': 0.35
            },
            "entry_threshold": 0.40,
            "timeframe": "1H",
            "profile_name": "ETH"
        }

        engine = FusionEngineV150(config)

        # Mock DataFrame with valid pattern
        df = pd.DataFrame({
            'close': [100, 101, 100, 102, 101, 103] + [102] * 14,  # Valid alternating + stable
            'high': [101, 102, 101, 103, 102, 104] + [103] * 14,
            'low': [99, 100, 99, 101, 100, 102] + [101] * 14,
            'volume': [1000] * 20
        })

        base_scores = {
            'wyckoff': 0.45, 'liquidity': 0.40, 'structure': 0.42,
            'momentum': 0.48, 'volume': 0.38, 'context': 0.45, 'mtf': 0.50
        }

        enhanced_scores = engine.apply_v150_alphas(df, base_scores)

        # Check that impacts are reduced (should be smaller changes)
        mtf_change = abs(enhanced_scores['mtf'] - base_scores['mtf'])
        structure_change = abs(enhanced_scores['structure'] - base_scores['structure'])
        volume_change = abs(enhanced_scores['volume'] - base_scores['volume'])

        # Changes should be modest (< 0.1) due to reduced alpha weights
        assert mtf_change < 0.1, f"MTF change too large: {mtf_change}"
        assert structure_change < 0.1, f"Structure change too large: {structure_change}"
        assert volume_change < 0.1, f"Volume change too large: {volume_change}"


def test_mtf_dl2_timeframe_thresholds():
    """Test that MTF DL2 uses relaxed thresholds for 4H."""
    try:
        from bull_machine.modules.mtf.mtf_sync import mtf_dl2_filter

        # Create data with moderate deviation
        np.random.seed(42)
        moderate_spike_data = pd.DataFrame({
            'close': [100] * 19 + [107]  # 7% spike - should fail 2.0 threshold but pass 2.5
        })

        # Should fail with default threshold
        assert mtf_dl2_filter(moderate_spike_data, "") is False

        # Should pass with 4H relaxed threshold
        assert mtf_dl2_filter(moderate_spike_data, "4H") is True
    except ImportError:
        # Skip if module not available
        pass


import numpy as np  # Add import for test