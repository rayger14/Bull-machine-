#!/usr/bin/env python3
"""
MVP Test Suite for Bull Archetypes

Tests the 5 priority bull archetypes:
1. Spring/UTAD (A)
2. Order Block Retest (B)
3. BOS/CHOCH Reversal (C)
4. Liquidity Sweep (G)
5. Trap Within Trend (H)

Acceptance criteria for each archetype:
- Produces >5 signals on 2022-2024 data
- Has clear entry logic (not just fusion score)
- Uses domain engines (Wyckoff/SMC confirmation)
- Has safety vetoes
- Is backtestable

Author: Claude Code (Backend Architect)
Date: 2025-12-12
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from engine.strategies.archetypes.bull import (
    SpringUTADArchetype,
    OrderBlockRetestArchetype,
    BOSCHOCHReversalArchetype,
    LiquiditySweepArchetype,
    TrapWithinTrendArchetype
)


# ============================================================================
# Test Fixtures
# ============================================================================

@pytest.fixture
def sample_bull_bar():
    """Create a sample bullish bar with all required features."""
    return pd.Series({
        # OHLCV
        'open': 30000.0,
        'high': 30500.0,
        'low': 29800.0,
        'close': 30400.0,
        'volume': 1000.0,

        # Technical indicators
        'rsi_14': 55.0,
        'adx_14': 25.0,
        'macd': 50.0,
        'macd_signal': 40.0,
        'macd_hist': 10.0,
        'volume_zscore': 1.5,

        # Wyckoff features
        'wyckoff_spring_a': True,
        'wyckoff_spring_a_confidence': 0.7,
        'wyckoff_spring_b': False,
        'wyckoff_spring_b_confidence': 0.0,
        'wyckoff_lps': True,
        'wyckoff_lps_confidence': 0.6,
        'wyckoff_sos': False,
        'wyckoff_sos_confidence': 0.0,
        'wyckoff_phase_abc': 'C',

        # SMC features
        'smc_demand_zone': True,
        'smc_liquidity_sweep': True,
        'smc_choch': False,
        'tf1h_ob_bull_bottom': 29900.0,
        'tf1h_ob_bull_top': 30100.0,
        'tf1h_bos_bullish': True,
        'tf4h_bos_bullish': False,
        'tf4h_bos_bearish': False,
        'tf1h_fvg_bull': True,

        # MTF features
        'tf4h_trend_direction': 1,
        'tf4h_fusion_score': 0.6,

        # Other
        'bearish_divergence_detected': False,
        'capitulation_depth': -0.05,
    })


@pytest.fixture
def sample_bear_bar():
    """Create a sample bearish bar (should NOT trigger bull signals)."""
    return pd.Series({
        # OHLCV
        'open': 30000.0,
        'high': 30200.0,
        'low': 29500.0,
        'close': 29600.0,
        'volume': 800.0,

        # Technical indicators
        'rsi_14': 30.0,
        'adx_14': 35.0,
        'macd': -50.0,
        'macd_signal': -40.0,
        'macd_hist': -10.0,
        'volume_zscore': -0.5,

        # Wyckoff features
        'wyckoff_spring_a': False,
        'wyckoff_lps': False,
        'wyckoff_phase_abc': 'neutral',

        # SMC features
        'smc_demand_zone': False,
        'smc_liquidity_sweep': False,
        'tf1h_bos_bullish': False,
        'tf4h_bos_bearish': True,

        # MTF features
        'tf4h_trend_direction': -1,
        'tf4h_fusion_score': 0.3,

        # Other
        'bearish_divergence_detected': True,
        'capitulation_depth': -0.02,
    })


# ============================================================================
# Spring/UTAD Tests
# ============================================================================

class TestSpringUTAD:
    """Test Spring/UTAD archetype (A)."""

    def test_initialization(self):
        """Test archetype initializes correctly."""
        archetype = SpringUTADArchetype()
        assert archetype.min_fusion_score == 0.35
        assert archetype.wyckoff_weight == 0.30

    def test_detects_spring_signal(self, sample_bull_bar):
        """Test detection of valid spring pattern."""
        archetype = SpringUTADArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        assert name == 'spring_utad', "Should detect spring pattern"
        assert confidence > 0.35, "Confidence should exceed threshold"
        assert metadata['wyckoff_score'] > 0.0, "Wyckoff score should be positive"
        assert metadata['pattern_type'] == 'spring_long'

    def test_vetoes_overbought(self, sample_bull_bar):
        """Test veto for overbought RSI."""
        sample_bull_bar['rsi_14'] = 75.0
        archetype = SpringUTADArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        assert name is None, "Should veto overbought entry"
        assert 'veto_reason' in metadata
        assert 'overbought' in metadata['veto_reason']

    def test_vetoes_strong_downtrend(self, sample_bull_bar):
        """Test veto for strong downtrend."""
        sample_bull_bar['tf4h_trend_direction'] = -1
        sample_bull_bar['adx_14'] = 30.0
        archetype = SpringUTADArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        assert name is None, "Should veto strong downtrend"
        assert 'veto_reason' in metadata

    def test_no_signal_on_bear_bar(self, sample_bear_bar):
        """Test no false signal on bearish bar."""
        archetype = SpringUTADArchetype()
        name, confidence, metadata = archetype.detect(sample_bear_bar, 'risk_on')

        assert name is None, "Should not detect spring on bearish bar"


# ============================================================================
# Order Block Retest Tests
# ============================================================================

class TestOrderBlockRetest:
    """Test Order Block Retest archetype (B)."""

    def test_initialization(self):
        """Test archetype initializes correctly."""
        archetype = OrderBlockRetestArchetype()
        assert archetype.min_fusion_score == 0.35
        assert archetype.smc_weight == 0.35

    def test_detects_ob_retest(self, sample_bull_bar):
        """Test detection of order block retest."""
        archetype = OrderBlockRetestArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        assert name == 'order_block_retest', "Should detect OB retest"
        assert confidence > 0.35, "Confidence should exceed threshold"
        assert metadata['smc_score'] > 0.0, "SMC score should be positive"
        assert metadata['pattern_type'] == 'order_block_retest_long'

    def test_vetoes_broken_support(self, sample_bull_bar):
        """Test veto for broken OB support."""
        # Close below OB bottom
        sample_bull_bar['close'] = 29800.0
        sample_bull_bar['tf1h_ob_bull_bottom'] = 29900.0

        archetype = OrderBlockRetestArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        assert name is None, "Should veto broken support"
        assert 'veto_reason' in metadata

    def test_no_signal_without_ob(self, sample_bull_bar):
        """Test no signal when OB not present."""
        sample_bull_bar['tf1h_ob_bull_bottom'] = None
        sample_bull_bar['tf1h_ob_bull_top'] = None

        archetype = OrderBlockRetestArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        assert name is None, "Should not detect without OB"


# ============================================================================
# BOS/CHOCH Reversal Tests
# ============================================================================

class TestBOSCHOCHReversal:
    """Test BOS/CHOCH Reversal archetype (C)."""

    def test_initialization(self):
        """Test archetype initializes correctly."""
        archetype = BOSCHOCHReversalArchetype()
        assert archetype.min_fusion_score == 0.35
        assert archetype.smc_weight == 0.40

    def test_detects_bos_signal(self, sample_bull_bar):
        """Test detection of BOS pattern."""
        archetype = BOSCHOCHReversalArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        assert name == 'bos_choch_reversal', "Should detect BOS"
        assert confidence > 0.35, "Confidence should exceed threshold"
        assert metadata['smc_score'] > 0.0, "SMC score should be positive"
        assert metadata['momentum_score'] > 0.0, "Momentum score should be positive"

    def test_vetoes_extreme_overbought(self, sample_bull_bar):
        """Test veto for extreme overbought."""
        sample_bull_bar['rsi_14'] = 85.0
        archetype = BOSCHOCHReversalArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        assert name is None, "Should veto extreme overbought"
        assert 'veto_reason' in metadata

    def test_vetoes_weak_volume(self, sample_bull_bar):
        """Test veto for weak volume on breakout."""
        sample_bull_bar['volume_zscore'] = -0.8
        archetype = BOSCHOCHReversalArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        assert name is None, "Should veto weak volume breakout"


# ============================================================================
# Liquidity Sweep Tests
# ============================================================================

class TestLiquiditySweep:
    """Test Liquidity Sweep archetype (G)."""

    def test_initialization(self):
        """Test archetype initializes correctly."""
        archetype = LiquiditySweepArchetype()
        assert archetype.min_fusion_score == 0.35
        assert archetype.smc_weight == 0.35

    def test_detects_sweep_signal(self, sample_bull_bar):
        """Test detection of liquidity sweep."""
        archetype = LiquiditySweepArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        assert name == 'liquidity_sweep', "Should detect sweep"
        assert confidence > 0.35, "Confidence should exceed threshold"
        assert metadata['price_action_score'] > 0.0, "Price action score should be positive"

    def test_requires_wick_rejection(self, sample_bull_bar):
        """Test requires deep lower wick."""
        # Create bar with no lower wick
        sample_bull_bar['low'] = 30000.0
        sample_bull_bar['open'] = 30000.0

        archetype = LiquiditySweepArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        # Should have lower score without wick
        assert metadata.get('price_action_score', 0) < 0.5

    def test_vetoes_weak_volume(self, sample_bull_bar):
        """Test veto for weak volume."""
        sample_bull_bar['volume_zscore'] = 0.3
        archetype = LiquiditySweepArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        assert name is None, "Should veto weak volume"


# ============================================================================
# Trap Within Trend Tests
# ============================================================================

class TestTrapWithinTrend:
    """Test Trap Within Trend archetype (H)."""

    def test_initialization(self):
        """Test archetype initializes correctly."""
        archetype = TrapWithinTrendArchetype()
        assert archetype.min_fusion_score == 0.35
        assert archetype.momentum_weight == 0.35

    def test_detects_trap_signal(self, sample_bull_bar):
        """Test detection of trap within uptrend."""
        archetype = TrapWithinTrendArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        assert name == 'trap_within_trend', "Should detect trap"
        assert confidence > 0.35, "Confidence should exceed threshold"
        assert metadata['momentum_score'] > 0.0, "Momentum score should be positive"

    def test_requires_uptrend(self, sample_bull_bar):
        """Test requires uptrend on higher timeframe."""
        sample_bull_bar['tf4h_trend_direction'] = -1
        archetype = TrapWithinTrendArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        assert name is None, "Should veto without uptrend"
        assert metadata.get('momentum_score', 1) == 0.0, "Momentum score should be 0 without uptrend"

    def test_vetoes_weak_momentum(self, sample_bull_bar):
        """Test veto for weak momentum."""
        sample_bull_bar['adx_14'] = 12.0
        archetype = TrapWithinTrendArchetype()
        name, confidence, metadata = archetype.detect(sample_bull_bar, 'risk_on')

        assert name is None, "Should veto weak momentum"


# ============================================================================
# Integration Tests
# ============================================================================

class TestArchetypeIntegration:
    """Integration tests across all archetypes."""

    def test_all_archetypes_importable(self):
        """Test all archetypes can be imported."""
        from engine.strategies.archetypes.bull import (
            SpringUTADArchetype,
            OrderBlockRetestArchetype,
            BOSCHOCHReversalArchetype,
            LiquiditySweepArchetype,
            TrapWithinTrendArchetype
        )

        assert SpringUTADArchetype is not None
        assert OrderBlockRetestArchetype is not None
        assert BOSCHOCHReversalArchetype is not None
        assert LiquiditySweepArchetype is not None
        assert TrapWithinTrendArchetype is not None

    def test_all_archetypes_have_detect_method(self):
        """Test all archetypes implement detect() method."""
        archetypes = [
            SpringUTADArchetype(),
            OrderBlockRetestArchetype(),
            BOSCHOCHReversalArchetype(),
            LiquiditySweepArchetype(),
            TrapWithinTrendArchetype()
        ]

        for archetype in archetypes:
            assert hasattr(archetype, 'detect')
            assert callable(archetype.detect)

    def test_crisis_regime_vetoes(self, sample_bull_bar):
        """Test crisis regime properly vetoes signals (unless extreme)."""
        archetypes = [
            SpringUTADArchetype(),
            OrderBlockRetestArchetype(),
            BOSCHOCHReversalArchetype(),
            TrapWithinTrendArchetype()
        ]

        # Normal capitulation depth should veto in crisis
        sample_bull_bar['capitulation_depth'] = -0.08

        for archetype in archetypes:
            name, _, metadata = archetype.detect(sample_bull_bar, 'crisis')
            # Most should veto (except LiquiditySweep which allows some crisis trades)
            if archetype.__class__.__name__ != 'LiquiditySweepArchetype':
                assert name is None, f"{archetype.__class__.__name__} should veto crisis"


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == '__main__':
    """
    Run tests with pytest.

    Usage:
        python3 tests/archetypes/test_bull_archetypes_mvp.py

    Or with pytest:
        pytest tests/archetypes/test_bull_archetypes_mvp.py -v
    """
    pytest.main([__file__, '-v', '--tb=short'])
