#!/usr/bin/env python3
"""
Unit Tests for Temporal Fusion Engine

Tests the 4-component temporal confluence system.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from dataclasses import dataclass

from engine.temporal.temporal_fusion import (
    TemporalFusionEngine,
    compute_bars_since_wyckoff_events
)


@dataclass
class MockContext:
    """Mock RuntimeContext for testing."""
    row: pd.Series


@pytest.fixture
def config():
    """Default temporal fusion config."""
    return {
        'enabled': True,
        'temporal_weights': {
            'fib_time': 0.40,
            'gann_cycles': 0.30,
            'volatility_cycles': 0.20,
            'emotional_cycles': 0.10
        },
        'temporal_adjustment_range': [0.85, 1.15],
        'fib_levels': [13, 21, 34, 55, 89, 144],
        'gann_vibrations': [3, 7, 9, 12, 21, 36, 45, 72, 90, 144],
        'fib_tolerance_bars': 3,
        'gann_tolerance_bars': 2,
        'vol_compression_threshold': 0.75,
        'vol_expansion_threshold': 1.25,
        'emotional_rsi_thresholds': {
            'extreme_fear': 25,
            'hope_lower': 35,
            'hope_upper': 45,
            'greed': 65,
            'extreme_greed': 75
        }
    }


@pytest.fixture
def engine(config):
    """Temporal fusion engine instance."""
    return TemporalFusionEngine(config)


class TestFibonacciTimeScoring:
    """Test Fibonacci time cluster scoring."""

    def test_no_recent_events(self, engine):
        """Test scoring when no recent Wyckoff events."""
        row = pd.Series({
            'bars_since_sc': 999,
            'bars_since_ar': 999,
            'bars_since_st': 999,
            'bars_since_sos_long': 999,
            'bars_since_sos_short': 999,
        })
        context = MockContext(row=row)

        score = engine._compute_fib_cluster_score(context)
        assert score == 0.20  # Base score for no events

    def test_single_fib_hit(self, engine):
        """Test scoring with single Fibonacci hit."""
        row = pd.Series({
            'bars_since_sc': 89,  # 89 is a Fib level
            'bars_since_ar': 999,
            'bars_since_st': 999,
            'bars_since_sos_long': 999,
            'bars_since_sos_short': 999,
        })
        context = MockContext(row=row)

        score = engine._compute_fib_cluster_score(context)
        assert score == 0.40  # 1 hit

    def test_multiple_fib_hits(self, engine):
        """Test scoring with multiple Fibonacci hits (cluster)."""
        row = pd.Series({
            'bars_since_sc': 89,   # Fib hit
            'bars_since_ar': 55,   # Fib hit
            'bars_since_st': 34,   # Fib hit
            'bars_since_sos_long': 999,
            'bars_since_sos_short': 999,
        })
        context = MockContext(row=row)

        score = engine._compute_fib_cluster_score(context)
        assert score == 0.80  # 3+ hits = strong cluster

    def test_fib_tolerance(self, engine):
        """Test tolerance window for Fibonacci hits."""
        # 91 is within ±3 of 89 (Fib level)
        row = pd.Series({
            'bars_since_sc': 91,
            'bars_since_ar': 999,
            'bars_since_st': 999,
            'bars_since_sos_long': 999,
            'bars_since_sos_short': 999,
        })
        context = MockContext(row=row)

        score = engine._compute_fib_cluster_score(context)
        assert score == 0.40  # Should count as hit within tolerance


class TestGannCycleScoring:
    """Test Gann cycle vibration scoring."""

    def test_no_reference_point(self, engine):
        """Test when no SC/AR reference available."""
        row = pd.Series({
            'bars_since_sc': 999,
            'bars_since_ar': 999,
        })
        context = MockContext(row=row)

        score = engine._compute_gann_cycle_score(context)
        assert score == 0.20  # Base score

    def test_major_vibration_hit(self, engine):
        """Test major vibration (90, 144) detection."""
        row = pd.Series({
            'bars_since_sc': 144,  # Major Gann vibration
            'bars_since_ar': 999,
        })
        context = MockContext(row=row)

        score = engine._compute_gann_cycle_score(context)
        assert score == 0.90  # Major vibration score

    def test_strong_vibration_hit(self, engine):
        """Test strong vibration (45, 72) detection."""
        row = pd.Series({
            'bars_since_sc': 45,  # Strong Gann vibration
            'bars_since_ar': 999,
        })
        context = MockContext(row=row)

        score = engine._compute_gann_cycle_score(context)
        assert score == 0.75  # Strong vibration score

    def test_medium_vibration_hit(self, engine):
        """Test medium vibration (21, 36) detection."""
        row = pd.Series({
            'bars_since_sc': 21,  # Medium Gann vibration
            'bars_since_ar': 999,
        })
        context = MockContext(row=row)

        score = engine._compute_gann_cycle_score(context)
        assert score == 0.60  # Medium vibration score


class TestVolatilityCycleScoring:
    """Test volatility cycle scoring."""

    def test_compression_phase(self, engine):
        """Test high score during compression."""
        row = pd.Series({
            'atr': 100,
            'atr_ma_20': 150,  # ATR is 0.67 × MA (< 0.75 threshold)
        })
        context = MockContext(row=row)

        score = engine._compute_volatility_cycle_score(context)
        assert score == 0.90  # Compression = coiled spring

    def test_expansion_phase(self, engine):
        """Test low score during expansion climax."""
        row = pd.Series({
            'atr': 200,
            'atr_ma_20': 150,  # ATR is 1.33 × MA (> 1.25 threshold)
        })
        context = MockContext(row=row)

        score = engine._compute_volatility_cycle_score(context)
        assert score == 0.10  # Expansion = avoid

    def test_normal_volatility(self, engine):
        """Test neutral score during normal volatility."""
        row = pd.Series({
            'atr': 100,
            'atr_ma_20': 100,  # ATR = MA (normal)
        })
        context = MockContext(row=row)

        score = engine._compute_volatility_cycle_score(context)
        assert score == 0.50  # Neutral


class TestEmotionalCycleScoring:
    """Test emotional cycle scoring."""

    def test_extreme_fear(self, engine):
        """Test capitulation (extreme fear) detection."""
        row = pd.Series({
            'rsi': 20,      # Below 25 threshold
            'funding': -0.03,  # Negative funding
        })
        context = MockContext(row=row)

        score = engine._compute_emotional_cycle_score(context)
        assert score == 0.95  # Extreme fear = best entry

    def test_extreme_greed(self, engine):
        """Test euphoria (extreme greed) detection."""
        row = pd.Series({
            'rsi': 80,      # Above 75 threshold
            'funding': 0.04,  # High positive funding
        })
        context = MockContext(row=row)

        score = engine._compute_emotional_cycle_score(context)
        assert score == 0.05  # Extreme greed = avoid

    def test_hope_zone(self, engine):
        """Test hope/relief zone detection."""
        row = pd.Series({
            'rsi': 40,  # Between 35-45
            'funding': 0.01,
        })
        context = MockContext(row=row)

        score = engine._compute_emotional_cycle_score(context)
        assert score == 0.70  # Hope zone = good entry

    def test_neutral_zone(self, engine):
        """Test neutral emotional state."""
        row = pd.Series({
            'rsi': 50,  # Neutral
            'funding': 0.0,
        })
        context = MockContext(row=row)

        score = engine._compute_emotional_cycle_score(context)
        assert score == 0.50  # Neutral


class TestTemporalConfluence:
    """Test overall temporal confluence calculation."""

    def test_high_confluence(self, engine):
        """Test high confluence scenario (all components aligned)."""
        row = pd.Series({
            # Fib cluster (3 hits → 0.80)
            'bars_since_sc': 89,
            'bars_since_ar': 55,
            'bars_since_st': 34,
            'bars_since_sos_long': 999,
            'bars_since_sos_short': 999,
            # Gann (major vibration → 0.90)
            # (uses bars_since_sc = 89, close to 90)
            # Vol (compression → 0.90)
            'atr': 100,
            'atr_ma_20': 150,
            # Emotional (extreme fear → 0.95)
            'rsi': 20,
            'funding': -0.03,
        })
        context = MockContext(row=row)

        confluence = engine.compute_temporal_confluence(context)
        assert confluence > 0.70  # High confluence expected

    def test_low_confluence(self, engine):
        """Test low confluence scenario (misaligned)."""
        row = pd.Series({
            # No fib cluster → 0.20
            'bars_since_sc': 999,
            'bars_since_ar': 999,
            'bars_since_st': 999,
            'bars_since_sos_long': 999,
            'bars_since_sos_short': 999,
            # No Gann hit → 0.20
            # Vol expansion → 0.10
            'atr': 200,
            'atr_ma_20': 150,
            # Emotional greed → 0.05
            'rsi': 80,
            'funding': 0.04,
        })
        context = MockContext(row=row)

        confluence = engine.compute_temporal_confluence(context)
        assert confluence < 0.30  # Low confluence expected


class TestFusionAdjustment:
    """Test fusion weight adjustment logic."""

    def test_high_confluence_boost(self, engine):
        """Test +15% boost for high confluence."""
        base_fusion = 0.65
        confluence = 0.85  # High

        adjusted = engine.adjust_fusion_weight(base_fusion, confluence)
        expected = 0.65 * 1.15
        assert abs(adjusted - expected) < 0.001

    def test_moderate_confluence_boost(self, engine):
        """Test +10% boost for moderate confluence."""
        base_fusion = 0.65
        confluence = 0.70  # Moderate

        adjusted = engine.adjust_fusion_weight(base_fusion, confluence)
        expected = 0.65 * 1.10
        assert abs(adjusted - expected) < 0.001

    def test_low_confluence_penalty(self, engine):
        """Test -15% penalty for low confluence."""
        base_fusion = 0.65
        confluence = 0.15  # Very low

        adjusted = engine.adjust_fusion_weight(base_fusion, confluence)
        expected = 0.65 * 0.85
        assert abs(adjusted - expected) < 0.001

    def test_neutral_confluence_no_adjustment(self, engine):
        """Test no adjustment in neutral zone."""
        base_fusion = 0.65
        confluence = 0.50  # Neutral

        adjusted = engine.adjust_fusion_weight(base_fusion, confluence)
        assert adjusted == base_fusion  # No change


class TestBarsComputationSince:
    """Test bars_since_* feature computation."""

    def test_compute_bars_since(self):
        """Test computation of bars_since_* from event flags."""
        # Create test dataframe
        df = pd.DataFrame({
            'wyckoff_sc': [False, False, True, False, False, False],
            'wyckoff_ar': [False, False, False, False, True, False],
        })

        # Compute bars_since features
        df = compute_bars_since_wyckoff_events(df)

        # Check bars_since_sc
        assert df['bars_since_sc'].iloc[2] == 0  # Event at idx 2
        assert df['bars_since_sc'].iloc[3] == 1  # 1 bar since
        assert df['bars_since_sc'].iloc[4] == 2  # 2 bars since
        assert df['bars_since_sc'].iloc[5] == 3  # 3 bars since

        # Check bars_since_ar
        assert df['bars_since_sos_long'].iloc[4] == 0  # Event at idx 4
        assert df['bars_since_sos_long'].iloc[5] == 1  # 1 bar since


class TestEngineDisabled:
    """Test engine behavior when disabled."""

    def test_disabled_returns_neutral(self, config):
        """Test that disabled engine returns neutral confluence."""
        config['enabled'] = False
        engine = TemporalFusionEngine(config)

        row = pd.Series({
            'bars_since_sc': 89,
            'rsi': 20,
            'atr': 100,
            'atr_ma_20': 150,
            'funding': -0.03,
        })
        context = MockContext(row=row)

        confluence = engine.compute_temporal_confluence(context)
        assert confluence == 0.5  # Neutral when disabled

    def test_disabled_no_adjustment(self, config):
        """Test that disabled engine doesn't adjust fusion."""
        config['enabled'] = False
        engine = TemporalFusionEngine(config)

        base_fusion = 0.65
        confluence = 0.85  # Would normally boost

        adjusted = engine.adjust_fusion_weight(base_fusion, confluence)
        assert adjusted == base_fusion  # No adjustment when disabled


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
