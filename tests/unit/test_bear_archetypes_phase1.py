#!/usr/bin/env python3
"""
Unit tests for Phase 1 Bear Archetypes (S2 + S5)

Tests the approved bear patterns:
- S2: Failed Rally Rejection
- S5: Long Squeeze Cascade (with corrected funding logic)
"""

import pytest
import pandas as pd
import numpy as np
from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.runtime.context import RuntimeContext


@pytest.fixture
def archetype_config():
    """Minimal config for bear archetypes."""
    return {
        'use_archetypes': True,
        'enable_S2': True,
        'enable_S5': True,
        'thresholds': {
            'failed_rally': {
                'fusion_threshold': 0.36,
                'wick_ratio_min': 2.0,
            },
            'long_squeeze': {
                'fusion_threshold': 0.35,
                'funding_z_min': 1.2,
                'rsi_min': 70,
                'liquidity_max': 0.25,
            }
        }
    }


@pytest.fixture
def logic(archetype_config):
    """Create ArchetypeLogic instance."""
    return ArchetypeLogic(archetype_config)


def make_context(row_data, regime='neutral'):
    """Helper to create RuntimeContext from dict."""
    row = pd.Series(row_data)
    row.name = pd.Timestamp('2024-01-01')

    return RuntimeContext(
        ts=row.name,
        row=row,
        regime_probs={regime: 1.0},
        regime_label=regime,
        adapted_params={},
        thresholds={
            'failed_rally': {
                'fusion_threshold': 0.36,
                'wick_ratio_min': 2.0,
            },
            'long_squeeze': {
                'fusion_threshold': 0.35,
                'funding_z_min': 1.2,
                'rsi_min': 70,
                'liquidity_max': 0.25,
            }
        }
    )


class TestS2FailedRally:
    """Test S2: Failed Rally Rejection archetype."""

    def test_perfect_failed_rally_signal(self, logic):
        """Test perfect failed rally conditions."""
        row_data = {
            'tf1h_ob_high': 100.0,
            'close': 99.5,
            'high': 101.0,
            'low': 98.0,
            'open': 99.0,
            'rsi_14': 72.0,
            'volume_zscore': 0.2,
            'tf4h_external_trend': -1,
        }
        context = make_context(row_data)

        matched, score, meta = logic._check_S2(context)

        assert matched, f"Should match perfect failed rally signal. Meta: {meta}"
        assert score >= 0.36, f"Score {score} should exceed threshold 0.36"
        assert 'components' in meta
        assert meta['components']['ob_retest'] == 1.0
        assert meta['wick_ratio'] >= 2.0

    def test_no_ob_retest(self, logic):
        """Test rejection when order block not retested."""
        row_data = {
            'tf1h_ob_high': 100.0,
            'close': 95.0,  # Too far below OB
            'high': 96.0,
            'low': 94.0,
            'open': 95.5,
            'rsi_14': 72.0,
            'volume_zscore': 0.2,
            'tf4h_external_trend': -1,
        }
        context = make_context(row_data)

        matched, score, meta = logic._check_S2(context)

        assert not matched
        assert meta['reason'] == 'no_ob_retest'

    def test_weak_rejection_wick(self, logic):
        """Test rejection when wick ratio too small."""
        row_data = {
            'tf1h_ob_high': 100.0,
            'close': 99.5,
            'high': 99.8,  # Small wick
            'low': 98.0,
            'open': 99.0,
            'rsi_14': 72.0,
            'volume_zscore': 0.2,
            'tf4h_external_trend': -1,
        }
        context = make_context(row_data)

        matched, score, meta = logic._check_S2(context)

        assert not matched
        assert meta['reason'] == 'weak_rejection'
        assert meta['wick_ratio'] < 2.0

    def test_missing_ob_high(self, logic):
        """Test graceful handling when ob_high missing."""
        row_data = {
            'tf1h_ob_high': None,  # Missing
            'close': 99.5,
            'high': 101.0,
            'low': 98.0,
            'open': 99.0,
            'rsi_14': 72.0,
            'volume_zscore': 0.2,
            'tf4h_external_trend': -1,
        }
        context = make_context(row_data)

        matched, score, meta = logic._check_S2(context)

        assert not matched
        assert meta['reason'] == 'no_ob_retest'


class TestS5LongSqueeze:
    """Test S5: Long Squeeze Cascade archetype."""

    def test_perfect_long_squeeze_signal(self, logic):
        """Test perfect long squeeze conditions."""
        row_data = {
            'funding_Z': 1.8,  # High positive funding (longs overcrowded)
            'oi_change_24h': 0.12,  # 12% OI increase
            'rsi_14': 75.0,
            'liquidity_score': 0.18,  # Low liquidity
        }
        context = make_context(row_data)

        matched, score, meta = logic._check_S5(context)

        assert matched, f"Should match perfect long squeeze signal. Meta: {meta}"
        assert score >= 0.35, f"Score {score} should exceed threshold 0.35"
        assert 'mechanism' in meta
        assert meta['mechanism'] == 'longs_overcrowded_cascade_risk'
        assert meta['funding_z'] > 1.2

    def test_funding_not_extreme(self, logic):
        """Test rejection when funding not extreme enough."""
        row_data = {
            'funding_Z': 0.5,  # Not extreme
            'oi_change_24h': 0.12,
            'rsi_14': 75.0,
            'liquidity_score': 0.18,
        }
        context = make_context(row_data)

        matched, score, meta = logic._check_S5(context)

        assert not matched
        assert meta['reason'] == 'funding_not_extreme'
        assert meta['funding_z'] < 1.2

    def test_rsi_not_overbought(self, logic):
        """Test rejection when RSI not overbought."""
        row_data = {
            'funding_Z': 1.8,
            'oi_change_24h': 0.12,
            'rsi_14': 55.0,  # Not overbought
            'liquidity_score': 0.18,
        }
        context = make_context(row_data)

        matched, score, meta = logic._check_S5(context)

        assert not matched
        assert meta['reason'] == 'rsi_not_overbought'
        assert meta['rsi'] < 70.0

    def test_funding_logic_corrected(self, logic):
        """
        CRITICAL TEST: Verify funding logic is corrected.

        Positive funding = longs pay shorts = LONG SQUEEZE DOWN (bearish)
        Negative funding = shorts pay longs = SHORT SQUEEZE UP (bullish)
        """
        # This should trigger (positive funding = bearish signal)
        row_data_bearish = {
            'funding_Z': 2.0,  # HIGH POSITIVE = longs overcrowded
            'oi_change_24h': 0.10,
            'rsi_14': 75.0,
            'liquidity_score': 0.20,
        }
        context_bearish = make_context(row_data_bearish)
        matched_bearish, _, meta_bearish = logic._check_S5(context_bearish)

        assert matched_bearish, "Positive funding should trigger LONG squeeze (bearish)"

        # This should NOT trigger (negative funding = bullish condition)
        row_data_bullish = {
            'funding_Z': -2.0,  # HIGH NEGATIVE = shorts overcrowded (wrong for bear pattern)
            'oi_change_24h': 0.10,
            'rsi_14': 75.0,
            'liquidity_score': 0.20,
        }
        context_bullish = make_context(row_data_bullish)
        matched_bullish, _, meta_bullish = logic._check_S5(context_bullish)

        assert not matched_bullish, "Negative funding should NOT trigger (that's SHORT squeeze UP)"
        assert meta_bullish['reason'] == 'funding_not_extreme'


class TestIntegration:
    """Integration tests for bear archetypes."""

    def test_both_patterns_enabled(self, logic):
        """Test that both S2 and S5 are enabled."""
        assert logic.enabled['S2'] == True
        assert logic.enabled['S5'] == True

    def test_rejected_patterns_disabled(self, logic):
        """Test that rejected patterns (S6, S7) are disabled."""
        assert logic.enabled['S6'] == False
        assert logic.enabled['S7'] == False

    def test_archetype_map_names(self, logic):
        """Test that archetype names are correctly mapped."""
        # Check via detect_all to see archetype_map
        from engine import feature_flags as features
        old_flag = features.EVALUATE_ALL_ARCHETYPES
        features.EVALUATE_ALL_ARCHETYPES = True

        try:
            # Trigger S2
            row_data_s2 = {
                'tf1h_ob_high': 100.0,
                'close': 99.5,
                'high': 101.0,
                'low': 98.0,
                'open': 99.0,
                'rsi_14': 72.0,
                'volume_zscore': 0.2,
                'tf4h_external_trend': -1,
                'liquidity_score': 0.35,
            }
            context_s2 = make_context(row_data_s2)
            archetype, score, liq = logic.detect(context_s2)

            assert archetype == 'failed_rally', f"Expected 'failed_rally', got '{archetype}'"

            # Trigger S5
            row_data_s5 = {
                'funding_Z': 1.8,
                'oi_change_24h': 0.12,
                'rsi_14': 75.0,
                'liquidity_score': 0.35,
            }
            context_s5 = make_context(row_data_s5)
            archetype, score, liq = logic.detect(context_s5)

            assert archetype == 'long_squeeze', f"Expected 'long_squeeze', got '{archetype}'"

        finally:
            features.EVALUATE_ALL_ARCHETYPES = old_flag


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
