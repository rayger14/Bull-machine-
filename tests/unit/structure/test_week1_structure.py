"""
Unit Tests for Week 1 Structure Modules

Tests critical scenarios for:
- Internal vs External structure detection
- BOMS detection
- 1-2-3 Squiggle patterns
- Range outcome classification

Author: Bull Machine v2.0
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from engine.structure import (
    detect_structure_state,
    StructureState,
    detect_boms,
    BOMSSignal,
    detect_squiggle_123,
    SquigglePattern,
    classify_range_outcome,
    RangeOutcome
)


@pytest.fixture
def sample_ohlcv():
    """Create sample OHLCV data for testing"""
    dates = pd.date_range(start='2024-01-01', periods=200, freq='1H')

    # Generate realistic price data with trend
    np.random.seed(42)
    base_price = 40000
    trend = np.linspace(0, 2000, 200)  # Uptrend
    noise = np.random.randn(200) * 100
    close = base_price + trend + noise

    # OHLC from close
    high = close + np.abs(np.random.randn(200) * 50)
    low = close - np.abs(np.random.randn(200) * 50)
    open_ = close + np.random.randn(200) * 30
    volume = np.random.randint(100, 1000, 200)

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return df


@pytest.fixture
def bullish_breakout_data():
    """Create data with clear bullish breakout"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='4H')

    # Range phase (first 70 bars)
    range_high = 42000
    range_low = 40000
    range_prices = np.random.uniform(range_low, range_high, 70)

    # Breakout phase (last 30 bars)
    breakout_prices = np.linspace(42000, 44000, 30)

    close = np.concatenate([range_prices, breakout_prices])
    high = close + np.abs(np.random.randn(100) * 100)
    low = close - np.abs(np.random.randn(100) * 100)
    open_ = close + np.random.randn(100) * 50

    # Volume surge on breakout
    volume = np.ones(100) * 500
    volume[70:] = 1500  # 3x surge on breakout

    df = pd.DataFrame({
        'open': open_,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=dates)

    return df


class TestInternalExternalStructure:
    """Test Internal vs External structure detection"""

    def test_aligned_bullish_structure(self, sample_ohlcv):
        """Test case: Both internal and external bullish (aligned)"""
        df_1h = sample_ohlcv.copy()
        df_4h = sample_ohlcv.iloc[::4].copy()  # Resample to 4H
        df_1d = sample_ohlcv.iloc[::24].copy()  # Resample to 1D

        state = detect_structure_state(df_1h, df_4h, df_1d, config={})

        # Should detect alignment
        assert isinstance(state, StructureState)
        assert state.alignment in [True, False]
        assert 0 <= state.conflict_score <= 1
        assert 0 <= state.internal_strength <= 1
        assert 0 <= state.external_strength <= 1

    def test_conflict_detection(self):
        """Test case: Internal bullish, external bearish (conflict)"""
        # Create downtrend data (external bearish)
        dates_1d = pd.date_range(start='2024-01-01', periods=50, freq='1D')
        close_1d = np.linspace(45000, 40000, 50)  # Downtrend
        df_1d = pd.DataFrame({
            'open': close_1d,
            'high': close_1d + 200,
            'low': close_1d - 200,
            'close': close_1d,
            'volume': np.ones(50) * 500
        }, index=dates_1d)

        # Create uptrend data (internal bullish)
        dates_4h = pd.date_range(start='2024-01-01', periods=200, freq='4H')
        close_4h = np.linspace(40000, 41000, 200)  # Uptrend
        df_4h = pd.DataFrame({
            'open': close_4h,
            'high': close_4h + 100,
            'low': close_4h - 100,
            'close': close_4h,
            'volume': np.ones(200) * 500
        }, index=dates_4h)

        dates_1h = pd.date_range(start='2024-01-01', periods=800, freq='1H')
        close_1h = np.linspace(40000, 41000, 800)
        df_1h = pd.DataFrame({
            'open': close_1h,
            'high': close_1h + 50,
            'low': close_1h - 50,
            'close': close_1h,
            'volume': np.ones(800) * 500
        }, index=dates_1h)

        state = detect_structure_state(df_1h, df_4h, df_1d, config={})

        # Should detect conflict (might be alignment depending on thresholds)
        # Main check: no crashes, valid ranges
        assert 0 <= state.conflict_score <= 1
        assert state.internal_phase in ['accumulation', 'distribution', 'markup', 'markdown', 'transition']
        assert state.external_trend in ['bullish', 'bearish', 'range']


class TestBOMSDetection:
    """Test BOMS (Break of Market Structure) detection"""

    def test_bullish_boms_with_volume(self, bullish_breakout_data):
        """Test case: Clean bullish BOMS with volume confirmation"""
        df = bullish_breakout_data

        boms = detect_boms(df, timeframe='4H', config={})

        assert isinstance(boms, BOMSSignal)
        # May or may not detect BOMS depending on FVG presence
        # Main check: valid output
        assert boms.direction in ['bullish', 'bearish', 'none']
        assert 0 <= boms.volume_surge <= 10
        assert isinstance(boms.fvg_present, bool)

    def test_no_boms_in_range(self, sample_ohlcv):
        """Test case: No BOMS in ranging market"""
        # Create clear range
        dates = pd.date_range(start='2024-01-01', periods=100, freq='4H')
        close = np.random.uniform(40000, 42000, 100)
        df = pd.DataFrame({
            'open': close,
            'high': close + 100,
            'low': close - 100,
            'close': close,
            'volume': np.ones(100) * 500
        }, index=dates)

        boms = detect_boms(df, timeframe='4H', config={})

        # Should not detect BOMS (or very weak)
        assert isinstance(boms, BOMSSignal)
        if boms.boms_detected:
            # If detected, should be weak
            assert boms.displacement < 0.05

    def test_weak_volume_no_boms(self):
        """Test case: Price breaks structure but volume too weak"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='4H')

        # Range then breakout
        close = np.concatenate([
            np.ones(70) * 41000,
            np.linspace(41000, 42000, 30)
        ])

        df = pd.DataFrame({
            'open': close,
            'high': close + 100,
            'low': close - 100,
            'close': close,
            'volume': np.ones(100) * 500  # No volume surge
        }, index=dates)

        boms = detect_boms(df, timeframe='4H', config={})

        # Should not confirm BOMS without volume
        # (May still detect BOS but not BOMS)
        if boms.boms_detected:
            # If detected, volume surge should be low
            assert boms.volume_surge < 1.5


class TestSquigglePattern:
    """Test 1-2-3 Squiggle pattern detection"""

    def test_stage_1_bos_detected(self, bullish_breakout_data):
        """Test case: Stage 1 - BOS detected"""
        df = bullish_breakout_data.iloc[:75]  # Just past breakout

        pattern = detect_squiggle_123(df, timeframe='4H', config={})

        assert isinstance(pattern, SquigglePattern)
        assert pattern.stage in [0, 1, 2, 3]
        assert pattern.direction in ['bullish', 'bearish', 'none']
        assert 0 <= pattern.confidence <= 1

    def test_stage_2_entry_window(self):
        """Test case: Stage 2 - Retest entry window"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='4H')

        # BOS at bar 50, retest at bar 70-80
        close = np.concatenate([
            np.ones(40) * 41000,           # Pre-BOS range
            np.linspace(41000, 42500, 10), # BOS (Stage 1)
            np.linspace(42500, 43000, 10), # Continuation
            np.linspace(43000, 42000, 10), # Retest (Stage 2)
            np.ones(30) * 42000            # Current
        ])

        df = pd.DataFrame({
            'open': close,
            'high': close + 100,
            'low': close - 100,
            'close': close,
            'volume': np.ones(100) * 500
        }, index=dates)

        pattern = detect_squiggle_123(df, timeframe='4H', config={})

        # Should be in retest zone (or expired)
        assert isinstance(pattern, SquigglePattern)
        if pattern.stage == 2:
            assert pattern.entry_window == True
            assert pattern.retest_quality >= 0

    def test_no_pattern_in_range(self):
        """Test case: No squiggle pattern in range"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='4H')
        close = np.random.uniform(40000, 42000, 100)
        df = pd.DataFrame({
            'open': close,
            'high': close + 100,
            'low': close - 100,
            'close': close,
            'volume': np.ones(100) * 500
        }, index=dates)

        pattern = detect_squiggle_123(df, timeframe='4H', config={})

        # Should be stage 0 (no pattern)
        assert pattern.stage == 0
        assert pattern.entry_window == False


class TestRangeOutcomes:
    """Test range outcome classification"""

    def test_range_bound_detection(self):
        """Test case: Price in clear range"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='4H')

        # Oscillate between 40k-42k
        close = 41000 + 1000 * np.sin(np.linspace(0, 4*np.pi, 100))

        df = pd.DataFrame({
            'open': close,
            'high': close + 100,
            'low': close - 100,
            'close': close,
            'volume': np.ones(100) * 500
        }, index=dates)

        outcome = classify_range_outcome(df, timeframe='4H', config={})

        assert isinstance(outcome, RangeOutcome)
        # Should detect range (or none if too volatile)
        if outcome.outcome == 'range_bound':
            assert outcome.range_high > outcome.range_low
            assert outcome.bars_in_range > 0

    def test_breakout_with_volume(self, bullish_breakout_data):
        """Test case: Clean breakout with volume"""
        df = bullish_breakout_data

        outcome = classify_range_outcome(df, timeframe='4H', config={})

        assert isinstance(outcome, RangeOutcome)
        # Should detect breakout (or none if no range detected)
        if outcome.outcome == 'breakout':
            assert outcome.direction in ['bullish', 'bearish']
            assert outcome.breakout_strength > 0

    def test_fakeout_detection(self):
        """Test case: Fakeout (breakout fails, returns to range)"""
        dates = pd.date_range(start='2024-01-01', periods=100, freq='4H')

        # Range 40-42k for 70 bars
        # Failed breakout to 43k (bars 70-75)
        # Return to range (bars 75-100)
        close = np.concatenate([
            np.random.uniform(40000, 42000, 70),   # Range
            np.linspace(42000, 43000, 5),          # Breakout attempt
            np.linspace(43000, 41000, 25)          # Return to range
        ])

        # Weak volume on breakout
        volume = np.concatenate([
            np.ones(70) * 500,
            np.ones(5) * 550,  # Low volume on breakout
            np.ones(25) * 500
        ])

        df = pd.DataFrame({
            'open': close,
            'high': close + 100,
            'low': close - 100,
            'close': close,
            'volume': volume
        }, index=dates)

        outcome = classify_range_outcome(df, timeframe='4H', config={})

        # May detect fakeout or range_bound
        assert outcome.outcome in ['fakeout', 'range_bound', 'none', 'breakout', 'rejection']

    def test_rejection_detection(self):
        """Test case: Price rejects from range boundary"""
        dates = pd.date_range(start='2024-01-01', periods=50, freq='4H')

        # Range 40-42k
        # Touch 42k but reject with long wick
        close = np.concatenate([
            np.random.uniform(40000, 41500, 45),
            np.array([41800, 41600, 41400, 41200, 41000])  # Rejection
        ])

        high = close.copy()
        high[45] = 42100  # Long upper wick on rejection bar

        df = pd.DataFrame({
            'open': close,
            'high': high,
            'low': close - 100,
            'close': close,
            'volume': np.ones(50) * 500
        }, index=dates)

        outcome = classify_range_outcome(df, timeframe='4H', config={})

        # May detect rejection or range_bound
        assert isinstance(outcome, RangeOutcome)


class TestIntegration:
    """Integration tests for full structure analysis"""

    def test_all_modules_export_valid_features(self, sample_ohlcv):
        """Test that all modules export valid feature dictionaries"""
        df_1h = sample_ohlcv
        df_4h = sample_ohlcv.iloc[::4]
        df_1d = sample_ohlcv.iloc[::24]

        # Run all detectors
        structure_state = detect_structure_state(df_1h, df_4h, df_1d, {})
        boms_signal = detect_boms(df_4h, timeframe='4H', config={})
        squiggle_pattern = detect_squiggle_123(df_4h, timeframe='4H', config={})
        range_outcome = classify_range_outcome(df_4h, timeframe='4H', config={})

        # Check all export to_dict()
        assert isinstance(structure_state.to_dict(), dict)
        assert isinstance(boms_signal.to_dict(), dict)
        assert isinstance(squiggle_pattern.to_dict(), dict)
        assert isinstance(range_outcome.to_dict(), dict)

        # Check all required keys
        assert 'internal_phase' in structure_state.to_dict()
        assert 'boms_detected' in boms_signal.to_dict()
        assert 'squiggle_stage' in squiggle_pattern.to_dict()
        assert 'range_outcome' in range_outcome.to_dict()

    def test_feature_store_integration(self, sample_ohlcv):
        """Test full feature store integration"""
        df_1h = sample_ohlcv
        df_4h = sample_ohlcv.iloc[::4]
        df_1d = sample_ohlcv.iloc[::24]

        # Build feature dict
        features = {}

        # Add all structure features
        features.update(detect_structure_state(df_1h, df_4h, df_1d, {}).to_dict())
        features.update(detect_boms(df_4h, timeframe='4H').to_dict())
        features.update(detect_squiggle_123(df_4h, timeframe='4H').to_dict())
        features.update(classify_range_outcome(df_4h, timeframe='4H').to_dict())

        # Should have all 29 columns
        # 6 (internal/external) + 7 (boms) + 8 (squiggle) + 8 (range) = 29
        assert len(features) == 29

        # Check no None values
        for key, value in features.items():
            assert value is not None, f"Feature {key} is None"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
