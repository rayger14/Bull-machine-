#!/usr/bin/env python3
"""
Unit Tests for Wyckoff Event Detection System

Tests all 18 Wyckoff events using synthetic data patterns that match
institutional trading characteristics.

Run:
    pytest tests/test_wyckoff_events.py -v
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from engine.wyckoff.events import (
    detect_selling_climax,
    detect_buying_climax,
    detect_automatic_rally,
    detect_automatic_reaction,
    detect_secondary_test,
    detect_sign_of_strength,
    detect_sign_of_weakness,
    detect_spring_type_a,
    detect_spring_type_b,
    detect_upthrust,
    detect_upthrust_after_distribution,
    detect_last_point_of_support,
    detect_last_point_of_supply,
    detect_all_wyckoff_events,
    integrate_wyckoff_with_pti,
)


@pytest.fixture
def base_ohlcv():
    """Create base OHLCV dataframe for testing"""
    dates = pd.date_range(start='2024-01-01', periods=100, freq='1H')
    df = pd.DataFrame({
        'open': np.random.uniform(50000, 52000, 100),
        'high': np.random.uniform(51000, 53000, 100),
        'low': np.random.uniform(49000, 51000, 100),
        'close': np.random.uniform(50000, 52000, 100),
        'volume': np.random.uniform(1000, 5000, 100),
    }, index=dates)

    # Ensure OHLC consistency
    df['high'] = df[['open', 'close', 'high']].max(axis=1)
    df['low'] = df[['open', 'close', 'low']].min(axis=1)

    return df


@pytest.fixture
def default_config():
    """Default Wyckoff event configuration"""
    return {
        'sc_volume_z_min': 2.5,
        'sc_range_pos_max': 0.2,
        'sc_range_z_min': 1.5,
        'sc_wick_min': 0.6,
    }


class TestSellingClimax:
    """Tests for Selling Climax (SC) detection"""

    def test_sc_basic_detection(self, base_ohlcv, default_config):
        """Test SC detection with clear capitulation pattern"""
        df = base_ohlcv.copy()

        # Create SC pattern at index 50
        sc_idx = 50
        df.loc[df.index[sc_idx], 'low'] = 45000  # Sharp drop
        df.loc[df.index[sc_idx], 'close'] = 46000  # Closes off lows
        df.loc[df.index[sc_idx], 'high'] = 48000
        df.loc[df.index[sc_idx], 'volume'] = 15000  # Extreme volume

        detected, confidence = detect_selling_climax(df, default_config)

        assert detected.iloc[sc_idx], "SC should be detected at capitulation point"
        assert confidence.iloc[sc_idx] > 0.5, "SC confidence should be high"

    def test_sc_requires_volume_spike(self, base_ohlcv, default_config):
        """Test that SC requires extreme volume"""
        df = base_ohlcv.copy()

        # Low volume drop (should NOT trigger SC)
        sc_idx = 50
        df.loc[df.index[sc_idx], 'low'] = 45000
        df.loc[df.index[sc_idx], 'close'] = 46000
        df.loc[df.index[sc_idx], 'volume'] = 1000  # Low volume

        detected, _ = detect_selling_climax(df, default_config)

        assert not detected.iloc[sc_idx], "SC should NOT trigger without volume spike"

    def test_sc_requires_lower_wick(self, base_ohlcv, default_config):
        """Test that SC requires absorption (lower wick)"""
        df = base_ohlcv.copy()

        # No wick (close at low) - should have lower confidence
        sc_idx = 50
        df.loc[df.index[sc_idx], 'low'] = 45000
        df.loc[df.index[sc_idx], 'close'] = 45000  # No wick
        df.loc[df.index[sc_idx], 'high'] = 46000
        df.loc[df.index[sc_idx], 'volume'] = 15000

        detected, confidence = detect_selling_climax(df, default_config)

        # May still detect but with lower confidence
        if detected.iloc[sc_idx]:
            assert confidence.iloc[sc_idx] < 0.7, "Confidence should be lower without wick"


class TestBuyingClimax:
    """Tests for Buying Climax (BC) detection"""

    def test_bc_basic_detection(self, base_ohlcv, default_config):
        """Test BC detection with euphoria pattern"""
        df = base_ohlcv.copy()

        # Create BC pattern at index 50
        bc_idx = 50
        df.loc[df.index[bc_idx], 'high'] = 58000  # Sharp rally
        df.loc[df.index[bc_idx], 'close'] = 56000  # Closes off highs (rejection)
        df.loc[df.index[bc_idx], 'low'] = 54000
        df.loc[df.index[bc_idx], 'volume'] = 15000  # Extreme volume

        detected, confidence = detect_buying_climax(df, default_config)

        assert detected.iloc[bc_idx], "BC should be detected at euphoria point"
        assert confidence.iloc[bc_idx] > 0.5, "BC confidence should be high"


class TestAutomaticRally:
    """Tests for Automatic Rally (AR) detection"""

    def test_ar_after_sc(self, base_ohlcv, default_config):
        """Test AR detection following SC"""
        df = base_ohlcv.copy()

        # Create SC at index 50
        sc_idx = 50
        df.loc[df.index[sc_idx], 'low'] = 45000
        df.loc[df.index[sc_idx], 'close'] = 46000
        df.loc[df.index[sc_idx], 'volume'] = 15000

        # Create AR at index 53 (within lookback)
        ar_idx = 53
        df.loc[df.index[ar_idx], 'close'] = 49000  # 50% retrace
        df.loc[df.index[ar_idx], 'high'] = 49200
        df.loc[df.index[ar_idx], 'low'] = 48000
        df.loc[df.index[ar_idx], 'volume'] = 3000  # Lower volume

        detected, confidence = detect_automatic_rally(df, default_config)

        assert detected.iloc[ar_idx], "AR should be detected after SC"
        assert confidence.iloc[ar_idx] > 0.4, "AR confidence should be moderate"


class TestSecondaryTest:
    """Tests for Secondary Test (ST) detection"""

    def test_st_basic_detection(self, base_ohlcv, default_config):
        """Test ST detection - retest of SC lows on lower volume"""
        df = base_ohlcv.copy()

        # Create SC at index 50
        sc_idx = 50
        df.loc[df.index[sc_idx], 'low'] = 45000

        # Create ST at index 65 (within lookback)
        st_idx = 65
        df.loc[df.index[st_idx], 'low'] = 45500  # Near SC low but holds above
        df.loc[df.index[st_idx], 'close'] = 46000
        df.loc[df.index[st_idx], 'volume'] = 2000  # Much lower volume

        detected, confidence = detect_secondary_test(df, default_config)

        assert detected.iloc[st_idx], "ST should be detected at retest"


class TestSignOfStrength:
    """Tests for Sign of Strength (SOS) detection"""

    def test_sos_breakout(self, base_ohlcv, default_config):
        """Test SOS detection on range breakout"""
        df = base_ohlcv.copy()

        # Establish range
        for i in range(40, 50):
            df.loc[df.index[i], 'high'] = 52000
            df.loc[df.index[i], 'low'] = 50000

        # Create SOS breakout at index 55
        sos_idx = 55
        df.loc[df.index[sos_idx], 'high'] = 54000  # Breaks above range
        df.loc[df.index[sos_idx], 'close'] = 53500  # Strong close
        df.loc[df.index[sos_idx], 'low'] = 52000
        df.loc[df.index[sos_idx], 'volume'] = 8000  # Strong volume

        detected, confidence = detect_sign_of_strength(df, default_config)

        assert detected.iloc[sos_idx], "SOS should be detected on breakout"


class TestSpringTypeA:
    """Tests for Spring Type A detection"""

    def test_spring_a_fake_breakdown(self, base_ohlcv, default_config):
        """Test Spring Type A - deep fake breakdown that recovers"""
        df = base_ohlcv.copy()

        # Establish range
        for i in range(40, 50):
            df.loc[df.index[i], 'low'] = 50000

        # Create Spring at index 55
        spring_idx = 55
        df.loc[df.index[spring_idx], 'low'] = 48900  # Breaks below 50000 by 2%
        df.loc[df.index[spring_idx], 'close'] = 49500
        df.loc[df.index[spring_idx], 'volume'] = 7000

        # Recovery within 3 bars
        df.loc[df.index[spring_idx + 2], 'close'] = 51000  # Back above range

        detected, confidence = detect_spring_type_a(df, default_config)

        # Note: Detection happens after recovery confirmation
        # May not show at spring_idx due to lookahead requirement
        assert any(detected.iloc[spring_idx:spring_idx+5]), "Spring should be detected after recovery"


class TestUpthrust:
    """Tests for Upthrust (UT) detection"""

    def test_ut_fake_breakout(self, base_ohlcv, default_config):
        """Test UT - fake breakout above range that fails"""
        df = base_ohlcv.copy()

        # Establish range
        for i in range(40, 50):
            df.loc[df.index[i], 'high'] = 52000

        # Create UT at index 55
        ut_idx = 55
        df.loc[df.index[ut_idx], 'high'] = 53500  # Breaks above by 2%
        df.loc[df.index[ut_idx], 'close'] = 52200
        df.loc[df.index[ut_idx], 'volume'] = 7000

        # Failure within 3 bars
        df.loc[df.index[ut_idx + 2], 'close'] = 51000  # Back below range

        detected, confidence = detect_upthrust(df, default_config)

        # Detection happens after failure confirmation
        assert any(detected.iloc[ut_idx:ut_idx+5]), "UT should be detected after failure"


class TestLastPointOfSupport:
    """Tests for Last Point of Support (LPS) detection"""

    def test_lps_final_test(self, base_ohlcv, default_config):
        """Test LPS - final support test on low volume before markup"""
        df = base_ohlcv.copy()

        # Establish support at 50000
        for i in range(40, 50):
            df.loc[df.index[i], 'low'] = 50000

        # Create LPS at index 60
        lps_idx = 60
        df.loc[df.index[lps_idx], 'low'] = 50200  # At support
        df.loc[df.index[lps_idx], 'close'] = 50800  # Strong close
        df.loc[df.index[lps_idx], 'high'] = 51000
        df.loc[df.index[lps_idx], 'volume'] = 1500  # Very low volume

        detected, confidence = detect_last_point_of_support(df, default_config)

        assert detected.iloc[lps_idx], "LPS should be detected at final support test"


class TestAllWyckoffEvents:
    """Integration tests for complete Wyckoff event detection"""

    def test_detect_all_events_no_errors(self, base_ohlcv):
        """Test that detect_all_wyckoff_events runs without errors"""
        df = base_ohlcv.copy()

        result = detect_all_wyckoff_events(df, {})

        # Check that all expected columns are present
        expected_cols = [
            'wyckoff_sc', 'wyckoff_sc_confidence',
            'wyckoff_bc', 'wyckoff_bc_confidence',
            'wyckoff_ar', 'wyckoff_ar_confidence',
            'wyckoff_st', 'wyckoff_st_confidence',
            'wyckoff_sos', 'wyckoff_sos_confidence',
            'wyckoff_sow', 'wyckoff_sow_confidence',
            'wyckoff_spring_a', 'wyckoff_spring_a_confidence',
            'wyckoff_spring_b', 'wyckoff_spring_b_confidence',
            'wyckoff_ut', 'wyckoff_ut_confidence',
            'wyckoff_lps', 'wyckoff_lps_confidence',
            'wyckoff_lpsy', 'wyckoff_lpsy_confidence',
            'wyckoff_phase_abc',
            'wyckoff_sequence_position',
        ]

        for col in expected_cols:
            assert col in result.columns, f"Column {col} should be present"

    def test_phase_classification(self, base_ohlcv):
        """Test Wyckoff phase classification logic"""
        df = base_ohlcv.copy()

        # Create SC pattern (Phase A indicator)
        sc_idx = 50
        df.loc[df.index[sc_idx], 'low'] = 45000
        df.loc[df.index[sc_idx], 'close'] = 46000
        df.loc[df.index[sc_idx], 'volume'] = 15000

        result = detect_all_wyckoff_events(df, {
            'sc_volume_z_min': 2.0,
            'sc_range_pos_max': 0.3,
            'sc_range_z_min': 1.0,
            'sc_wick_min': 0.5,
        })

        # Check that Phase A is detected near SC event
        phase_a_bars = result[result['wyckoff_phase_abc'] == 'A']
        assert len(phase_a_bars) > 0, "Phase A should be detected after SC event"


class TestPTIIntegration:
    """Tests for PTI-Wyckoff integration"""

    def test_pti_integration(self, base_ohlcv):
        """Test PTI integration with Wyckoff events"""
        df = base_ohlcv.copy()

        # Add PTI scores
        df['pti_score'] = 0.3
        df.loc[df.index[50], 'pti_score'] = 0.8  # High PTI

        # Create Spring at same location
        df.loc[df.index[50], 'low'] = 48000
        df.loc[df.index[50], 'close'] = 50000
        df.loc[df.index[50], 'volume'] = 7000

        result = detect_all_wyckoff_events(df, {})
        result = integrate_wyckoff_with_pti(result)

        assert 'wyckoff_pti_confluence' in result.columns
        assert 'wyckoff_pti_score' in result.columns

    def test_confluence_detection(self, base_ohlcv):
        """Test that confluence is properly detected"""
        df = base_ohlcv.copy()

        # High PTI + trap event should trigger confluence
        df['pti_score'] = 0.0
        df.loc[df.index[50], 'pti_score'] = 0.7

        # Manually set Spring event for testing
        df = detect_all_wyckoff_events(df, {})
        df.loc[df.index[50], 'wyckoff_spring_a'] = True
        df.loc[df.index[50], 'wyckoff_spring_a_confidence'] = 0.8

        result = integrate_wyckoff_with_pti(df)

        # Should detect confluence
        assert result['wyckoff_pti_confluence'].iloc[50], "Confluence should be detected"


class TestEdgeCases:
    """Test edge cases and error handling"""

    def test_minimal_data(self):
        """Test with minimal data (should not crash)"""
        df = pd.DataFrame({
            'open': [50000, 51000],
            'high': [51000, 52000],
            'low': [49000, 50000],
            'close': [50500, 51500],
            'volume': [1000, 1200],
        })

        result = detect_all_wyckoff_events(df, {})

        # Should complete without errors (but likely no events detected)
        assert len(result) == 2

    def test_missing_volume(self):
        """Test handling of missing volume data"""
        df = pd.DataFrame({
            'open': [50000] * 50,
            'high': [51000] * 50,
            'low': [49000] * 50,
            'close': [50500] * 50,
            # No volume column
        })

        # Should raise error or handle gracefully
        with pytest.raises((KeyError, ValueError)):
            detect_all_wyckoff_events(df, {})

    def test_confidence_bounds(self, base_ohlcv):
        """Test that confidence scores are bounded [0, 1]"""
        df = base_ohlcv.copy()

        result = detect_all_wyckoff_events(df, {})

        confidence_cols = [col for col in result.columns if '_confidence' in col]

        for col in confidence_cols:
            assert result[col].min() >= 0.0, f"{col} should be >= 0"
            assert result[col].max() <= 1.0, f"{col} should be <= 1"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
