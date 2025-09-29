"""
Unit tests for PO3 (Preliminary Order 3) detection in Bull Machine v1.6.1
Tests PO3 sweep and break patterns with Wyckoff integration
"""

import pandas as pd
import pytest
import numpy as np
from bull_machine.strategy.po3_detection import detect_po3, reverses, has_bojan_high


class TestPO3Detection:
    """Test PO3 pattern detection."""

    def test_basic_po3_detection(self):
        """Test basic PO3 detection with sweep and break."""
        # Create test data with PO3 pattern
        df = pd.DataFrame({
            'high': [100, 105, 110, 108, 112],
            'low': [95, 100, 105, 102, 107],
            'close': [98, 104, 108, 106, 111],
            'volume': [1000, 1200, 2000, 1800, 2500]  # Volume spike during sweep
        })

        irh = 110  # Initial Range High
        irl = 95   # Initial Range Low

        po3 = detect_po3(df, irh, irl, vol_spike_threshold=1.4)

        assert po3 is not None
        assert po3['strength'] >= 0.70
        assert po3['po3_type'] in ['low_sweep_high_break', 'high_sweep_low_break', 'high_sweep_high_break', 'low_sweep_low_break']

    def test_reversal_detection(self):
        """Test reversal pattern detection."""
        # Strong reversal bar
        df = pd.DataFrame({
            'high': [100, 101, 102],
            'low': [95, 96, 97],
            'close': [99, 100, 101],  # Close near highs
            'volume': [1000, 1500, 2000]
        })

        assert reverses(df, n_bars=3) is True

        # Weak reversal bar
        df_weak = pd.DataFrame({
            'high': [100, 101, 102],
            'low': [95, 96, 97],
            'close': [96, 97, 98],  # Close near lows
            'volume': [1000, 1500, 2000]
        })

        assert reverses(df_weak, n_bars=3) is False

    def test_bojan_high_detection(self):
        """Test Bojan high pattern detection (wick magnets)."""
        # Bojan high pattern
        df = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [105, 110, 115],  # Long upper wicks
            'low': [99, 100, 101],
            'close': [101, 102, 103],
            'volume': [1000, 1500, 2000]
        })

        assert has_bojan_high(df) is True

        # No Bojan high
        df_no_bojan = pd.DataFrame({
            'open': [100, 101, 102],
            'high': [101, 102, 103],  # No significant wicks
            'low': [99, 100, 101],
            'close': [100.5, 101.5, 102.5],
            'volume': [1000, 1500, 2000]
        })

        assert has_bojan_high(df_no_bojan) is False

    def test_low_sweep_high_break(self):
        """Test low sweep followed by high break pattern."""
        df = pd.DataFrame({
            'high': [100, 102, 104, 106, 108],
            'low': [95, 97, 99, 94, 103],  # Sweep below IRL at bar 4
            'close': [98, 100, 102, 107, 107],  # Break above IRH at bar 4
            'volume': [1000, 1200, 1400, 2500, 2000]  # Volume spike during sweep
        })

        irh = 104  # Initial Range High
        irl = 95   # Initial Range Low

        po3 = detect_po3(df, irh, irl, vol_spike_threshold=1.4)

        assert po3 is not None
        assert po3['po3_type'] == 'low_sweep_high_break'
        assert po3['strength'] >= 0.70

    def test_high_sweep_low_break(self):
        """Test high sweep followed by low break pattern."""
        df = pd.DataFrame({
            'high': [105, 107, 109, 112, 98],  # Sweep above IRH at bar 4
            'low': [100, 102, 104, 107, 92],   # Break below IRL at bar 5
            'close': [103, 105, 107, 98, 95],
            'volume': [1000, 1200, 1400, 2500, 2000]  # Volume spike during sweep
        })

        irh = 109  # Initial Range High
        irl = 100  # Initial Range Low

        po3 = detect_po3(df, irh, irl, vol_spike_threshold=1.4)

        assert po3 is not None
        assert po3['po3_type'] == 'high_sweep_low_break'
        assert po3['strength'] >= 0.70

    def test_no_po3_pattern(self):
        """Test when no PO3 pattern is present."""
        # Normal range-bound price action
        df = pd.DataFrame({
            'high': [102, 103, 104, 103, 102],
            'low': [98, 99, 100, 99, 98],
            'close': [100, 101, 102, 101, 100],
            'volume': [1000, 1100, 1200, 1100, 1000]  # No volume spikes
        })

        irh = 104
        irl = 98

        po3 = detect_po3(df, irh, irl, vol_spike_threshold=1.4)

        assert po3 is None

    def test_insufficient_data(self):
        """Test PO3 detection with insufficient data."""
        df = pd.DataFrame({
            'high': [100],
            'low': [95],
            'close': [98],
            'volume': [1000]
        })

        irh = 100
        irl = 95

        po3 = detect_po3(df, irh, irl)

        assert po3 is None

    def test_bojan_high_boost(self):
        """Test Bojan high pattern provides strength boost."""
        # Create PO3 pattern with Bojan high
        df = pd.DataFrame({
            'open': [98, 100, 102, 104, 106],
            'high': [101, 115, 107, 109, 111],  # Bojan high at bar 2
            'low': [95, 99, 101, 94, 105],     # Sweep at bar 4
            'close': [99, 103, 105, 108, 108],  # Break at bar 4
            'volume': [1000, 1200, 1400, 2500, 2000]
        })

        irh = 107
        irl = 95

        po3 = detect_po3(df, irh, irl, vol_spike_threshold=1.4)

        assert po3 is not None
        assert po3['strength'] >= 0.79  # Base 0.70 + 0.10 Bojan boost (allow for float precision)

    def test_volume_threshold_sensitivity(self):
        """Test PO3 detection sensitivity to volume threshold."""
        df = pd.DataFrame({
            'high': [100, 102, 104, 106, 108],
            'low': [95, 97, 99, 94, 103],
            'close': [98, 100, 102, 107, 107],
            'volume': [1000, 1200, 1400, 1600, 2000]  # Lower volume spike
        })

        irh = 104
        irl = 95

        # Strict threshold - no detection
        po3_strict = detect_po3(df, irh, irl, vol_spike_threshold=2.0)
        assert po3_strict is None

        # Loose threshold - detection
        po3_loose = detect_po3(df, irh, irl, vol_spike_threshold=1.2)
        assert po3_loose is not None


class TestPO3Integration:
    """Test PO3 integration with other systems."""

    def test_po3_with_wyckoff_phases(self):
        """Test PO3 detection works with different Wyckoff phases."""
        # This would be tested in integration tests with actual Wyckoff detection
        # For now, just ensure PO3 patterns are detected consistently

        phases_data = {
            'Phase_C': pd.DataFrame({
                'high': [100, 102, 104, 106, 108],
                'low': [95, 97, 99, 94, 103],
                'close': [98, 100, 102, 107, 107],
                'volume': [1000, 1200, 1400, 2500, 2000]
            }),
            'Phase_D': pd.DataFrame({
                'high': [105, 107, 109, 112, 98],
                'low': [100, 102, 104, 107, 92],
                'close': [103, 105, 107, 98, 95],
                'volume': [1000, 1200, 1400, 2500, 2000]
            })
        }

        for phase, df in phases_data.items():
            irh = df['high'].iloc[:-2].max()
            irl = df['low'].iloc[:-2].min()

            po3 = detect_po3(df, irh, irl, vol_spike_threshold=1.4)

            # PO3 should be detected in both Wyckoff phases
            assert po3 is not None, f"PO3 not detected in {phase}"
            assert po3['strength'] >= 0.70, f"Insufficient PO3 strength in {phase}"

    def test_edge_cases(self):
        """Test edge cases and error handling."""
        # Zero range bars
        df_zero_range = pd.DataFrame({
            'high': [100, 100, 100],
            'low': [100, 100, 100],
            'close': [100, 100, 100],
            'volume': [1000, 2000, 1500]
        })

        assert reverses(df_zero_range) is False
        assert has_bojan_high(df_zero_range) is False

        # NaN values
        df_nan = pd.DataFrame({
            'high': [100, np.nan, 104],
            'low': [95, 97, np.nan],
            'close': [98, 100, 102],
            'volume': [1000, 1500, 2000]
        })

        # Should handle gracefully without crashing
        try:
            po3 = detect_po3(df_nan, 104, 95)
            # May return None or valid result, but shouldn't crash
            assert po3 is None or isinstance(po3, dict)
        except Exception:
            pytest.fail("PO3 detection should handle NaN values gracefully")


# Integration test placeholder
def test_po3_confluence_tags():
    """Test that PO3 generates proper confluence tags."""
    df = pd.DataFrame({
        'high': [100, 102, 104, 106, 108],
        'low': [95, 97, 99, 94, 103],
        'close': [98, 100, 102, 107, 107],
        'volume': [1000, 1200, 1400, 2500, 2000]
    })

    irh = 104
    irl = 95

    po3 = detect_po3(df, irh, irl, vol_spike_threshold=1.5)

    assert po3 is not None
    assert 'po3_type' in po3
    assert 'strength' in po3
    assert po3['po3_type'] in ['low_sweep_high_break', 'high_sweep_low_break', 'high_sweep_high_break', 'low_sweep_low_break']