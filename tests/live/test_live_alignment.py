#!/usr/bin/env python3
"""
Test right-edge alignment and no future leak during live streaming
"""

import sys
import os
from pathlib import Path
import pytest
import pandas as pd
from datetime import datetime, timedelta

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bin.live.adapters import LiveDataAdapter


class TestLiveAlignment:
    """Test suite for live data alignment and temporal integrity."""

    def setup_method(self):
        """Setup test fixtures."""
        self.adapter = LiveDataAdapter()

    def test_update_ohlcv_right_edge(self):
        """Test that OHLCV updates maintain right-edge temporal ordering."""
        # Create empty DataFrame
        df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Add ticks in correct temporal order
        tick1 = {
            'timestamp': datetime(2025, 6, 1, 10, 0),
            'Open': 2000.0, 'High': 2010.0, 'Low': 1990.0, 'Close': 2005.0, 'Volume': 100
        }

        tick2 = {
            'timestamp': datetime(2025, 6, 1, 11, 0),
            'Open': 2005.0, 'High': 2020.0, 'Low': 2000.0, 'Close': 2015.0, 'Volume': 150
        }

        # Update DataFrame
        df = self.adapter.update_ohlcv(df, tick1)
        df = self.adapter.update_ohlcv(df, tick2)

        # Verify temporal ordering
        assert len(df) == 2
        assert df.index[0] < df.index[1]
        assert df.index.is_monotonic_increasing

    def test_no_future_leak_protection(self):
        """Test protection against future data leakage."""
        df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Add initial tick
        tick1 = {
            'timestamp': datetime(2025, 6, 1, 10, 0),
            'Open': 2000.0, 'High': 2010.0, 'Low': 1990.0, 'Close': 2005.0, 'Volume': 100
        }
        df = self.adapter.update_ohlcv(df, tick1)

        # Try to add tick from the past (should be rejected)
        tick_past = {
            'timestamp': datetime(2025, 6, 1, 9, 0),  # 1 hour in the past
            'Open': 1950.0, 'High': 1960.0, 'Low': 1940.0, 'Close': 1955.0, 'Volume': 80
        }

        df_updated = self.adapter.update_ohlcv(df, tick_past)

        # DataFrame should be unchanged (past tick rejected)
        assert len(df_updated) == 1
        assert df_updated.index[0] == tick1['timestamp']

    def test_drift_tolerance(self):
        """Test that small time drifts are allowed within tolerance."""
        df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Add initial tick
        tick1 = {
            'timestamp': datetime(2025, 6, 1, 10, 0),
            'Open': 2000.0, 'High': 2010.0, 'Low': 1990.0, 'Close': 2005.0, 'Volume': 100
        }
        df = self.adapter.update_ohlcv(df, tick1)

        # Add tick with small backward drift (within tolerance)
        tick_drift = {
            'timestamp': datetime(2025, 6, 1, 9, 58),  # 2 minutes back (within 5min tolerance)
            'Open': 2005.0, 'High': 2020.0, 'Low': 2000.0, 'Close': 2015.0, 'Volume': 120
        }

        df_updated = self.adapter.update_ohlcv(df, tick_drift)

        # Should accept tick within drift tolerance
        assert len(df_updated) == 2

    def test_mtf_alignment_tolerance(self):
        """Test multi-timeframe alignment with realistic tolerance."""
        # Create sample MTF data
        base_time = datetime(2025, 6, 1, 12, 0)

        # 1H data (latest)
        df_1h = pd.DataFrame({
            'Open': [2000, 2010],
            'High': [2010, 2025],
            'Low': [1990, 2005],
            'Close': [2005, 2020],
            'Volume': [100, 150]
        }, index=[base_time, base_time + timedelta(hours=1)])

        # 4H data (slightly lagged)
        df_4h = pd.DataFrame({
            'Open': [2000],
            'High': [2025],
            'Low': [1990],
            'Close': [2020],
            'Volume': [300]
        }, index=[base_time - timedelta(minutes=10)])  # 10 minutes lag

        # 1D data (more lagged)
        df_1d = pd.DataFrame({
            'Open': [1980],
            'High': [2050],
            'Low': [1970],
            'Close': [2020],
            'Volume': [1000]
        }, index=[base_time - timedelta(hours=2)])  # 2 hours lag

        # Test alignment
        aligned_1h, aligned_4h, aligned_1d = self.adapter.align_mtf(df_1h, df_4h, df_1d)

        # Should return all DataFrames (within reasonable lag tolerance)
        assert len(aligned_1h) == 2
        assert len(aligned_4h) == 1
        assert len(aligned_1d) == 1

    def test_max_bars_limit(self):
        """Test that max_bars limit is enforced."""
        df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Add more ticks than max_bars limit
        max_bars = 5
        for i in range(10):
            tick = {
                'timestamp': datetime(2025, 6, 1, 10, i),
                'Open': 2000.0 + i,
                'High': 2010.0 + i,
                'Low': 1990.0 + i,
                'Close': 2005.0 + i,
                'Volume': 100 + i
            }
            df = self.adapter.update_ohlcv(df, tick, max_bars=max_bars)

        # Should only keep last max_bars
        assert len(df) == max_bars
        assert df['Close'].iloc[-1] == 2014.0  # Last tick close price

    def test_empty_dataframe_handling(self):
        """Test proper handling of empty DataFrames."""
        df_empty = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Test MTF alignment with empty data
        aligned_1h, aligned_4h, aligned_1d = self.adapter.align_mtf(df_empty, df_empty, df_empty)

        assert len(aligned_1h) == 0
        assert len(aligned_4h) == 0
        assert len(aligned_1d) == 0

    def test_monotonic_index_preservation(self):
        """Test that DataFrame index remains monotonic after updates."""
        df = pd.DataFrame(columns=['Open', 'High', 'Low', 'Close', 'Volume'])

        # Add multiple ticks
        timestamps = [
            datetime(2025, 6, 1, 10, 0),
            datetime(2025, 6, 1, 11, 0),
            datetime(2025, 6, 1, 12, 0),
            datetime(2025, 6, 1, 13, 0)
        ]

        for i, ts in enumerate(timestamps):
            tick = {
                'timestamp': ts,
                'Open': 2000.0 + i,
                'High': 2010.0 + i,
                'Low': 1990.0 + i,
                'Close': 2005.0 + i,
                'Volume': 100 + i
            }
            df = self.adapter.update_ohlcv(df, tick)

        # Verify monotonic index
        assert df.index.is_monotonic_increasing
        assert all(df.index[i] <= df.index[i+1] for i in range(len(df)-1))


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])