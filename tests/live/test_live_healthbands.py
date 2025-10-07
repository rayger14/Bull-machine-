#!/usr/bin/env python3
"""
Test health band monitoring and rate calculations
"""

import sys
import os
from pathlib import Path
import pytest
from datetime import datetime

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from bin.live.health_monitor import HealthMonitor, HealthBands, VIXHysteresis


class TestHealthBands:
    """Test suite for health band monitoring."""

    def setup_method(self):
        """Setup test fixtures."""
        self.health_bands = HealthBands()
        self.monitor = HealthMonitor(window_size=20, health_bands=self.health_bands)

    def test_macro_veto_rate_calculation(self):
        """Test macro veto rate calculation."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Record signals with macro vetoes
        for i in range(10):
            signal_data = {'macro_vetoed': i < 3}  # 3 out of 10 vetoed = 30%
            domains_active = {'wyckoff': True, 'liquidity': False}
            self.monitor.record_signal(signal_data, domains_active, timestamp)

        metrics = self.monitor.get_current_metrics(timestamp)

        # Should be 30% macro veto rate
        assert abs(metrics.macro_veto_rate - 30.0) < 0.1

    def test_smc_2hit_rate_calculation(self):
        """Test SMC 2+ hit rate calculation."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Record signals with varying SMC domain activity
        test_cases = [
            {'liquidity': True, 'structure': True, 'orderflow': False},    # 2 hits
            {'liquidity': True, 'structure': False, 'orderflow': False},   # 1 hit
            {'liquidity': True, 'structure': True, 'orderflow': True},     # 3 hits
            {'liquidity': False, 'structure': False, 'orderflow': False},  # 0 hits
            {'liquidity': True, 'structure': True, 'orderflow': False},    # 2 hits
        ]

        for domains in test_cases:
            signal_data = {'macro_vetoed': False}
            self.monitor.record_signal(signal_data, domains, timestamp)

        metrics = self.monitor.get_current_metrics(timestamp)

        # 3 out of 5 signals had 2+ SMC hits = 60%
        assert abs(metrics.smc_2hit_rate - 60.0) < 0.1

    def test_health_band_violations(self):
        """Test detection of health band violations."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Create scenario with health violations
        # High macro veto rate (above max)
        for i in range(10):
            signal_data = {'macro_vetoed': i < 8}  # 80% veto rate (above 15% max)
            domains_active = {'wyckoff': True, 'liquidity': False}
            self.monitor.record_signal(signal_data, domains_active, timestamp)

        metrics = self.monitor.get_current_metrics(timestamp)

        # Should be unhealthy due to high macro veto rate
        assert not metrics.is_healthy
        assert any('macro veto rate too high' in warning.lower() for warning in metrics.warnings)

    def test_low_smc_hit_rate_warning(self):
        """Test warning for low SMC 2+ hit rate."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Record signals with mostly low SMC activity
        for i in range(10):
            # Only 2 out of 10 will have 2+ SMC hits = 20% (below 30% minimum)
            domains_active = {
                'liquidity': i < 2,
                'structure': i < 2,
                'orderflow': False
            }
            signal_data = {'macro_vetoed': False}
            self.monitor.record_signal(signal_data, domains_active, timestamp)

        metrics = self.monitor.get_current_metrics(timestamp)

        # Should be unhealthy due to low SMC hit rate
        assert not metrics.is_healthy
        assert any('smc 2+ hit rate too low' in warning.lower() for warning in metrics.warnings)

    def test_delta_breach_detection(self):
        """Test delta breach detection and counting."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Record signal with delta breaches
        signal_data = {
            'macro_vetoed': False,
            'macro_delta': 0.15,      # Breach (>0.10)
            'momentum_delta': 0.08,   # Breach (>0.06)
            'hob_delta': 0.03,        # OK (<0.05)
            'hps_delta': 0.02         # OK (<0.03)
        }
        domains_active = {'wyckoff': True}

        self.monitor.record_signal(signal_data, domains_active, timestamp)
        metrics = self.monitor.get_current_metrics(timestamp)

        # Should detect 2 delta breaches
        assert metrics.delta_breaches == 2

    def test_hob_relevance_tracking(self):
        """Test HOB relevance rate tracking."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Record signals with varying HOB activity
        for i in range(10):
            domains_active = {'hob': i < 4}  # 4 out of 10 = 40%
            signal_data = {'macro_vetoed': False}
            self.monitor.record_signal(signal_data, domains_active, timestamp)

        metrics = self.monitor.get_current_metrics(timestamp)

        # Should be 40% HOB relevance
        assert abs(metrics.hob_relevance_rate - 40.0) < 0.1

    def test_rolling_window_behavior(self):
        """Test that health monitor uses rolling window correctly."""
        timestamp = datetime(2025, 6, 1, 10, 0)
        window_size = 5
        monitor = HealthMonitor(window_size=window_size)

        # Fill beyond window size
        for i in range(10):
            signal_data = {'macro_vetoed': i < 8}  # High veto rate
            domains_active = {'wyckoff': True}
            monitor.record_signal(signal_data, domains_active, timestamp)

        # Check that only recent signals are considered
        assert len(monitor.macro_vetoes) == window_size
        assert len(monitor.signals_total) == window_size

    def test_health_summary_generation(self):
        """Test comprehensive health summary generation."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Record some signals
        for i in range(5):
            signal_data = {'macro_vetoed': i == 0}  # 20% veto rate
            domains_active = {
                'liquidity': True,
                'structure': True,
                'hob': i < 2  # 40% HOB relevance
            }
            self.monitor.record_signal(signal_data, domains_active, timestamp)

        # Log health metrics
        metrics = self.monitor.get_current_metrics(timestamp)
        self.monitor.log_health(metrics)

        # Get summary
        summary = self.monitor.get_health_summary()

        # Verify summary structure
        assert 'status' in summary
        assert 'current_metrics' in summary
        assert 'health_bands' in summary
        assert summary['total_checks'] > 0

    def test_vix_hysteresis_behavior(self):
        """Test VIX hysteresis state management."""
        vix = VIXHysteresis(on_threshold=22.0, off_threshold=18.0)

        # Initially inactive
        assert not vix.is_active

        # Activate when above threshold
        result = vix.update(25.0)
        assert result
        assert vix.is_active

        # Stay active when between thresholds
        result = vix.update(20.0)
        assert result
        assert vix.is_active

        # Deactivate when below off threshold
        result = vix.update(15.0)
        assert not result
        assert not vix.is_active

        # Stay inactive when between thresholds
        result = vix.update(20.0)
        assert not result
        assert not vix.is_active

    def test_empty_signals_handling(self):
        """Test proper handling when no signals have been recorded."""
        timestamp = datetime(2025, 6, 1, 10, 0)

        metrics = self.monitor.get_current_metrics(timestamp)

        # Should handle empty state gracefully
        assert metrics.macro_veto_rate == 0.0
        assert metrics.smc_2hit_rate == 0.0
        assert metrics.hob_relevance_rate == 0.0
        assert metrics.delta_breaches == 0


class TestHealthBandsIntegration:
    """Integration tests for health band monitoring."""

    def test_realistic_health_scenario(self):
        """Test realistic health monitoring scenario."""
        monitor = HealthMonitor(window_size=50)
        timestamp = datetime(2025, 6, 1, 10, 0)

        # Simulate realistic signal pattern
        for hour in range(24):  # 24 hours of signals
            current_time = timestamp.replace(hour=hour)

            # Vary signal patterns throughout day
            macro_veto_prob = 0.08 if hour < 12 else 0.12  # Higher vetoes in afternoon
            smc_activity_prob = 0.4 if hour % 4 == 0 else 0.3  # Periodic SMC activity

            signal_data = {
                'macro_vetoed': (hour * 7) % 100 < macro_veto_prob * 100,
                'macro_delta': 0.05 + (hour % 5) * 0.01,  # Varying delta
            }

            domains_active = {
                'liquidity': (hour * 3) % 100 < smc_activity_prob * 100,
                'structure': (hour * 5) % 100 < smc_activity_prob * 100,
                'hob': (hour * 11) % 100 < 25,  # 25% HOB relevance
            }

            monitor.record_signal(signal_data, domains_active, current_time)

        # Get final health assessment
        final_metrics = monitor.get_current_metrics(current_time)
        monitor.log_health(final_metrics)  # Need to log health for summary
        summary = monitor.get_health_summary()

        # Should have reasonable metrics
        assert 0 <= final_metrics.macro_veto_rate <= 100
        assert 0 <= final_metrics.smc_2hit_rate <= 100
        assert 0 <= final_metrics.hob_relevance_rate <= 100
        assert summary['status'] in ['healthy', 'warning', 'no_data']  # Accept no_data as valid


if __name__ == "__main__":
    # Run tests directly
    pytest.main([__file__, "-v"])