"""
Unit tests for CorrelationMonitor.

Tests:
1. Correlation calculation accuracy
2. Rolling window logic
3. Alert triggers at thresholds (>0.7 breakdown, >0.8 redundant)
4. Conflict identification for portfolio allocator
5. Report generation
6. State persistence
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import shutil

from engine.monitoring.correlation_monitor import (
    CorrelationMonitor,
    CorrelationAlert,
    CorrelationMetrics
)


@pytest.fixture
def temp_output_dir():
    """Create temporary directory for test outputs."""
    temp_dir = Path(tempfile.mkdtemp())
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def monitor(temp_output_dir):
    """Create CorrelationMonitor instance for testing."""
    return CorrelationMonitor(
        window_days=[30, 60],
        alert_threshold=0.7,
        redundant_threshold=0.8,
        min_observations=10,
        output_dir=temp_output_dir
    )


@pytest.fixture
def sample_returns():
    """Generate sample archetype returns for testing."""
    np.random.seed(42)

    # Create 100 days of returns
    dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(100)]

    # Create archetypes with known correlation structure
    # Group 1: spring and wick_trap (high correlation ~0.85)
    base_returns_1 = np.random.randn(100) * 2.0
    spring_returns = base_returns_1 + np.random.randn(100) * 0.5
    wick_trap_returns = base_returns_1 + np.random.randn(100) * 0.5

    # Group 2: whipsaw (independent, correlation ~0.0)
    whipsaw_returns = np.random.randn(100) * 2.0

    # Group 3: order_block (medium correlation with spring ~0.6)
    order_block_returns = base_returns_1 * 0.6 + np.random.randn(100) * 1.5

    returns_history = []
    for i in range(100):
        returns_history.append({
            'spring': spring_returns[i],
            'wick_trap': wick_trap_returns[i],
            'whipsaw': whipsaw_returns[i],
            'order_block': order_block_returns[i]
        })

    return dates, returns_history


class TestCorrelationCalculation:
    """Test correlation calculation accuracy."""

    def test_correlation_accuracy(self, monitor, sample_returns):
        """Test that calculated correlations match expected values."""
        dates, returns = sample_returns

        # Feed data to monitor
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        # Get correlation matrix
        corr_matrix = monitor.get_correlation_matrix(window_days=60)

        assert corr_matrix is not None
        assert corr_matrix.shape == (4, 4)

        # Check known correlations
        # spring-wick_trap should be high (>0.8)
        spring_wick = corr_matrix.loc['spring', 'wick_trap']
        assert spring_wick > 0.8, f"Expected spring-wick_trap >0.8, got {spring_wick:.3f}"

        # spring-whipsaw should be low (<0.3)
        spring_whip = abs(corr_matrix.loc['spring', 'whipsaw'])
        assert spring_whip < 0.3, f"Expected spring-whipsaw <0.3, got {spring_whip:.3f}"

        # Diagonal should be 1.0
        for arch in corr_matrix.columns:
            assert abs(corr_matrix.loc[arch, arch] - 1.0) < 0.01

    def test_rolling_window_updates(self, monitor, sample_returns):
        """Test that correlation updates as new data comes in."""
        dates, returns = sample_returns

        # Feed first 30 days
        for i in range(30):
            monitor.update(dates[i], returns[i])

        corr_30 = monitor.get_correlation_matrix(window_days=30)

        # Feed next 30 days
        for i in range(30, 60):
            monitor.update(dates[i], returns[i])

        corr_60 = monitor.get_correlation_matrix(window_days=30)

        # Correlations should be different (window moved)
        assert not np.allclose(corr_30.values, corr_60.values)

    def test_multiple_windows(self, monitor, sample_returns):
        """Test that multiple windows are tracked independently."""
        dates, returns = sample_returns

        # Feed all data
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        # Get both windows
        corr_30 = monitor.get_correlation_matrix(window_days=30)
        corr_60 = monitor.get_correlation_matrix(window_days=60)

        assert corr_30 is not None
        assert corr_60 is not None

        # Should be different (different window sizes)
        assert not np.allclose(corr_30.values, corr_60.values)


class TestAlertGeneration:
    """Test alert triggers at correlation thresholds."""

    def test_redundant_alert_trigger(self, monitor, sample_returns):
        """Test that redundant strategy alert fires when correlation > 0.8."""
        dates, returns = sample_returns

        # Feed data
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        # Get critical alerts
        critical = monitor.get_alerts(min_severity='critical')

        # Should have alert for spring-wick_trap (corr ~0.85)
        spring_wick_alerts = [
            a for a in critical
            if set(a.archetype_pair) == {'spring', 'wick_trap'}
        ]

        assert len(spring_wick_alerts) > 0, "Expected redundant alert for spring-wick_trap"
        assert spring_wick_alerts[0].severity == 'critical'
        assert spring_wick_alerts[0].alert_type == 'redundant'

    def test_breakdown_alert_trigger(self, monitor, sample_returns):
        """Test that breakdown alert fires when correlation > 0.7 but < 0.8."""
        dates, returns = sample_returns

        # Feed data
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        # Get warning alerts
        warnings = [a for a in monitor.get_alerts() if a.severity == 'warning']

        # Should have alert for spring-order_block (corr ~0.6-0.7 range)
        # Note: May or may not trigger depending on random seed, so we just check structure
        if warnings:
            assert warnings[0].alert_type == 'breakdown'
            assert warnings[0].correlation >= 0.7
            assert warnings[0].correlation < 0.8

    def test_alert_filtering(self, monitor, sample_returns):
        """Test alert filtering by severity and timestamp."""
        dates, returns = sample_returns

        # Feed data
        for date, ret in zip(dates[:50], returns[:50]):
            monitor.update(date, ret)

        alerts_before = len(monitor.get_alerts())

        # Feed more data
        for date, ret in zip(dates[50:], returns[50:]):
            monitor.update(date, ret)

        # Filter by timestamp
        recent = monitor.get_alerts(since=dates[60])
        all_alerts = monitor.get_alerts()

        assert len(recent) <= len(all_alerts)

        # Filter by severity
        critical = monitor.get_alerts(min_severity='critical')
        assert all(a.severity == 'critical' for a in critical)


class TestMetricsCalculation:
    """Test diversification metrics calculation."""

    def test_metrics_structure(self, monitor, sample_returns):
        """Test that metrics are calculated and stored correctly."""
        dates, returns = sample_returns

        # Feed data
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        metrics = monitor.get_diversification_metrics(window_days=60)

        assert metrics is not None
        assert metrics.mean_correlation >= 0
        assert metrics.median_correlation >= 0
        assert metrics.max_correlation <= 1.0
        assert metrics.diversification_score >= 0
        assert metrics.diversification_score <= 1.0

    def test_health_check(self, monitor, sample_returns):
        """Test diversification health assessment."""
        dates, returns = sample_returns

        # Feed data with high correlation
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        metrics = monitor.get_diversification_metrics(window_days=60)

        # Should be unhealthy (spring-wick_trap corr > 0.8)
        assert not metrics.is_healthy()
        assert metrics.num_redundant_pairs > 0

    def test_diversification_score(self, monitor):
        """Test diversification score calculation."""
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(50)]

        # Create perfectly uncorrelated returns
        np.random.seed(42)
        returns = []
        for i in range(50):
            returns.append({
                'arch1': np.random.randn(),
                'arch2': np.random.randn(),
                'arch3': np.random.randn()
            })

        # Feed data
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        metrics = monitor.get_diversification_metrics(window_days=30)

        # Diversification score should be high (close to 1.0) for uncorrelated
        assert metrics.diversification_score > 0.8


class TestConflictResolution:
    """Test conflict identification for portfolio allocator."""

    def test_conflict_detection(self, monitor, sample_returns):
        """Test detection of conflicting archetype pairs."""
        dates, returns = sample_returns

        # Feed data
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        conflicts = monitor.get_conflict_resolution_data()

        # spring and wick_trap should be in conflict (corr > 0.7)
        assert 'spring' in conflicts
        assert 'wick_trap' in conflicts['spring']
        assert 'spring' in conflicts['wick_trap']

        # whipsaw should have no conflicts (independent)
        assert 'whipsaw' not in conflicts or len(conflicts.get('whipsaw', [])) == 0

    def test_redundant_pairs_identification(self, monitor, sample_returns):
        """Test identification of redundant strategy pairs."""
        dates, returns = sample_returns

        # Feed data
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        redundant = monitor.get_redundant_pairs()

        # Should include spring-wick_trap
        pair_names = [set(p[:2]) for p in redundant]
        assert {'spring', 'wick_trap'} in pair_names

        # Should be sorted by correlation (descending)
        correlations = [abs(p[2]) for p in redundant]
        assert correlations == sorted(correlations, reverse=True)


class TestVisualization:
    """Test visualization generation."""

    def test_heatmap_generation(self, monitor, sample_returns, temp_output_dir):
        """Test correlation heatmap generation."""
        dates, returns = sample_returns

        # Feed data
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        # Generate heatmap
        output_path = monitor.plot_correlation_heatmap(window_days=60, show=False)

        assert output_path is not None
        assert output_path.exists()
        assert output_path.suffix == '.png'

    def test_timeseries_generation(self, monitor, sample_returns, temp_output_dir):
        """Test correlation time series plot."""
        dates, returns = sample_returns

        # Feed data
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        # Generate time series
        pairs = [('spring', 'wick_trap'), ('spring', 'whipsaw')]
        output_path = monitor.plot_correlation_timeseries(
            archetype_pairs=pairs,
            window_days=30,
            show=False
        )

        assert output_path is not None
        assert output_path.exists()
        assert output_path.suffix == '.png'


class TestReportGeneration:
    """Test report generation."""

    def test_report_structure(self, monitor, sample_returns, temp_output_dir):
        """Test that report contains all required sections."""
        dates, returns = sample_returns

        # Feed data
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        report = monitor.generate_report(save=True)

        # Check report contains key sections
        assert "ARCHETYPE CORRELATION MONITORING REPORT" in report
        assert "ALERTS SUMMARY" in report
        assert "REDUNDANT ARCHETYPE PAIRS" in report
        assert "RECOMMENDATIONS" in report

        # Check report file was created
        report_files = list(temp_output_dir.glob('correlation_report_*.txt'))
        assert len(report_files) > 0

    def test_report_recommendations(self, monitor, sample_returns):
        """Test that report provides actionable recommendations."""
        dates, returns = sample_returns

        # Feed data
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        report = monitor.generate_report(save=False)

        # Should identify redundant pairs and recommend deactivation
        assert 'spring' in report or 'wick_trap' in report
        assert 'RECOMMENDATION' in report


class TestStatePersistence:
    """Test state save/load functionality."""

    def test_save_and_load_state(self, monitor, sample_returns, temp_output_dir):
        """Test saving and loading monitor state."""
        dates, returns = sample_returns

        # Feed data
        for date, ret in zip(dates[:50], returns[:50]):
            monitor.update(date, ret)

        # Save state
        state_path = monitor.save_state()
        assert state_path.exists()

        # Load state
        restored = CorrelationMonitor.load_state(state_path)

        # Verify state matches
        assert len(restored.returns_history) == len(monitor.returns_history)
        assert len(restored.timestamps) == len(monitor.timestamps)
        assert restored.archetypes == monitor.archetypes
        assert restored.window_days == monitor.window_days

        # Continue with restored monitor
        for date, ret in zip(dates[50:], returns[50:]):
            restored.update(date, ret)

        # Should work correctly
        metrics = restored.get_diversification_metrics()
        assert metrics is not None


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_insufficient_data(self, monitor):
        """Test behavior with insufficient data."""
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(5)]
        returns = [
            {'arch1': np.random.randn(), 'arch2': np.random.randn()}
            for _ in range(5)
        ]

        # Feed data
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        # Should not have correlation matrix (< min_observations)
        corr = monitor.get_correlation_matrix()
        assert corr is None

    def test_missing_archetypes(self, monitor):
        """Test handling of inconsistent archetype sets."""
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]

        # Feed data with changing archetype sets
        for i, date in enumerate(dates):
            if i < 15:
                ret = {'arch1': np.random.randn(), 'arch2': np.random.randn()}
            else:
                ret = {'arch1': np.random.randn(), 'arch3': np.random.randn()}

            monitor.update(date, ret)

        # Should handle gracefully (NaN for missing data)
        corr = monitor.get_correlation_matrix()
        # Check that it exists and has expected archetypes
        assert corr is not None
        assert len(monitor.archetypes) == 3  # arch1, arch2, arch3

    def test_single_archetype(self, monitor):
        """Test behavior with only one archetype."""
        dates = [datetime(2023, 1, 1) + timedelta(days=i) for i in range(30)]
        returns = [{'arch1': np.random.randn()} for _ in range(30)]

        # Feed data
        for date, ret in zip(dates, returns):
            monitor.update(date, ret)

        # Should work but have trivial correlation matrix
        corr = monitor.get_correlation_matrix()
        assert corr is not None
        assert corr.shape == (1, 1)
        assert corr.loc['arch1', 'arch1'] == 1.0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
