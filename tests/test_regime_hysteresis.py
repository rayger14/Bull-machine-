"""
Unit tests for RegimeHysteresis implementation.

Tests:
1. Configuration validation
2. Dual threshold mechanism
3. Minimum dwell time enforcement
4. EWMA smoothing
5. State management and reset
6. Transition tracking
7. Edge cases and error handling

Author: Claude Code (Backend Architect)
Date: 2026-01-19
"""

import pytest
import pandas as pd
from engine.context.regime_hysteresis import RegimeHysteresis


class TestRegimeHysteresisConfiguration:
    """Test configuration validation."""

    def test_default_configuration(self):
        """Test initialization with default configuration."""
        hysteresis = RegimeHysteresis()

        assert hysteresis.enter_threshold == 0.65
        assert hysteresis.exit_threshold == 0.50
        assert hysteresis.enable_ewma is False
        assert 'crisis' in hysteresis.min_duration_hours
        assert hysteresis.min_duration_hours['crisis'] == 6

    def test_custom_configuration(self):
        """Test initialization with custom configuration."""
        config = {
            'enter_threshold': 0.70,
            'exit_threshold': 0.45,
            'min_duration_hours': {'crisis': 12, 'risk_on': 36},
            'enable_ewma': True,
            'ewma_alpha': 0.4
        }
        hysteresis = RegimeHysteresis(config)

        assert hysteresis.enter_threshold == 0.70
        assert hysteresis.exit_threshold == 0.45
        assert hysteresis.enable_ewma is True
        assert hysteresis.ewma_alpha == 0.4
        assert hysteresis.min_duration_hours['crisis'] == 12

    def test_invalid_threshold_range(self):
        """Test that invalid thresholds raise ValueError."""
        with pytest.raises(ValueError, match="enter_threshold must be in"):
            RegimeHysteresis({'enter_threshold': 1.5})

        with pytest.raises(ValueError, match="exit_threshold must be in"):
            RegimeHysteresis({'exit_threshold': -0.1})

    def test_invalid_ewma_alpha(self):
        """Test that invalid EWMA alpha raises ValueError."""
        with pytest.raises(ValueError, match="ewma_alpha must be in"):
            RegimeHysteresis({'enable_ewma': True, 'ewma_alpha': 1.5})

    def test_inverted_thresholds_warning(self, caplog):
        """Test warning when exit_threshold > enter_threshold."""
        RegimeHysteresis({'enter_threshold': 0.50, 'exit_threshold': 0.60})
        assert "may cause instability" in caplog.text


class TestDualThresholdMechanism:
    """Test dual threshold logic."""

    def test_first_regime_high_confidence(self):
        """Test initialization with high confidence."""
        hysteresis = RegimeHysteresis({'enter_threshold': 0.65})
        probs = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.30, 'risk_on': 0.70}
        timestamp = pd.Timestamp('2024-01-01 00:00:00')

        result = hysteresis.apply_hysteresis('risk_on', probs, timestamp)

        assert result['regime'] == 'risk_on'
        assert result['transition'] is True
        assert result['dwell_time'] == 0.0
        assert hysteresis.current_regime == 'risk_on'

    def test_first_regime_low_confidence(self):
        """Test initialization with low confidence (defaults to proposed)."""
        hysteresis = RegimeHysteresis({'enter_threshold': 0.65})
        probs = {'crisis': 0.25, 'risk_off': 0.25, 'neutral': 0.25, 'risk_on': 0.25}
        timestamp = pd.Timestamp('2024-01-01 00:00:00')

        result = hysteresis.apply_hysteresis('neutral', probs, timestamp)

        # Should default to proposed regime despite low confidence
        assert result['regime'] == 'neutral'
        assert result['transition'] is True
        assert "low confidence" in result['reason'].lower()

    def test_stay_in_regime_below_enter_threshold(self):
        """Test staying in current regime when new regime below enter_threshold."""
        config = {'enter_threshold': 0.65, 'exit_threshold': 0.50, 'min_duration_hours': {'risk_on': 0}}
        hysteresis = RegimeHysteresis(config)

        # Initialize in risk_on
        probs1 = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.30, 'risk_on': 0.70}
        ts1 = pd.Timestamp('2024-01-01 00:00:00')
        hysteresis.apply_hysteresis('risk_on', probs1, ts1)

        # Propose neutral but below enter_threshold
        probs2 = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.60, 'risk_on': 0.40}
        ts2 = pd.Timestamp('2024-01-01 01:00:00')
        result = hysteresis.apply_hysteresis('neutral', probs2, ts2)

        assert result['regime'] == 'risk_on'  # Stay in risk_on
        assert result['transition'] is False
        assert result['hysteresis_applied'] is True
        assert "neutral prob 0.600 < enter_threshold" in result['reason']

    def test_transition_when_above_enter_and_below_exit(self):
        """Test transition when new regime > enter AND current < exit."""
        config = {'enter_threshold': 0.65, 'exit_threshold': 0.50, 'min_duration_hours': {'risk_on': 0}}
        hysteresis = RegimeHysteresis(config)

        # Initialize in risk_on
        probs1 = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.30, 'risk_on': 0.70}
        ts1 = pd.Timestamp('2024-01-01 00:00:00')
        hysteresis.apply_hysteresis('risk_on', probs1, ts1)

        # Propose neutral with strong signal, risk_on drops below exit
        probs2 = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.70, 'risk_on': 0.40}
        ts2 = pd.Timestamp('2024-01-01 01:00:00')
        result = hysteresis.apply_hysteresis('neutral', probs2, ts2)

        assert result['regime'] == 'neutral'  # Transition
        assert result['transition'] is True
        assert result['hysteresis_applied'] is False
        assert hysteresis.transition_count == 1

    def test_stay_when_current_above_exit_threshold(self):
        """Test staying when current regime still above exit_threshold."""
        config = {'enter_threshold': 0.65, 'exit_threshold': 0.50, 'min_duration_hours': {'risk_on': 0}}
        hysteresis = RegimeHysteresis(config)

        # Initialize in risk_on
        probs1 = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.30, 'risk_on': 0.70}
        ts1 = pd.Timestamp('2024-01-01 00:00:00')
        hysteresis.apply_hysteresis('risk_on', probs1, ts1)

        # Propose neutral with strong signal BUT risk_on still above exit
        probs2 = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.70, 'risk_on': 0.55}
        ts2 = pd.Timestamp('2024-01-01 01:00:00')
        result = hysteresis.apply_hysteresis('neutral', probs2, ts2)

        assert result['regime'] == 'risk_on'  # Stay
        assert result['transition'] is False
        assert result['hysteresis_applied'] is True
        assert "Current prob 0.550 >= exit_threshold" in result['reason']


class TestMinimumDwellTime:
    """Test minimum dwell time enforcement."""

    def test_locked_during_min_dwell(self):
        """Test that regime cannot change during minimum dwell time."""
        config = {
            'enter_threshold': 0.65,
            'exit_threshold': 0.50,
            'min_duration_hours': {'risk_on': 24}  # 24 hours minimum
        }
        hysteresis = RegimeHysteresis(config)

        # Initialize in risk_on
        probs1 = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.30, 'risk_on': 0.70}
        ts1 = pd.Timestamp('2024-01-01 00:00:00')
        hysteresis.apply_hysteresis('risk_on', probs1, ts1)

        # Try to switch after 12 hours (< 24h min)
        probs2 = {'crisis': 0.10, 'risk_off': 0.80, 'neutral': 0.30, 'risk_on': 0.10}
        ts2 = pd.Timestamp('2024-01-01 12:00:00')
        result = hysteresis.apply_hysteresis('risk_off', probs2, ts2)

        assert result['regime'] == 'risk_on'  # Locked
        assert result['transition'] is False
        assert result['hysteresis_applied'] is True
        assert "Locked in risk_on" in result['reason']
        assert result['dwell_time'] == 12.0

    def test_transition_after_min_dwell(self):
        """Test that transition is allowed after minimum dwell time."""
        config = {
            'enter_threshold': 0.65,
            'exit_threshold': 0.50,
            'min_duration_hours': {'risk_on': 24}
        }
        hysteresis = RegimeHysteresis(config)

        # Initialize in risk_on
        probs1 = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.30, 'risk_on': 0.70}
        ts1 = pd.Timestamp('2024-01-01 00:00:00')
        hysteresis.apply_hysteresis('risk_on', probs1, ts1)

        # Try to switch after 25 hours (> 24h min)
        probs2 = {'crisis': 0.10, 'risk_off': 0.80, 'neutral': 0.30, 'risk_on': 0.10}
        ts2 = pd.Timestamp('2024-01-02 01:00:00')
        result = hysteresis.apply_hysteresis('risk_off', probs2, ts2)

        assert result['regime'] == 'risk_off'  # Transition allowed
        assert result['transition'] is True
        assert hysteresis.transition_count == 1

    def test_per_regime_dwell_times(self):
        """Test that different regimes have different minimum dwell times."""
        config = {
            'enter_threshold': 0.65,
            'exit_threshold': 0.50,
            'min_duration_hours': {
                'crisis': 6,
                'risk_off': 24,
                'neutral': 12,
                'risk_on': 48
            }
        }
        hysteresis = RegimeHysteresis(config)

        # Test crisis (6h minimum)
        probs1 = {'crisis': 0.80, 'risk_off': 0.10, 'neutral': 0.05, 'risk_on': 0.05}
        ts1 = pd.Timestamp('2024-01-01 00:00:00')
        hysteresis.apply_hysteresis('crisis', probs1, ts1)

        # Try to switch after 3 hours (< 6h)
        probs2 = {'crisis': 0.10, 'risk_off': 0.80, 'neutral': 0.05, 'risk_on': 0.05}
        ts2 = pd.Timestamp('2024-01-01 03:00:00')
        result = hysteresis.apply_hysteresis('risk_off', probs2, ts2)

        assert result['regime'] == 'crisis'  # Locked
        assert result['dwell_time'] == 3.0

        # Switch after 7 hours (> 6h)
        ts3 = pd.Timestamp('2024-01-01 07:00:00')
        result2 = hysteresis.apply_hysteresis('risk_off', probs2, ts3)

        assert result2['regime'] == 'risk_off'  # Transition allowed
        assert result2['transition'] is True


class TestEWMASmoothing:
    """Test EWMA probability smoothing."""

    def test_ewma_disabled_by_default(self):
        """Test that EWMA is disabled by default."""
        hysteresis = RegimeHysteresis()
        assert hysteresis.enable_ewma is False
        assert hysteresis.ewma_probs is None

    def test_ewma_smoothing_effect(self):
        """Test that EWMA smooths probabilities over time."""
        config = {'enable_ewma': True, 'ewma_alpha': 0.5, 'min_duration_hours': {'risk_on': 0}}
        hysteresis = RegimeHysteresis(config)

        # First call - should return raw probabilities
        probs1 = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.30, 'risk_on': 0.40}
        ts1 = pd.Timestamp('2024-01-01 00:00:00')
        result1 = hysteresis.apply_hysteresis('risk_on', probs1, ts1)

        # Second call - should smooth
        probs2 = {'crisis': 0.20, 'risk_off': 0.30, 'neutral': 0.40, 'risk_on': 0.10}
        ts2 = pd.Timestamp('2024-01-01 01:00:00')
        result2 = hysteresis.apply_hysteresis('risk_on', probs2, ts2)

        # Check that smoothing occurred (alpha=0.5, so average of prev and current)
        smoothed = result2['probs']
        # Expected: 0.5 * 0.20 + 0.5 * 0.10 = 0.15 for crisis (then normalized)
        assert smoothed['crisis'] > 0.10  # Should be between old (0.10) and new (0.20)
        assert smoothed['crisis'] < 0.20

    def test_ewma_normalization(self):
        """Test that EWMA probabilities sum to 1.0."""
        config = {'enable_ewma': True, 'ewma_alpha': 0.3, 'min_duration_hours': {'risk_on': 0}}
        hysteresis = RegimeHysteresis(config)

        probs = {'crisis': 0.25, 'risk_off': 0.25, 'neutral': 0.25, 'risk_on': 0.25}
        ts = pd.Timestamp('2024-01-01 00:00:00')

        for i in range(5):
            result = hysteresis.apply_hysteresis('risk_on', probs, ts + pd.Timedelta(hours=i))
            prob_sum = sum(result['probs'].values())
            assert abs(prob_sum - 1.0) < 1e-6  # Should sum to 1.0


class TestStateManagement:
    """Test state management and reset functionality."""

    def test_reset_clears_state(self):
        """Test that reset() clears all state."""
        config = {'min_duration_hours': {'risk_on': 0}}
        hysteresis = RegimeHysteresis(config)

        # Initialize state
        probs = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.30, 'risk_on': 0.70}
        ts = pd.Timestamp('2024-01-01 00:00:00')
        hysteresis.apply_hysteresis('risk_on', probs, ts)

        # Verify state exists
        assert hysteresis.current_regime == 'risk_on'
        assert hysteresis.regime_start_time is not None

        # Reset
        hysteresis.reset()

        # Verify state cleared
        assert hysteresis.current_regime is None
        assert hysteresis.regime_start_time is None
        assert hysteresis.transition_count == 0
        assert len(hysteresis.transition_history) == 0

    def test_get_statistics(self):
        """Test get_statistics() returns correct information."""
        config = {'enter_threshold': 0.70, 'min_duration_hours': {'risk_on': 24}}
        hysteresis = RegimeHysteresis(config)

        # Initialize
        probs = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.30, 'risk_on': 0.80}
        ts = pd.Timestamp('2024-01-01 00:00:00')
        hysteresis.apply_hysteresis('risk_on', probs, ts)

        stats = hysteresis.get_statistics()

        assert stats['current_regime'] == 'risk_on'
        assert stats['total_transitions'] == 0  # Initial state, no transitions yet
        assert stats['config']['enter_threshold'] == 0.70
        assert stats['config']['min_duration_hours']['risk_on'] == 24


class TestTransitionTracking:
    """Test transition tracking functionality."""

    def test_transition_count_increments(self):
        """Test that transition_count increments correctly."""
        config = {'enter_threshold': 0.65, 'exit_threshold': 0.50, 'min_duration_hours': {'risk_on': 0, 'risk_off': 0}}
        hysteresis = RegimeHysteresis(config)

        # Initialize
        probs1 = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.30, 'risk_on': 0.70}
        ts1 = pd.Timestamp('2024-01-01 00:00:00')
        hysteresis.apply_hysteresis('risk_on', probs1, ts1)

        assert hysteresis.transition_count == 0  # Initial state

        # Transition 1
        probs2 = {'crisis': 0.10, 'risk_off': 0.80, 'neutral': 0.30, 'risk_on': 0.10}
        ts2 = pd.Timestamp('2024-01-01 01:00:00')
        hysteresis.apply_hysteresis('risk_off', probs2, ts2)

        assert hysteresis.transition_count == 1

        # Transition 2
        probs3 = {'crisis': 0.10, 'risk_off': 0.10, 'neutral': 0.30, 'risk_on': 0.70}
        ts3 = pd.Timestamp('2024-01-01 02:00:00')
        hysteresis.apply_hysteresis('risk_on', probs3, ts3)

        assert hysteresis.transition_count == 2

    def test_transition_history_records(self):
        """Test that transition history is recorded correctly."""
        config = {'enter_threshold': 0.65, 'exit_threshold': 0.50, 'min_duration_hours': {'risk_on': 0, 'risk_off': 0}}
        hysteresis = RegimeHysteresis(config)

        # Initialize
        probs1 = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.30, 'risk_on': 0.70}
        ts1 = pd.Timestamp('2024-01-01 00:00:00')
        hysteresis.apply_hysteresis('risk_on', probs1, ts1)

        # Transition
        probs2 = {'crisis': 0.10, 'risk_off': 0.80, 'neutral': 0.30, 'risk_on': 0.10}
        ts2 = pd.Timestamp('2024-01-01 01:00:00')
        hysteresis.apply_hysteresis('risk_off', probs2, ts2)

        assert len(hysteresis.transition_history) == 1
        timestamp, old_regime, new_regime = hysteresis.transition_history[0]
        assert timestamp == ts2
        assert old_regime == 'risk_on'
        assert new_regime == 'risk_off'

    def test_validate_transitions_per_year(self):
        """Test transitions per year validation."""
        config = {'enter_threshold': 0.65, 'exit_threshold': 0.50, 'min_duration_hours': {'risk_on': 0, 'neutral': 0}}
        hysteresis = RegimeHysteresis(config)

        # Simulate 20 transitions over 1 year
        start_date = pd.Timestamp('2024-01-01 00:00:00')
        end_date = pd.Timestamp('2024-12-31 23:59:59')

        for i in range(20):
            regime = 'risk_on' if i % 2 == 0 else 'neutral'
            probs = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.70 if regime == 'neutral' else 0.30, 'risk_on': 0.70 if regime == 'risk_on' else 0.30}
            ts = start_date + pd.Timedelta(days=i * 18)
            hysteresis.apply_hysteresis(regime, probs, ts)

        result = hysteresis.validate_transitions_per_year(start_date, end_date)

        assert result['valid'] is True
        assert result['status'] == 'OPTIMAL'  # 19 transitions (20 calls - 1 init) is within 10-40
        assert 10 <= result['transitions_per_year'] <= 40


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_no_timestamp_provided(self):
        """Test behavior when no timestamp is provided."""
        hysteresis = RegimeHysteresis()
        probs = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.30, 'risk_on': 0.70}

        result = hysteresis.apply_hysteresis('risk_on', probs, timestamp=None)

        # Should work but dwell_time will be 0.0
        assert result['regime'] == 'risk_on'
        assert result['dwell_time'] == 0.0

    def test_empty_probabilities_dict(self):
        """Test behavior with empty probabilities."""
        hysteresis = RegimeHysteresis()
        probs = {}
        ts = pd.Timestamp('2024-01-01 00:00:00')

        result = hysteresis.apply_hysteresis('risk_on', probs, ts)

        # Should handle gracefully (prob defaults to 0.0)
        assert result['regime'] == 'risk_on'

    def test_missing_regime_in_probs(self):
        """Test behavior when proposed regime not in probs dict."""
        config = {'min_duration_hours': {'risk_on': 0}}
        hysteresis = RegimeHysteresis(config)

        # Initialize
        probs1 = {'risk_off': 0.50, 'neutral': 0.50}  # Missing risk_on
        ts1 = pd.Timestamp('2024-01-01 00:00:00')
        result = hysteresis.apply_hysteresis('risk_on', probs1, ts1)

        # Should default prob to 0.0 and handle gracefully
        assert result['regime'] == 'risk_on'

    def test_negative_dwell_time_protection(self):
        """Test protection against negative dwell times."""
        hysteresis = RegimeHysteresis()

        # Initialize
        probs = {'crisis': 0.10, 'risk_off': 0.20, 'neutral': 0.30, 'risk_on': 0.70}
        ts1 = pd.Timestamp('2024-01-01 12:00:00')
        hysteresis.apply_hysteresis('risk_on', probs, ts1)

        # Call with earlier timestamp (should not crash)
        ts2 = pd.Timestamp('2024-01-01 10:00:00')
        result = hysteresis.apply_hysteresis('risk_on', probs, ts2)

        assert result['dwell_time'] >= 0.0  # Should be non-negative


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
