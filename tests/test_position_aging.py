"""
Unit tests for position aging functionality.

Tests the exact bugs we killed in the position aging system:
- Positions properly age over time (bars_held increments)
- Time stops trigger at correct thresholds
- Scale-ins preserve original entry time
"""

import unittest
import pandas as pd
from unittest.mock import Mock, patch
import tempfile
import os

from bull_machine.backtest.broker import PaperBroker, Position
from bull_machine.strategy.exits.rules import TimeStopEvaluator
from bull_machine.strategy.exits.types import ExitSignal, ExitType


class TestPositionAging(unittest.TestCase):
    """Test position aging system works correctly."""

    def setUp(self):
        """Set up test fixtures."""
        self.broker = PaperBroker(fee_bps=10, slippage_bps=7, spread_bps=2)
        # Set broker context for aging tests
        self.broker._current_bar_idx = 0
        self.broker._current_timeframe = '1H'

    def test_bars_held_increments(self):
        """Test that bars_held increments correctly over time."""
        # Open a position
        fill = self.broker.submit(
            ts=pd.Timestamp('2025-01-01 10:00'),
            symbol='BTCUSD_1H',
            side='long',
            size=1.0,
            price_hint=50000.0
        )

        assert fill is not None
        assert 'BTCUSD_1H' in self.broker.positions

        # Initial position should have bars_held=0
        pos = self.broker.positions['BTCUSD_1H']
        assert pos.bars_held == 0
        assert pos.opened_at_idx is not None

        # Advance 6 bars and update aging
        for i in range(1, 7):
            self.broker.update_position_aging(current_bar_idx=pos.opened_at_idx + i, timeframe='1H')
            updated_pos = self.broker.positions['BTCUSD_1H']
            assert updated_pos.bars_held == i, f"Expected bars_held={i}, got {updated_pos.bars_held}"

    def test_scale_ins_preserve_opened_at(self):
        """Test that scale-ins preserve original opened_at fields."""
        # Open initial position
        fill1 = self.broker.submit(
            ts=pd.Timestamp('2025-01-01 10:00'),
            symbol='BTCUSD_1H',
            side='long',
            size=1.0,
            price_hint=50000.0
        )

        pos = self.broker.positions['BTCUSD_1H']
        original_opened_at_ts = pos.opened_at_ts
        original_opened_at_idx = pos.opened_at_idx
        original_entry_price = pos.entry

        # Age the position a few bars
        self.broker.update_position_aging(current_bar_idx=pos.opened_at_idx + 3, timeframe='1H')

        # Scale in (add to position)
        fill2 = self.broker.submit(
            ts=pd.Timestamp('2025-01-01 13:00'),
            symbol='BTCUSD_1H',
            side='long',
            size=0.5,
            price_hint=51000.0
        )

        # Check that opened_at fields are preserved
        pos_after_scale = self.broker.positions['BTCUSD_1H']
        assert pos_after_scale.opened_at_ts == original_opened_at_ts
        assert pos_after_scale.opened_at_idx == original_opened_at_idx
        assert pos_after_scale.bars_held == 3  # Should preserve aging
        assert pos_after_scale.size == 1.5  # Size should update
        # Entry price should be weighted average (accounting for broker costs)
        # We need to calculate expected average using actual fill prices after costs
        second_fill_price = 51000.0 + (7 + 2) * 1e-4 * 51000.0  # slippage + spread
        expected_avg = (1.0 * original_entry_price + 0.5 * second_fill_price) / 1.5
        assert abs(pos_after_scale.entry - expected_avg) < 1.0  # Allow for rounding differences

    def test_get_position_data_includes_aging_fields(self):
        """Test that get_position_data includes all aging fields."""
        # Open a position
        self.broker.submit(
            ts=pd.Timestamp('2025-01-01 10:00'),
            symbol='BTCUSD_1H',
            side='long',
            size=1.0,
            price_hint=50000.0
        )

        # Age it a few bars
        pos = self.broker.positions['BTCUSD_1H']
        self.broker.update_position_aging(current_bar_idx=pos.opened_at_idx + 5, timeframe='1H')

        # Get position data
        position_data = self.broker.get_position_data('BTCUSD_1H')

        assert position_data is not None
        assert 'bars_held' in position_data
        assert 'timeframe' in position_data
        assert 'opened_at_ts' in position_data
        assert 'opened_at_idx' in position_data
        assert position_data['bars_held'] == 5
        assert position_data['timeframe'] == '1H'


class TestTimeStopTriggers(unittest.TestCase):
    """Test that time stops trigger at correct thresholds."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = {
            'max_bars_1h': 6,
            'max_bars_4h': 12,
            'max_bars_1d': 6
        }
        self.evaluator = TimeStopEvaluator(self.config)

    def test_time_stop_triggers_at_threshold(self):
        """Test that time stop triggers when bars_held >= max_bars."""
        # Position data at exactly the threshold
        position_data = {
            'symbol': 'BTCUSD_1H',
            'bars_held': 6,
            'timeframe': '1H',
            'pnl_pct': -0.01
        }

        result = self.evaluator.evaluate('BTCUSD_1H', position_data, pd.Timestamp('2025-01-01 10:00'))

        assert result is not None
        assert isinstance(result, ExitSignal)
        assert result.exit_type == ExitType.TIME_STOP
        assert result.confidence > 0.5

    def test_time_stop_does_not_trigger_below_threshold(self):
        """Test that time stop does not trigger below threshold."""
        # Position data below threshold
        position_data = {
            'symbol': 'BTCUSD_1H',
            'bars_held': 4,
            'timeframe': '1H',
            'pnl_pct': -0.01
        }

        result = self.evaluator.evaluate('BTCUSD_1H', position_data, pd.Timestamp('2025-01-01 10:00'))

        assert result is None

    def test_time_stop_backward_compatibility(self):
        """Test backward compatibility with legacy bars_max config."""
        # Legacy config with bars_max
        legacy_config = {'bars_max': 10}
        evaluator = TimeStopEvaluator(legacy_config)

        position_data = {
            'symbol': 'BTCUSD_1H',
            'bars_held': 10,
            'timeframe': '1H',
            'pnl_pct': -0.01
        }

        result = evaluator.evaluate('BTCUSD_1H', position_data, pd.Timestamp('2025-01-01 10:00'))

        assert result is not None
        assert isinstance(result, ExitSignal)


class TestRegressionEntryExitFlow(unittest.TestCase):
    """Regression test for complete entry-exit flow."""

    def test_complete_entry_exit_flow(self):
        """Test that we can complete a full entry->aging->exit cycle."""
        # Create a minimal config
        config = {
            'run_id': 'test',
            'symbols': ['BTCUSD_1H'],
            'timeframes': ['1H'],
            'strategy': {'enter_threshold': 0.3},
            'exit_signals': {
                'time_stop': {'max_bars_1h': 3}
            },
            'risk': {'base_risk_pct': 0.01},
            'engine': {'lookback_bars': 10}
        }

        # Create sample data
        dates = pd.date_range('2025-01-01 00:00', periods=20, freq='h')
        df = pd.DataFrame({
            'timestamp': dates,
            'open': [50000.0] * 20,
            'high': [50100.0] * 20,
            'low': [49900.0] * 20,
            'close': [50000.0] * 20,
            'volume': [1000.0] * 20
        })
        df.set_index('timestamp', inplace=True)

        # Mock components
        broker = PaperBroker(fee_bps=10, slippage_bps=5, spread_bps=2)

        # Set current bar index context for broker
        broker._current_bar_idx = 5
        broker._current_timeframe = '1H'

        # Simulate entry
        fill = broker.submit(
            ts=dates[5],
            symbol='BTCUSD_1H',
            side='long',
            size=1.0,
            price_hint=50000.0
        )

        assert fill is not None
        assert 'BTCUSD_1H' in broker.positions

        # Simulate aging over several bars
        pos = broker.positions['BTCUSD_1H']
        opened_at_idx = pos.opened_at_idx or 5

        # Age position to trigger time stop
        for i in range(1, 5):
            broker.update_position_aging(current_bar_idx=opened_at_idx + i, timeframe='1H')

        # Verify position aged correctly
        aged_pos = broker.positions['BTCUSD_1H']
        assert aged_pos.bars_held == 4

        # Test that position data includes aging fields
        position_data = broker.get_position_data('BTCUSD_1H')
        assert position_data['bars_held'] == 4

        # Test time stop would trigger
        evaluator = TimeStopEvaluator({'max_bars_1h': 3})
        exit_signal = evaluator.evaluate('BTCUSD_1H', position_data, dates[10])

        assert exit_signal is not None
        assert exit_signal.exit_type == ExitType.TIME_STOP


if __name__ == '__main__':
    unittest.main()