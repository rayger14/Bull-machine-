"""
Unit tests for archetype-specific exit logic system.

Tests cover:
1. Invalidation exits (pattern breaks)
2. Scale-out exits (R-multiple targets)
3. Time-based exits (max hold period)
4. Reason-gone exits (entry condition reversal)
5. Trailing stop updates
"""

import pytest
import pandas as pd
from datetime import datetime, timedelta

from engine.archetypes.exit_logic import (
    ExitLogic,
    ExitSignal,
    ExitType,
    create_default_exit_config
)
from engine.models.base import Position
from engine.runtime.context import RuntimeContext


class TestExitLogicInitialization:
    """Test exit logic initialization and configuration."""

    def test_initialization_with_default_config(self):
        """Test initialization with default configuration."""
        config = create_default_exit_config()
        exit_logic = ExitLogic(config)

        assert exit_logic.enable_scale_outs is True
        assert exit_logic.enable_time_exits is True
        assert exit_logic.enable_trailing is True
        assert len(exit_logic.exit_rules) > 0

    def test_archetype_rules_loaded(self):
        """Test that archetype-specific rules are loaded."""
        config = create_default_exit_config()
        exit_logic = ExitLogic(config)

        # Check S1 rules
        s1_rules = exit_logic.exit_rules.get('liquidity_vacuum')
        assert s1_rules is not None
        assert s1_rules['max_hold_hours'] == 120
        assert s1_rules['trailing_start_r'] == 0.5

        # Check S4 rules
        s4_rules = exit_logic.exit_rules.get('funding_divergence')
        assert s4_rules is not None
        assert s4_rules['max_hold_hours'] == 240

    def test_custom_config_override(self):
        """Test custom configuration overrides."""
        config = create_default_exit_config()
        config['exit_rules']['liquidity_vacuum']['max_hold_hours'] = 96

        exit_logic = ExitLogic(config)
        assert exit_logic.exit_rules['liquidity_vacuum']['max_hold_hours'] == 96


class TestInvalidationExits:
    """Test pattern invalidation exit conditions."""

    @pytest.fixture
    def exit_logic(self):
        """Create exit logic instance."""
        return ExitLogic(create_default_exit_config())

    @pytest.fixture
    def context(self):
        """Create mock runtime context."""
        bar = pd.Series({
            'close': 50000.0,
            'atr_14': 500.0
        }, name=pd.Timestamp('2024-01-01 12:00'))

        return RuntimeContext(
            ts=bar.name,
            row=bar,
            regime_probs={'risk_off': 1.0},
            regime_label='risk_off',
            adapted_params={},
            thresholds={}
        )

    def test_s1_invalidation_previous_low_break(self, exit_logic, context):
        """S1 should exit when previous low is taken out."""
        position = Position(
            direction='long',
            entry_price=50000.0,
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            size=1000.0,
            stop_loss=49000.0,
            metadata={'entry_prev_low': 49500.0}
        )

        bar = pd.Series({
            'close': 49000.0,  # Below previous low
            'atr_14': 500.0
        }, name=pd.Timestamp('2024-01-01 12:00'))

        exit_signal = exit_logic.check_exit(
            bar=bar,
            position=position,
            archetype='liquidity_vacuum',
            context=context
        )

        assert exit_signal is not None
        assert exit_signal.exit_type == ExitType.INVALIDATION.value
        assert exit_signal.exit_pct == 1.0
        assert 'S1 invalidation' in exit_signal.reason

    def test_s4_funding_flip_invalidation(self, exit_logic, context):
        """S4 should exit when funding normalizes."""
        position = Position(
            direction='short',
            entry_price=50000.0,
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            size=1000.0,
            stop_loss=51000.0,
            metadata={'entry_funding_z': -6.0}  # Extreme negative
        )

        bar = pd.Series({
            'close': 49500.0,
            'atr_14': 500.0,
            'funding_z_score': -0.5  # Normalized
        }, name=pd.Timestamp('2024-01-01 12:00'))

        exit_signal = exit_logic.check_exit(
            bar=bar,
            position=position,
            archetype='funding_divergence',
            context=context
        )

        assert exit_signal is not None
        assert exit_signal.exit_type == ExitType.INVALIDATION.value
        assert 'funding normalized' in exit_signal.reason.lower()

    def test_s5_oi_rebound_invalidation(self, exit_logic, context):
        """S5 should exit when OI rebounds after entry on drop."""
        position = Position(
            direction='short',
            entry_price=50000.0,
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            size=1000.0,
            stop_loss=51000.0,
            metadata={'entry_oi_delta': -12.0}  # Entered on OI drop
        )

        bar = pd.Series({
            'close': 49500.0,
            'atr_14': 500.0,
            'oi_delta_pct': 7.0  # OI rebounded
        }, name=pd.Timestamp('2024-01-01 12:00'))

        exit_signal = exit_logic.check_exit(
            bar=bar,
            position=position,
            archetype='long_squeeze',
            context=context
        )

        assert exit_signal is not None
        assert exit_signal.exit_type == ExitType.INVALIDATION.value
        assert 'OI rebounded' in exit_signal.reason


class TestProfitTargetExits:
    """Test profit target and scale-out exit conditions."""

    @pytest.fixture
    def exit_logic(self):
        return ExitLogic(create_default_exit_config())

    @pytest.fixture
    def context(self):
        bar = pd.Series({'close': 50000.0, 'atr_14': 500.0},
                       name=pd.Timestamp('2024-01-01 12:00'))
        return RuntimeContext(
            ts=bar.name,
            row=bar,
            regime_probs={'risk_off': 1.0},
            regime_label='risk_off',
            adapted_params={},
            thresholds={}
        )

    def test_scale_out_at_05r(self, exit_logic, context):
        """Test scale-out at +0.5R."""
        position = Position(
            direction='long',
            entry_price=50000.0,
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            size=1000.0,
            stop_loss=49000.0,  # 1000 risk
            metadata={'executed_scale_outs': []}
        )

        # Price at +0.6R (50000 + 600 = 50600)
        bar = pd.Series({
            'close': 50600.0,
            'atr_14': 500.0
        }, name=pd.Timestamp('2024-01-01 12:00'))

        exit_signal = exit_logic.check_exit(
            bar=bar,
            position=position,
            archetype='liquidity_vacuum',
            context=context
        )

        assert exit_signal is not None
        assert exit_signal.exit_type == ExitType.PROFIT_TARGET.value
        assert exit_signal.exit_pct == 0.2  # 20% scale-out
        assert 0.5 in position.metadata['executed_scale_outs']

    def test_scale_out_at_10r(self, exit_logic, context):
        """Test scale-out at +1.0R."""
        position = Position(
            direction='long',
            entry_price=50000.0,
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            size=1000.0,
            stop_loss=49000.0,
            metadata={'executed_scale_outs': [0.5]}  # Already scaled at 0.5R
        )

        # Price at +1.2R
        bar = pd.Series({
            'close': 51200.0,
            'atr_14': 500.0
        }, name=pd.Timestamp('2024-01-01 12:00'))

        exit_signal = exit_logic.check_exit(
            bar=bar,
            position=position,
            archetype='liquidity_vacuum',
            context=context
        )

        assert exit_signal is not None
        assert exit_signal.exit_pct == 0.2  # Second 20% scale-out
        assert 1.0 in position.metadata['executed_scale_outs']

    def test_no_duplicate_scale_outs(self, exit_logic, context):
        """Test that same R-level doesn't trigger multiple times."""
        position = Position(
            direction='long',
            entry_price=50000.0,
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            size=1000.0,
            stop_loss=49000.0,
            metadata={'executed_scale_outs': [0.5, 1.0]}  # Already scaled
        )

        # Price still at +1.2R
        bar = pd.Series({
            'close': 51200.0,
            'atr_14': 500.0
        }, name=pd.Timestamp('2024-01-01 12:00'))

        exit_signal = exit_logic.check_exit(
            bar=bar,
            position=position,
            archetype='liquidity_vacuum',
            context=context
        )

        # Should not trigger profit target (already executed)
        assert exit_signal is None or exit_signal.exit_type != ExitType.PROFIT_TARGET.value


class TestTimeBasedExits:
    """Test max hold period exit conditions."""

    @pytest.fixture
    def exit_logic(self):
        return ExitLogic(create_default_exit_config())

    @pytest.fixture
    def context(self):
        bar = pd.Series({'close': 50000.0, 'atr_14': 500.0},
                       name=pd.Timestamp('2024-01-01 12:00'))
        return RuntimeContext(
            ts=bar.name,
            row=bar,
            regime_probs={'risk_off': 1.0},
            regime_label='risk_off',
            adapted_params={},
            thresholds={}
        )

    def test_time_exit_max_hold_exceeded(self, exit_logic, context):
        """Test exit when max hold period exceeded."""
        # S1 max hold: 120 hours
        entry_time = pd.Timestamp('2024-01-01 10:00')
        current_time = entry_time + timedelta(hours=125)

        position = Position(
            direction='long',
            entry_price=50000.0,
            entry_time=entry_time,
            size=1000.0,
            stop_loss=49000.0,
            metadata={}
        )

        bar = pd.Series({
            'close': 50500.0,
            'atr_14': 500.0
        }, name=current_time)

        exit_signal = exit_logic.check_exit(
            bar=bar,
            position=position,
            archetype='liquidity_vacuum',
            context=context
        )

        assert exit_signal is not None
        assert exit_signal.exit_type == ExitType.TIME_EXIT.value
        assert exit_signal.exit_pct == 1.0

    def test_no_time_exit_within_max_hold(self, exit_logic, context):
        """Test no exit when within max hold period."""
        entry_time = pd.Timestamp('2024-01-01 10:00')
        current_time = entry_time + timedelta(hours=100)  # < 120h

        position = Position(
            direction='long',
            entry_price=50000.0,
            entry_time=entry_time,
            size=1000.0,
            stop_loss=49000.0,
            metadata={}
        )

        bar = pd.Series({
            'close': 50500.0,
            'atr_14': 500.0
        }, name=current_time)

        exit_signal = exit_logic.check_exit(
            bar=bar,
            position=position,
            archetype='liquidity_vacuum',
            context=context
        )

        # Should not trigger time exit
        assert exit_signal is None or exit_signal.exit_type != ExitType.TIME_EXIT.value


class TestReasonGoneExits:
    """Test reason-gone exit conditions."""

    @pytest.fixture
    def exit_logic(self):
        return ExitLogic(create_default_exit_config())

    @pytest.fixture
    def context(self):
        bar = pd.Series({'close': 50000.0, 'atr_14': 500.0},
                       name=pd.Timestamp('2024-01-01 12:00'))
        return RuntimeContext(
            ts=bar.name,
            row=bar,
            regime_probs={'risk_off': 1.0},
            regime_label='risk_off',
            adapted_params={},
            thresholds={}
        )

    def test_volume_fade_exit(self, exit_logic, context):
        """Test exit on volume fade for volume archetypes."""
        position = Position(
            direction='long',
            entry_price=50000.0,
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            size=1000.0,
            stop_loss=49000.0,
            metadata={'entry_volume': 1000000}
        )

        bar = pd.Series({
            'close': 50500.0,
            'atr_14': 500.0,
            'volume': 400000  # < 50% of entry volume
        }, name=pd.Timestamp('2024-01-01 12:00'))

        exit_signal = exit_logic.check_exit(
            bar=bar,
            position=position,
            archetype='volume_exhaustion',
            context=context
        )

        assert exit_signal is not None
        assert exit_signal.exit_type == ExitType.REASON_GONE.value
        assert 'volume fade' in exit_signal.reason.lower()

    def test_momentum_fade_partial_exit(self, exit_logic, context):
        """Test partial exit on momentum fade."""
        position = Position(
            direction='long',
            entry_price=50000.0,
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            size=1000.0,
            stop_loss=49000.0,
            metadata={'entry_adx': 32.0}
        )

        bar = pd.Series({
            'close': 50500.0,
            'atr_14': 500.0,
            'adx_14': 22.0  # Below 25 threshold
        }, name=pd.Timestamp('2024-01-01 12:00'))

        exit_signal = exit_logic.check_exit(
            bar=bar,
            position=position,
            archetype='trap_within_trend',
            context=context
        )

        assert exit_signal is not None
        assert exit_signal.exit_type == ExitType.REASON_GONE.value
        assert exit_signal.exit_pct == 0.5  # 50% partial exit


class TestTrailingStopUpdates:
    """Test trailing stop logic."""

    @pytest.fixture
    def exit_logic(self):
        return ExitLogic(create_default_exit_config())

    @pytest.fixture
    def context(self):
        bar = pd.Series({'close': 50000.0, 'atr_14': 500.0},
                       name=pd.Timestamp('2024-01-01 12:00'))
        return RuntimeContext(
            ts=bar.name,
            row=bar,
            regime_probs={'risk_off': 1.0},
            regime_label='risk_off',
            adapted_params={},
            thresholds={}
        )

    def test_trailing_activates_after_threshold(self, exit_logic, context):
        """Test trailing stop activates after reaching +0.5R for S1."""
        position = Position(
            direction='long',
            entry_price=50000.0,
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            size=1000.0,
            stop_loss=49000.0,
            metadata={}
        )

        # Price at +0.6R
        bar = pd.Series({
            'close': 50600.0,
            'atr_14': 500.0
        }, name=pd.Timestamp('2024-01-01 12:00'))

        original_stop = position.stop_loss

        exit_logic.check_exit(
            bar=bar,
            position=position,
            archetype='liquidity_vacuum',
            context=context
        )

        # Stop should be updated (trailing active)
        # New stop = 50600 - (2.0 * 500) = 49600
        assert position.stop_loss > original_stop

    def test_trailing_only_moves_up(self, exit_logic, context):
        """Test trailing stop only moves up, never down."""
        position = Position(
            direction='long',
            entry_price=50000.0,
            entry_time=pd.Timestamp('2024-01-01 10:00'),
            size=1000.0,
            stop_loss=49800.0,  # Already trailed
            metadata={}
        )

        # Price dropped slightly
        bar = pd.Series({
            'close': 50200.0,
            'atr_14': 500.0
        }, name=pd.Timestamp('2024-01-01 12:00'))

        original_stop = position.stop_loss

        exit_logic.check_exit(
            bar=bar,
            position=position,
            archetype='liquidity_vacuum',
            context=context
        )

        # Stop should NOT move down
        assert position.stop_loss == original_stop


class TestExitPriority:
    """Test exit priority ordering."""

    @pytest.fixture
    def exit_logic(self):
        return ExitLogic(create_default_exit_config())

    @pytest.fixture
    def context(self):
        bar = pd.Series({'close': 50000.0, 'atr_14': 500.0},
                       name=pd.Timestamp('2024-01-01 12:00'))
        return RuntimeContext(
            ts=bar.name,
            row=bar,
            regime_probs={'risk_off': 1.0},
            regime_label='risk_off',
            adapted_params={},
            thresholds={}
        )

    def test_invalidation_overrides_profit_target(self, exit_logic, context):
        """Invalidation should take priority over profit targets."""
        entry_time = pd.Timestamp('2024-01-01 10:00')

        position = Position(
            direction='long',
            entry_price=50000.0,
            entry_time=entry_time,
            size=1000.0,
            stop_loss=49000.0,
            metadata={
                'entry_prev_low': 49500.0,
                'executed_scale_outs': []
            }
        )

        # Price is profitable (+1R) BUT also broke previous low
        bar = pd.Series({
            'close': 49200.0,  # Below previous low (invalidation)
            'atr_14': 500.0
        }, name=pd.Timestamp('2024-01-01 12:00'))

        exit_signal = exit_logic.check_exit(
            bar=bar,
            position=position,
            archetype='liquidity_vacuum',
            context=context
        )

        # Should trigger invalidation, NOT profit target
        assert exit_signal.exit_type == ExitType.INVALIDATION.value


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
