"""
Exit Signal Evaluators
Coordinates multiple exit detection systems and provides unified evaluation.
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List
import logging

from .types import ExitSignal, ExitEvaluationResult
from .rules import CHoCHAgainstDetector, MomentumFadeDetector, TimeStopEvaluator


class ExitSignalEvaluator:
    """
    Main coordinator for all exit signal detection.
    Evaluates multiple exit conditions and prioritizes signals.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize with configuration for all exit detectors.

        Args:
            config: Exit system configuration
        """
        self.config = config

        # Initialize detectors
        choch_config = config.get('choch_against', {})
        momentum_config = config.get('momentum_fade', {})
        time_config = config.get('time_stop', {})

        self.choch_detector = CHoCHAgainstDetector(choch_config)
        self.momentum_detector = MomentumFadeDetector(momentum_config)
        self.time_evaluator = TimeStopEvaluator(time_config)

        # Global exit settings
        self.enabled_exits = config.get('enabled_exits', [
            'choch_against', 'momentum_fade', 'time_stop'
        ])
        self.min_confidence = config.get('min_confidence', 0.6)
        self.priority_order = config.get('priority_order', [
            'choch_against', 'momentum_fade', 'time_stop'
        ])

        logging.info(f"ExitSignalEvaluator initialized with exits: {self.enabled_exits}")

    def evaluate_exits(self, symbol: str, position_data: Dict[str, Any],
                      mtf_data: Dict[str, pd.DataFrame],
                      current_bar: pd.Timestamp) -> ExitEvaluationResult:
        """
        Evaluate all exit conditions for a position.

        Args:
            symbol: Trading symbol
            position_data: Current position information including bias, entry_time, pnl_pct
            mtf_data: Multi-timeframe OHLCV data
            current_bar: Current timestamp

        Returns:
            ExitEvaluationResult with all detected signals
        """
        result = ExitEvaluationResult()
        position_bias = position_data.get('bias', 'long')

        try:
            # 1. CHoCH-Against Detection
            if 'choch_against' in self.enabled_exits:
                choch_signal = self.choch_detector.evaluate(
                    symbol, position_bias, mtf_data, current_bar
                )
                if choch_signal and choch_signal.confidence >= self.min_confidence:
                    result.add_signal(choch_signal)
                    logging.debug(f"CHoCH-Against signal: {choch_signal.confidence:.2f}")

            # 2. Momentum Fade Detection
            if 'momentum_fade' in self.enabled_exits:
                momentum_signal = self.momentum_detector.evaluate(
                    symbol, position_bias, mtf_data, current_bar
                )
                if momentum_signal and momentum_signal.confidence >= self.min_confidence:
                    result.add_signal(momentum_signal)
                    logging.debug(f"Momentum fade signal: {momentum_signal.confidence:.2f}")

            # 3. Time Stop Evaluation
            if 'time_stop' in self.enabled_exits:
                time_signal = self.time_evaluator.evaluate(
                    symbol, position_data, current_bar
                )
                if time_signal and time_signal.confidence >= self.min_confidence:
                    result.add_signal(time_signal)
                    logging.debug(f"Time stop signal: {time_signal.confidence:.2f}")

        except Exception as e:
            logging.error(f"Exit evaluation error for {symbol}: {e}")

        # Log results
        if result.has_signals():
            logging.info(f"Exit signals for {symbol}: {len(result.signals)} signals, "
                        f"max confidence: {result.max_confidence:.2f}")
        else:
            logging.debug(f"No exit signals for {symbol}")

        return result

    def get_action_recommendation(self, result: ExitEvaluationResult) -> Optional[ExitSignal]:
        """
        Get the recommended action from evaluation results.
        Prioritizes signals based on urgency and type priority.

        Args:
            result: Exit evaluation results

        Returns:
            Recommended ExitSignal or None
        """
        if not result.has_signals():
            return None

        # Sort signals by priority and urgency
        prioritized_signals = self._prioritize_signals(result.signals)

        if prioritized_signals:
            recommended = prioritized_signals[0]
            logging.info(f"Recommended exit: {recommended.exit_type.value} "
                        f"(confidence: {recommended.confidence:.2f}, "
                        f"urgency: {recommended.urgency:.2f})")
            return recommended

        return None

    def _prioritize_signals(self, signals: List[ExitSignal]) -> List[ExitSignal]:
        """
        Prioritize signals based on type priority and urgency.

        Args:
            signals: List of exit signals

        Returns:
            Sorted list with highest priority first
        """
        def signal_priority_score(signal: ExitSignal) -> float:
            # Type priority (lower index = higher priority)
            type_priority = 1.0
            try:
                type_index = self.priority_order.index(signal.exit_type.value)
                type_priority = 1.0 - (type_index * 0.2)  # Higher priority gets higher score
            except ValueError:
                type_priority = 0.5  # Unknown types get medium priority

            # Combine type priority, confidence, and urgency
            return (type_priority * 0.4 +
                   signal.confidence * 0.4 +
                   signal.urgency * 0.2)

        return sorted(signals, key=signal_priority_score, reverse=True)


class MTFDesyncEvaluator:
    """
    Evaluates multi-timeframe synchronization and detects when
    timeframes become desynchronized (suggesting position exit).
    """

    def __init__(self, config: Dict[str, Any]):
        self.min_desync_score = config.get('min_desync_score', 0.7)
        self.critical_desync_score = config.get('critical_desync_score', 0.85)

    def evaluate(self, symbol: str, position_bias: str,
                 sync_report: Optional[Dict[str, Any]],
                 current_bar: pd.Timestamp) -> Optional[ExitSignal]:
        """
        Evaluate MTF desynchronization for exit signals.

        Args:
            symbol: Trading symbol
            position_bias: Current position direction
            sync_report: MTF synchronization report from engine
            current_bar: Current timestamp

        Returns:
            ExitSignal if desync detected, None otherwise
        """
        if not sync_report:
            return None

        try:
            # Extract desync metrics from sync report
            desync_score = self._calculate_desync_score(sync_report, position_bias)

            if desync_score >= self.min_desync_score:
                confidence = min(0.9, desync_score)

                if desync_score >= self.critical_desync_score:
                    urgency = 0.9
                    action_type = ExitAction.FULL_EXIT
                else:
                    urgency = 0.6
                    action_type = ExitAction.PARTIAL_EXIT

                return ExitSignal(
                    timestamp=current_bar,
                    symbol=symbol,
                    exit_type=ExitType.MTF_DESYNC,
                    action=action_type,
                    confidence=confidence,
                    urgency=urgency,
                    reasons=[f"MTF desync score: {desync_score:.2f}"],
                    context={'desync_score': desync_score, 'sync_report': sync_report}
                )

        except Exception as e:
            logging.error(f"MTF desync evaluation error: {e}")

        return None

    def _calculate_desync_score(self, sync_report: Dict[str, Any], position_bias: str) -> float:
        """Calculate desynchronization score from sync report."""
        # This would need to be implemented based on your sync report structure
        # For now, return a placeholder that integrates with existing sync logic

        sync_score = sync_report.get('sync_score', 1.0)
        bias_agreement = sync_report.get('bias_agreement', 1.0)

        # Desync is inverse of sync
        desync_score = 1.0 - (sync_score * bias_agreement)

        return max(0.0, min(1.0, desync_score))


def create_default_exit_config() -> Dict[str, Any]:
    """
    Create default configuration for exit signal system.

    Returns:
        Default exit configuration
    """
    return {
        'enabled_exits': ['choch_against', 'momentum_fade', 'time_stop'],
        'min_confidence': 0.6,
        'priority_order': ['choch_against', 'momentum_fade', 'time_stop'],

        'choch_against': {
            'min_break_strength': 0.6,
            'confirmation_bars': 2,
            'volume_confirmation': True
        },

        'momentum_fade': {
            'rsi_period': 14,
            'rsi_divergence_threshold': 0.7,
            'volume_decline_threshold': 0.3,
            'velocity_threshold': 0.4
        },

        'time_stop': {
            'max_bars_1h': 168,  # 1 week
            'max_bars_4h': 42,   # 1 week
            'max_bars_1d': 10,   # 10 days
            'performance_threshold': 0.1,  # 10% gain to justify time
            'time_decay_start': 0.7  # Start decay at 70% of max time
        },

        'mtf_desync': {
            'min_desync_score': 0.7,
            'critical_desync_score': 0.85
        }
    }