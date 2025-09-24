"""
Exit Signal Evaluators
Coordinates multiple exit detection systems and provides unified evaluation.
"""

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .rules import CHoCHAgainstDetector, MomentumFadeDetector, TimeStopEvaluator
from .types import ExitEvaluationResult, ExitSignal


class ExitSignalEvaluator:
    """
    Main coordinator for all exit signal detection.
    Evaluates multiple exit conditions and prioritizes signals.
    """

    def __init__(self, config: Dict[str, Any], out_dir: str = "."):
        """
        Initialize with configuration for all exit detectors.

        Args:
            config: Exit system configuration
            out_dir: Output directory for diagnostic files
        """
        self.config = config

        # DIAGNOSTIC: Log and dump config to prove it reaches evaluator
        logging.info("EXIT_EVAL_INIT cfg=%s", json.dumps(config, sort_keys=True))
        try:
            config_dump_path = os.path.join(out_dir, "exit_cfg_applied.json")
            with open(config_dump_path, "w") as f:
                json.dump(config, f, indent=2, sort_keys=True)
            logging.info("EXIT_EVAL_INIT config dumped to %s", config_dump_path)
        except Exception as e:
            logging.warning("EXIT_EVAL_INIT failed to dump config: %s", e)

        # Initialize detectors
        choch_config = config.get("choch_against", {})
        momentum_config = config.get("momentum_fade", {})
        time_config = config.get("time_stop", {})

        self.choch_detector = CHoCHAgainstDetector(choch_config)
        self.momentum_detector = MomentumFadeDetector(momentum_config)
        self.time_evaluator = TimeStopEvaluator(time_config)

        # Global exit settings
        self.enabled_exits = config.get(
            "enabled_exits", ["choch_against", "momentum_fade", "time_stop"]
        )
        self.min_confidence = config.get("min_confidence", 0.6)
        self.priority_order = config.get(
            "priority_order", ["choch_against", "momentum_fade", "time_stop"]
        )

        # Exit tracking for telemetry
        self.exit_counts = {"choch_against": 0, "momentum_fade": 0, "time_stop": 0, "none": 0}

        # Enhanced telemetry tracking
        self.exit_applications = []  # Track each application with details
        self.parameter_usage = {
            "choch_against": {"triggered": 0, "evaluated": 0, "params_used": choch_config},
            "momentum_fade": {"triggered": 0, "evaluated": 0, "params_used": momentum_config},
            "time_stop": {"triggered": 0, "evaluated": 0, "params_used": time_config},
        }
        self.out_dir = out_dir

        logging.info(f"ExitSignalEvaluator initialized with exits: {self.enabled_exits}")

    def evaluate_exits(
        self,
        symbol: str,
        position_data: Dict[str, Any],
        mtf_data: Dict[str, pd.DataFrame],
        current_bar: pd.Timestamp,
    ) -> ExitEvaluationResult:
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
        position_bias = position_data.get("bias", "long")

        # Track signals for telemetry
        choch_sig = None
        mom_sig = None
        time_sig = None

        try:
            # 1. CHoCH-Against Detection
            if "choch_against" in self.enabled_exits:
                self.parameter_usage["choch_against"]["evaluated"] += 1
                choch_sig = self.choch_detector.evaluate(
                    symbol, position_bias, mtf_data, current_bar
                )
                if choch_sig and choch_sig.confidence >= self.min_confidence:
                    result.add_signal(choch_sig)
                    self.parameter_usage["choch_against"]["triggered"] += 1
                    logging.debug(f"CHoCH-Against signal: {choch_sig.confidence:.2f}")

            # 2. Momentum Fade Detection
            if "momentum_fade" in self.enabled_exits:
                self.parameter_usage["momentum_fade"]["evaluated"] += 1
                mom_sig = self.momentum_detector.evaluate(
                    symbol, position_bias, mtf_data, current_bar
                )
                if mom_sig and mom_sig.confidence >= self.min_confidence:
                    result.add_signal(mom_sig)
                    self.parameter_usage["momentum_fade"]["triggered"] += 1
                    logging.debug(f"Momentum fade signal: {mom_sig.confidence:.2f}")

            # 3. Time Stop Evaluation
            if "time_stop" in self.enabled_exits:
                self.parameter_usage["time_stop"]["evaluated"] += 1
                time_sig = self.time_evaluator.evaluate(symbol, position_data, current_bar)
                if time_sig and time_sig.confidence >= self.min_confidence:
                    result.add_signal(time_sig)
                    self.parameter_usage["time_stop"]["triggered"] += 1
                    logging.debug(f"Time stop signal: {time_sig.confidence:.2f}")

        except Exception as e:
            logging.error(f"Exit evaluation error for {symbol}: {e}")

        # EXIT_SCAN telemetry log
        chosen = self.get_action_recommendation(result) if result.has_signals() else None
        logging.debug(
            f"[EXIT_SCAN] sym={symbol} bar={current_bar} side={position_bias} "
            f"choch={choch_sig.exit_type.value if choch_sig else '-'} "
            f"mom={mom_sig.exit_type.value if mom_sig else '-'} "
            f"time={time_sig.exit_type.value if time_sig else '-'} -> "
            f"chosen={chosen.exit_type.value if chosen else '-'}"
        )

        # Log results
        if result.has_signals():
            logging.info(
                f"Exit signals for {symbol}: {len(result.signals)} signals, "
                f"max confidence: {result.max_confidence:.2f}"
            )
        else:
            logging.debug(f"No exit signals for {symbol}")
            self.exit_counts["none"] += 1

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
            # Track exit counts for telemetry
            exit_type = recommended.exit_type.value
            if exit_type in self.exit_counts:
                self.exit_counts[exit_type] += 1

            # Track detailed application
            application_record = {
                "timestamp": recommended.timestamp.isoformat()
                if hasattr(recommended.timestamp, "isoformat")
                else str(recommended.timestamp),
                "symbol": recommended.symbol,
                "exit_type": exit_type,
                "confidence": recommended.confidence,
                "urgency": recommended.urgency,
                "reasons": recommended.reasons,
                "context": recommended.context,
            }
            self.exit_applications.append(application_record)

            logging.info(
                f"Recommended exit: {recommended.exit_type.value} "
                f"(confidence: {recommended.confidence:.2f}, "
                f"urgency: {recommended.urgency:.2f})"
            )
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
            return type_priority * 0.4 + signal.confidence * 0.4 + signal.urgency * 0.2

        return sorted(signals, key=signal_priority_score, reverse=True)

    def save_exit_counts(self, output_dir: str):
        """Save exit counts telemetry to JSON file."""
        output_path = Path(output_dir) / "exit_counts.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            json.dump(self.exit_counts, f, indent=2)

        logging.info(f"Exit counts saved to {output_path}: {self.exit_counts}")

    def save_comprehensive_telemetry(self, output_dir: str):
        """Save comprehensive exit telemetry including parameter usage and applications."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. Save exits_applied.json - detailed applications
        exits_applied_path = output_dir / "exits_applied.json"
        exits_applied_data = {
            "total_applications": len(self.exit_applications),
            "applications_by_type": {
                exit_type: len(
                    [app for app in self.exit_applications if app["exit_type"] == exit_type]
                )
                for exit_type in ["choch_against", "momentum_fade", "time_stop"]
            },
            "detailed_applications": self.exit_applications,
            "summary": self.exit_counts,
        }

        with open(exits_applied_path, "w") as f:
            json.dump(exits_applied_data, f, indent=2)
        logging.info(f"Exits applied telemetry saved to {exits_applied_path}")

        # 2. Save parameter_usage.json - prove parameters are used
        parameter_usage_path = output_dir / "parameter_usage.json"
        parameter_telemetry = {
            "parameter_effectiveness": {
                exit_type: {
                    "evaluated_count": data["evaluated"],
                    "triggered_count": data["triggered"],
                    "trigger_rate": data["triggered"] / data["evaluated"]
                    if data["evaluated"] > 0
                    else 0.0,
                    "parameters_applied": data["params_used"],
                }
                for exit_type, data in self.parameter_usage.items()
            },
            "global_stats": {
                "total_evaluations": sum(
                    data["evaluated"] for data in self.parameter_usage.values()
                ),
                "total_triggers": sum(data["triggered"] for data in self.parameter_usage.values()),
                "overall_trigger_rate": (
                    sum(data["triggered"] for data in self.parameter_usage.values())
                    / sum(data["evaluated"] for data in self.parameter_usage.values())
                    if sum(data["evaluated"] for data in self.parameter_usage.values()) > 0
                    else 0.0
                ),
            },
            "enabled_exits": self.enabled_exits,
            "min_confidence": self.min_confidence,
            "priority_order": self.priority_order,
        }

        with open(parameter_usage_path, "w") as f:
            json.dump(parameter_telemetry, f, indent=2)
        logging.info(f"Parameter usage telemetry saved to {parameter_usage_path}")

        # 3. Save layer_masks.json - fusion probe data (placeholder for now)
        layer_masks_path = output_dir / "layer_masks.json"
        layer_masks_data = {
            "fusion_layers": {
                "choch_layer": {
                    "active": "choch_against" in self.enabled_exits,
                    "evaluation_count": self.parameter_usage["choch_against"]["evaluated"],
                    "trigger_count": self.parameter_usage["choch_against"]["triggered"],
                    "mask_applied": self.parameter_usage["choch_against"]["triggered"] > 0,
                },
                "momentum_layer": {
                    "active": "momentum_fade" in self.enabled_exits,
                    "evaluation_count": self.parameter_usage["momentum_fade"]["evaluated"],
                    "trigger_count": self.parameter_usage["momentum_fade"]["triggered"],
                    "mask_applied": self.parameter_usage["momentum_fade"]["triggered"] > 0,
                },
                "time_layer": {
                    "active": "time_stop" in self.enabled_exits,
                    "evaluation_count": self.parameter_usage["time_stop"]["evaluated"],
                    "trigger_count": self.parameter_usage["time_stop"]["triggered"],
                    "mask_applied": self.parameter_usage["time_stop"]["triggered"] > 0,
                },
            },
            "layer_interaction": {
                "total_layers": len(self.enabled_exits),
                "active_layers": len(
                    [
                        exit
                        for exit in self.enabled_exits
                        if self.parameter_usage.get(exit, {}).get("triggered", 0) > 0
                    ]
                ),
                "fusion_effectiveness": len(
                    [
                        exit
                        for exit in self.enabled_exits
                        if self.parameter_usage.get(exit, {}).get("triggered", 0) > 0
                    ]
                )
                / len(self.enabled_exits)
                if self.enabled_exits
                else 0.0,
            },
        }

        with open(layer_masks_path, "w") as f:
            json.dump(layer_masks_data, f, indent=2)
        logging.info(f"Layer masks telemetry saved to {layer_masks_path}")

        # Also save traditional exit_counts.json for backward compatibility
        self.save_exit_counts(str(output_dir))

        print("\n🔍 TELEMETRY SAVED:")
        print(f"  • exits_applied.json - {len(self.exit_applications)} applications tracked")
        print(
            f"  • parameter_usage.json - {sum(data['evaluated'] for data in self.parameter_usage.values())} evaluations"
        )
        print(f"  • layer_masks.json - {len(self.enabled_exits)} fusion layers")
        print(f"  • exit_counts.json - {sum(self.exit_counts.values())} total exits")


class MTFDesyncEvaluator:
    """
    Evaluates multi-timeframe synchronization and detects when
    timeframes become desynchronized (suggesting position exit).
    """

    def __init__(self, config: Dict[str, Any]):
        self.min_desync_score = config.get("min_desync_score", 0.7)
        self.critical_desync_score = config.get("critical_desync_score", 0.85)

    def evaluate(
        self,
        symbol: str,
        position_bias: str,
        sync_report: Optional[Dict[str, Any]],
        current_bar: pd.Timestamp,
    ) -> Optional[ExitSignal]:
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
                    context={"desync_score": desync_score, "sync_report": sync_report},
                )

        except Exception as e:
            logging.error(f"MTF desync evaluation error: {e}")

        return None

    def _calculate_desync_score(self, sync_report: Dict[str, Any], position_bias: str) -> float:
        """Calculate desynchronization score from sync report."""
        # This would need to be implemented based on your sync report structure
        # For now, return a placeholder that integrates with existing sync logic

        sync_score = sync_report.get("sync_score", 1.0)
        bias_agreement = sync_report.get("bias_agreement", 1.0)

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
        "enabled_exits": ["choch_against", "momentum_fade", "time_stop"],
        "min_confidence": 0.6,
        "priority_order": ["choch_against", "momentum_fade", "time_stop"],
        "choch_against": {
            "min_break_strength": 0.6,
            "confirmation_bars": 2,
            "volume_confirmation": True,
        },
        "momentum_fade": {
            "rsi_period": 14,
            "rsi_divergence_threshold": 0.7,
            "volume_decline_threshold": 0.3,
            "velocity_threshold": 0.4,
        },
        "time_stop": {
            "max_bars_1h": 168,  # 1 week
            "max_bars_4h": 42,  # 1 week
            "max_bars_1d": 10,  # 10 days
            "performance_threshold": 0.1,  # 10% gain to justify time
            "time_decay_start": 0.7,  # Start decay at 70% of max time
        },
        "mtf_desync": {"min_desync_score": 0.7, "critical_desync_score": 0.85},
    }
