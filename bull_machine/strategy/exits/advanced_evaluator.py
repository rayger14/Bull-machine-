"""
Advanced Exit Signal Evaluator for Bull Machine v1.4.1
Master monitor that orchestrates all exit rules and updates trade plans.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

from .advanced_rules import (
    BojanExtremeProtection,
    ExitDecision,
    GlobalVeto,
    MarkdownSOSSpringFlip,
    MarkupExhaustion,
    MarkupSOWUTWarning,
    MarkupUTADRejection,
    MoneytaurTrailing,
)


class AdvancedExitEvaluator:
    """
    Master exit evaluator that manages all exit rules and telemetry.
    """

    def __init__(self, config_path: str = None):
        """Initialize with config file or defaults."""
        if config_path:
            self.cfg = self._load_config(config_path)
        else:
            self.cfg = self._get_default_config()

        self._validate_config()
        self.rules = self._build_rules()
        self.telemetry = []

    def _load_config(self, path: str) -> Dict:
        """Load config from JSON file."""
        with open(path, "r") as f:
            return json.load(f)

    def _get_default_config(self) -> Dict:
        """Default v1.4.1 exit configuration."""
        return {
            "enabled": True,
            "order": [
                "global_veto",
                "bojan_extreme_protection",
                "markup_sow_ut_warning",
                "markup_utad_rejection",
                "markup_exhaustion",
                "markdown_sos_spring_flip",
                "moneytaur_trailing",
            ],
            "shared": {
                "atr_period": 14,
                "vol_sma": 10,
                "range_lookback": 20,
                "aggregate_floor": 0.35,
                "mtf_desync_floor": 0.6,
                "cooldown_bars": 8,
            },
            "markup_sow_ut_warning": {
                "enabled": True,
                "premium_floor": 0.70,
                "vol_divergence_ratio": 0.70,
                "wick_atr_mult": 1.5,
                "mtf_desync_floor": 0.6,
                "veto_needed": 3,
                "partial_pct": 0.25,
                "trail_atr_buffer_R": 0.5,
            },
            "markup_utad_rejection": {
                "enabled": True,
                "wick_close_frac": 0.5,
                "fib_retrace": 0.618,
                "partial_pct": 0.5,
                "trail_to_structure_minus_atr": True,
            },
            "markup_exhaustion": {
                "enabled": True,
                "min_bars_since_entry": 20,
                "retest_frac": 0.95,
                "wyckoff_drop_floor": 0.40,
                "aggregate_floor": 0.35,
            },
            "markdown_sos_spring_flip": {
                "enabled": True,
                "discount_ceiling": 0.30,
                "vol_surge_mult": 1.5,
                "sos_green_in_6": 4,
                "wyckoff_flip_floor": 0.70,
                "partial_pct": 0.5,
            },
            "moneytaur_trailing": {
                "enabled": True,
                "activate_after_R": 1.0,
                "trail_rule": "max(BE + 0.5R, structure_pivot - 1*ATR)",
                "update_every_bars": 3,
            },
            "global_veto": {
                "enabled": True,
                "aggregate_floor": 0.40,
                "context_floor": 0.30,
                "cooldown_bars": 8,
            },
            "bojan_extreme_protection": {
                "enabled": False,  # Phase-gated for v2.x
                "wick_atr_mult": 2.0,
                "vol_under_sma_mult": 0.5,
                "exit_pct": 0.75,
                "require_htf_alignment": True,
            },
        }

    def _validate_config(self):
        """Validate configuration has all required fields."""
        if not self.cfg.get("enabled"):
            logging.warning("Exit evaluator is disabled in config")

        if "order" not in self.cfg:
            raise ValueError("Config missing 'order' field")

        if "shared" not in self.cfg:
            raise ValueError("Config missing 'shared' field")

    def _build_rules(self) -> List:
        """Build rule instances based on config order."""
        rule_mapping = {
            "markup_sow_ut_warning": MarkupSOWUTWarning,
            "markup_utad_rejection": MarkupUTADRejection,
            "markup_exhaustion": MarkupExhaustion,
            "markdown_sos_spring_flip": MarkdownSOSSpringFlip,
            "moneytaur_trailing": MoneytaurTrailing,
            "global_veto": GlobalVeto,
            "bojan_extreme_protection": BojanExtremeProtection,
        }

        rules = []
        for rule_name in self.cfg["order"]:
            if rule_name in rule_mapping:
                rule_cfg = self.cfg.get(rule_name, {})
                if rule_cfg.get("enabled", False):
                    rule_class = rule_mapping[rule_name]
                    rule = rule_class(rule_cfg, self.cfg["shared"])
                    rules.append(rule)
                    logging.info(f"Loaded exit rule: {rule_name}")

        return rules

    def evaluate_exits(
        self,
        df: pd.DataFrame,
        trade_plan: Dict,
        confluence_scores: Dict,
        bars_since_entry: int,
        mtf_context: Dict = None,
    ) -> Dict:
        """
        Main evaluation method - checks all rules and updates trade plan.

        Args:
            df: OHLCV DataFrame
            trade_plan: Current trade plan dict
            confluence_scores: 7-layer scores dict
            bars_since_entry: Bars since position opened
            mtf_context: Multi-timeframe context data

        Returns:
            Updated trade_plan dict
        """

        if not self.cfg.get("enabled"):
            return trade_plan

        # Initialize exits section if needed
        if "exits" not in trade_plan:
            trade_plan["exits"] = {
                "history": [],
                "current_action": None,
                "cooldown_bars": 0,
                "current_sl": trade_plan.get("sl"),
                "current_tp": trade_plan.get("tp"),
            }

        # Check cooldown
        if trade_plan["exits"].get("cooldown_bars", 0) > 0:
            trade_plan["exits"]["cooldown_bars"] -= 1
            return trade_plan

        # Evaluate each rule in order
        for rule in self.rules:
            try:
                decision = rule.evaluate(df, trade_plan, confluence_scores, bars_since_entry, mtf_context)

                if decision:
                    # Log telemetry
                    self._log_telemetry(rule.name, decision, trade_plan, confluence_scores, bars_since_entry)

                    # Apply the decision
                    trade_plan = self._apply_decision(trade_plan, decision, bars_since_entry)

                    # Stop after first matching rule (ordered precedence)
                    break

            except Exception as e:
                logging.error(f"Error in rule {rule.name}: {e}")
                continue

        return trade_plan

    def _apply_decision(self, trade_plan: Dict, decision: ExitDecision, bars_since_entry: int) -> Dict:
        """Apply exit decision to trade plan."""

        # Record in history
        exit_record = {
            "bar": bars_since_entry,
            "action": decision.action,
            "size_pct": decision.size_pct,
            "reason": decision.reason,
            "metadata": decision.metadata,
        }

        trade_plan["exits"]["history"].append(exit_record)
        trade_plan["exits"]["current_action"] = decision.action

        # Apply specific actions
        if decision.action == "partial":
            trade_plan["exits"]["partial_exit_pct"] = decision.size_pct
            if decision.new_sl:
                trade_plan["exits"]["current_sl"] = decision.new_sl

        elif decision.action == "full":
            trade_plan["exits"]["full_exit"] = True
            if decision.cooldown_bars:
                trade_plan["exits"]["cooldown_bars"] = decision.cooldown_bars

        elif decision.action == "trail":
            if decision.new_sl:
                trade_plan["exits"]["current_sl"] = decision.new_sl

        # Handle position flip
        if decision.flip_bias:
            trade_plan["exits"]["flip_to"] = decision.flip_bias

        return trade_plan

    def _log_telemetry(self, rule_name: str, decision: ExitDecision, trade_plan: Dict, scores: Dict, bars: int):
        """Log telemetry for exit evaluation."""

        telemetry_entry = {
            "timestamp": pd.Timestamp.now().isoformat(),
            "rule": rule_name,
            "evaluated": True,
            "triggered": decision is not None,
            "action": decision.action if decision else None,
            "params": {
                "bars_since_entry": bars,
                "bias": trade_plan["bias"],
                "entry_price": trade_plan["entry_price"],
            },
            "scores": scores,
            "reason": decision.reason if decision else None,
        }

        self.telemetry.append(telemetry_entry)

        # Log to file
        logging.info(f"EXIT_EVAL_APPLIED {json.dumps(telemetry_entry)}")

    def save_telemetry(self, output_dir: str):
        """Save telemetry to JSON files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        # Save detailed telemetry
        with open(output_path / "exits_applied.jsonl", "a") as f:
            for entry in self.telemetry:
                f.write(json.dumps(entry) + "\n")

        # Save summary
        summary = {
            "total_evaluations": len(self.telemetry),
            "total_triggers": sum(1 for t in self.telemetry if t["triggered"]),
            "rules_triggered": {},
        }

        for entry in self.telemetry:
            if entry["triggered"]:
                rule = entry["rule"]
                if rule not in summary["rules_triggered"]:
                    summary["rules_triggered"][rule] = 0
                summary["rules_triggered"][rule] += 1

        with open(output_path / "exit_counts.json", "w") as f:
            json.dump(summary, f, indent=2)

        # Clear telemetry after saving
        self.telemetry = []
