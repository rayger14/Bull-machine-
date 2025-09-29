#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - Metric Gates and Stability Scoring
Quality gates to filter stable, tradeable configurations
Prevent overfitting and ensure robustness
"""

import numpy as np
import pandas as pd
import json
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

@dataclass
class QualityGates:
    """Configuration quality gate thresholds"""
    # Minimum requirements
    min_trades_per_fold: int = 5
    min_total_trades: int = 15
    min_win_rate: float = 0.4
    min_profit_factor: float = 1.1

    # Maximum risk tolerance
    max_drawdown_pct: float = 25.0
    max_consecutive_losses: int = 8

    # Stability requirements
    max_win_rate_cv: float = 0.3  # Coefficient of variation
    max_pnl_cv: float = 0.5
    max_drawdown_cv: float = 0.4

    # Performance targets
    target_sharpe: float = 1.5
    target_calmar: float = 1.0  # Annual return / Max DD
    target_trades_per_month: float = 0.5

    # Robustness requirements
    min_stable_folds: int = 3
    max_catastrophic_folds: int = 0  # Folds with >30% DD

class MetricGateValidator:
    """Validates trading configurations against quality gates"""

    def __init__(self, gates: QualityGates = None):
        self.gates = gates or QualityGates()

    def validate_single_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Validate a single backtest result"""
        validation = {
            "passes_gates": True,
            "violations": [],
            "warnings": [],
            "quality_score": 0.0,
            "result": result
        }

        # Extract metrics
        trades = result.get("total_trades", 0)
        win_rate = result.get("win_rate", 0.0)
        pnl_pct = result.get("total_pnl_pct", 0.0)
        profit_factor = result.get("profit_factor", 0.0)
        max_dd = result.get("max_drawdown_pct", 0.0)
        sharpe = result.get("sharpe_ratio", 0.0)
        trades_per_month = result.get("trades_per_month", 0.0)

        # Check minimum requirements
        violations = []

        if trades < self.gates.min_total_trades:
            violations.append(f"Insufficient trades: {trades} < {self.gates.min_total_trades}")

        if win_rate < self.gates.min_win_rate:
            violations.append(f"Low win rate: {win_rate:.1%} < {self.gates.min_win_rate:.1%}")

        if profit_factor < self.gates.min_profit_factor:
            violations.append(f"Low profit factor: {profit_factor:.2f} < {self.gates.min_profit_factor:.2f}")

        if max_dd > self.gates.max_drawdown_pct:
            violations.append(f"Excessive drawdown: {max_dd:.1f}% > {self.gates.max_drawdown_pct:.1f}%")

        # Check performance targets (warnings only)
        warnings = []

        if sharpe < self.gates.target_sharpe:
            warnings.append(f"Low Sharpe ratio: {sharpe:.2f} < {self.gates.target_sharpe:.2f}")

        if trades_per_month < self.gates.target_trades_per_month:
            warnings.append(f"Low trade frequency: {trades_per_month:.2f} < {self.gates.target_trades_per_month:.2f}")

        # Calculate quality score
        quality_score = self._calculate_quality_score(result)

        validation.update({
            "passes_gates": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "quality_score": quality_score
        })

        return validation

    def validate_fold_stability(self, fold_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Validate stability across multiple folds"""
        if len(fold_results) < self.gates.min_stable_folds:
            return {
                "passes_gates": False,
                "violations": [f"Insufficient folds: {len(fold_results)} < {self.gates.min_stable_folds}"],
                "stability_score": 0.0
            }

        # Extract metrics across folds
        metrics = self._extract_fold_metrics(fold_results)

        violations = []
        warnings = []

        # Check stability (coefficient of variation)
        if metrics["win_rate_cv"] > self.gates.max_win_rate_cv:
            violations.append(f"Unstable win rate: CV={metrics['win_rate_cv']:.3f} > {self.gates.max_win_rate_cv:.3f}")

        if metrics["pnl_cv"] > self.gates.max_pnl_cv:
            violations.append(f"Unstable PnL: CV={metrics['pnl_cv']:.3f} > {self.gates.max_pnl_cv:.3f}")

        if metrics["drawdown_cv"] > self.gates.max_drawdown_cv:
            violations.append(f"Unstable drawdown: CV={metrics['drawdown_cv']:.3f} > {self.gates.max_drawdown_cv:.3f}")

        # Check catastrophic folds
        catastrophic_folds = sum(1 for r in fold_results if r.get("max_drawdown_pct", 0) > 30)
        if catastrophic_folds > self.gates.max_catastrophic_folds:
            violations.append(f"Catastrophic folds: {catastrophic_folds} > {self.gates.max_catastrophic_folds}")

        # Check minimum trades per fold
        min_trades = min(r.get("total_trades", 0) for r in fold_results)
        if min_trades < self.gates.min_trades_per_fold:
            violations.append(f"Insufficient trades per fold: {min_trades} < {self.gates.min_trades_per_fold}")

        # Calculate stability score
        stability_score = self._calculate_stability_score(metrics)

        return {
            "passes_gates": len(violations) == 0,
            "violations": violations,
            "warnings": warnings,
            "stability_score": stability_score,
            "fold_count": len(fold_results),
            "metrics": metrics
        }

    def _extract_fold_metrics(self, fold_results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Extract aggregated metrics across folds"""
        # Collect all metric values
        win_rates = [r.get("win_rate", 0) for r in fold_results]
        pnl_values = [r.get("total_pnl_pct", 0) for r in fold_results]
        drawdowns = [r.get("max_drawdown_pct", 0) for r in fold_results]
        sharpe_values = [r.get("sharpe_ratio", 0) for r in fold_results]
        trade_counts = [r.get("total_trades", 0) for r in fold_results]

        def safe_cv(values):
            """Safe coefficient of variation calculation"""
            if not values or np.mean(values) == 0:
                return 0
            return np.std(values) / abs(np.mean(values))

        return {
            # Central tendencies
            "mean_win_rate": np.mean(win_rates),
            "mean_pnl": np.mean(pnl_values),
            "mean_drawdown": np.mean(drawdowns),
            "mean_sharpe": np.mean(sharpe_values),
            "mean_trades": np.mean(trade_counts),

            # Stability measures (CV)
            "win_rate_cv": safe_cv(win_rates),
            "pnl_cv": safe_cv(pnl_values),
            "drawdown_cv": safe_cv(drawdowns),
            "sharpe_cv": safe_cv(sharpe_values),
            "trades_cv": safe_cv(trade_counts),

            # Risk measures
            "max_drawdown": max(drawdowns) if drawdowns else 0,
            "min_trades": min(trade_counts) if trade_counts else 0,
            "worst_pnl": min(pnl_values) if pnl_values else 0,

            # Consistency measures
            "positive_folds": sum(1 for pnl in pnl_values if pnl > 0),
            "negative_folds": sum(1 for pnl in pnl_values if pnl < 0),
        }

    def _calculate_quality_score(self, result: Dict[str, Any]) -> float:
        """Calculate overall quality score for a single result"""
        trades = result.get("total_trades", 0)
        win_rate = result.get("win_rate", 0.0)
        pnl_pct = result.get("total_pnl_pct", 0.0)
        profit_factor = result.get("profit_factor", 0.0)
        max_dd = result.get("max_drawdown_pct", 0.0)
        sharpe = result.get("sharpe_ratio", 0.0)

        # Minimum viability check
        if trades < 5 or profit_factor < 1.0:
            return 0.0

        # Normalize metrics to 0-1 scale
        n_trades = min(1.0, trades / 30)  # 30 trades = perfect
        n_win_rate = win_rate
        n_pnl = max(0, min(1, pnl_pct / 30))  # 30% = perfect
        n_pf = max(0, min(1, (profit_factor - 1) / 2))  # PF 3.0 = perfect
        n_sharpe = max(0, min(1, sharpe / 3))  # Sharpe 3 = perfect

        # Penalties
        dd_penalty = min(1, max_dd / 30) ** 2  # Quadratic penalty for DD

        # Weighted score
        score = (
            0.25 * n_trades +
            0.20 * n_win_rate +
            0.25 * n_pnl +
            0.15 * n_pf +
            0.15 * n_sharpe -
            0.30 * dd_penalty
        )

        return max(0, min(1, score))

    def _calculate_stability_score(self, metrics: Dict[str, float]) -> float:
        """Calculate stability score across folds"""
        # Invert CV values (lower CV = higher stability)
        wr_stability = 1 / (1 + metrics["win_rate_cv"])
        pnl_stability = 1 / (1 + metrics["pnl_cv"])
        dd_stability = 1 / (1 + metrics["drawdown_cv"])

        # Performance component
        performance = max(0, min(1, metrics["mean_pnl"] / 20))  # 20% = perfect

        # Risk component
        risk_penalty = min(1, metrics["max_drawdown"] / 30)

        # Consistency component
        total_folds = metrics["positive_folds"] + metrics["negative_folds"]
        consistency = metrics["positive_folds"] / max(1, total_folds)

        # Weighted stability score
        stability = (
            0.25 * wr_stability +
            0.25 * pnl_stability +
            0.20 * dd_stability +
            0.15 * performance +
            0.15 * consistency -
            0.20 * risk_penalty
        )

        return max(0, min(1, stability))

class QualityReport:
    """Generate comprehensive quality reports"""

    def __init__(self, validator: MetricGateValidator = None):
        self.validator = validator or MetricGateValidator()

    def analyze_results_file(self, results_file: str) -> Dict[str, Any]:
        """Analyze a complete results file"""
        try:
            with open(results_file, 'r') as f:
                results = []
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        if result.get("status") == "ok":
                            results.append(result)
                    except:
                        continue

            return self.analyze_results(results)

        except Exception as e:
            return {"error": f"Failed to analyze results file: {e}"}

    def analyze_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze a list of results"""
        if not results:
            return {"error": "No valid results to analyze"}

        # Group results by configuration
        config_groups = {}
        for result in results:
            config_key = self._get_config_key(result)
            if config_key not in config_groups:
                config_groups[config_key] = []
            config_groups[config_key].append(result)

        # Analyze each configuration
        config_analysis = {}
        for config_key, fold_results in config_groups.items():
            if len(fold_results) == 1:
                # Single result validation
                validation = self.validator.validate_single_result(fold_results[0])
            else:
                # Multi-fold stability validation
                validation = self.validator.validate_fold_stability(fold_results)

            config_analysis[config_key] = validation

        # Generate summary statistics
        total_configs = len(config_analysis)
        passing_configs = sum(1 for v in config_analysis.values() if v.get("passes_gates", False))

        # Best performers
        best_configs = sorted(
            [(k, v) for k, v in config_analysis.items() if v.get("passes_gates", False)],
            key=lambda x: x[1].get("quality_score", x[1].get("stability_score", 0)),
            reverse=True
        )[:10]

        return {
            "summary": {
                "total_configs": total_configs,
                "passing_configs": passing_configs,
                "pass_rate": passing_configs / max(1, total_configs),
                "total_results": len(results)
            },
            "best_configs": [
                {
                    "config_key": config_key,
                    "score": validation.get("quality_score", validation.get("stability_score", 0)),
                    "violations": validation.get("violations", []),
                    "fold_count": validation.get("fold_count", 1)
                }
                for config_key, validation in best_configs
            ],
            "config_analysis": config_analysis,
            "gates_used": {
                "min_trades_per_fold": self.validator.gates.min_trades_per_fold,
                "min_win_rate": self.validator.gates.min_win_rate,
                "max_drawdown_pct": self.validator.gates.max_drawdown_pct,
                "min_profit_factor": self.validator.gates.min_profit_factor
            }
        }

    def _get_config_key(self, result: Dict[str, Any]) -> str:
        """Generate configuration key for grouping"""
        config = result.get("config", {})
        key_params = [
            "entry_threshold", "min_active_domains", "cooldown_days",
            "risk_pct", "sl_atr_multiplier", "tp_atr_multiplier"
        ]

        key_parts = []
        for param in key_params:
            if param in config:
                value = config[param]
                if isinstance(value, float):
                    key_parts.append(f"{param}={value:.3f}")
                else:
                    key_parts.append(f"{param}={value}")

        return "|".join(key_parts)

    def generate_report(self, results_file: str, output_file: str = None) -> str:
        """Generate a comprehensive quality report"""
        analysis = self.analyze_results_file(results_file)

        if "error" in analysis:
            return analysis["error"]

        # Generate report text
        report_lines = [
            "=" * 60,
            "BULL MACHINE QUALITY GATE ANALYSIS",
            "=" * 60,
            f"Generated: {datetime.now().isoformat()}",
            f"Source: {results_file}",
            "",
            "SUMMARY:",
            f"  Total configurations tested: {analysis['summary']['total_configs']}",
            f"  Configurations passing gates: {analysis['summary']['passing_configs']}",
            f"  Pass rate: {analysis['summary']['pass_rate']:.1%}",
            f"  Total backtest results: {analysis['summary']['total_results']}",
            "",
            "QUALITY GATES:",
            f"  Minimum trades per fold: {analysis['gates_used']['min_trades_per_fold']}",
            f"  Minimum win rate: {analysis['gates_used']['min_win_rate']:.1%}",
            f"  Maximum drawdown: {analysis['gates_used']['max_drawdown_pct']:.1f}%",
            f"  Minimum profit factor: {analysis['gates_used']['min_profit_factor']:.2f}",
            "",
            "TOP PERFORMING CONFIGURATIONS:",
            ""
        ]

        for i, config in enumerate(analysis["best_configs"][:5], 1):
            report_lines.extend([
                f"{i}. Score: {config['score']:.3f} | Folds: {config['fold_count']}",
                f"   Config: {config['config_key'][:100]}...",
                f"   Violations: {len(config['violations'])}",
                ""
            ])

        report_text = "\n".join(report_lines)

        # Save to file if requested
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_text)

        return report_text

# Example usage
if __name__ == "__main__":
    print("Testing metric gates...")

    # Create test result
    test_result = {
        "total_trades": 20,
        "win_rate": 0.6,
        "total_pnl_pct": 15.5,
        "profit_factor": 2.1,
        "max_drawdown_pct": 8.5,
        "sharpe_ratio": 2.2,
        "trades_per_month": 1.5,
        "config": {"entry_threshold": 0.3, "risk_pct": 0.025}
    }

    # Test validation
    validator = MetricGateValidator()
    validation = validator.validate_single_result(test_result)

    print(f"Passes gates: {validation['passes_gates']}")
    print(f"Quality score: {validation['quality_score']:.3f}")
    print(f"Violations: {validation['violations']}")

    print("Metric gates test complete.")