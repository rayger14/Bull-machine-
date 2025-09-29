#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - Optimization Strategies
Stage A: Coarse grid search
Stage B: Bayesian refinement
Stage C: Walk-forward validation
"""

import numpy as np
import pandas as pd
import json
import itertools
import random
from typing import List, Dict, Tuple, Any, Optional
from pathlib import Path
import time
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class StageAStrategy:
    """Stage A: Coarse grid search over major parameters"""

    def __init__(self, assets: List[str] = None, max_configs: int = 200):
        self.assets = assets or ["ETH", "BTC"]
        self.max_configs = max_configs

    def generate_grid(self) -> List[Dict[str, Any]]:
        """Generate coarse parameter grid"""

        # Define parameter ranges
        entry_thresholds = [0.2, 0.3, 0.4, 0.5]
        min_domains = [2, 3, 4]
        cooldown_days = [3, 7, 10, 14]
        risk_pcts = [0.015, 0.025, 0.035, 0.05]
        sl_multipliers = [1.0, 1.4, 1.8, 2.2]
        tp_multipliers = [2.0, 2.5, 3.0, 4.0]
        trail_multipliers = [0.6, 0.8, 1.0, 1.2]

        # Generate all combinations
        grid = []
        for asset in self.assets:
            for thresh, min_dom, cd, risk, sl, tp, trail in itertools.product(
                entry_thresholds, min_domains, cooldown_days,
                risk_pcts, sl_multipliers, tp_multipliers, trail_multipliers
            ):

                # Skip invalid combinations
                if sl >= tp:  # TP must be > SL
                    continue
                if thresh >= 0.5 and min_dom < 3:  # High threshold needs more domains
                    continue
                if risk >= 0.05 and sl < 1.4:  # High risk needs tighter stops
                    continue

                config = {
                    "asset": asset,
                    "entry_threshold": thresh,
                    "min_active_domains": min_dom,
                    "cooldown_days": cd,
                    "risk_pct": risk,
                    "sl_atr_multiplier": sl,
                    "tp_atr_multiplier": tp,
                    "trail_atr_multiplier": trail,
                    "strategy": "stage_a"
                }
                grid.append(config)

        # Limit total configurations
        if len(grid) > self.max_configs:
            random.shuffle(grid)
            grid = grid[:self.max_configs]

        return grid

    def create_walk_forward_folds(self, start_date: str = "2022-01-01",
                                  end_date: str = "2024-12-31") -> List[Tuple[str, str]]:
        """Create walk-forward fold pairs"""
        start = datetime.strptime(start_date, "%Y-%m-%d")
        end = datetime.strptime(end_date, "%Y-%m-%d")

        folds = []
        current = start
        fold_months = 6  # 6-month folds

        while current + timedelta(days=fold_months*30*2) <= end:
            fold_end = current + timedelta(days=fold_months*30)
            val_end = fold_end + timedelta(days=fold_months*30)

            folds.append((
                current.strftime("%Y-%m-%d"),
                val_end.strftime("%Y-%m-%d")
            ))

            current += timedelta(days=fold_months*30//2)  # 50% overlap

        return folds

class StageBStrategy:
    """Stage B: Bayesian optimization around best Stage A results"""

    def __init__(self, stage_a_results: str, top_n: int = 20, max_iterations: int = 50):
        self.stage_a_results = stage_a_results
        self.top_n = top_n
        self.max_iterations = max_iterations

    def load_stage_a_winners(self) -> List[Dict]:
        """Load top performers from Stage A"""
        try:
            with open(self.stage_a_results, 'r') as f:
                results = []
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        if result.get("status") == "ok":
                            results.append(result)
                    except:
                        continue

            # Sort by utility score (or custom metric)
            results.sort(key=lambda x: self._calculate_utility(x), reverse=True)
            return results[:self.top_n]

        except Exception as e:
            print(f"Error loading Stage A results: {e}")
            return []

    def _calculate_utility(self, result: Dict) -> float:
        """Calculate utility score for ranking"""
        trades = result.get("total_trades", 0)
        wr = result.get("win_rate", 0)
        pnl = result.get("total_pnl_pct", 0)
        pf = result.get("profit_factor", 0)
        dd = result.get("max_drawdown_pct", 0)
        sharpe = result.get("sharpe_ratio", 0)

        # Minimum trade filter
        if trades < 5:
            return 0

        # Normalize components
        n_pnl = max(0, min(1, pnl / 50))  # 50% PnL = perfect
        n_pf = max(0, min(1, (pf - 1) / 2))  # PF 3.0 = perfect
        n_wr = max(0, min(1, wr))
        n_sharpe = max(0, min(1, sharpe / 3))  # Sharpe 3 = perfect

        # Penalties
        dd_penalty = (dd / 30) ** 2  # 30% DD = maximum penalty
        freq_penalty = 0
        if trades < 10:
            freq_penalty = (10 - trades) / 10

        utility = (1.0 * n_pnl + 0.8 * n_pf + 0.6 * n_wr + 0.4 * n_sharpe -
                  1.2 * dd_penalty - 0.6 * freq_penalty)

        return max(0, utility)

    def generate_bayesian_suggestions(self, iteration: int) -> List[Dict]:
        """Generate parameter suggestions using simple Bayesian approach"""
        winners = self.load_stage_a_winners()
        if not winners:
            return []

        suggestions = []

        # Extract parameter distributions from winners
        param_stats = self._analyze_winner_params(winners)

        # Generate suggestions around winner means with exploration
        for i in range(min(20, self.max_iterations - iteration)):
            config = {}

            # Sample around successful parameter ranges
            for param, stats in param_stats.items():
                if param in ["asset", "strategy"]:
                    continue

                mean = stats["mean"]
                std = stats["std"]

                # Exploration decreases over iterations
                exploration_factor = max(0.1, 1.0 - iteration / self.max_iterations)

                if param == "entry_threshold":
                    value = np.clip(np.random.normal(mean, std * exploration_factor), 0.1, 0.6)
                elif param == "min_active_domains":
                    value = int(np.clip(np.random.normal(mean, std * exploration_factor), 2, 5))
                elif param == "cooldown_days":
                    value = int(np.clip(np.random.normal(mean, std * exploration_factor), 1, 21))
                elif param == "risk_pct":
                    value = np.clip(np.random.normal(mean, std * exploration_factor), 0.005, 0.1)
                elif param == "sl_atr_multiplier":
                    value = np.clip(np.random.normal(mean, std * exploration_factor), 0.5, 3.0)
                elif param == "tp_atr_multiplier":
                    value = np.clip(np.random.normal(mean, std * exploration_factor), 1.0, 6.0)
                elif param == "trail_atr_multiplier":
                    value = np.clip(np.random.normal(mean, std * exploration_factor), 0.3, 2.0)
                else:
                    value = mean

                config[param] = value

            # Add metadata
            config["asset"] = random.choice(self.assets if hasattr(self, 'assets') else ["ETH", "BTC"])
            config["strategy"] = "stage_b"
            config["iteration"] = iteration

            suggestions.append(config)

        return suggestions

    def _analyze_winner_params(self, winners: List[Dict]) -> Dict:
        """Analyze parameter distributions from winners"""
        param_stats = {}

        numeric_params = [
            "entry_threshold", "min_active_domains", "cooldown_days",
            "risk_pct", "sl_atr_multiplier", "tp_atr_multiplier", "trail_atr_multiplier"
        ]

        for param in numeric_params:
            values = []
            for winner in winners:
                config = winner.get("config", {})
                if param in config:
                    values.append(config[param])

            if values:
                param_stats[param] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "min": np.min(values),
                    "max": np.max(values),
                    "count": len(values)
                }

        return param_stats

class StageCStrategy:
    """Stage C: Walk-forward validation and stability testing"""

    def __init__(self, stage_b_results: str, stability_threshold: float = 0.8):
        self.stage_b_results = stage_b_results
        self.stability_threshold = stability_threshold

    def create_walk_forward_schedule(self, train_months: int = 12,
                                   val_months: int = 6) -> List[Dict]:
        """Create walk-forward validation schedule"""
        start_date = datetime(2022, 1, 1)
        end_date = datetime(2024, 12, 31)

        schedule = []
        current = start_date

        while current + timedelta(days=(train_months + val_months) * 30) <= end_date:
            train_start = current
            train_end = current + timedelta(days=train_months * 30)
            val_start = train_end
            val_end = val_start + timedelta(days=val_months * 30)

            schedule.append({
                "train_start": train_start.strftime("%Y-%m-%d"),
                "train_end": train_end.strftime("%Y-%m-%d"),
                "val_start": val_start.strftime("%Y-%m-%d"),
                "val_end": val_end.strftime("%Y-%m-%d"),
                "fold_id": len(schedule) + 1
            })

            # Move forward by val_months
            current = val_start

        return schedule

    def analyze_stability(self, results_file: str) -> Dict:
        """Analyze configuration stability across folds"""
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

            # Group by configuration
            config_results = {}
            for result in results:
                config_key = self._get_config_key(result.get("config", {}))
                if config_key not in config_results:
                    config_results[config_key] = []
                config_results[config_key].append(result)

            # Analyze stability for each config
            stability_analysis = {}
            for config_key, fold_results in config_results.items():
                if len(fold_results) >= 3:  # Need at least 3 folds
                    stability_analysis[config_key] = self._calculate_stability_metrics(fold_results)

            return stability_analysis

        except Exception as e:
            print(f"Error analyzing stability: {e}")
            return {}

    def _get_config_key(self, config: Dict) -> str:
        """Generate unique key for configuration"""
        key_params = [
            "entry_threshold", "min_active_domains", "cooldown_days",
            "risk_pct", "sl_atr_multiplier", "tp_atr_multiplier"
        ]
        key_parts = []
        for param in key_params:
            if param in config:
                key_parts.append(f"{param}={config[param]}")
        return "|".join(key_parts)

    def _calculate_stability_metrics(self, fold_results: List[Dict]) -> Dict:
        """Calculate stability metrics across folds"""
        metrics = ["win_rate", "total_pnl_pct", "profit_factor", "max_drawdown_pct"]

        stability = {}
        for metric in metrics:
            values = [r.get(metric, 0) for r in fold_results]
            if values:
                stability[f"{metric}_mean"] = np.mean(values)
                stability[f"{metric}_std"] = np.std(values)
                stability[f"{metric}_cv"] = np.std(values) / max(abs(np.mean(values)), 1e-6)
                stability[f"{metric}_min"] = np.min(values)
                stability[f"{metric}_max"] = np.max(values)

        # Overall stability score
        cv_scores = [stability.get(f"{m}_cv", 1) for m in metrics]
        stability["overall_stability"] = 1 / (1 + np.mean(cv_scores))

        # Catastrophic failure check
        dd_values = [r.get("max_drawdown_pct", 0) for r in fold_results]
        stability["max_catastrophic_dd"] = max(dd_values) if dd_values else 0
        stability["catastrophic_folds"] = sum(1 for dd in dd_values if dd > 25)

        # Trade frequency stability
        trade_counts = [r.get("total_trades", 0) for r in fold_results]
        stability["min_trades_per_fold"] = min(trade_counts) if trade_counts else 0
        stability["trades_cv"] = np.std(trade_counts) / max(np.mean(trade_counts), 1)

        stability["fold_count"] = len(fold_results)
        stability["is_stable"] = (
            stability["overall_stability"] > self.stability_threshold and
            stability["catastrophic_folds"] == 0 and
            stability["min_trades_per_fold"] >= 3
        )

        return stability

    def get_production_candidates(self, stability_analysis: Dict) -> List[Dict]:
        """Get configurations ready for production"""
        candidates = []

        for config_key, stability in stability_analysis.items():
            if stability.get("is_stable", False):
                candidates.append({
                    "config_key": config_key,
                    "stability_score": stability["overall_stability"],
                    "mean_pnl": stability.get("total_pnl_pct_mean", 0),
                    "mean_sharpe": stability.get("sharpe_ratio_mean", 0),
                    "max_dd": stability.get("max_drawdown_pct_max", 0),
                    "fold_count": stability["fold_count"],
                    "min_trades": stability["min_trades_per_fold"]
                })

        # Sort by stability score
        candidates.sort(key=lambda x: x["stability_score"], reverse=True)
        return candidates

# Factory function to create optimization strategies
def create_optimization_strategy(stage: str, **kwargs) -> Any:
    """Factory function to create optimization strategies"""
    if stage.upper() == "A":
        return StageAStrategy(**kwargs)
    elif stage.upper() == "B":
        return StageBStrategy(**kwargs)
    elif stage.upper() == "C":
        return StageCStrategy(**kwargs)
    else:
        raise ValueError(f"Unknown optimization stage: {stage}")

# Example usage
if __name__ == "__main__":
    print("Testing optimization strategies...")

    # Test Stage A
    stage_a = StageAStrategy(assets=["ETH"], max_configs=10)
    grid = stage_a.generate_grid()
    print(f"Stage A grid: {len(grid)} configurations")

    # Test walk-forward folds
    folds = stage_a.create_walk_forward_folds()
    print(f"Walk-forward folds: {len(folds)}")

    print("Optimization strategies test complete.")