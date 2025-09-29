#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - Stage C Walk-Forward Validation
Time-based stability analysis for production deployment readiness
"""

import numpy as np
import pandas as pd
import json
import random
from typing import List, Dict, Any, Tuple
from datetime import datetime, timedelta
import time
from safe_grid_runner import SafeGridRunner
from pathlib import Path

class StageCValidator(SafeGridRunner):
    """Stage C validation with walk-forward analysis"""

    def __init__(self):
        super().__init__(max_workers=2, timeout_s=300)  # Conservative for validation

    def create_time_folds(self, start_date: str, end_date: str,
                         train_months: int = 12, validate_months: int = 6) -> List[Tuple[str, str, str, str]]:
        """Create overlapping time folds for walk-forward validation"""

        start = pd.to_datetime(start_date)
        end = pd.to_datetime(end_date)

        folds = []
        current_start = start
        fold_id = 1

        while current_start + timedelta(days=365 + 180) <= end:  # Need at least train + validate period
            train_start = current_start.strftime("%Y-%m-%d")
            train_end = (current_start + timedelta(days=365)).strftime("%Y-%m-%d")
            validate_start = (current_start + timedelta(days=365)).strftime("%Y-%m-%d")
            validate_end = (current_start + timedelta(days=365 + 180)).strftime("%Y-%m-%d")

            # Ensure validate_end doesn't exceed overall end date
            if pd.to_datetime(validate_end) > end:
                validate_end = end.strftime("%Y-%m-%d")

            folds.append((f"fold_{fold_id}", train_start, train_end, validate_start, validate_end))

            # Move forward by 6 months for next fold
            current_start += timedelta(days=180)
            fold_id += 1

        return folds

    def get_stage_b_winners(self, min_trades: int = 5, top_n: int = 10) -> List[Dict]:
        """Extract top performing configurations from Stage A/B results"""

        results_file = Path("reports/opt/results.jsonl")
        if not results_file.exists():
            print("❌ No results file found. Run Stage A/B first.")
            return []

        # Load all results with trades
        trading_results = []
        with open(results_file, 'r') as f:
            for line in f:
                try:
                    result = json.loads(line.strip())
                    if (result.get("status") == "ok" and
                        result.get("trades", 0) >= min_trades):
                        trading_results.append(result)
                except:
                    continue

        if not trading_results:
            print("❌ No profitable configurations found in Stage A/B results.")
            return []

        # Calculate utility scores and rank
        def calculate_utility(result):
            trades = result.get("trades", 0)
            wr = result.get("wr", 0)
            pnl = result.get("pnl_pct", 0)
            pf = result.get("pf", 0)
            dd = result.get("dd_pct", 0)
            sharpe = result.get("sharpe", 0)

            if trades < min_trades:
                return 0

            # Normalized utility with penalties
            n_pnl = max(0, min(1, pnl / 30))
            n_pf = max(0, min(1, (pf - 1) / 2))
            n_wr = wr / 100 if wr > 1 else wr  # Handle percentage format
            n_sharpe = max(0, min(1, sharpe / 3))

            dd_penalty = min(1, dd / 25) ** 2
            freq_penalty = max(0, (min_trades - trades) / min_trades) if trades < 10 else 0

            utility = (1.0 * n_pnl + 0.8 * n_pf + 0.6 * n_wr + 0.4 * n_sharpe -
                      1.2 * dd_penalty - 0.6 * freq_penalty)

            return max(0, utility)

        # Sort by utility and take top N
        scored_results = [(calculate_utility(r), r) for r in trading_results]
        scored_results.sort(reverse=True)

        winners = []
        for score, result in scored_results[:top_n]:
            winners.append({
                "cfg": result["cfg"],
                "asset": result["asset"],
                "utility_score": score,
                "trades": result["trades"],
                "wr": result["wr"],
                "pnl_pct": result["pnl_pct"],
                "pf": result["pf"],
                "sharpe": result["sharpe"],
                "dd_pct": result["dd_pct"]
            })

        return winners

    def run_walk_forward_validation(self, config: Dict, folds: List[Tuple], asset: str = "ETH") -> Dict:
        """Run walk-forward validation on a single configuration"""

        validation_results = {
            "config": config["cfg"],
            "asset": asset,
            "folds": [],
            "stability_metrics": {},
            "deployment_ready": False
        }

        fold_pnls = []
        fold_sharpes = []
        fold_trade_counts = []
        fold_wrs = []

        print(f"\n🔬 Walk-Forward Validation: {config['cfg'][:40]}...")

        for fold_name, train_start, train_end, val_start, val_end in folds:
            print(f"   {fold_name}: Train {train_start}->{train_end}, Validate {val_start}->{val_end}")

            # Run validation period
            val_task = (asset, "1D", f"{val_start}..{val_end}", config["cfg"])
            val_result = self.run_one_backtest(val_task, seed=42)

            if val_result.get("status") == "ok":
                trades = val_result.get("trades", 0)
                wr = val_result.get("wr", 0)
                pnl = val_result.get("pnl_pct", 0)
                pf = val_result.get("pf", 0)
                sharpe = val_result.get("sharpe", 0)
                dd = val_result.get("dd_pct", 0)

                fold_result = {
                    "fold": fold_name,
                    "period": f"{val_start} to {val_end}",
                    "trades": trades,
                    "wr": wr,
                    "pnl_pct": pnl,
                    "profit_factor": pf,
                    "sharpe_ratio": sharpe,
                    "max_dd_pct": dd,
                    "status": "ok"
                }

                # Track for stability analysis
                if trades > 0:
                    fold_pnls.append(pnl)
                    fold_sharpes.append(sharpe)
                    fold_trade_counts.append(trades)
                    fold_wrs.append(wr)

                print(f"      → {trades}T, {wr:.1f}% WR, {pnl:+.2f}% PnL, {sharpe:.2f} Sharpe")

            else:
                fold_result = {
                    "fold": fold_name,
                    "period": f"{val_start} to {val_end}",
                    "status": val_result.get("status", "error"),
                    "error": val_result.get("error", "Unknown error")
                }
                print(f"      → ERROR: {fold_result['status']}")

            validation_results["folds"].append(fold_result)

        # Calculate stability metrics
        if len(fold_pnls) >= 2:
            stability_metrics = {
                "total_folds": len(folds),
                "successful_folds": len(fold_pnls),
                "avg_pnl_pct": np.mean(fold_pnls),
                "std_pnl_pct": np.std(fold_pnls),
                "cv_pnl": np.std(fold_pnls) / abs(np.mean(fold_pnls)) if np.mean(fold_pnls) != 0 else 999,
                "avg_sharpe": np.mean(fold_sharpes),
                "std_sharpe": np.std(fold_sharpes),
                "avg_trades_per_fold": np.mean(fold_trade_counts),
                "positive_folds": len([p for p in fold_pnls if p > 0]),
                "negative_folds": len([p for p in fold_pnls if p < 0]),
                "consistency_ratio": len([p for p in fold_pnls if p > 0]) / len(fold_pnls),
                "max_fold_pnl": max(fold_pnls),
                "min_fold_pnl": min(fold_pnls)
            }

            # Deployment readiness criteria
            is_stable = (
                stability_metrics["cv_pnl"] < 2.0 and  # Low volatility
                stability_metrics["consistency_ratio"] >= 0.6 and  # 60%+ positive periods
                stability_metrics["avg_pnl_pct"] > 0.5 and  # Positive average return
                stability_metrics["successful_folds"] >= len(folds) * 0.8  # 80%+ successful runs
            )

            validation_results["stability_metrics"] = stability_metrics
            validation_results["deployment_ready"] = is_stable

            print(f"   📊 Stability: CV={stability_metrics['cv_pnl']:.2f}, "
                  f"Consistency={stability_metrics['consistency_ratio']:.1%}, "
                  f"Avg PnL={stability_metrics['avg_pnl_pct']:+.2f}%")
            print(f"   {'✅ DEPLOYMENT READY' if is_stable else '❌ NEEDS IMPROVEMENT'}")

        else:
            validation_results["stability_metrics"] = {"error": "Insufficient successful folds"}
            validation_results["deployment_ready"] = False
            print(f"   ❌ INSUFFICIENT DATA: Only {len(fold_pnls)} successful folds")

        return validation_results

    def run_stage_c_validation(self):
        """Main Stage C validation entry point"""

        print(f"🔬 Bull Machine v1.6.2 - Stage C Walk-Forward Validation")
        print(f"Git commit: {self.git_commit}")
        print(f"System: {self.system_info}")
        print("-" * 70)

        # Get top performers from Stage A/B
        winners = self.get_stage_b_winners(min_trades=5, top_n=5)

        if not winners:
            print("❌ No Stage A/B winners found. Run Stage A/B optimization first.")
            return

        print(f"📈 Found {len(winners)} configurations for validation")
        for i, config in enumerate(winners, 1):
            print(f"   {i}. {config['cfg'][:50]}... "
                  f"(Utility: {config['utility_score']:.3f})")

        # Create time folds for walk-forward validation
        folds = self.create_time_folds("2023-01-01", "2025-01-01",
                                     train_months=12, validate_months=6)

        print(f"\n📅 Created {len(folds)} time folds for validation:")
        for fold_name, train_start, train_end, val_start, val_end in folds:
            print(f"   {fold_name}: Train {train_start}->{train_end}, Test {val_start}->{val_end}")

        # Run validation for each configuration
        all_results = []
        deployment_ready_configs = []

        for i, config in enumerate(winners, 1):
            print(f"\n{'='*70}")
            print(f"VALIDATION {i}/{len(winners)}: {config['cfg']}")
            print(f"{'='*70}")

            validation_result = self.run_walk_forward_validation(config, folds)
            all_results.append(validation_result)

            if validation_result["deployment_ready"]:
                deployment_ready_configs.append(validation_result)

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = f"reports/opt/stage_c_validation_{timestamp}.json"

        final_report = {
            "timestamp": datetime.now().isoformat(),
            "git_commit": self.git_commit,
            "validation_summary": {
                "total_configs_tested": len(winners),
                "deployment_ready_count": len(deployment_ready_configs),
                "time_folds": len(folds)
            },
            "all_results": all_results,
            "deployment_ready": deployment_ready_configs
        }

        with open(output_file, 'w') as f:
            json.dump(final_report, f, indent=2, default=str)

        # Print final summary
        print(f"\n{'='*70}")
        print(f"🎯 STAGE C VALIDATION COMPLETE")
        print(f"{'='*70}")
        print(f"📊 Configurations tested: {len(winners)}")
        print(f"✅ Deployment ready: {len(deployment_ready_configs)}")
        print(f"📁 Results saved: {output_file}")

        if deployment_ready_configs:
            print(f"\n🚀 PRODUCTION-READY CONFIGURATIONS:")
            for i, config in enumerate(deployment_ready_configs, 1):
                stability = config["stability_metrics"]
                print(f"   {i}. {config['config'][:50]}...")
                print(f"      Consistency: {stability['consistency_ratio']:.1%}")
                print(f"      Avg PnL: {stability['avg_pnl_pct']:+.2f}%")
                print(f"      Volatility (CV): {stability['cv_pnl']:.2f}")
        else:
            print(f"\n⚠️  NO CONFIGURATIONS READY FOR DEPLOYMENT")
            print(f"   Consider adjusting parameters or extending validation period")

def main():
    """Main entry point for Stage C validation"""
    validator = StageCValidator()
    validator.run_stage_c_validation()

if __name__ == "__main__":
    main()