#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - Multi-Asset Optimization
Parallel optimization across BTC, ETH, SOL, XRP for portfolio construction
"""

import numpy as np
import pandas as pd
import json
import time
from typing import List, Dict, Any, Tuple
from datetime import datetime
from safe_grid_runner import SafeGridRunner
from pathlib import Path
import concurrent.futures
from itertools import product

class MultiAssetOptimizer(SafeGridRunner):
    """Multi-asset optimization with portfolio-aware parameter selection"""

    def __init__(self):
        super().__init__(max_workers=6, timeout_s=300)  # Higher parallelism for multi-asset

        # Asset-specific data paths
        self.asset_data_paths = {
            'ETH': {
                '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv',
                '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv',
                '1H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv'
            },
            'BTC': {
                '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 1D_85c84.csv',
                '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 240_c2b76.csv',
                '1H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 60_50ad4.csv'
            }
            # SOL and XRP paths would go here when data is available
        }

    def create_multi_asset_grid(self, assets: List[str] = None) -> List[Tuple]:
        """Create parameter grid for multiple assets"""

        if assets is None:
            assets = ['ETH', 'BTC']  # Start with available data

        # Core parameter ranges
        entry_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4]
        min_domains = [3]  # Keep focused on 3+ domains
        cooldown_days = [5, 7, 10]
        risk_levels = [0.015, 0.02, 0.025, 0.03]  # Conservative for multi-asset
        sl_multipliers = [1.2, 1.4, 1.6, 1.8, 2.0]
        tp_multipliers = [2.0, 2.5, 3.0, 3.5, 4.0]
        trail_multipliers = [0.8]  # Keep simple for now

        # Time period for optimization
        periods = ["2023-01-01..2025-01-01"]

        tasks = []
        for asset in assets:
            if asset not in self.asset_data_paths:
                print(f"⚠️  Skipping {asset}: No data paths configured")
                continue

            for (entry, min_dom, cd, risk, sl, tp, trail, period) in product(
                entry_thresholds, min_domains, cooldown_days, risk_levels,
                sl_multipliers, tp_multipliers, trail_multipliers, periods
            ):
                if tp > sl:  # TP must be greater than SL
                    cfg = f"thresh{entry}_min{min_dom}_cd{cd}_r{risk}_sl{sl}_tp{tp}_tr{trail}"
                    tasks.append((asset, "1D", period, cfg))

        return tasks

    def analyze_asset_performance(self, results: List[Dict]) -> Dict[str, Any]:
        """Analyze performance by asset and find correlation patterns"""

        asset_stats = {}

        # Group results by asset
        for result in results:
            if result.get("status") != "ok" or result.get("trades", 0) == 0:
                continue

            asset = result.get("asset")
            if asset not in asset_stats:
                asset_stats[asset] = []
            asset_stats[asset].append(result)

        analysis = {
            "asset_summary": {},
            "cross_asset_insights": {},
            "portfolio_recommendations": []
        }

        # Per-asset analysis
        for asset, asset_results in asset_stats.items():
            if not asset_results:
                continue

            pnls = [r.get("pnl_pct", 0) for r in asset_results]
            sharpes = [r.get("sharpe", 0) for r in asset_results]
            trade_counts = [r.get("trades", 0) for r in asset_results]
            win_rates = [r.get("wr", 0) for r in asset_results]

            # Find best configuration for this asset
            best_result = max(asset_results, key=lambda x: self.calculate_portfolio_utility(x))

            analysis["asset_summary"][asset] = {
                "total_configs": len(asset_results),
                "avg_pnl": np.mean(pnls),
                "std_pnl": np.std(pnls),
                "avg_sharpe": np.mean(sharpes),
                "avg_trades": np.mean(trade_counts),
                "avg_win_rate": np.mean(win_rates),
                "best_config": {
                    "cfg": best_result.get("cfg"),
                    "pnl_pct": best_result.get("pnl_pct", 0),
                    "sharpe": best_result.get("sharpe", 0),
                    "trades": best_result.get("trades", 0),
                    "wr": best_result.get("wr", 0),
                    "utility": self.calculate_portfolio_utility(best_result)
                }
            }

        # Cross-asset correlation analysis
        if len(asset_stats) >= 2:
            analysis["cross_asset_insights"] = self.analyze_cross_asset_patterns(asset_stats)

        # Portfolio construction recommendations
        analysis["portfolio_recommendations"] = self.generate_portfolio_recommendations(asset_stats)

        return analysis

    def calculate_portfolio_utility(self, result: Dict) -> float:
        """Portfolio-focused utility calculation emphasizing stability and Sharpe"""

        trades = result.get("trades", 0)
        wr = result.get("wr", 0)
        pnl = result.get("pnl_pct", 0)
        pf = result.get("pf", 0)
        dd = result.get("dd_pct", 0)
        sharpe = result.get("sharpe", 0)

        if trades < 3:
            return 0

        # Normalize win rate format
        wr_normalized = wr / 100 if wr > 1 else wr

        # Portfolio utility emphasizes consistency and risk-adjusted returns
        sharpe_score = min(sharpe / 5.0, 1.0)  # Cap at 5.0 Sharpe
        pnl_score = max(0, min(pnl / 20.0, 1.0))  # Cap at 20% PnL
        consistency_score = wr_normalized
        stability_score = max(0, (1 - dd / 30.0))  # Penalize drawdowns > 30%

        # Portfolio weights: favor Sharpe and stability
        utility = (
            0.4 * sharpe_score +
            0.3 * stability_score +
            0.2 * consistency_score +
            0.1 * pnl_score
        )

        return utility

    def analyze_cross_asset_patterns(self, asset_stats: Dict[str, List]) -> Dict:
        """Analyze patterns across different assets"""

        insights = {
            "parameter_effectiveness": {},
            "risk_level_analysis": {},
            "entry_threshold_analysis": {}
        }

        # Analyze which parameters work best across assets
        all_results = []
        for asset_results in asset_stats.values():
            all_results.extend(asset_results)

        # Group by parameter types
        param_analysis = {}
        for result in all_results:
            cfg = result.get("cfg", "")
            utility = self.calculate_portfolio_utility(result)

            # Extract key parameters
            if "thresh" in cfg:
                thresh = cfg.split("_")[0].replace("thresh", "")
                if thresh not in param_analysis:
                    param_analysis[thresh] = []
                param_analysis[thresh].append(utility)

        # Find universally good parameters
        for param, utilities in param_analysis.items():
            if len(utilities) >= 2:  # At least 2 data points
                insights["parameter_effectiveness"][param] = {
                    "avg_utility": np.mean(utilities),
                    "std_utility": np.std(utilities),
                    "sample_count": len(utilities)
                }

        return insights

    def generate_portfolio_recommendations(self, asset_stats: Dict[str, List]) -> List[str]:
        """Generate portfolio construction recommendations"""

        recommendations = []

        if not asset_stats:
            recommendations.append("❌ No successful configurations found across any assets")
            return recommendations

        # Find best configuration per asset
        best_configs = {}
        for asset, results in asset_stats.items():
            if results:
                best = max(results, key=lambda x: self.calculate_portfolio_utility(x))
                best_configs[asset] = best

        if len(best_configs) >= 2:
            recommendations.append("🎯 MULTI-ASSET PORTFOLIO READY")

            total_utility = 0
            for asset, config in best_configs.items():
                utility = self.calculate_portfolio_utility(config)
                total_utility += utility
                pnl = config.get("pnl_pct", 0)
                sharpe = config.get("sharpe", 0)
                trades = config.get("trades", 0)

                recommendations.append(
                    f"   {asset}: {config.get('cfg')[:30]}... "
                    f"({trades}T, {pnl:+.1f}% PnL, {sharpe:.1f} Sharpe)"
                )

            recommendations.append(f"📊 Portfolio Utility Score: {total_utility:.3f}")

            # Risk recommendations
            avg_sharpe = np.mean([config.get("sharpe", 0) for config in best_configs.values()])
            if avg_sharpe > 3.0:
                recommendations.append("✅ Portfolio shows strong risk-adjusted returns")
            else:
                recommendations.append("⚠️  Consider optimizing for higher Sharpe ratios")

        else:
            recommendations.append("⚠️  Single asset portfolio - consider expanding coverage")

        return recommendations

    def run_multi_asset_optimization(self, assets: List[str] = None):
        """Main multi-asset optimization entry point"""

        print(f"🌍 Bull Machine v1.6.2 - Multi-Asset Optimization")
        print(f"Git commit: {self.git_commit}")
        print(f"System: {self.system_info}")
        print(f"Assets: {assets or ['ETH', 'BTC']}")
        print("-" * 70)

        # Create parameter grid
        tasks = self.create_multi_asset_grid(assets)
        print(f"📊 Total configurations: {len(tasks)}")

        # Check existing results
        results_file = Path("reports/opt/multi_asset_results.jsonl")
        done_keys = set()

        if results_file.exists():
            with open(results_file, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        key = (result["asset"], result["fold"], result["cfg"])
                        done_keys.add(key)
                    except:
                        continue

        # Filter remaining tasks
        todo_tasks = []
        for task in tasks:
            asset, tf, fold, cfg = task
            if (asset, fold, cfg) not in done_keys:
                todo_tasks.append(task)

        print(f"✅ Already completed: {len(tasks) - len(todo_tasks)}")
        print(f"🔄 Remaining: {len(todo_tasks)}")

        if not todo_tasks:
            print("🎉 All multi-asset optimizations already complete!")
        else:
            # Run optimizations with progress tracking
            completed = 0
            successful = 0
            errors = 0

            with open(results_file, "a", encoding="utf-8") as f:
                for task in todo_tasks:
                    try:
                        result = self.run_one_backtest(task, seed=42)
                        result["optimization_type"] = "multi_asset"

                        # Write result immediately
                        f.write(json.dumps(result) + "\\n")
                        f.flush()

                        completed += 1
                        status = result.get("status", "unknown")

                        if status == "ok":
                            trades = result.get("trades", 0)
                            if trades > 0:
                                successful += 1
                                pnl = result.get("pnl_pct", 0)
                                wr = result.get("wr", 0)
                                sharpe = result.get("sharpe", 0)
                                asset = result.get("asset")
                                print(f"✅ [{completed}/{len(todo_tasks)}] {asset} → "
                                      f"{trades}T, {wr:.1f}% WR, {pnl:+.1f}% PnL, {sharpe:.1f} Sharpe")
                            else:
                                print(f"⚪ [{completed}/{len(todo_tasks)}] {task[0]} → 0 trades")
                        else:
                            errors += 1
                            print(f"❌ [{completed}/{len(todo_tasks)}] {task[0]} → {status}")

                    except Exception as e:
                        errors += 1
                        print(f"💥 [{completed+1}/{len(todo_tasks)}] {task[0]} → {str(e)[:50]}...")
                        completed += 1

        # Load all results for analysis
        all_results = []
        if results_file.exists():
            with open(results_file, 'r') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        if result.get("status") == "ok":
                            all_results.append(result)
                    except:
                        continue

        # Analyze results
        analysis = self.analyze_asset_performance(all_results)

        # Save analysis
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        analysis_file = f"reports/opt/multi_asset_analysis_{timestamp}.json"

        with open(analysis_file, 'w') as f:
            json.dump(analysis, f, indent=2, default=str)

        # Print summary
        print(f"\\n{'='*70}")
        print(f"🌍 MULTI-ASSET OPTIMIZATION COMPLETE")
        print(f"{'='*70}")

        if "asset_summary" in analysis:
            for asset, stats in analysis["asset_summary"].items():
                best = stats["best_config"]
                print(f"📈 {asset}: {stats['total_configs']} configs, "
                      f"best: {best['pnl_pct']:+.1f}% PnL, {best['sharpe']:.1f} Sharpe")

        print(f"\\n💡 PORTFOLIO RECOMMENDATIONS:")
        for rec in analysis.get("portfolio_recommendations", []):
            print(f"   {rec}")

        print(f"\\n📁 Results: {results_file}")
        print(f"📁 Analysis: {analysis_file}")

def main():
    """Main entry point for multi-asset optimization"""
    optimizer = MultiAssetOptimizer()
    optimizer.run_multi_asset_optimization(['ETH', 'BTC'])

if __name__ == "__main__":
    main()