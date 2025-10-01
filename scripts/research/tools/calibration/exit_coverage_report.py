#!/usr/bin/env python3
"""
Exit Coverage Report: Aggregate exit impact & coverage across multiple datasets

Runs exit parameter tests across mixed regimes (BTC/ETH) to force CHoCH/Momentum
coverage and measure parameter effectiveness across different market conditions.
"""

import os
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


class ExitCoverageReport:
    def __init__(self):
        self.datasets = [
            {
                "name": "btc_1h",
                "path": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_BTCUSD, 60_50ad4.csv",
                "regime": "mixed",
            },
            {
                "name": "eth_1h",
                "path": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv",
                "regime": "mixed",
            },
            {
                "name": "eth_4h",
                "path": "/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv",
                "regime": "trending",
            },
        ]

        # More aggressive parameters to force CHoCH/Momentum triggers
        self.test_configs = [
            {"name": "baseline", "params": {}},
            {
                "name": "choch_sensitive",
                "params": {
                    "exit_signals.choch_against.bars_confirm": 1,
                    "exit_signals.choch_against.min_break_strength": 0.02,
                },
            },
            {
                "name": "momentum_sensitive",
                "params": {
                    "exit_signals.momentum_fade.drop_pct": 0.12,
                    "exit_signals.momentum_fade.min_bars_in_pos": 2,
                },
            },
            {
                "name": "all_aggressive",
                "params": {
                    "exit_signals.choch_against.bars_confirm": 1,
                    "exit_signals.choch_against.min_break_strength": 0.015,
                    "exit_signals.momentum_fade.drop_pct": 0.10,
                    "exit_signals.momentum_fade.min_bars_in_pos": 1,
                    "exit_signals.time_stop.max_bars_1h": 18,
                },
            },
        ]

    def create_base_config(self, dataset: Dict[str, str]) -> Dict[str, Any]:
        """Create base config for dataset."""
        return {
            "run_id": f"coverage_{dataset['name']}",
            "data": {
                "sources": {f"{dataset['name'].upper()}_1H": dataset["path"]},
                "timeframes": ["1H"],
                "schema": {
                    "timestamp": {"name": "time", "unit": "s"},
                    "open": "open",
                    "high": "high",
                    "low": "low",
                    "close": "close",
                    "volume": "volume",
                },
            },
            "broker": {"fee_bps": 10, "slippage_bps": 5, "spread_bps": 2, "partial_fill": True},
            "portfolio": {"starting_cash": 100000, "exposure_cap_pct": 0.60, "max_positions": 4},
            "engine": {"lookback_bars": 50, "seed": 42},
            "strategy": {
                "version": "v1.4",
                "config": "bull_machine/configs/diagnostic_v14_step4_config.json",
            },
            "risk": {"base_risk_pct": 0.008, "max_risk_per_trade": 0.02, "min_stop_pct": 0.001},
            "exit_signals": {
                "enabled": True,
                "enabled_exits": ["choch_against", "momentum_fade", "time_stop"],
                "emit_exit_edge_logs": True,
                "choch_against": {
                    "swing_lookback": 3,
                    "bars_confirm": 2,
                    "min_break_strength": 0.05,
                },
                "momentum_fade": {"lookback": 6, "drop_pct": 0.20, "min_bars_in_pos": 4},
                "time_stop": {"max_bars_1h": 24, "max_bars_4h": 8, "max_bars_1d": 4},
                "emit_exit_debug": True,
            },
            "logging": {"level": "INFO", "emit_fusion_debug": False, "emit_exit_debug": True},
        }

    def run_test(self, dataset: Dict[str, str], test_config: Dict[str, Any]) -> Dict[str, Any]:
        """Run single test combination."""
        config = self.create_base_config(dataset)
        test_name = f"{dataset['name']}_{test_config['name']}"
        config["run_id"] = test_name

        # Apply parameter overrides
        for path, value in test_config["params"].items():
            keys = path.split(".")
            current = config
            for key in keys[:-1]:
                current = current[key]
            current[keys[-1]] = value

        # Create temp files
        temp_dir = tempfile.mkdtemp(prefix=f"coverage_{test_name}_")
        config_path = Path(temp_dir) / f"{test_name}.json"
        result_dir = Path(temp_dir) / "results"

        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        print(f"🧪 Running {test_name}...")

        try:
            result = subprocess.run(
                [
                    "python3",
                    "-m",
                    "bull_machine.app.main_backtest",
                    "--config",
                    str(config_path),
                    "--out",
                    str(result_dir),
                ],
                capture_output=True,
                text=True,
                timeout=90,
            )

            if result.returncode == 0:
                # Collect results
                test_result = {
                    "dataset": dataset["name"],
                    "regime": dataset["regime"],
                    "config": test_config["name"],
                    "status": "success",
                    "exit_counts": {},
                    "performance": {},
                    "params_used": test_config["params"],
                }

                # Read exit counts
                exit_counts_path = result_dir / "exit_counts.json"
                if exit_counts_path.exists():
                    with open(exit_counts_path, "r") as f:
                        test_result["exit_counts"] = json.load(f)

                # Read performance summary
                summary_path = result_dir / f"{test_name}_summary.json"
                if summary_path.exists():
                    with open(summary_path, "r") as f:
                        summary = json.load(f)
                        test_result["performance"] = {
                            "trades": summary.get("total_trades", 0),
                            "win_rate": summary.get("win_rate", 0),
                            "expectancy": summary.get("expectancy", 0),
                            "max_dd": summary.get("max_drawdown", 0),
                        }

                print(f"✅ {test_name}: {test_result['exit_counts']}")
                return test_result

            else:
                print(f"❌ {test_name}: Failed with code {result.returncode}")
                return {
                    "dataset": dataset["name"],
                    "config": test_config["name"],
                    "status": "failed",
                    "error": result.stderr[:200],
                }

        except subprocess.TimeoutExpired:
            print(f"❌ {test_name}: Timeout")
            return {"dataset": dataset["name"], "config": test_config["name"], "status": "timeout"}
        finally:
            # Cleanup
            try:
                import shutil

                shutil.rmtree(temp_dir)
            except:
                pass

    def run_full_coverage_report(self) -> Dict[str, Any]:
        """Run coverage tests across all datasets and configs."""
        print("🔬 EXIT COVERAGE REPORT - Mixed Regime Analysis")
        print("=" * 60)
        print("Testing CHoCH/Momentum coverage across BTC/ETH datasets")
        print()

        all_results = []

        for dataset in self.datasets:
            print(f"\n📊 Dataset: {dataset['name']} ({dataset['regime']} regime)")
            print("-" * 40)

            for test_config in self.test_configs:
                result = self.run_test(dataset, test_config)
                all_results.append(result)

        return self.analyze_coverage(all_results)

    def analyze_coverage(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze exit coverage across all tests."""
        print("\n" + "=" * 60)
        print("📊 EXIT COVERAGE ANALYSIS")
        print("=" * 60)

        successful_results = [r for r in results if r.get("status") == "success"]

        if not successful_results:
            print("❌ No successful runs to analyze")
            return {"status": "failed", "results": results}

        # Create coverage matrix
        coverage_matrix = {}
        for result in successful_results:
            dataset = result["dataset"]
            config = result["config"]
            exit_counts = result.get("exit_counts", {})

            if dataset not in coverage_matrix:
                coverage_matrix[dataset] = {}
            coverage_matrix[dataset][config] = exit_counts

        # Print coverage matrix
        print(
            f"\n{'Dataset':<12} {'Config':<18} {'CHoCH':<8} {'Momentum':<10} {'TimeStop':<10} {'None':<8}"
        )
        print("-" * 75)

        for dataset, configs in coverage_matrix.items():
            for config, counts in configs.items():
                choch = counts.get("choch_against", 0)
                momentum = counts.get("momentum_fade", 0)
                timestop = counts.get("time_stop", 0)
                none_exits = counts.get("none", 0)

                print(
                    f"{dataset:<12} {config:<18} {choch:<8} {momentum:<10} {timestop:<10} {none_exits:<8}"
                )

        # Analyze exit type coverage
        print(f"\n🎯 EXIT TYPE COVERAGE ANALYSIS:")

        all_choch = [
            counts.get("choch_against", 0)
            for r in successful_results
            for counts in [r.get("exit_counts", {})]
        ]
        all_momentum = [
            counts.get("momentum_fade", 0)
            for r in successful_results
            for counts in [r.get("exit_counts", {})]
        ]
        all_timestop = [
            counts.get("time_stop", 0)
            for r in successful_results
            for counts in [r.get("exit_counts", {})]
        ]

        choch_coverage = sum(1 for x in all_choch if x > 0)
        momentum_coverage = sum(1 for x in all_momentum if x > 0)
        timestop_coverage = sum(1 for x in all_timestop if x > 0)
        total_tests = len(successful_results)

        print(
            f"CHoCH coverage: {choch_coverage}/{total_tests} tests ({choch_coverage / total_tests * 100:.1f}%)"
        )
        print(
            f"Momentum coverage: {momentum_coverage}/{total_tests} tests ({momentum_coverage / total_tests * 100:.1f}%)"
        )
        print(
            f"TimeStop coverage: {timestop_coverage}/{total_tests} tests ({timestop_coverage / total_tests * 100:.1f}%)"
        )

        # Parameter effectiveness analysis
        print(f"\n🎯 PARAMETER EFFECTIVENESS:")

        baseline_results = [r for r in successful_results if r["config"] == "baseline"]
        choch_results = [r for r in successful_results if r["config"] == "choch_sensitive"]
        momentum_results = [r for r in successful_results if r["config"] == "momentum_sensitive"]

        if baseline_results and choch_results:
            baseline_choch = sum(
                r.get("exit_counts", {}).get("choch_against", 0) for r in baseline_results
            )
            sensitive_choch = sum(
                r.get("exit_counts", {}).get("choch_against", 0) for r in choch_results
            )
            print(f"CHoCH sensitivity effect: {baseline_choch} → {sensitive_choch} exits")

        if baseline_results and momentum_results:
            baseline_momentum = sum(
                r.get("exit_counts", {}).get("momentum_fade", 0) for r in baseline_results
            )
            sensitive_momentum = sum(
                r.get("exit_counts", {}).get("momentum_fade", 0) for r in momentum_results
            )
            print(f"Momentum sensitivity effect: {baseline_momentum} → {sensitive_momentum} exits")

        # Success criteria
        success_criteria = {
            "choch_triggers": choch_coverage > 0,
            "momentum_triggers": momentum_coverage > 0,
            "timestop_variance": len(set(all_timestop)) > 1,
            "parameter_effects": True,  # Will enhance this
        }

        overall_success = all(success_criteria.values())

        print(f"\n🏆 COVERAGE REPORT SUMMARY:")
        print(f"✅ CHoCH triggers: {'PASS' if success_criteria['choch_triggers'] else 'FAIL'}")
        print(
            f"✅ Momentum triggers: {'PASS' if success_criteria['momentum_triggers'] else 'FAIL'}"
        )
        print(
            f"✅ TimeStop variance: {'PASS' if success_criteria['timestop_variance'] else 'FAIL'}"
        )
        print(f"Overall: {'SUCCESS' if overall_success else 'PARTIAL SUCCESS'}")

        return {
            "status": "success" if overall_success else "partial",
            "coverage_matrix": coverage_matrix,
            "success_criteria": success_criteria,
            "results": successful_results,
            "recommendations": self.generate_recommendations(successful_results, success_criteria),
        }

    def generate_recommendations(
        self, results: List[Dict[str, Any]], criteria: Dict[str, bool]
    ) -> List[str]:
        """Generate recommendations based on coverage analysis."""
        recommendations = []

        if not criteria["choch_triggers"]:
            recommendations.append(
                "CHoCH not triggering - consider more aggressive parameters or different market periods"
            )

        if not criteria["momentum_triggers"]:
            recommendations.append(
                "Momentum not triggering - try lower drop_pct thresholds or shorter lookbacks"
            )

        if criteria["timestop_variance"]:
            recommendations.append(
                "TimeStop working correctly - use as baseline for parameter validation"
            )

        # Check for dataset differences
        dataset_performance = {}
        for result in results:
            dataset = result["dataset"]
            if dataset not in dataset_performance:
                dataset_performance[dataset] = []
            dataset_performance[dataset].append(result)

        if len(dataset_performance) > 1:
            for dataset, dataset_results in dataset_performance.items():
                total_exits = sum(
                    sum(r.get("exit_counts", {}).values()) - r.get("exit_counts", {}).get("none", 0)
                    for r in dataset_results
                )
                if total_exits > 0:
                    recommendations.append(
                        f"Dataset {dataset} shows exit activity - good for parameter tuning"
                    )

        return recommendations


def main():
    reporter = ExitCoverageReport()
    report = reporter.run_full_coverage_report()

    # Save report
    with open("exit_coverage_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\n📄 Full report saved to: exit_coverage_report.json")

    if report.get("recommendations"):
        print(f"\n💡 RECOMMENDATIONS:")
        for rec in report["recommendations"]:
            print(f"• {rec}")


if __name__ == "__main__":
    main()
