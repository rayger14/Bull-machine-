#!/usr/bin/env python3
"""
Domain Engine Wiring Verification Test
========================================

Tests whether domain engine wiring (Step 4) actually affects archetype performance.

MISSION:
Prove that domain engines change behavior by comparing Core vs Full variants.

TEST PROTOCOL:
1. Run Core variants (minimal engines): S1_core, S4_core, S5_core
2. Run Full variants (all engines): S1_full, S4_full, S5_full
3. Compare results: Full should differ from Core

EXPECTED PATTERN:
BEFORE WIRING: Core PF = Full PF (engines ignored)
AFTER WIRING:  Core PF ≠ Full PF (engines active)

ACCEPTANCE CRITERIA:
✅ Trade counts differ (Core ≠ Full)
✅ Full variant PF > Core variant PF (by at least 10%)
✅ Domain engines appear in logs
✅ Feature flags control behavior
"""

import sys
import os
import json
import subprocess
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add engine to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test configuration
TEST_CONFIGS = {
    "S1": {
        "core": "configs/variants/s1_core.json",
        "full": "configs/variants/s1_full.json",
        "archetype": "liquidity_vacuum",
        "expected_engines_core": ["wyckoff"],
        "expected_engines_full": ["wyckoff", "smc", "temporal", "hob", "fusion", "macro"]
    },
    "S4": {
        "core": "configs/variants/s4_core.json",
        "full": "configs/variants/s4_full.json",
        "archetype": "funding_divergence",
        "expected_engines_core": ["funding_core"],
        "expected_engines_full": ["macro"]
    },
    "S5": {
        "core": "configs/variants/s5_core.json",
        "full": "configs/variants/s5_full.json",
        "archetype": "long_squeeze",
        "expected_engines_core": ["funding_core", "rsi_momentum"],
        "expected_engines_full": ["wyckoff", "smc", "temporal", "hob", "fusion", "macro"]
    }
}

# Test parameters (2022 bear market for consistent baseline)
START_DATE = "2022-01-01"
END_DATE = "2022-12-31"
ASSET = "BTC"

class DomainWiringTest:
    """Systematic test of domain engine wiring"""

    def __init__(self):
        self.results = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = Path(f"artifacts/domain_wiring_test_{self.timestamp}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run_backtest(self, config_path: str, variant_name: str) -> Dict:
        """Run single backtest and extract results"""
        print(f"\n{'='*80}")
        print(f"Running: {variant_name}")
        print(f"Config: {config_path}")
        print(f"Period: {START_DATE} to {END_DATE}")
        print(f"{'='*80}\n")

        # Run backtest
        cmd = [
            "python3", "bin/backtest_knowledge_v2.py",
            "--asset", ASSET,
            "--config", config_path,
            "--start", START_DATE,
            "--end", END_DATE,
            "--export-trades", str(self.output_dir / variant_name / "trades.csv")
        ]

        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )

            if result.returncode != 0:
                print(f"❌ FAILED: {variant_name}")
                print(f"Error: {result.stderr}")
                return {"status": "failed", "error": result.stderr}

            # Parse output for key metrics
            output = result.stdout
            metrics = self._parse_metrics(output)
            metrics["status"] = "success"
            metrics["variant"] = variant_name

            # Save detailed output
            output_file = self.output_dir / f"{variant_name}_output.txt"
            with open(output_file, "w") as f:
                f.write(output)

            print(f"✅ SUCCESS: {variant_name}")
            print(f"   Trades: {metrics.get('trades', 0)}")
            print(f"   PF: {metrics.get('profit_factor', 0):.2f}")
            print(f"   Win Rate: {metrics.get('win_rate', 0):.1%}")

            return metrics

        except subprocess.TimeoutExpired:
            print(f"⏱️  TIMEOUT: {variant_name}")
            return {"status": "timeout", "variant": variant_name}
        except Exception as e:
            print(f"💥 ERROR: {variant_name} - {str(e)}")
            return {"status": "error", "error": str(e), "variant": variant_name}

    def _parse_metrics(self, output: str) -> Dict:
        """Extract metrics from backtest output"""
        metrics = {
            "trades": 0,
            "profit_factor": 0.0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "max_dd": 0.0,
            "total_pnl": 0.0
        }

        # Parse key metrics from output
        for line in output.split('\n'):
            if "Total Trades:" in line:
                try:
                    metrics["trades"] = int(line.split(':')[-1].strip())
                except:
                    pass
            elif "Profit Factor:" in line:
                try:
                    metrics["profit_factor"] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif "Win Rate:" in line:
                try:
                    val = line.split(':')[-1].strip().replace('%', '')
                    metrics["win_rate"] = float(val) / 100
                except:
                    pass
            elif "Sharpe Ratio:" in line:
                try:
                    metrics["sharpe"] = float(line.split(':')[-1].strip())
                except:
                    pass
            elif "Max Drawdown:" in line:
                try:
                    val = line.split(':')[-1].strip().replace('%', '')
                    metrics["max_dd"] = float(val) / 100
                except:
                    pass
            elif "Total PnL:" in line:
                try:
                    val = line.split(':')[-1].strip().replace('$', '').replace(',', '')
                    metrics["total_pnl"] = float(val)
                except:
                    pass

        return metrics

    def verify_config_flags(self, config_path: str, expected_engines: List[str]) -> bool:
        """Verify that config has correct domain engine flags"""
        with open(config_path) as f:
            config = json.load(f)

        flags = config.get("feature_flags", {})

        print(f"\nVerifying config: {config_path}")
        print(f"Expected engines: {expected_engines}")
        print(f"Feature flags: {json.dumps(flags, indent=2)}")

        # Check that expected engines are enabled
        for engine in expected_engines:
            flag_name = f"enable_{engine}"
            if flag_name in flags and not flags[flag_name]:
                print(f"⚠️  WARNING: {flag_name} is disabled but expected to be enabled")
                return False

        return True

    def compare_variants(self, archetype: str, core_metrics: Dict, full_metrics: Dict) -> Dict:
        """Compare core vs full variant performance"""
        comparison = {
            "archetype": archetype,
            "core": core_metrics,
            "full": full_metrics
        }

        # Calculate differences
        if core_metrics.get("status") == "success" and full_metrics.get("status") == "success":
            core_pf = core_metrics.get("profit_factor", 0)
            full_pf = full_metrics.get("profit_factor", 0)
            core_trades = core_metrics.get("trades", 0)
            full_trades = full_metrics.get("trades", 0)

            if core_pf > 0:
                pf_improvement = ((full_pf - core_pf) / core_pf) * 100
            else:
                pf_improvement = 0

            trade_diff = full_trades - core_trades

            comparison["pf_improvement_pct"] = pf_improvement
            comparison["trade_count_diff"] = trade_diff

            # Verification checks
            comparison["checks"] = {
                "trades_differ": abs(trade_diff) > 0,
                "pf_improved": pf_improvement > 10,  # At least 10% improvement
                "full_better": full_pf > core_pf,
                "config_flags_correct": True  # Set externally
            }

            # Overall status
            all_checks = all(comparison["checks"].values())
            comparison["wiring_verified"] = all_checks

        else:
            comparison["wiring_verified"] = False
            comparison["checks"] = {
                "error": "One or both tests failed"
            }

        return comparison

    def run_all_tests(self):
        """Run complete test suite"""
        print("\n" + "="*80)
        print("DOMAIN ENGINE WIRING VERIFICATION TEST")
        print("="*80)
        print(f"Test Period: {START_DATE} to {END_DATE}")
        print(f"Output Dir: {self.output_dir}")
        print("="*80 + "\n")

        all_comparisons = {}

        for archetype, configs in TEST_CONFIGS.items():
            print(f"\n{'#'*80}")
            print(f"# Testing {archetype}")
            print(f"{'#'*80}\n")

            # Verify config flags
            core_flags_ok = self.verify_config_flags(
                configs["core"],
                configs["expected_engines_core"]
            )
            full_flags_ok = self.verify_config_flags(
                configs["full"],
                configs["expected_engines_full"]
            )

            # Run core variant
            core_metrics = self.run_backtest(configs["core"], f"{archetype}_core")

            # Run full variant
            full_metrics = self.run_backtest(configs["full"], f"{archetype}_full")

            # Compare results
            comparison = self.compare_variants(archetype, core_metrics, full_metrics)
            comparison["checks"]["config_flags_correct"] = core_flags_ok and full_flags_ok

            all_comparisons[archetype] = comparison

            # Print summary
            self.print_comparison(comparison)

        # Generate final report
        self.generate_report(all_comparisons)

        return all_comparisons

    def print_comparison(self, comparison: Dict):
        """Print comparison results"""
        print(f"\n{'='*80}")
        print(f"COMPARISON: {comparison['archetype']}")
        print(f"{'='*80}")

        core = comparison.get("core", {})
        full = comparison.get("full", {})

        print("\nCORE VARIANT:")
        print(f"  Trades: {core.get('trades', 0)}")
        print(f"  PF: {core.get('profit_factor', 0):.2f}")
        print(f"  Win Rate: {core.get('win_rate', 0):.1%}")
        print(f"  Sharpe: {core.get('sharpe', 0):.2f}")

        print("\nFULL VARIANT:")
        print(f"  Trades: {full.get('trades', 0)}")
        print(f"  PF: {full.get('profit_factor', 0):.2f}")
        print(f"  Win Rate: {full.get('win_rate', 0):.1%}")
        print(f"  Sharpe: {full.get('sharpe', 0):.2f}")

        if "pf_improvement_pct" in comparison:
            print(f"\nIMPROVEMENT:")
            print(f"  PF Change: {comparison['pf_improvement_pct']:+.1f}%")
            print(f"  Trade Diff: {comparison['trade_count_diff']:+d}")

        print(f"\nVERIFICATION CHECKS:")
        for check, status in comparison.get("checks", {}).items():
            icon = "✅" if status else "❌"
            print(f"  {icon} {check}: {status}")

        status_icon = "✅" if comparison.get("wiring_verified") else "❌"
        print(f"\n{status_icon} WIRING STATUS: {'VERIFIED' if comparison.get('wiring_verified') else 'FAILED'}")
        print("="*80)

    def generate_report(self, comparisons: Dict):
        """Generate markdown report"""
        report_path = self.output_dir / "DOMAIN_WIRING_VERIFICATION_REPORT.md"

        with open(report_path, "w") as f:
            f.write("# Domain Engine Wiring Verification Report\n\n")
            f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"**Test Period:** {START_DATE} to {END_DATE}\n")
            f.write(f"**Asset:** {ASSET}\n\n")

            f.write("## Executive Summary\n\n")

            verified_count = sum(1 for c in comparisons.values() if c.get("wiring_verified"))
            total_count = len(comparisons)

            f.write(f"**Status:** {verified_count}/{total_count} archetypes verified\n\n")

            if verified_count == total_count:
                f.write("✅ **DOMAIN WIRING VERIFIED** - All archetypes show expected behavior\n\n")
            else:
                f.write("❌ **DOMAIN WIRING ISSUES** - Some archetypes not showing expected behavior\n\n")

            # Results table
            f.write("## Results Summary\n\n")
            f.write("| Archetype | Core PF | Full PF | Improvement | Core Trades | Full Trades | Wiring Status |\n")
            f.write("|-----------|---------|---------|-------------|-------------|-------------|---------------|\n")

            for archetype, comp in comparisons.items():
                core = comp.get("core", {})
                full = comp.get("full", {})

                core_pf = core.get("profit_factor", 0)
                full_pf = full.get("profit_factor", 0)
                core_trades = core.get("trades", 0)
                full_trades = full.get("trades", 0)
                improvement = comp.get("pf_improvement_pct", 0)
                status = "✅" if comp.get("wiring_verified") else "❌"

                f.write(f"| {archetype} | {core_pf:.2f} | {full_pf:.2f} | {improvement:+.1f}% | "
                       f"{core_trades} | {full_trades} | {status} |\n")

            # Detailed results
            f.write("\n## Detailed Results\n\n")

            for archetype, comp in comparisons.items():
                f.write(f"### {archetype}\n\n")

                core = comp.get("core", {})
                full = comp.get("full", {})

                f.write("**Core Variant:**\n")
                f.write(f"- Trades: {core.get('trades', 0)}\n")
                f.write(f"- Profit Factor: {core.get('profit_factor', 0):.2f}\n")
                f.write(f"- Win Rate: {core.get('win_rate', 0):.1%}\n")
                f.write(f"- Sharpe: {core.get('sharpe', 0):.2f}\n\n")

                f.write("**Full Variant:**\n")
                f.write(f"- Trades: {full.get('trades', 0)}\n")
                f.write(f"- Profit Factor: {full.get('profit_factor', 0):.2f}\n")
                f.write(f"- Win Rate: {full.get('win_rate', 0):.1%}\n")
                f.write(f"- Sharpe: {full.get('sharpe', 0):.2f}\n\n")

                f.write("**Verification Checks:**\n")
                for check, status in comp.get("checks", {}).items():
                    icon = "✅" if status else "❌"
                    f.write(f"- {icon} {check}: {status}\n")

                f.write(f"\n**Status:** {'✅ VERIFIED' if comp.get('wiring_verified') else '❌ FAILED'}\n\n")
                f.write("---\n\n")

            # Recommendations
            f.write("## Recommendations\n\n")

            if verified_count == total_count:
                f.write("✅ Domain engine wiring is working correctly. Proceed with:\n")
                f.write("1. Re-optimization with working domain engines\n")
                f.write("2. Individual engine impact analysis\n")
                f.write("3. Production deployment preparation\n\n")
            else:
                f.write("⚠️ Domain engine wiring needs attention:\n")

                for archetype, comp in comparisons.items():
                    if not comp.get("wiring_verified"):
                        f.write(f"\n**{archetype}:**\n")
                        for check, status in comp.get("checks", {}).items():
                            if not status:
                                f.write(f"- Fix: {check}\n")

        print(f"\n📄 Report saved to: {report_path}")

        # Also save JSON results
        json_path = self.output_dir / "results.json"
        with open(json_path, "w") as f:
            json.dump(comparisons, f, indent=2)

        print(f"📊 JSON results saved to: {json_path}")


def main():
    """Run domain wiring verification test"""
    tester = DomainWiringTest()
    results = tester.run_all_tests()

    # Exit code based on results
    all_verified = all(r.get("wiring_verified", False) for r in results.values())
    sys.exit(0 if all_verified else 1)


if __name__ == "__main__":
    main()
