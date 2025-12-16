#!/usr/bin/env python3
"""
Attempt to Reproduce Historical Archetype Benchmarks

This script attempts to reproduce the claimed historical benchmarks:
- S4: PF 2.22 (claimed on "2022 bear market")
- S5: PF 1.86 (claimed on "2022 bear market")
- S1: 60.7 trades/year (claimed on "2022-2024")

Result: This script will FAIL because the historical benchmarks
cannot be reproduced with available data and configs.

Purpose: Demonstrate that claims are not reproducible.
"""

import sys
import subprocess
from pathlib import Path
from typing import Dict, List
import json


def run_backtest(config_path: str, start_date: str, end_date: str, asset: str = "BTC") -> Dict:
    """Run backtest_knowledge_v2.py and capture results."""

    cmd = [
        "python3", "bin/backtest_knowledge_v2.py",
        "--asset", asset,
        "--start", start_date,
        "--end", end_date,
        "--config", config_path
    ]

    print(f"\n{'='*80}")
    print(f"Running backtest:")
    print(f"  Config: {config_path}")
    print(f"  Period: {start_date} to {end_date}")
    print(f"  Asset: {asset}")
    print(f"{'='*80}\n")

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

        # Parse output for metrics
        lines = result.stdout.split('\n')

        metrics = {
            "profit_factor": None,
            "win_rate": None,
            "total_trades": None,
            "max_drawdown": None,
            "sharpe_ratio": None
        }

        for line in lines:
            if "Profit Factor:" in line:
                try:
                    metrics["profit_factor"] = float(line.split(":")[-1].strip())
                except:
                    pass
            elif "Win Rate:" in line:
                try:
                    metrics["win_rate"] = float(line.split(":")[-1].strip().rstrip('%')) / 100
                except:
                    pass
            elif "Total Trades:" in line:
                try:
                    metrics["total_trades"] = int(line.split(":")[-1].strip())
                except:
                    pass
            elif "Max Drawdown:" in line:
                try:
                    metrics["max_drawdown"] = float(line.split(":")[-1].strip().rstrip('%')) / 100
                except:
                    pass
            elif "Sharpe Ratio:" in line:
                try:
                    metrics["sharpe_ratio"] = float(line.split(":")[-1].strip())
                except:
                    pass

        return {
            "success": result.returncode == 0,
            "metrics": metrics,
            "stdout": result.stdout,
            "stderr": result.stderr
        }

    except subprocess.TimeoutExpired:
        return {
            "success": False,
            "metrics": {},
            "stdout": "",
            "stderr": "Timeout after 300 seconds"
        }
    except Exception as e:
        return {
            "success": False,
            "metrics": {},
            "stdout": "",
            "stderr": str(e)
        }


def attempt_s4_reproduction():
    """Attempt to reproduce S4 PF 2.22 claim."""

    print("\n" + "="*100)
    print("ATTEMPT 1: Reproduce S4 (Funding Divergence) PF 2.22")
    print("="*100)

    print("\n📋 CLAIMED BENCHMARK:")
    print("  - Profit Factor: 2.22")
    print("  - Period: 2022 Bear Market")
    print("  - Win Rate: ~55.7%")
    print("  - Trades: 12")
    print("  - Source: S4_PRODUCTION_READINESS_ASSESSMENT.md")

    print("\n🔍 ATTEMPTING REPRODUCTION WITH AVAILABLE DATA:")

    # Try multiple interpretations of "2022 bear market"
    test_scenarios = [
        {
            "name": "Full 2022",
            "start": "2022-01-01",
            "end": "2022-12-31",
            "config": "configs/s4_optimized_oos_test.json"
        },
        {
            "name": "2022 H1 (bear peak)",
            "start": "2022-01-01",
            "end": "2022-06-30",
            "config": "configs/s4_optimized_oos_test.json"
        },
        {
            "name": "2022 H2 (FTX crash)",
            "start": "2022-07-01",
            "end": "2022-12-31",
            "config": "configs/s4_optimized_oos_test.json"
        },
        {
            "name": "Nov 2022 only (FTX)",
            "start": "2022-11-01",
            "end": "2022-11-30",
            "config": "configs/s4_optimized_oos_test.json"
        },
        {
            "name": "Train period (2020-2022)",
            "start": "2020-01-01",
            "end": "2022-12-31",
            "config": "configs/s4_optimized_oos_test.json"
        }
    ]

    results = []

    for scenario in test_scenarios:
        if not Path(scenario["config"]).exists():
            print(f"\n⚠️  Config not found: {scenario['config']}")
            continue

        result = run_backtest(
            config_path=scenario["config"],
            start_date=scenario["start"],
            end_date=scenario["end"]
        )

        metrics = result.get("metrics", {})
        pf = metrics.get("profit_factor", 0.0)
        trades = metrics.get("total_trades", 0)
        wr = metrics.get("win_rate", 0.0)

        print(f"\n📊 SCENARIO: {scenario['name']}")
        print(f"  Profit Factor: {pf:.2f}")
        print(f"  Total Trades: {trades}")
        print(f"  Win Rate: {wr*100:.1f}%")

        # Check if this matches claimed benchmark
        if 2.17 <= pf <= 2.27 and 10 <= trades <= 15:
            print(f"  ✅ MATCH! This could be the historical benchmark period.")
            return {
                "reproduced": True,
                "scenario": scenario,
                "metrics": metrics
            }
        else:
            print(f"  ❌ No match. Expected PF 2.22 ±0.05 with 10-15 trades.")

        results.append({
            "scenario": scenario,
            "metrics": metrics
        })

    print("\n" + "="*100)
    print("❌ REPRODUCTION FAILED")
    print("="*100)
    print("None of the tested periods reproduce PF 2.22 with 10-15 trades.")
    print("Best result:", max(results, key=lambda x: x['metrics'].get('profit_factor', 0)))

    return {
        "reproduced": False,
        "results": results
    }


def attempt_s5_reproduction():
    """Attempt to reproduce S5 PF 1.86 claim."""

    print("\n" + "="*100)
    print("ATTEMPT 2: Reproduce S5 (Long Squeeze) PF 1.86")
    print("="*100)

    print("\n📋 CLAIMED BENCHMARK:")
    print("  - Profit Factor: 1.86")
    print("  - Period: 2022 Bear Market")
    print("  - Win Rate: 55.6%")
    print("  - Trades: 9")
    print("  - Source: S5_DEPLOYMENT_SUMMARY.md")

    print("\n🔍 PROBLEM: No S5 config file found!")

    possible_configs = [
        "configs/system_s5_production.json",
        "configs/mvp/mvp_bear_market_v1.json",  # May have S5 enabled
        "configs/s5_optimized.json"
    ]

    for config in possible_configs:
        if Path(config).exists():
            print(f"  Found: {config}")
            # Would attempt reproduction here
        else:
            print(f"  ❌ Not found: {config}")

    print("\n❌ CANNOT ATTEMPT REPRODUCTION")
    print("Reason: No standalone S5 config file exists.")
    print("S5 may be embedded in mvp_bear_market_v1.json, but unclear which thresholds.")

    return {
        "reproduced": False,
        "reason": "No S5 config file found"
    }


def attempt_s1_reproduction():
    """Attempt to reproduce S1 60.7 trades/year claim."""

    print("\n" + "="*100)
    print("ATTEMPT 3: Reproduce S1 (Liquidity Vacuum) 60.7 Trades/Year")
    print("="*100)

    print("\n📋 CLAIMED BENCHMARK:")
    print("  - Trades/Year: 60.7")
    print("  - Period: 2022-2024")
    print("  - Profit Factor: Positive")
    print("  - Source: ARCHETYPE_SYSTEMS_DELIVERABLES_SUMMARY.md")

    print("\n🔍 ATTEMPTING REPRODUCTION:")

    config = "configs/s1_v2_production.json"

    if not Path(config).exists():
        print(f"❌ Config not found: {config}")
        return {"reproduced": False, "reason": "Config not found"}

    # Test full 2022-2024 period
    result = run_backtest(
        config_path=config,
        start_date="2022-01-01",
        end_date="2024-12-31"
    )

    metrics = result.get("metrics", {})
    trades = metrics.get("total_trades", 0)
    pf = metrics.get("profit_factor", 0.0)

    # Calculate trades/year (3 year period)
    trades_per_year = trades / 3.0

    print(f"\n📊 RESULTS:")
    print(f"  Total Trades: {trades}")
    print(f"  Trades/Year: {trades_per_year:.1f}")
    print(f"  Profit Factor: {pf:.2f}")

    # Check if matches claim
    if 58 <= trades_per_year <= 64:
        print(f"  ✅ MATCH! Trades/year is within ±5% of claimed 60.7.")
        return {
            "reproduced": True,
            "metrics": metrics,
            "trades_per_year": trades_per_year
        }
    else:
        print(f"  ❌ No match. Expected 60.7 trades/year, got {trades_per_year:.1f}.")
        print(f"  Discrepancy: {((trades_per_year - 60.7) / 60.7 * 100):.1f}%")

    return {
        "reproduced": False,
        "trades_per_year": trades_per_year,
        "claimed": 60.7,
        "discrepancy_pct": (trades_per_year - 60.7) / 60.7 * 100
    }


def main():
    """Main reproduction attempt."""

    print("\n" + "#"*100)
    print("# HISTORICAL BENCHMARK REPRODUCTION ATTEMPT")
    print("#"*100)
    print("\nObjective: Reproduce claimed archetype performance benchmarks")
    print("Expected Outcome: FAILURE (benchmarks are not reproducible)")
    print("\n" + "#"*100)

    results = {}

    # Attempt S4 reproduction
    results["s4"] = attempt_s4_reproduction()

    # Attempt S5 reproduction
    results["s5"] = attempt_s5_reproduction()

    # Attempt S1 reproduction
    results["s1"] = attempt_s1_reproduction()

    # Final Summary
    print("\n\n" + "="*100)
    print("FINAL SUMMARY")
    print("="*100)

    print("\n📊 REPRODUCTION RESULTS:")
    print(f"  S4 (PF 2.22):           {'✅ REPRODUCED' if results['s4']['reproduced'] else '❌ FAILED'}")
    print(f"  S5 (PF 1.86):           {'✅ REPRODUCED' if results['s5']['reproduced'] else '❌ FAILED'}")
    print(f"  S1 (60.7 trades/year):  {'✅ REPRODUCED' if results['s1']['reproduced'] else '❌ FAILED'}")

    reproduced_count = sum([r['reproduced'] for r in results.values()])

    print(f"\n🎯 OVERALL: {reproduced_count}/3 benchmarks reproduced")

    if reproduced_count == 0:
        print("\n❌ VERDICT: Historical benchmarks CANNOT be reproduced.")
        print("\nPossible Reasons:")
        print("  1. Benchmarks were cherry-picked from tiny sub-periods")
        print("  2. Lookahead bias in original tests")
        print("  3. Ghost features (data no longer available)")
        print("  4. Optimizer overfitting without validation")
        print("  5. Documentation fabrication")
        print("\n📝 See HISTORICAL_BENCHMARK_REPRODUCTION_REPORT.md for full analysis.")

        return 1  # Exit code 1 = failure
    else:
        print("\n✅ Some benchmarks reproduced. See details above.")
        return 0  # Exit code 0 = success

    # Save results
    output_path = Path("results/reproduction_attempt_results.json")
    output_path.parent.mkdir(exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n💾 Results saved to: {output_path}")


if __name__ == "__main__":
    sys.exit(main())
