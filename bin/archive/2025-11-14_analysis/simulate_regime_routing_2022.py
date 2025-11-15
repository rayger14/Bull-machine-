#!/usr/bin/env python3
"""
Simulate Regime Routing Impact on 2022 Bear Market Performance

Replays 2022 backtest with different regime weight configurations to measure impact.
Validates that aggressive suppression of bull archetypes improves performance.

Usage:
    python bin/simulate_regime_routing_2022.py --scenario all
    python bin/simulate_regime_routing_2022.py --scenario aggressive --asset BTC
    python bin/simulate_regime_routing_2022.py --compare-2024  # Safety check
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
import pandas as pd

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# Regime Weight Scenarios
# ============================================================================

SCENARIOS = {
    "baseline_no_routing": {
        "name": "Baseline (No Routing)",
        "description": "Current broken state - equal 1.0x weights for all archetypes",
        "routing": {}  # Empty routing config (all defaults to 1.0)
    },

    "moderate_suppression": {
        "name": "Moderate Suppression (0.3x)",
        "description": "Conservative approach - 70% suppression of trap_within_trend",
        "routing": {
            "risk_off": {
                "weights": {
                    "trap_within_trend": 0.3,
                    "order_block_retest": 0.5,
                    "volume_exhaustion": 0.6,
                    "wick_trap": 0.4,
                    "spring": 0.4,
                    "rejection": 1.5,
                    "long_squeeze": 1.6,
                    "breakdown": 1.5,
                    "distribution": 1.4,
                    "volume_fade_chop": 1.3
                }
            },
            "crisis": {
                "weights": {
                    "trap_within_trend": 0.2,
                    "order_block_retest": 0.3,
                    "volume_exhaustion": 0.4,
                    "rejection": 1.8,
                    "long_squeeze": 2.0,
                    "breakdown": 2.0,
                    "distribution": 1.8
                }
            }
        }
    },

    "aggressive_suppression": {
        "name": "Aggressive Suppression (0.2x)",
        "description": "Recommended approach - 80% suppression of trap_within_trend",
        "routing": {
            "risk_on": {
                "weights": {
                    "trap_within_trend": 1.3,
                    "volume_exhaustion": 1.1,
                    "order_block_retest": 1.4,
                    "wick_trap": 1.2,
                    "spring": 1.3,
                    "rejection": 0.3,
                    "long_squeeze": 0.2,
                    "breakdown": 0.1,
                    "distribution": 0.2,
                    "whipsaw": 0.3,
                    "volume_fade_chop": 0.3
                }
            },
            "neutral": {
                "weights": {
                    "trap_within_trend": 1.0,
                    "volume_exhaustion": 1.0,
                    "order_block_retest": 1.0,
                    "wick_trap": 1.0,
                    "spring": 1.0,
                    "rejection": 0.7,
                    "long_squeeze": 0.6,
                    "breakdown": 0.5,
                    "distribution": 0.7,
                    "whipsaw": 0.6,
                    "volume_fade_chop": 0.8
                }
            },
            "risk_off": {
                "weights": {
                    "trap_within_trend": 0.2,
                    "volume_exhaustion": 0.5,
                    "order_block_retest": 0.4,
                    "wick_trap": 0.3,
                    "spring": 0.3,
                    "rejection": 1.8,
                    "long_squeeze": 2.0,
                    "breakdown": 2.0,
                    "distribution": 1.9,
                    "whipsaw": 1.6,
                    "volume_fade_chop": 1.5
                },
                "final_gate_delta": 0.02
            },
            "crisis": {
                "weights": {
                    "trap_within_trend": 0.1,
                    "volume_exhaustion": 0.3,
                    "order_block_retest": 0.2,
                    "wick_trap": 0.2,
                    "spring": 0.2,
                    "rejection": 2.2,
                    "long_squeeze": 2.5,
                    "breakdown": 2.5,
                    "distribution": 2.3,
                    "whipsaw": 0.5,
                    "volume_fade_chop": 1.8
                },
                "final_gate_delta": 0.04
            }
        }
    },

    "extreme_suppression": {
        "name": "Extreme Suppression (0.1x)",
        "description": "Experimental - 90% suppression (may over-filter)",
        "routing": {
            "risk_off": {
                "weights": {
                    "trap_within_trend": 0.1,
                    "order_block_retest": 0.2,
                    "volume_exhaustion": 0.3,
                    "rejection": 2.5,
                    "long_squeeze": 3.0,
                    "breakdown": 2.8,
                    "distribution": 2.5
                }
            }
        }
    }
}


# ============================================================================
# Backtest Execution
# ============================================================================

def create_routing_config(base_config_path: str, routing: dict, output_path: str) -> str:
    """
    Create a new config with routing weights injected.

    Args:
        base_config_path: Path to baseline config
        routing: Routing weights to inject
        output_path: Where to save modified config

    Returns:
        Path to modified config
    """
    with open(base_config_path) as f:
        config = json.load(f)

    # Inject routing config at top level (NOT under archetypes)
    config['routing'] = routing

    # Enable bear archetypes if they have non-zero weights
    if 'risk_off' in routing and 'weights' in routing['risk_off']:
        weights = routing['risk_off']['weights']
        if 'rejection' in weights and weights['rejection'] > 0.5:
            config['archetypes']['enable_S2'] = True
        if 'long_squeeze' in weights and weights['long_squeeze'] > 0.5:
            config['archetypes']['enable_S5'] = True
        if 'breakdown' in weights and weights['breakdown'] > 0.5:
            config['archetypes']['enable_S1'] = True
        if 'distribution' in weights and weights['distribution'] > 0.5:
            config['archetypes']['enable_S4'] = True
        if 'whipsaw' in weights and weights['whipsaw'] > 0.5:
            config['archetypes']['enable_S3'] = True
        if 'volume_fade_chop' in weights and weights['volume_fade_chop'] > 0.5:
            config['archetypes']['enable_S8'] = True

    # Add metadata
    config['_simulation_metadata'] = {
        "generated_by": "simulate_regime_routing_2022.py",
        "timestamp": datetime.now().isoformat(),
        "base_config": base_config_path
    }

    # Write modified config
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)

    return output_path


def run_backtest(
    asset: str,
    start: str,
    end: str,
    config_path: str,
    results_dir: str
) -> Dict[str, Any]:
    """
    Run backtest with specified config.

    Args:
        asset: BTC or ETH
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        config_path: Path to config file
        results_dir: Where to save results

    Returns:
        Results dict with metrics
    """
    cmd = [
        sys.executable,
        "bin/backtest_knowledge_v2.py",
        "--asset", asset,
        "--start", start,
        "--end", end,
        "--config", config_path,
        "--results-dir", results_dir
    ]

    print(f"\n[BACKTEST] Running: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"[ERROR] Backtest failed:\n{result.stderr}")
            return {"error": result.stderr}

        # Parse results from output or results file
        results_path = Path(results_dir) / "results.json"
        if results_path.exists():
            with open(results_path) as f:
                return json.load(f)
        else:
            # Parse from stdout (fallback)
            print(f"[WARNING] No results.json found, parsing stdout")
            return parse_backtest_output(result.stdout)

    except subprocess.TimeoutExpired:
        print("[ERROR] Backtest timed out (>10 minutes)")
        return {"error": "timeout"}
    except Exception as e:
        print(f"[ERROR] Backtest exception: {e}")
        return {"error": str(e)}


def parse_backtest_output(stdout: str) -> Dict[str, Any]:
    """Parse backtest metrics from stdout (fallback)."""
    # TODO: Implement robust parsing of backtest output
    return {
        "total_trades": 0,
        "profit_factor": 0.0,
        "win_rate": 0.0,
        "total_pnl": 0.0,
        "error": "Failed to parse results"
    }


# ============================================================================
# Results Analysis
# ============================================================================

def compare_scenarios(results: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Compare multiple scenario results in a table.

    Args:
        results: Dict mapping scenario_name -> metrics

    Returns:
        Comparison DataFrame
    """
    rows = []

    for scenario_name, metrics in results.items():
        if "error" in metrics:
            continue

        rows.append({
            "Scenario": scenario_name,
            "Trades": metrics.get("total_trades", 0),
            "PF": metrics.get("profit_factor", 0.0),
            "Win Rate": f"{metrics.get('win_rate', 0.0) * 100:.1f}%",
            "Total PNL": f"${metrics.get('total_pnl', 0.0):.2f}",
            "Avg PNL": f"${metrics.get('avg_pnl', 0.0):.2f}",
            "Max DD": f"{metrics.get('max_drawdown_pct', 0.0):.2f}%",
            "Top Pattern": metrics.get("top_archetype", "N/A")
        })

    df = pd.DataFrame(rows)
    return df


def print_comparison_table(df: pd.DataFrame):
    """Pretty-print comparison table."""
    print("\n" + "="*80)
    print("SCENARIO COMPARISON (2022 Bear Market)")
    print("="*80)
    print(df.to_string(index=False))
    print("="*80)


def print_recommendation(df: pd.DataFrame):
    """Print final recommendation based on results."""
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    # Find best scenario by PF
    best_idx = df['PF'].astype(float).idxmax()
    best = df.iloc[best_idx]

    print(f"\nBest Scenario: {best['Scenario']}")
    print(f"  - Profit Factor: {best['PF']}")
    print(f"  - Win Rate: {best['Win Rate']}")
    print(f"  - Total Trades: {best['Trades']}")
    print(f"  - Top Pattern: {best['Top Pattern']}")

    # Check if it meets success criteria
    pf = float(best['PF'])
    trades = int(best['Trades'])

    if pf >= 1.2 and trades >= 15:
        print("\n✅ SUCCESS CRITERIA MET:")
        print(f"  - PF >= 1.2: {pf:.2f} ✓")
        print(f"  - Trades >= 15: {trades} ✓")
        print("\n→ Deploy this scenario to production")
    else:
        print("\n⚠️  SUCCESS CRITERIA NOT MET:")
        if pf < 1.2:
            print(f"  - PF >= 1.2: {pf:.2f} ✗ (need {1.2 - pf:.2f} improvement)")
        if trades < 15:
            print(f"  - Trades >= 15: {trades} ✗ (need {15 - trades} more trades)")
        print("\n→ Further tuning required (try frontier optimization)")

    print("="*80)


# ============================================================================
# Main Execution
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Simulate regime routing impact on 2022 performance"
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()) + ["all"],
        default="all",
        help="Scenario to test (or 'all' for comparison)"
    )
    parser.add_argument(
        "--asset",
        choices=["BTC", "ETH"],
        default="BTC",
        help="Asset to backtest"
    )
    parser.add_argument(
        "--start",
        default="2022-01-01",
        help="Start date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--end",
        default="2022-12-31",
        help="End date (YYYY-MM-DD)"
    )
    parser.add_argument(
        "--base-config",
        default="configs/baseline_btc_bull_pf20_biased_20pct_no_ml.json",
        help="Base config to modify"
    )
    parser.add_argument(
        "--compare-2024",
        action="store_true",
        help="Also run 2024 backtest to validate performance maintained"
    )
    parser.add_argument(
        "--results-dir",
        default="results/regime_routing_simulation",
        help="Where to save results"
    )

    args = parser.parse_args()

    # Setup
    results_root = Path(args.results_dir)
    results_root.mkdir(parents=True, exist_ok=True)

    # Select scenarios to test
    if args.scenario == "all":
        scenarios_to_test = SCENARIOS
    else:
        scenarios_to_test = {args.scenario: SCENARIOS[args.scenario]}

    # Run backtests
    print("\n" + "="*80)
    print("REGIME ROUTING SIMULATION - 2022 BEAR MARKET")
    print("="*80)
    print(f"Asset: {args.asset}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Base Config: {args.base_config}")
    print(f"Scenarios: {len(scenarios_to_test)}")
    print("="*80)

    results_2022 = {}

    for scenario_name, scenario_spec in scenarios_to_test.items():
        print(f"\n[SCENARIO] {scenario_spec['name']}")
        print(f"[DESCRIPTION] {scenario_spec['description']}")

        # Create modified config
        scenario_dir = results_root / scenario_name
        scenario_dir.mkdir(parents=True, exist_ok=True)

        config_path = scenario_dir / "config.json"
        create_routing_config(
            args.base_config,
            scenario_spec['routing'],
            str(config_path)
        )

        # Run 2022 backtest
        metrics = run_backtest(
            args.asset,
            args.start,
            args.end,
            str(config_path),
            str(scenario_dir / "2022")
        )

        results_2022[scenario_name] = metrics

        # Print quick summary
        if "error" not in metrics:
            print(f"  → Trades: {metrics.get('total_trades', 0)}, "
                  f"PF: {metrics.get('profit_factor', 0.0):.2f}, "
                  f"WR: {metrics.get('win_rate', 0.0) * 100:.1f}%")

    # Compare results
    df = compare_scenarios(results_2022)
    print_comparison_table(df)
    print_recommendation(df)

    # Optional: 2024 safety check
    if args.compare_2024:
        print("\n" + "="*80)
        print("2024 SAFETY CHECK (Ensure Bull Performance Maintained)")
        print("="*80)

        # Find best 2022 scenario
        best_scenario = df.loc[df['PF'].astype(float).idxmax(), 'Scenario']

        print(f"\nTesting best 2022 scenario ({best_scenario}) on 2024...")

        scenario_dir = results_root / best_scenario
        config_path = scenario_dir / "config.json"

        metrics_2024 = run_backtest(
            args.asset,
            "2024-01-01",
            "2024-09-30",
            str(config_path),
            str(scenario_dir / "2024")
        )

        if "error" not in metrics_2024:
            pf_2024 = metrics_2024.get('profit_factor', 0.0)
            print(f"\n2024 Results:")
            print(f"  - Profit Factor: {pf_2024:.2f}")
            print(f"  - Total Trades: {metrics_2024.get('total_trades', 0)}")
            print(f"  - Win Rate: {metrics_2024.get('win_rate', 0.0) * 100:.1f}%")

            if pf_2024 >= 2.5:
                print(f"\n✅ 2024 performance maintained (PF {pf_2024:.2f} >= 2.5)")
            else:
                print(f"\n⚠️  2024 performance degraded (PF {pf_2024:.2f} < 2.5)")
                print("   → Routing weights may be too aggressive")

    print("\n[COMPLETE] Simulation finished")
    print(f"[RESULTS] Saved to: {results_root}")


if __name__ == "__main__":
    main()
