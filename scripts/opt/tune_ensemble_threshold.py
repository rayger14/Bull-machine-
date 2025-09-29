#!/usr/bin/env python3
"""
Bull Machine v1.5.1 - Ensemble Threshold Tuning
Systematic sweep to find optimal entry threshold for 2-4 trades/month
"""

import pandas as pd
import numpy as np
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from run_eth_ensemble_backtest import load_and_align_eth_data, run_eth_ensemble_backtest, get_eth_ensemble_config

def sweep_thresholds():
    """Sweep entry thresholds to find optimal trading frequency."""

    print("🎯 Bull Machine v1.5.1 - Ensemble Threshold Tuning")
    print("=" * 60)
    print("Target: Find threshold for 2-4 trades/month")
    print("Range: 0.38 (30.67 trades/mo) → 0.50 (0 trades)")
    print("=" * 60)

    # Load ETH data once
    print("\n📊 Loading ETH data...")
    df_aligned = load_and_align_eth_data()

    # Threshold sweep range
    thresholds = [0.38, 0.40, 0.42, 0.43, 0.44, 0.45, 0.46, 0.48, 0.50]

    results_summary = []

    for threshold in thresholds:
        print(f"\n🔄 Testing threshold: {threshold}")

        # Get base config and update threshold
        config = get_eth_ensemble_config()
        config["entry_threshold"] = threshold

        # Also reduce consensus penalty for intermediate thresholds
        if 0.40 <= threshold <= 0.46:
            config["ensemble"]["consensus_penalty"] = 0.02  # Less penalty
            config["ensemble"]["rolling_k"] = 2  # Less strict rolling
        else:
            config["ensemble"]["consensus_penalty"] = 0.05  # More penalty
            config["ensemble"]["rolling_k"] = 3  # More strict rolling

        # Update config in backtest function temporarily
        import run_eth_ensemble_backtest
        original_config_fn = run_eth_ensemble_backtest.get_eth_ensemble_config

        def temp_config():
            return config

        run_eth_ensemble_backtest.get_eth_ensemble_config = temp_config

        try:
            # Run backtest
            results = run_eth_ensemble_backtest.run_eth_ensemble_backtest(df_aligned)

            # Store summary
            summary = {
                "threshold": threshold,
                "consensus_penalty": config["ensemble"]["consensus_penalty"],
                "rolling_k": config["ensemble"]["rolling_k"],
                "signals": results["signals_generated"],
                "trades": results["total_trades"],
                "trades_per_month": results["trades_per_month"],
                "win_rate": results["win_rate"],
                "total_return": results["total_return_pct"],
                "max_drawdown": results["max_drawdown"],
                "profit_factor": results["profit_factor"],
                "sharpe_ratio": results["sharpe_ratio"]
            }

            results_summary.append(summary)

            print(f"   Signals: {results['signals_generated']}")
            print(f"   Trades: {results['total_trades']}")
            print(f"   Frequency: {results['trades_per_month']:.2f}/month")
            print(f"   Win Rate: {results['win_rate']:.1f}%")
            print(f"   Return: {results['total_return_pct']:+.2f}%")

            # Check if in target range
            target_met = 2.0 <= results['trades_per_month'] <= 4.0
            if target_met:
                print(f"   ✅ IN TARGET RANGE!")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            summary = {
                "threshold": threshold,
                "consensus_penalty": config["ensemble"]["consensus_penalty"],
                "rolling_k": config["ensemble"]["rolling_k"],
                "signals": 0,
                "trades": 0,
                "trades_per_month": 0,
                "win_rate": 0,
                "total_return": 0,
                "max_drawdown": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "error": str(e)
            }
            results_summary.append(summary)

        finally:
            # Restore original config function
            run_eth_ensemble_backtest.get_eth_ensemble_config = original_config_fn

    # Analysis and recommendations
    print(f"\n{'='*80}")
    print(f"📊 THRESHOLD SWEEP RESULTS")
    print(f"{'='*80}")

    print(f"{'Threshold':<10} {'Trades':<7} {'Freq/Mo':<8} {'WR%':<6} {'Return%':<9} {'PF':<6} {'Target?'}")
    print("-" * 80)

    best_candidates = []

    for r in results_summary:
        if 'error' in r:
            print(f"{r['threshold']:<10} ERROR")
            continue

        target_met = 2.0 <= r['trades_per_month'] <= 4.0
        target_indicator = "✅" if target_met else "❌"

        print(f"{r['threshold']:<10} {r['trades']:<7} {r['trades_per_month']:<8.2f} "
              f"{r['win_rate']:<6.1f} {r['total_return']:<9.2f} {r['profit_factor']:<6.2f} {target_indicator}")

        if target_met:
            best_candidates.append(r)

    # Recommend best threshold
    print(f"\n🎯 RECOMMENDATIONS:")

    if best_candidates:
        # Sort by profit factor, then win rate
        best_candidates.sort(key=lambda x: (x['profit_factor'], x['win_rate']), reverse=True)
        best = best_candidates[0]

        print(f"\n🏆 OPTIMAL THRESHOLD: {best['threshold']}")
        print(f"   📊 Trades/Month: {best['trades_per_month']:.2f}")
        print(f"   🏆 Win Rate: {best['win_rate']:.1f}%")
        print(f"   📈 Return: {best['total_return']:+.2f}%")
        print(f"   ⚖️  Profit Factor: {best['profit_factor']:.2f}")
        print(f"   📉 Max DD: {best['max_drawdown']:.2f}%")

        # RC target analysis
        print(f"\n🎯 RC TARGET ANALYSIS FOR THRESHOLD {best['threshold']}:")
        targets_met = 0

        freq_ok = 2.0 <= best['trades_per_month'] <= 4.0
        wr_ok = best['win_rate'] >= 50.0
        return_ok = best['total_return'] >= 10.0
        dd_ok = best['max_drawdown'] <= 9.2
        pf_ok = best['profit_factor'] >= 1.3
        sharpe_ok = best['sharpe_ratio'] >= 2.2

        print(f"   ✅ Frequency: {best['trades_per_month']:.2f}/mo {'✅' if freq_ok else '❌'} (Target: 2-4)")
        print(f"   ✅ Win Rate: {best['win_rate']:.1f}% {'✅' if wr_ok else '❌'} (Target: ≥50%)")
        print(f"   ✅ Return: {best['total_return']:+.2f}% {'✅' if return_ok else '❌'} (Target: ≥10%)")
        print(f"   ✅ Max DD: {best['max_drawdown']:.2f}% {'✅' if dd_ok else '❌'} (Target: ≤9.2%)")
        print(f"   ✅ Profit Factor: {best['profit_factor']:.2f} {'✅' if pf_ok else '❌'} (Target: ≥1.3)")
        print(f"   ✅ Sharpe: {best['sharpe_ratio']:.2f} {'✅' if sharpe_ok else '❌'} (Target: ≥2.2)")

        targets_met = sum([freq_ok, wr_ok, return_ok, dd_ok, pf_ok, sharpe_ok])
        print(f"\n🏆 RC TARGETS MET: {targets_met}/6")

        if targets_met >= 5:
            print("🎉 READY FOR RC PROMOTION!")
        elif targets_met >= 3:
            print("🔧 NEEDS MINOR TUNING (consider profit ladder optimization)")
        else:
            print("⚠️  NEEDS MORE OPTIMIZATION")

    else:
        print("❌ No thresholds achieved 2-4 trades/month target")
        print("💡 Suggestions:")
        print("   - Consider lowering quality floors")
        print("   - Reduce consensus penalty further")
        print("   - Adjust rolling requirements (2/5 instead of 3/5)")

    # Save detailed results
    output_file = f"threshold_sweep_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump(results_summary, f, indent=2)

    print(f"\n💾 Detailed results saved to: {output_file}")

    return results_summary, best_candidates

if __name__ == "__main__":
    sweep_thresholds()