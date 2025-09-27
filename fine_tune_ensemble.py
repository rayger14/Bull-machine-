#!/usr/bin/env python3
"""
Bull Machine v1.5.1 - Fine-Grained Ensemble Tuning
Address the sharp cliff between 0.42 (29 trades) and 0.43 (0 trades)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from run_eth_ensemble_backtest import load_and_align_eth_data, run_eth_ensemble_backtest, get_eth_ensemble_config

def fine_tune_parameters():
    """Fine-tune ensemble parameters to achieve 2-4 trades/month."""

    print("🎯 Bull Machine v1.5.1 - Fine-Grained Ensemble Tuning")
    print("=" * 65)
    print("Problem: Sharp cliff between 0.42 (19 trades/mo) and 0.43 (0 trades)")
    print("Solution: Adjust multiple parameters to create gradual tuning")
    print("=" * 65)

    # Load ETH data once
    print("\n📊 Loading ETH data...")
    df_aligned = load_and_align_eth_data()

    # Parameter combinations to test
    test_configs = [
        # Slightly more restrictive than 0.42
        {"entry_threshold": 0.415, "consensus_penalty": 0.03, "rolling_k": 2, "rolling_n": 5, "quality_adjustment": 0.0},
        {"entry_threshold": 0.420, "consensus_penalty": 0.03, "rolling_k": 2, "rolling_n": 5, "quality_adjustment": 0.0},
        {"entry_threshold": 0.425, "consensus_penalty": 0.03, "rolling_k": 2, "rolling_n": 5, "quality_adjustment": 0.0},

        # Adjust quality floors slightly upward
        {"entry_threshold": 0.420, "consensus_penalty": 0.02, "rolling_k": 2, "rolling_n": 5, "quality_adjustment": 0.01},
        {"entry_threshold": 0.420, "consensus_penalty": 0.02, "rolling_k": 2, "rolling_n": 5, "quality_adjustment": 0.02},

        # More restrictive rolling requirements
        {"entry_threshold": 0.420, "consensus_penalty": 0.02, "rolling_k": 3, "rolling_n": 6, "quality_adjustment": 0.0},
        {"entry_threshold": 0.415, "consensus_penalty": 0.02, "rolling_k": 3, "rolling_n": 7, "quality_adjustment": 0.0},

        # Combined adjustments
        {"entry_threshold": 0.418, "consensus_penalty": 0.025, "rolling_k": 2, "rolling_n": 6, "quality_adjustment": 0.005},
        {"entry_threshold": 0.416, "consensus_penalty": 0.025, "rolling_k": 3, "rolling_n": 6, "quality_adjustment": 0.01},
    ]

    results_summary = []

    for i, params in enumerate(test_configs, 1):
        print(f"\n🔄 Test {i}/{len(test_configs)}: Threshold={params['entry_threshold']}, "
              f"Penalty={params['consensus_penalty']}, Rolling={params['rolling_k']}/{params['rolling_n']}, "
              f"QF+{params['quality_adjustment']}")

        # Get base config and apply modifications
        config = get_eth_ensemble_config()
        config["entry_threshold"] = params["entry_threshold"]
        config["ensemble"]["consensus_penalty"] = params["consensus_penalty"]
        config["ensemble"]["rolling_k"] = params["rolling_k"]
        config["ensemble"]["rolling_n"] = params["rolling_n"]

        # Adjust quality floors slightly if needed
        if params["quality_adjustment"] > 0:
            for key in config["quality_floors"]:
                config["quality_floors"][key] += params["quality_adjustment"]

        # Override config function temporarily
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
                **params,
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

                # Additional quality checks
                quality_score = 0
                if results['win_rate'] >= 45: quality_score += 1
                if results['total_return_pct'] >= 5: quality_score += 1
                if results['max_drawdown'] <= 10: quality_score += 1
                if results['profit_factor'] >= 1.0: quality_score += 1

                print(f"   📊 Quality Score: {quality_score}/4")

        except Exception as e:
            print(f"   ❌ Error: {e}")
            summary = {**params, "error": str(e)}
            results_summary.append(summary)

        finally:
            # Restore original config function
            run_eth_ensemble_backtest.get_eth_ensemble_config = original_config_fn

    # Analysis and recommendations
    print(f"\n{'='*80}")
    print(f"📊 FINE-TUNING RESULTS")
    print(f"{'='*80}")

    print(f"{'Test':<5} {'Threshold':<10} {'Penalty':<8} {'Roll':<6} {'QF+':<5} "
          f"{'Trades':<7} {'Freq/Mo':<8} {'WR%':<6} {'Ret%':<7} {'Target?'}")
    print("-" * 80)

    best_candidates = []

    for i, r in enumerate(results_summary, 1):
        if 'error' in r:
            print(f"{i:<5} ERROR")
            continue

        target_met = 2.0 <= r['trades_per_month'] <= 4.0
        target_indicator = "✅" if target_met else "❌"

        rolling_str = f"{r['rolling_k']}/{r['rolling_n']}"
        print(f"{i:<5} {r['entry_threshold']:<10} {r['consensus_penalty']:<8} {rolling_str:<6} "
              f"{r['quality_adjustment']:<5} {r['trades']:<7} {r['trades_per_month']:<8.2f} "
              f"{r['win_rate']:<6.1f} {r['total_return']:<7.2f} {target_indicator}")

        if target_met:
            best_candidates.append(r)

    # Final recommendations
    print(f"\n🎯 FINAL RECOMMENDATIONS:")

    if best_candidates:
        # Sort by combined score: profit factor, then win rate, then return
        best_candidates.sort(key=lambda x: (x['profit_factor'], x['win_rate'], x['total_return']), reverse=True)
        best = best_candidates[0]

        print(f"\n🏆 OPTIMAL CONFIGURATION:")
        print(f"   Entry Threshold: {best['entry_threshold']}")
        print(f"   Consensus Penalty: {best['consensus_penalty']}")
        print(f"   Rolling Requirement: {best['rolling_k']}/{best['rolling_n']} bars")
        print(f"   Quality Floor Adjustment: +{best['quality_adjustment']}")
        print(f"\n📊 Performance:")
        print(f"   Trades/Month: {best['trades_per_month']:.2f}")
        print(f"   Win Rate: {best['win_rate']:.1f}%")
        print(f"   Return: {best['total_return']:+.2f}%")
        print(f"   Profit Factor: {best['profit_factor']:.2f}")
        print(f"   Max Drawdown: {best['max_drawdown']:.2f}%")

    else:
        print("❌ Still no configurations achieve 2-4 trades/month")
        print("\n💡 Next Steps:")
        print("   1. Lower quality floors more significantly (0.24-0.26 range)")
        print("   2. Consider reducing min_consensus to 1 (any single TF can trigger)")
        print("   3. Investigate why ensemble scores cluster tightly around 0.42")
        print("   4. Review base scoring functions for more granular distribution")

    return results_summary, best_candidates

if __name__ == "__main__":
    fine_tune_parameters()