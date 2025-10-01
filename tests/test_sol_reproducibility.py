#!/usr/bin/env python3
"""
SOL Reproducibility Test - Bull Machine v1.7.2 Adaptive Engine
Test consistency of adaptive backtest results
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from run_adaptive_backtest import AdaptiveBullMachine
import json
from datetime import datetime

def run_sol_reproducibility_test():
    """Run SOL backtest twice to test reproducibility."""
    print("🔬 SOL REPRODUCIBILITY TEST - Bull Machine v1.7.2")
    print("=" * 70)
    print("Testing consistency of adaptive backtest engine...")
    print()

    results = []

    for run_num in range(1, 3):
        print(f"🧪 Running SOL Test #{run_num}...")
        print("-" * 50)

        try:
            # Create fresh instance
            backtest = AdaptiveBullMachine('SOLUSD', 'COINBASE')

            # Run backtest
            result = backtest.run_backtest('SOLUSD')

            if result and 'error' not in result:
                results.append(result)

                perf = result['performance']
                print(f"✅ Test #{run_num} Complete:")
                print(f"   Final Balance: ${perf['final_balance']:,.2f}")
                print(f"   Total Return: {perf['total_return']:+.2f}%")
                print(f"   Total Trades: {perf['total_trades']}")
                print(f"   Win Rate: {perf['win_rate']:.1f}%")
                print(f"   Profit Factor: {perf['profit_factor']:.2f}")
                print()

                # Save individual result
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'sol_adaptive_test_{run_num}_{timestamp}.json'

                with open(filename, 'w') as f:
                    json.dump(result, f, indent=2, default=str)
                print(f"💾 Test #{run_num} saved to: {filename}")
                print()

            else:
                print(f"❌ Test #{run_num} failed: {result.get('error', 'Unknown error')}")

        except Exception as e:
            print(f"❌ Test #{run_num} error: {e}")

    # Compare results
    if len(results) == 2:
        compare_results(results)
    else:
        print("❌ Could not complete both tests for comparison")

def compare_results(results):
    """Compare two backtest results for consistency."""
    print("🔍 REPRODUCIBILITY ANALYSIS")
    print("=" * 70)

    test1, test2 = results
    perf1 = test1['performance']
    perf2 = test2['performance']

    # Key metrics comparison
    metrics = [
        ('Final Balance', 'final_balance'),
        ('Total Return', 'total_return'),
        ('Total Trades', 'total_trades'),
        ('Winning Trades', 'winning_trades'),
        ('Losing Trades', 'losing_trades'),
        ('Win Rate', 'win_rate'),
        ('Avg Win', 'avg_win'),
        ('Avg Loss', 'avg_loss'),
        ('Profit Factor', 'profit_factor')
    ]

    print("📊 METRIC COMPARISON:")
    print("-" * 70)
    print(f"{'Metric':<15} {'Test #1':<12} {'Test #2':<12} {'Match':<8} {'Diff'}")
    print("-" * 70)

    all_match = True

    for label, key in metrics:
        val1 = perf1[key]
        val2 = perf2[key]

        # Check if values match (within floating point precision)
        if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
            match = abs(val1 - val2) < 1e-10
            diff = abs(val1 - val2)

            if key in ['final_balance', 'total_return']:
                val1_str = f"${val1:,.2f}" if 'balance' in key else f"{val1:+.2f}%"
                val2_str = f"${val2:,.2f}" if 'balance' in key else f"{val2:+.2f}%"
            else:
                val1_str = f"{val1:.2f}" if isinstance(val1, float) else str(val1)
                val2_str = f"{val2:.2f}" if isinstance(val2, float) else str(val2)
        else:
            match = val1 == val2
            diff = 0 if match else 1
            val1_str = str(val1)
            val2_str = str(val2)

        match_str = "✅ YES" if match else "❌ NO"
        diff_str = f"{diff:.2e}" if diff < 0.01 else f"{diff:.2f}"

        print(f"{label:<15} {val1_str:<12} {val2_str:<12} {match_str:<8} {diff_str}")

        if not match:
            all_match = False

    print()

    # Engine usage comparison
    print("🔧 ENGINE USAGE COMPARISON:")
    print("-" * 70)

    engines1 = test1['engine_usage']
    engines2 = test2['engine_usage']

    engine_match = True
    for engine in engines1:
        count1 = engines1[engine]
        count2 = engines2.get(engine, 0)
        match = count1 == count2

        if not match:
            engine_match = False

        match_str = "✅" if match else "❌"
        print(f"{engine:<20} {count1:<8} {count2:<8} {match_str}")

    print()

    # Trade-by-trade comparison (if available)
    trades1 = test1.get('trades', [])
    trades2 = test2.get('trades', [])

    if len(trades1) == len(trades2) and len(trades1) > 0:
        print("📋 TRADE-BY-TRADE COMPARISON:")
        print("-" * 70)

        trade_match = True
        for i, (t1, t2) in enumerate(zip(trades1, trades2)):
            if abs(t1.get('pnl_pct', 0) - t2.get('pnl_pct', 0)) > 1e-10:
                trade_match = False
                print(f"❌ Trade {i+1}: PnL differs ({t1.get('pnl_pct', 0):.2f}% vs {t2.get('pnl_pct', 0):.2f}%)")
                break

        if trade_match:
            print("✅ All individual trades match exactly")

    print()

    # Final verdict
    print("🎯 REPRODUCIBILITY VERDICT:")
    print("=" * 70)

    overall_match = all_match and engine_match

    if overall_match:
        print("✅ PERFECT REPRODUCIBILITY CONFIRMED!")
        print("   All metrics, engine usage, and trades match exactly")
        print("   Bull Machine v1.7.2 Adaptive Engine is fully deterministic")
    else:
        print("❌ REPRODUCIBILITY ISSUES DETECTED")
        print("   Some metrics or engine usage differs between runs")
        print("   Investigate potential sources of non-determinism")

    print()

    # Summary
    period1 = test1['period']
    print("📈 TEST SUMMARY:")
    print(f"   Period: {period1['start']} to {period1['end']} ({period1['days']} days)")
    print(f"   Asset: SOL/USD with adaptive configuration")
    print(f"   Engine: Bull Machine v1.7.2 with asset-specific parameters")
    print(f"   Reproducibility: {'✅ CONFIRMED' if overall_match else '❌ FAILED'}")

    return overall_match

if __name__ == "__main__":
    run_sol_reproducibility_test()