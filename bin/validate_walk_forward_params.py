#!/usr/bin/env python3
"""
Validate Walk-Forward Parameters

Tests the walk-forward validated parameters on 2023 and 2024
to verify they solve the -66.4% degradation issue.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from engine.backtesting.engine import BacktestEngine
from engine.models.archetype_model import ArchetypeModel

def run_validation():
    print("="*80)
    print("WALK-FORWARD PARAMETERS VALIDATION")
    print("="*80)
    print("\nTesting walk-forward validated parameters on 2023 and 2024")
    print("Goal: Verify reduction in -66.4% degradation from previous config\n")

    # Load data
    df = pd.read_parquet(PROJECT_ROOT / 'data/btcusd_1h_features.parquet')

    config_path = PROJECT_ROOT / 'configs/test_optimized_no_funding.json'

    archetypes = ['B', 'K', 'A', 'S1']
    periods = {
        '2023': ('2023-01-01', '2023-12-31'),
        '2024': ('2024-01-01', '2024-12-31')
    }

    results = {}

    for archetype_id in archetypes:
        print(f"\n{'='*80}")
        print(f"ARCHETYPE {archetype_id}")
        print(f"{'='*80}")

        archetype_results = {}

        for period_name, (start, end) in periods.items():
            print(f"\n{period_name}: {start} to {end}")

            try:
                model = ArchetypeModel(
                    config_path=str(config_path),
                    archetype_name=archetype_id,
                    name=f"WFV_{archetype_id}_{period_name}"
                )

                engine = BacktestEngine(
                    model=model,
                    data=df,
                    initial_capital=10000.0,
                    commission_pct=0.001
                )

                bt_results = engine.run(start=start, end=end, verbose=False)

                metrics = {
                    'trades': bt_results.total_trades,
                    'pnl': bt_results.total_pnl,
                    'return_pct': bt_results.total_pnl / 10000 * 100,
                    'pf': bt_results.profit_factor if bt_results.profit_factor else 0,
                    'win_rate': bt_results.win_rate,
                    'max_dd': bt_results.max_drawdown if hasattr(bt_results, 'max_drawdown') else 0
                }

                print(f"  Trades: {metrics['trades']}")
                print(f"  PnL: ${metrics['pnl']:.2f} ({metrics['return_pct']:.1f}%)")
                print(f"  Profit Factor: {metrics['pf']:.2f}")
                print(f"  Win Rate: {metrics['win_rate']:.1f}%")

                archetype_results[period_name] = metrics

            except Exception as e:
                print(f"  Error: {e}")
                archetype_results[period_name] = {
                    'trades': 0, 'pnl': 0, 'return_pct': 0,
                    'pf': 0, 'win_rate': 0, 'max_dd': 0
                }

        # Calculate degradation
        if archetype_results['2023']['pnl'] > 0:
            degradation = ((archetype_results['2024']['pnl'] - archetype_results['2023']['pnl'])
                          / archetype_results['2023']['pnl'] * 100)
            print(f"\n  Degradation: {degradation:+.1f}%")
            archetype_results['degradation'] = degradation
        else:
            archetype_results['degradation'] = None

        results[archetype_id] = archetype_results

    # Summary
    print("\n" + "="*80)
    print("PERFORMANCE SUMMARY")
    print("="*80)

    total_2023 = sum(r['2023']['pnl'] for r in results.values())
    total_2024 = sum(r['2024']['pnl'] for r in results.values())

    print(f"\nPortfolio Performance:")
    print(f"  2023 Total PnL: ${total_2023:.2f}")
    print(f"  2024 Total PnL: ${total_2024:.2f}")

    if total_2023 > 0:
        portfolio_degradation = ((total_2024 - total_2023) / total_2023 * 100)
        print(f"  Degradation: {portfolio_degradation:+.1f}%")

        print(f"\n{'='*80}")
        print("COMPARISON TO PREVIOUS CONFIG (stop_loss_focus)")
        print(f"{'='*80}")
        print(f"Previous degradation: -66.4%")
        print(f"New degradation: {portfolio_degradation:+.1f}%")

        if portfolio_degradation > -66.4:
            improvement = abs(portfolio_degradation) - abs(-66.4)
            print(f"\n✅ IMPROVEMENT: {improvement:+.1f} percentage points")
            print(f"   Walk-forward validation successfully reduced overfitting!")
        else:
            print(f"\n⚠️  Performance worse than previous config")

    print("\n" + "="*80)
    print("ARCHETYPE BREAKDOWN")
    print("="*80)

    for arch_id, arch_results in results.items():
        print(f"\n{arch_id}:")
        print(f"  2023: ${arch_results['2023']['pnl']:.0f} ({arch_results['2023']['trades']} trades)")
        print(f"  2024: ${arch_results['2024']['pnl']:.0f} ({arch_results['2024']['trades']} trades)")
        if arch_results.get('degradation') is not None:
            print(f"  Degradation: {arch_results['degradation']:+.1f}%")

    print("\n" + "="*80)
    print("VALIDATION COMPLETE")
    print("="*80)

if __name__ == '__main__':
    run_validation()
