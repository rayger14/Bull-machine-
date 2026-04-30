#!/usr/bin/env python3
"""Validate K archetype Optuna parameters deployment."""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from engine.backtesting.engine import BacktestEngine
from engine.models.archetype_model import ArchetypeModel

def main():
    print("=" * 80)
    print("K ARCHETYPE OPTUNA PARAMETERS VALIDATION")
    print("=" * 80)

    # Load data
    df = pd.read_parquet(PROJECT_ROOT / 'data/btcusd_1h_features.parquet')
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    # Test K with Optuna params
    print("\nTesting K archetype with Optuna-optimized parameters...")
    print("Parameters:")
    print("  fusion_threshold: 0.506 (was 0.52)")
    print("  adx_min: 29.51 (was 25.0)")
    print("  wick_threshold: 0.610 (was 0.55)")
    print("  atr_stop_mult: 3.099 (was 4.0)")
    print("  atr_tp_mult: 3.156 (was 4.5)")
    print()

    config_path = PROJECT_ROOT / 'configs/test_optimized_no_funding.json'

    model = ArchetypeModel(
        config_path=str(config_path),
        archetype_name='K',
        name="K_OPTUNA"
    )

    engine = BacktestEngine(
        model=model,
        data=df,
        initial_capital=10000.0,
        commission_pct=0.001
    )

    results = engine.run(start=start_date, end=end_date, verbose=False)

    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print(f"\nTotal Trades: {results.total_trades}")
    print(f"Winners: {results.winning_trades}")
    print(f"Losers: {results.losing_trades}")
    print(f"Win Rate: {results.win_rate:.1f}%")
    print(f"Profit Factor: {results.profit_factor:.2f}" if results.profit_factor else "Profit Factor: N/A")
    print(f"Total PnL: ${results.total_pnl:.2f}")
    print(f"Return: {results.total_return_pct:.1f}%")
    print(f"Sharpe Ratio: {results.sharpe_ratio:.2f}" if results.sharpe_ratio else "Sharpe Ratio: N/A")

    try:
        print(f"Max Drawdown: {results.max_drawdown:.2f}%")
    except AttributeError:
        print("Max Drawdown: N/A")

    print("\n" + "=" * 80)
    print("COMPARISON TO EXPECTED")
    print("=" * 80)

    expected_from_optuna = {
        'win_rate': 54.3,
        'pf': 2.21,
        'trades': 60,  # ~60-65 expected
    }

    previous_stop_loss_focus = {
        'trades': 61,
        'win_rate': 47.5,
        'pnl': 970.08,
        'pf': 2.06
    }

    print(f"\n{'Metric':<20} {'Optuna Expected':<20} {'Actual':<20} {'Previous':<20}")
    print("-" * 80)
    print(f"{'Trades':<20} {expected_from_optuna['trades']:<20} {results.total_trades:<20} {previous_stop_loss_focus['trades']:<20}")
    print(f"{'Win Rate':<20} {expected_from_optuna['win_rate']:<19.1f}% {results.win_rate:<19.1f}% {previous_stop_loss_focus['win_rate']:<19.1f}%")
    print(f"{'Profit Factor':<20} {expected_from_optuna['pf']:<20.2f} {results.profit_factor if results.profit_factor else 0:<20.2f} {previous_stop_loss_focus['pf']:<20.2f}")
    print(f"{'PnL':<20} {'~$1,100-1,200':<20} ${results.total_pnl:<19.2f} ${previous_stop_loss_focus['pnl']:<19.2f}")

    print("\n" + "=" * 80)
    print("ASSESSMENT")
    print("=" * 80)

    pnl_vs_previous = results.total_pnl - previous_stop_loss_focus['pnl']
    wr_vs_optuna = results.win_rate - expected_from_optuna['win_rate']

    print(f"\nPnL vs stop_loss_focus: ${pnl_vs_previous:+.2f} ({(pnl_vs_previous/previous_stop_loss_focus['pnl']*100):+.1f}%)")
    print(f"Win Rate vs Optuna expected: {wr_vs_optuna:+.1f}pp")

    if results.win_rate >= 52.0 and results.total_pnl > previous_stop_loss_focus['pnl']:
        print("\n✅ EXCELLENT: Optuna params delivering expected improvements!")
    elif results.total_pnl > previous_stop_loss_focus['pnl']:
        print("\n✅ GOOD: PnL improved, WR slightly below Optuna target (acceptable)")
    elif abs(pnl_vs_previous) < 50:
        print("\n⚠️  NEUTRAL: Similar performance to stop_loss_focus (no significant change)")
    else:
        print("\n❌ DEGRADATION: Optuna params underperform stop_loss_focus")
        print("   Consider reverting or using hybrid approach")

    # Notes
    print("\n" + "=" * 80)
    print("NOTES")
    print("=" * 80)
    print("\nOptuna optimization was done on 3 mini walk-forward folds, not full 2023.")
    print("Slight variance from expected is normal.")
    print("\nKey improvements:")
    print("  - Higher ADX filter (29.5 vs 25.0) ensures stronger trends")
    print("  - Higher wick threshold (0.61 vs 0.55) requires stronger rejection")
    print("  - Tighter stops (3.1 vs 4.0) but better quality signals should compensate")

if __name__ == '__main__':
    main()
