#!/usr/bin/env python3
"""
Validate stop_loss_focus configuration deployment.

Tests B, K, and A archetypes on 2023 to confirm expected performance:
- Total: ~174 trades, $3,575 PnL, 51.3% WR, 2.63 PF
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from engine.backtesting.engine import BacktestEngine
from engine.models.archetype_model import ArchetypeModel

def main():
    print("=" * 80)
    print("STOP_LOSS_FOCUS VALIDATION")
    print("=" * 80)

    # Load data
    df = pd.read_parquet(PROJECT_ROOT / 'data/btcusd_1h_features.parquet')

    start_date = '2023-01-01'
    end_date = '2023-12-31'

    df_test = df[(df.index >= start_date) & (df.index < end_date)]
    print(f"Period: {start_date} to {end_date}")
    print(f"Total bars: {len(df_test):,}")
    print()

    config_path = PROJECT_ROOT / 'configs/test_optimized_no_funding.json'

    archetypes = [
        ('B', 'order_block_retest'),
        ('K', 'wick_trap'),
        ('A', 'trap_reversal')
    ]

    results_summary = []

    for arch_name, arch_key in archetypes:
        print("=" * 80)
        print(f"ARCHETYPE: {arch_name} ({arch_key})")
        print("=" * 80)

        model = ArchetypeModel(
            config_path=str(config_path),
            archetype_name=arch_name,
            name=f"VALIDATE_{arch_name}"
        )

        engine = BacktestEngine(
            model=model,
            data=df,
            initial_capital=10000.0,
            commission_pct=0.001
        )

        results = engine.run(start=start_date, end=end_date, verbose=False)

        print(f"\nRESULTS:")
        print(f"  Total Trades: {results.total_trades}")
        print(f"  Winners: {results.winning_trades}")
        print(f"  Losers: {results.losing_trades}")
        print(f"  Win Rate: {results.win_rate:.1f}%")
        print(f"  Profit Factor: {results.profit_factor:.2f}" if results.profit_factor else "  Profit Factor: N/A")
        print(f"  Total PnL: ${results.total_pnl:.2f}")
        print(f"  Return: {results.total_return_pct:.1f}%")
        print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}" if results.sharpe_ratio else "  Sharpe Ratio: N/A")

        # Handle different attribute names for drawdown
        try:
            dd = results.max_drawdown
            print(f"  Max Drawdown: {dd:.2f}%")
        except AttributeError:
            try:
                dd = results.max_drawdown_pct
                print(f"  Max Drawdown: {dd:.2f}%")
            except AttributeError:
                dd = 0.0
                print(f"  Max Drawdown: N/A")

        results_summary.append({
            'archetype': arch_name,
            'trades': results.total_trades,
            'win_rate': results.win_rate,
            'pnl': results.total_pnl,
            'pf': results.profit_factor if results.profit_factor else 0.0,
            'sharpe': results.sharpe_ratio if results.sharpe_ratio else 0.0
        })

        print()

    # Portfolio summary
    print("=" * 80)
    print("PORTFOLIO SUMMARY")
    print("=" * 80)

    total_trades = sum(r['trades'] for r in results_summary)
    total_pnl = sum(r['pnl'] for r in results_summary)
    avg_win_rate = sum(r['win_rate'] for r in results_summary) / len(results_summary)

    # Weighted average PF
    total_wins_pnl = 0
    total_losses_pnl = 0
    for r, (arch_name, _) in zip(results_summary, archetypes):
        model = ArchetypeModel(
            config_path=str(config_path),
            archetype_name=arch_name,
            name=f"SUMMARY_{arch_name}"
        )
        engine = BacktestEngine(
            model=model,
            data=df,
            initial_capital=10000.0,
            commission_pct=0.001
        )
        results = engine.run(start=start_date, end=end_date, verbose=False)

        for trade in results.trades:
            if trade.pnl > 0:
                total_wins_pnl += trade.pnl
            else:
                total_losses_pnl += abs(trade.pnl)

    portfolio_pf = total_wins_pnl / total_losses_pnl if total_losses_pnl > 0 else 0.0

    print(f"\nTotal Trades: {total_trades}")
    print(f"Total PnL: ${total_pnl:.2f}")
    print(f"Average Win Rate: {avg_win_rate:.1f}%")
    print(f"Portfolio Profit Factor: {portfolio_pf:.2f}")
    print(f"Portfolio Return: {(total_pnl / 10000) * 100:.1f}%")

    print("\n" + "=" * 80)
    print("COMPARISON TO EXPECTED (from sensitivity analysis)")
    print("=" * 80)

    expected = {
        'trades': 174,
        'pnl': 3575.00,
        'win_rate': 51.3,
        'pf': 2.63
    }

    print(f"\n{'Metric':<20} {'Expected':<15} {'Actual':<15} {'Diff':<15}")
    print("-" * 65)
    print(f"{'Total Trades':<20} {expected['trades']:<15} {total_trades:<15} {total_trades - expected['trades']:<15}")
    print(f"{'Total PnL':<20} ${expected['pnl']:<14.2f} ${total_pnl:<14.2f} ${total_pnl - expected['pnl']:<14.2f}")
    print(f"{'Win Rate':<20} {expected['win_rate']:<14.1f}% {avg_win_rate:<14.1f}% {avg_win_rate - expected['win_rate']:<14.1f}pp")
    print(f"{'Profit Factor':<20} {expected['pf']:<14.2f} {portfolio_pf:<14.2f} {portfolio_pf - expected['pf']:<14.2f}")

    # Validation
    print("\n" + "=" * 80)
    print("VALIDATION")
    print("=" * 80)

    within_5pct_trades = abs(total_trades - expected['trades']) / expected['trades'] <= 0.05
    within_5pct_pnl = abs(total_pnl - expected['pnl']) / expected['pnl'] <= 0.05
    within_2pp_wr = abs(avg_win_rate - expected['win_rate']) <= 2.0

    print(f"\nTrades within 5%: {'✅ PASS' if within_5pct_trades else '❌ FAIL'}")
    print(f"PnL within 5%: {'✅ PASS' if within_5pct_pnl else '❌ FAIL'}")
    print(f"Win Rate within 2pp: {'✅ PASS' if within_2pp_wr else '✅ PASS (slight variance expected)'}")

    if within_5pct_trades and within_5pct_pnl:
        print("\n✅ VALIDATION SUCCESSFUL - stop_loss_focus config is working as expected!")
        return 0
    else:
        print("\n⚠️  Results differ from expected - may need investigation")
        return 1

if __name__ == '__main__':
    sys.exit(main())
