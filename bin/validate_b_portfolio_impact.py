#!/usr/bin/env python3
"""
Validate B Archetype Portfolio Impact

Tests all archetypes individually using the dedicated test scripts
to measure B archetype's contribution to portfolio performance.
"""

import sys
from pathlib import Path
import subprocess
import json
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from engine.models.archetype_model import ArchetypeModel
from engine.backtesting.engine import BacktestEngine


def test_individual_archetype(arch_id, arch_name, df, start, end):
    """Test single archetype in isolation."""

    print(f"\n{'='*80}")
    print(f"ARCHETYPE: {arch_id} ({arch_name})")
    print(f"{'='*80}")

    # Create model
    model = ArchetypeModel(
        config_path=str(PROJECT_ROOT / 'configs/test_optimized_no_funding.json'),
        archetype_name=arch_id,
        name=f"TEST_{arch_id}"
    )

    # Run backtest
    engine = BacktestEngine(
        model=model,
        data=df,
        initial_capital=10000.0,
        commission_pct=0.001
    )

    results = engine.run(start=start, end=end, verbose=False)

    # Print results
    print(f"\nRESULTS:")
    print(f"  Trades: {results.total_trades}")
    print(f"  Winners: {results.winning_trades}")
    print(f"  Losers: {results.losing_trades}")
    print(f"  Win Rate: {results.win_rate:.1f}%")
    print(f"  PnL: ${results.total_pnl:.2f}")
    print(f"  Return: {results.total_return_pct:.1f}%")

    if results.profit_factor:
        print(f"  Profit Factor: {results.profit_factor:.2f}")

    if results.sharpe_ratio:
        print(f"  Sharpe Ratio: {results.sharpe_ratio:.2f}")

    print(f"  Max Drawdown: {results.max_drawdown:.1f}%")
    print(f"  Avg Duration: {results.avg_trade_duration_hours:.1f} hours")

    # Exit reasons
    exit_reasons = {}
    for trade in results.trades:
        reason = trade.exit_reason
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    if exit_reasons:
        print(f"\n  Exit Reasons:")
        for reason, count in sorted(exit_reasons.items(), key=lambda x: -x[1]):
            print(f"    {reason}: {count} ({count/results.total_trades*100:.1f}%)")

    return {
        'archetype': arch_id,
        'name': arch_name,
        'trades': results.total_trades,
        'winners': results.winning_trades,
        'losers': results.losing_trades,
        'win_rate': results.win_rate,
        'pnl': results.total_pnl,
        'return_pct': results.total_return_pct,
        'profit_factor': results.profit_factor if results.profit_factor else 0,
        'sharpe': results.sharpe_ratio if results.sharpe_ratio else 0,
        'max_dd': results.max_drawdown,
        'avg_hours': results.avg_trade_duration_hours,
        'trades_list': results.trades
    }


def check_trade_overlap(all_results):
    """Check temporal overlap between archetype trades."""

    print(f"\n{'='*80}")
    print("TRADE OVERLAP ANALYSIS")
    print(f"{'='*80}\n")

    archetypes = [r['archetype'] for r in all_results]

    for i, arch1 in enumerate(archetypes):
        for j, arch2 in enumerate(archetypes):
            if i >= j:
                continue

            trades1 = all_results[i]['trades_list']
            trades2 = all_results[j]['trades_list']

            # Check for overlapping time windows
            overlaps = 0
            for t1 in trades1:
                for t2 in trades2:
                    # Check if time ranges overlap
                    if (t1.entry_time <= t2.exit_time and t1.exit_time >= t2.entry_time):
                        overlaps += 1

            total_pairs = len(trades1) + len(trades2)
            if total_pairs > 0:
                overlap_pct = overlaps / total_pairs * 100
                print(f"{arch1} vs {arch2}: {overlaps} overlaps ({overlap_pct:.1f}%)")


def main():
    print("="*80)
    print("B ARCHETYPE PORTFOLIO VALIDATION")
    print("="*80)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Objective: Validate $904 PnL improvement from B archetype fix\n")

    # Load data
    feature_store = PROJECT_ROOT / "data/btcusd_1h_features.parquet"
    print(f"Loading: {feature_store}")
    df = pd.read_parquet(feature_store)

    # Test period - 2023
    start_date = '2023-01-01'
    end_date = '2023-12-31'

    print(f"Period: {start_date} to {end_date}")
    print(f"Total bars: {len(df.loc[start_date:end_date]):,}\n")

    # Archetypes to test
    archetypes = [
        ('S1', 'liquidity_vacuum'),
        ('H', 'trap_within_trend'),
        ('B', 'order_block_retest'),
        ('K', 'wick_trap'),
    ]

    # Run individual tests
    all_results = []

    for arch_id, arch_name in archetypes:
        try:
            result = test_individual_archetype(arch_id, arch_name, df, start_date, end_date)
            all_results.append(result)
        except Exception as e:
            print(f"\n✗ ERROR testing {arch_id}: {e}")
            import traceback
            traceback.print_exc()

    # Portfolio summary
    if all_results:
        print(f"\n{'='*80}")
        print("PORTFOLIO SUMMARY (2023)")
        print(f"{'='*80}\n")

        summary_df = pd.DataFrame(all_results)
        print(summary_df[['archetype', 'trades', 'win_rate', 'pnl', 'profit_factor', 'max_dd']].to_string(index=False))

        # Aggregate stats
        total_trades = summary_df['trades'].sum()
        total_pnl = summary_df['pnl'].sum()
        avg_win_rate = (summary_df['trades'] * summary_df['win_rate']).sum() / total_trades if total_trades > 0 else 0

        print(f"\n{'='*80}")
        print("AGGREGATE STATISTICS")
        print(f"{'='*80}")
        print(f"Total Trades: {total_trades}")
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Portfolio Return: {total_pnl/10000*100:.1f}%")
        print(f"Weighted Avg Win Rate: {avg_win_rate:.1f}%")

        # Before/After comparison
        print(f"\n{'='*80}")
        print("BEFORE/AFTER COMPARISON")
        print(f"{'='*80}")

        # Calculate without B
        without_b = summary_df[summary_df['archetype'] != 'B']
        trades_without_b = without_b['trades'].sum()
        pnl_without_b = without_b['pnl'].sum()

        # Calculate with B
        b_row = summary_df[summary_df['archetype'] == 'B']
        b_trades = b_row['trades'].values[0] if len(b_row) > 0 else 0
        b_pnl = b_row['pnl'].values[0] if len(b_row) > 0 else 0

        print(f"\nBEFORE B FIX:")
        print(f"  Total Trades: {trades_without_b}")
        print(f"  Portfolio PnL: ${pnl_without_b:.2f}")
        print(f"  Portfolio Return: {pnl_without_b/10000*100:.1f}%")

        print(f"\nAFTER B FIX:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Portfolio PnL: ${total_pnl:.2f}")
        print(f"  Portfolio Return: {total_pnl/10000*100:.1f}%")

        print(f"\nIMPROVEMENT:")
        print(f"  Additional Trades: +{b_trades}")
        print(f"  PnL Improvement: +${b_pnl:.2f}")
        print(f"  Return Improvement: +{b_pnl/10000*100:.1f}%")
        if pnl_without_b > 0:
            print(f"  Percentage Gain: +{(b_pnl/pnl_without_b)*100:.1f}%")

        # B archetype specific
        if len(b_row) > 0:
            print(f"\n{'='*80}")
            print("B ARCHETYPE CONTRIBUTION")
            print(f"{'='*80}")
            print(f"  Trades: {b_trades}")
            print(f"  Win Rate: {b_row['win_rate'].values[0]:.1f}%")
            print(f"  PnL: ${b_pnl:.2f}")
            print(f"  Profit Factor: {b_row['profit_factor'].values[0]:.2f}")
            print(f"  Max Drawdown: {b_row['max_dd'].values[0]:.1f}%")
            print(f"  Sharpe Ratio: {b_row['sharpe'].values[0]:.2f}")
            print(f"  Portfolio Contribution: {(b_pnl/total_pnl)*100:.1f}%")

        # Trade overlap analysis
        if len(all_results) > 1:
            check_trade_overlap(all_results)

        # Risk metrics
        print(f"\n{'='*80}")
        print("RISK METRICS (PORTFOLIO)")
        print(f"{'='*80}")
        max_dd_portfolio = summary_df['max_dd'].max()
        avg_dd = summary_df['max_dd'].mean()
        print(f"Max Drawdown (worst archetype): {max_dd_portfolio:.1f}%")
        print(f"Average Drawdown: {avg_dd:.1f}%")

        # Best performers
        print(f"\n{'='*80}")
        print("BEST PERFORMERS")
        print(f"{'='*80}")

        if len(summary_df) > 0:
            best_pf = summary_df.loc[summary_df['profit_factor'].idxmax()]
            print(f"Best Profit Factor: {best_pf['archetype']} ({best_pf['profit_factor']:.2f})")

            best_pnl = summary_df.loc[summary_df['pnl'].idxmax()]
            print(f"Best PnL: {best_pnl['archetype']} (${best_pnl['pnl']:.2f})")

            best_wr = summary_df.loc[summary_df['win_rate'].idxmax()]
            print(f"Best Win Rate: {best_wr['archetype']} ({best_wr['win_rate']:.1f}%)")

            most_trades = summary_df.loc[summary_df['trades'].idxmax()]
            print(f"Most Active: {most_trades['archetype']} ({int(most_trades['trades'])} trades)")

        # Recommendation
        print(f"\n{'='*80}")
        print("RECOMMENDATION")
        print(f"{'='*80}\n")

        if b_pnl > 700 and b_trades > 50 and max_dd_portfolio < 5.0:
            print("✅ RECOMMENDATION: KEEP B ARCHETYPE ENABLED")
            print(f"\nReasons:")
            print(f"  - PnL improvement: ${b_pnl:.2f} (target: $700-900) ✅")
            print(f"  - Trade count: {b_trades} (target: 50+) ✅")
            print(f"  - Max drawdown: {max_dd_portfolio:.1f}% (limit: <5%) ✅")
            print(f"  - Portfolio boost: +{(b_pnl/pnl_without_b)*100:.1f}%")
        else:
            print("⚠️  RECOMMENDATION: REVIEW B ARCHETYPE")
            print(f"\nIssues:")
            if b_pnl < 700:
                print(f"  - PnL below target: ${b_pnl:.2f} (target: $700-900)")
            if b_trades < 50:
                print(f"  - Trade count low: {b_trades} (target: 50+)")
            if max_dd_portfolio >= 5.0:
                print(f"  - Drawdown too high: {max_dd_portfolio:.1f}% (limit: <5%)")

    print(f"\n{'='*80}")
    print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*80}\n")

    return 0


if __name__ == '__main__':
    sys.exit(main())
