#!/usr/bin/env python3
"""
Bull Machine Year-Long Performance Report Generator
Comprehensive analysis of long-horizon backtest results
"""

import json
import pandas as pd
import numpy as np
from datetime import datetime

def load_summary(filepath: str):
    """Load year summary JSON."""
    with open(filepath, 'r') as f:
        return json.load(f)

def load_trades(filepath: str):
    """Load trades CSV."""
    df = pd.read_csv(filepath)
    df['entry_time'] = pd.to_datetime(df['entry_time'])
    return df

def generate_comprehensive_report():
    """Generate comprehensive year-long performance report."""

    # Load data
    summary = load_summary('reports/long_run/ETH_2024_2025/year_summary.json')
    trades_df = load_trades('reports/long_run/ETH_2024_2025/year_trades.csv')

    print("🚀 BULL MACHINE v1.7.1 - YEAR-LONG PERFORMANCE ANALYSIS")
    print("=" * 80)
    print(f"Analysis Period: {summary['summary']['period_start']} → {summary['summary']['period_end']}")
    print(f"Configuration: {summary['summary']['config_version']}")
    print(f"Total Chunks Processed: {summary['summary']['total_chunks']}")
    print("=" * 80)

    # Core Performance Metrics
    perf = summary['performance']
    print("\n📊 CORE PERFORMANCE METRICS:")
    print("-" * 80)
    print(f"{'Metric':<25} {'Value':<20} {'Assessment':<30}")
    print("-" * 80)
    print(f"{'Total Trades':<25} {perf['total_trades']:<20} {'High frequency' if perf['total_trades'] > 100 else 'Moderate frequency':<30}")
    print(f"{'Starting Balance':<25} {'$10,000.00':<20} {'Standard capital':<30}")
    print(f"{'Final Balance':<25} ${perf['final_balance']:,.2f}{'':<14} {'Significant loss' if perf['final_balance'] < 8000 else 'Acceptable':<30}")
    print(f"{'Total Return':<25} {perf['total_return']:.2f}%{'':<15} {'Poor performance' if perf['total_return'] < -10 else 'Good performance':<30}")
    print(f"{'Win Rate':<25} {perf['win_rate']:.1f}%{'':<16} {'Below target' if perf['win_rate'] < 55 else 'Good':<30}")
    print(f"{'Profit Factor':<25} {perf['profit_factor']:.2f}{'':<17} {'Poor' if perf['profit_factor'] < 1.3 else 'Good':<30}")
    print(f"{'Max Drawdown':<25} {perf['max_drawdown']:.2f}%{'':<15} {'Excessive' if perf['max_drawdown'] > 15 else 'Acceptable':<30}")
    print(f"{'Sharpe Ratio':<25} {perf['sharpe_ratio']:.2f}{'':<17} {'Poor' if perf['sharpe_ratio'] < 0.5 else 'Good':<30}")
    print(f"{'Average R-Multiple':<25} {perf['avg_r_multiple']:.2f}{'':<17} {'Below 1.0' if perf['avg_r_multiple'] < 1.0 else 'Good':<30}")

    # Trading Statistics
    print(f"\n💰 TRADING STATISTICS:")
    print("-" * 80)
    print(f"Winning Trades: {perf['winning_trades']} ({perf['win_rate']:.1f}%)")
    print(f"Losing Trades: {perf['losing_trades']} ({100-perf['win_rate']:.1f}%)")
    print(f"Average Win: {perf['avg_win']:.2f}%")
    print(f"Average Loss: {perf['avg_loss']:.2f}%")
    print(f"Largest Win: {trades_df['pnl'].max():.2f}%")
    print(f"Largest Loss: {trades_df['pnl'].min():.2f}%")
    print(f"R > 1.0 Trades: {perf['r_gt_1_count']} ({perf['r_gt_1_count']/perf['total_trades']*100:.1f}%)")
    print(f"R > 2.0 Trades: {perf['r_gt_2_count']} ({perf['r_gt_2_count']/perf['total_trades']*100:.1f}%)")

    # Engine Utilization Analysis
    engine_util = summary['engine_utilization']
    total_engine_signals = sum(engine_util.values())

    print(f"\n🔧 ENGINE UTILIZATION ANALYSIS:")
    print("-" * 80)
    print(f"{'Engine':<15} {'Signals':<10} {'Percentage':<12} {'Assessment':<30}")
    print("-" * 80)

    for engine, count in engine_util.items():
        if total_engine_signals > 0:
            pct = count / total_engine_signals * 100
            if engine.upper() == 'SMC':
                assessment = 'Dominant engine' if pct > 30 else 'Good utilization'
            elif engine.upper() == 'WYCKOFF':
                assessment = 'Well utilized' if pct > 15 else 'Underutilized'
            elif engine.upper() == 'HOB':
                assessment = f'Within target (<30%)' if pct <= 30 else 'Over-utilized'
            elif 'VETO' in engine.upper():
                assessment = 'No vetos (issue)' if count == 0 else 'Healthy filtering'
            else:
                assessment = 'Active participation'

            print(f"{engine.upper():<15} {count:<10} {pct:.1f}%{'':<8} {assessment:<30}")

    # Exit Type Analysis
    exit_types = summary['exit_type_breakdown']
    total_exits = sum(exit_types.values())

    print(f"\n🚪 EXIT TYPE BREAKDOWN:")
    print("-" * 80)
    for exit_type, count in exit_types.items():
        pct = count / total_exits * 100 if total_exits > 0 else 0
        print(f"{exit_type.replace('_', ' ').title()}: {count} ({pct:.1f}%)")

    # Monthly Analysis
    trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
    monthly_stats = trades_df.groupby('month').agg({
        'pnl': ['count', 'sum', 'mean'],
        'is_winner': 'mean'
    }).round(2)

    print(f"\n📅 MONTHLY PERFORMANCE BREAKDOWN:")
    print("-" * 80)
    print(f"{'Month':<12} {'Trades':<8} {'Total PnL':<12} {'Avg PnL':<10} {'Win Rate':<10}")
    print("-" * 80)

    for month in monthly_stats.index:
        trades_count = int(monthly_stats.loc[month, ('pnl', 'count')])
        total_pnl = monthly_stats.loc[month, ('pnl', 'sum')]
        avg_pnl = monthly_stats.loc[month, ('pnl', 'mean')]
        win_rate = monthly_stats.loc[month, ('is_winner', 'mean')] * 100

        print(f"{str(month):<12} {trades_count:<8} {total_pnl:+.2f}%{'':<7} {avg_pnl:+.2f}%{'':<5} {win_rate:.1f}%")

    # Engine Combination Analysis
    engine_combos = summary['engine_combinations']
    print(f"\n🤝 ENGINE COMBINATION ANALYSIS:")
    print("-" * 80)
    print(f"{'Combination':<25} {'Count':<8} {'Percentage':<12}")
    print("-" * 80)

    for combo, count in sorted(engine_combos.items(), key=lambda x: x[1], reverse=True)[:10]:
        pct = count / perf['total_trades'] * 100
        print(f"{combo:<25} {count:<8} {pct:.1f}%")

    # Health Band Assessment
    print(f"\n🏥 INSTITUTIONAL HEALTH BAND ASSESSMENT:")
    print("-" * 80)

    health_checks = [
        ("Profit Factor", perf['profit_factor'], "≥ 1.30", perf['profit_factor'] >= 1.30),
        ("Max Drawdown", perf['max_drawdown'], "≤ 8.0%", perf['max_drawdown'] <= 8.0),
        ("Win Rate", perf['win_rate'], "≥ 55.0%", perf['win_rate'] >= 55.0),
        ("SMC Utilization", engine_util.get('smc', 0)/total_engine_signals*100 if total_engine_signals > 0 else 0, "≥ 30.0%", (engine_util.get('smc', 0)/total_engine_signals*100 if total_engine_signals > 0 else 0) >= 30.0),
        ("HOB Relevance", engine_util.get('hob', 0)/total_engine_signals*100 if total_engine_signals > 0 else 0, "≤ 30.0%", (engine_util.get('hob', 0)/total_engine_signals*100 if total_engine_signals > 0 else 0) <= 30.0)
    ]

    passed_checks = 0
    for check_name, value, target, passed in health_checks:
        status = "✅ PASS" if passed else "❌ FAIL"
        if passed:
            passed_checks += 1

        if "%" in target:
            print(f"{check_name:<20} {value:.1f}% {target:<10} {status}")
        else:
            print(f"{check_name:<20} {value:.2f} {target:<12} {status}")

    print(f"\nOverall Health Score: {passed_checks}/{len(health_checks)} ({passed_checks/len(health_checks)*100:.1f}%)")

    # Key Insights and Recommendations
    print(f"\n🔍 KEY INSIGHTS & RECOMMENDATIONS:")
    print("-" * 80)

    insights = []

    if perf['profit_factor'] < 1.0:
        insights.append("• System is losing money overall - urgent refinement needed")
    elif perf['profit_factor'] < 1.3:
        insights.append("• Profit factor below institutional threshold - improve signal quality")

    if perf['max_drawdown'] > 15:
        insights.append("• Excessive drawdown - strengthen risk management and position sizing")

    if perf['win_rate'] < 50:
        insights.append("• Low win rate - enhance entry criteria and confluence requirements")

    if engine_util.get('macro_veto', 0) == 0:
        insights.append("• No macro vetos triggered - verify macro filtering is active")

    if perf['avg_r_multiple'] < 1.0:
        insights.append("• Average R-multiple below 1.0 - improve risk/reward management")

    # Monthly consistency check
    monthly_returns = trades_df.groupby('month')['pnl'].sum()
    negative_months = (monthly_returns < -5).sum()
    if negative_months > len(monthly_returns) * 0.4:
        insights.append("• High number of negative months - improve consistency")

    if not insights:
        insights.append("• System performance within acceptable parameters")
        insights.append("• Consider minor optimizations for enhanced performance")

    for insight in insights:
        print(insight)

    # Risk Assessment
    print(f"\n⚠️  RISK ASSESSMENT:")
    print("-" * 80)

    # Calculate volatility of returns
    monthly_returns = trades_df.groupby('month')['pnl'].sum()
    volatility = monthly_returns.std()

    risk_level = "HIGH" if perf['max_drawdown'] > 20 else "MODERATE" if perf['max_drawdown'] > 10 else "LOW"
    consistency = "POOR" if volatility > 15 else "MODERATE" if volatility > 8 else "GOOD"

    print(f"Risk Level: {risk_level}")
    print(f"Consistency: {consistency}")
    print(f"Monthly Volatility: {volatility:.2f}%")
    print(f"Risk-Adjusted Return: {perf['total_return']/perf['max_drawdown']:.2f}" if perf['max_drawdown'] > 0 else "N/A")

    # Final Verdict
    print(f"\n🎯 FINAL VERDICT:")
    print("=" * 80)

    if passed_checks >= len(health_checks) * 0.8 and perf['total_return'] > 0:
        verdict = "✅ APPROVED FOR PRODUCTION"
        details = "System meets institutional standards and is ready for live deployment."
    elif passed_checks >= len(health_checks) * 0.6:
        verdict = "⚠️  CONDITIONAL APPROVAL"
        details = "System shows promise but requires refinements before production."
    else:
        verdict = "❌ NOT APPROVED"
        details = "System requires significant improvements before consideration for production."

    print(f"Status: {verdict}")
    print(f"Assessment: {details}")

    if perf['total_return'] < -50:
        print("\n🚨 CRITICAL ISSUE: System showing severe losses. Immediate review required.")

    print(f"\n📊 Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

if __name__ == "__main__":
    generate_comprehensive_report()