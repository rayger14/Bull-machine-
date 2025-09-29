#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - 2024 ETH Performance Analysis
Real P&L analysis with $10k starting balance
"""

from run_complete_confluence_system import load_multi_timeframe_data, run_complete_confluence_backtest
import json

def analyze_2024_performance():
    print('🔬 Bull Machine v1.6.2 - 2024 ETH Performance Analysis')
    print('=' * 60)

    # Load ETH data
    data_paths = {
        '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv',
        '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv',
        '1H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv'
    }

    data = load_multi_timeframe_data('ETH', data_paths)

    # Production configuration for 2024 only
    config = {
        'entry_threshold': 0.3,
        'min_active_domains': 3,
        'cooldown_days': 7,
        'risk_pct': 0.025,  # 2.5% risk per trade
        'sl_atr_multiplier': 1.4,
        'tp_atr_multiplier': 2.5,
        'trail_atr_multiplier': 0.8,
        'start_date': '2024-01-01',
        'end_date': '2024-12-31'
    }

    print(f'Testing period: {config["start_date"]} to {config["end_date"]}')
    print(f'Starting capital: $10,000')
    print(f'Risk per trade: {config["risk_pct"]*100:.1f}%')
    print()

    result = run_complete_confluence_backtest('ETH', data, config)

    if result and 'trades' in result:
        trades = result['trades']
        metrics = result.get('metrics', {})

        # Starting balance
        starting_balance = 10000

        print(f'📊 TRADE-BY-TRADE BREAKDOWN')
        print('-' * 40)

        winners = []
        losers = []
        running_balance = starting_balance

        for i, trade in enumerate(trades, 1):
            entry_time = trade['entry_time'].strftime('%Y-%m-%d')
            exit_time = trade['exit_time'].strftime('%Y-%m-%d')
            pnl_pct = trade['pnl']
            score = trade['score']
            grade = trade['confluence_grade']

            # Calculate dollar P&L based on risk per trade
            trade_risk_amount = running_balance * config['risk_pct']
            dollar_pnl = trade_risk_amount * (pnl_pct / 100)
            running_balance += dollar_pnl

            status = '✅ WIN' if pnl_pct > 0 else '❌ LOSS'

            if pnl_pct > 0:
                winners.append((pnl_pct, dollar_pnl))
            else:
                losers.append((pnl_pct, dollar_pnl))

            print(f'Trade #{i:2d}: {entry_time} → {exit_time}')
            print(f'   Score: {score:.2f} | Grade: {grade}')
            print(f'   P&L: {pnl_pct:+6.2f}% = ${dollar_pnl:+8.2f} | Balance: ${running_balance:,.2f} | {status}')
            print()

        print(f'📈 SUMMARY STATISTICS')
        print('-' * 30)
        print(f'Total trades: {len(trades)}')
        print(f'Winners: {len(winners)} ({len(winners)/len(trades)*100:.1f}%)')
        print(f'Losers: {len(losers)} ({len(losers)/len(trades)*100:.1f}%)')
        print()

        total_winnings = sum(pnl[1] for pnl in winners)
        total_losses = sum(pnl[1] for pnl in losers)
        net_pnl = total_winnings + total_losses  # losses are negative

        print(f'💰 FINANCIAL PERFORMANCE')
        print('-' * 25)
        print(f'Starting balance: ${starting_balance:,.2f}')
        print(f'Ending balance:   ${running_balance:,.2f}')
        print(f'Net P&L:          ${net_pnl:+,.2f}')
        print(f'Total return:     {(running_balance/starting_balance - 1)*100:+.2f}%')
        print()
        print(f'Gross winnings:   ${total_winnings:+,.2f}')
        print(f'Gross losses:     ${total_losses:+,.2f}')

        # Calculate profit factor safely
        if total_losses != 0:
            profit_factor = abs(total_winnings/total_losses)
        else:
            profit_factor = float('inf')
        print(f'Profit factor:    {profit_factor:.2f}')
        print()

        if winners:
            avg_winner = sum(pnl[1] for pnl in winners) / len(winners)
            best_winner = max(winners, key=lambda x: x[1])
            print(f'Average winner:   ${avg_winner:+,.2f} ({sum(pnl[0] for pnl in winners)/len(winners):+.2f}%)')
            print(f'Best winner:      ${best_winner[1]:+,.2f} ({best_winner[0]:+.2f}%)')

        if losers:
            avg_loser = sum(pnl[1] for pnl in losers) / len(losers)
            worst_loser = min(losers, key=lambda x: x[1])
            print(f'Average loser:    ${avg_loser:+,.2f} ({sum(pnl[0] for pnl in losers)/len(losers):+.2f}%)')
            print(f'Worst loser:      ${worst_loser[1]:+,.2f} ({worst_loser[0]:+.2f}%)')

        print()
        print(f'🎯 RISK METRICS')
        print('-' * 15)
        max_dd_pct = metrics.get('max_drawdown_pct', 0)
        sharpe = metrics.get('sharpe_ratio', 0)
        print(f'Max drawdown:     {max_dd_pct:.2f}%')
        print(f'Sharpe ratio:     {sharpe:.2f}')
        print(f'Risk per trade:   ${starting_balance * config["risk_pct"]:,.0f} ({config["risk_pct"]*100:.1f}%)')

        # Additional insights
        print()
        print(f'📊 ADDITIONAL INSIGHTS')
        print('-' * 22)

        # Filter trades by year to see 2024 specifically
        trades_2024 = [t for t in trades if t['entry_time'].year == 2024]
        if trades_2024:
            print(f'Trades in 2024: {len(trades_2024)}')
            wins_2024 = len([t for t in trades_2024 if t['pnl'] > 0])
            print(f'2024 win rate: {wins_2024/len(trades_2024)*100:.1f}%')

        # Monthly breakdown
        monthly_trades = {}
        for trade in trades:
            month_key = trade['entry_time'].strftime('%Y-%m')
            if month_key not in monthly_trades:
                monthly_trades[month_key] = []
            monthly_trades[month_key].append(trade)

        print(f'Monthly activity:')
        for month, month_trades in sorted(monthly_trades.items()):
            wins = len([t for t in month_trades if t['pnl'] > 0])
            print(f'  {month}: {len(month_trades)} trades ({wins} wins)')

    else:
        print('❌ No trades generated or error in backtest')

if __name__ == "__main__":
    analyze_2024_performance()