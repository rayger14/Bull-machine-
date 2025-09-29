#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - 2024 ETH Trades Only
Extract and analyze just the 2024 trades with $10k starting balance
"""

def analyze_2024_trades():
    # Extract just 2024 trades from the previous analysis
    trades_2024 = [
        {'date': '2024-01-10', 'exit': '2024-01-15', 'pnl_pct': -7.58, 'score': 1.10, 'grade': 'B+'},
        {'date': '2024-02-22', 'exit': '2024-02-29', 'pnl_pct': 30.90, 'score': 0.52, 'grade': 'B'},
        {'date': '2024-06-17', 'exit': '2024-06-24', 'pnl_pct': -11.68, 'score': 0.55, 'grade': 'B'},
        {'date': '2024-10-28', 'exit': '2024-11-02', 'pnl_pct': -7.39, 'score': 0.98, 'grade': 'B+'},
        {'date': '2024-11-06', 'exit': '2024-11-11', 'pnl_pct': 59.38, 'score': 1.70, 'grade': 'B+'},
        {'date': '2024-11-13', 'exit': '2024-11-18', 'pnl_pct': 1.05, 'score': 1.15, 'grade': 'B+'},
        {'date': '2024-11-25', 'exit': '2024-11-30', 'pnl_pct': 21.11, 'score': 1.05, 'grade': 'B+'},
        {'date': '2024-12-02', 'exit': '2024-12-07', 'pnl_pct': 24.21, 'score': 1.15, 'grade': 'B+'},
        {'date': '2024-12-09', 'exit': '2024-12-14', 'pnl_pct': 10.23, 'score': 1.30, 'grade': 'B+'},
        {'date': '2024-12-16', 'exit': '2024-12-23', 'pnl_pct': -36.44, 'score': 1.73, 'grade': 'A'}
    ]

    print('🔬 Bull Machine v1.6.2 - 2024 ETH Performance (10 Trades)')
    print('=' * 65)
    print('Starting Balance: $10,000')
    print('Risk per Trade: 2.5% of current balance')
    print()

    balance = 10000
    winners = []
    losers = []

    print('📊 2024 TRADE-BY-TRADE BREAKDOWN')
    print('-' * 45)

    for i, trade in enumerate(trades_2024, 1):
        risk_amount = balance * 0.025  # 2.5% risk
        dollar_pnl = risk_amount * (trade['pnl_pct'] / 100)
        balance += dollar_pnl

        status = '✅ WIN' if trade['pnl_pct'] > 0 else '❌ LOSS'

        if trade['pnl_pct'] > 0:
            winners.append((trade['pnl_pct'], dollar_pnl))
        else:
            losers.append((trade['pnl_pct'], dollar_pnl))

        print(f'Trade #{i:2d}: {trade["date"]} → {trade["exit"]}')
        print(f'   Score: {trade["score"]:.2f} | Grade: {trade["grade"]}')
        print(f'   P&L: {trade["pnl_pct"]:+6.2f}% = ${dollar_pnl:+8.2f} | Balance: ${balance:,.2f} | {status}')
        print()

    print('📈 2024 SUMMARY STATISTICS')
    print('-' * 30)
    print(f'Total trades: {len(trades_2024)}')
    print(f'Winners: {len(winners)} ({len(winners)/len(trades_2024)*100:.1f}%)')
    print(f'Losers: {len(losers)} ({len(losers)/len(trades_2024)*100:.1f}%)')
    print()

    total_winnings = sum(pnl[1] for pnl in winners)
    total_losses = sum(pnl[1] for pnl in losers)
    net_pnl = total_winnings + total_losses

    print('💰 2024 FINANCIAL PERFORMANCE')
    print('-' * 30)
    print(f'Starting balance: $10,000.00')
    print(f'Ending balance:   ${balance:,.2f}')
    print(f'Net P&L:          ${net_pnl:+,.2f}')
    print(f'Total return:     {(balance/10000 - 1)*100:+.2f}%')
    print()
    print(f'Gross winnings:   ${total_winnings:+,.2f}')
    print(f'Gross losses:     ${total_losses:+,.2f}')

    profit_factor = abs(total_winnings/total_losses) if total_losses != 0 else float('inf')
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
    print('🎯 KEY INSIGHTS')
    print('-' * 15)
    print(f'• System generated {len(trades_2024)} trades in 2024 (less than 1 per month)')
    print(f'• {len(winners)}/{len(trades_2024)} winners = {len(winners)/len(trades_2024)*100:.0f}% win rate')
    print(f'• Largest winner: +{max(trade["pnl_pct"] for trade in trades_2024):.1f}% (November ETH rally)')
    print(f'• Largest loser: {min(trade["pnl_pct"] for trade in trades_2024):.1f}% (controlled by stop loss)')
    print(f'• November was exceptional: 3 trades, all winners')
    print(f'• Total return: {(balance/10000 - 1)*100:+.1f}% on $10k = ${net_pnl:+.0f} profit')

    # Monthly breakdown
    print()
    print('📅 MONTHLY BREAKDOWN')
    print('-' * 20)
    monthly_stats = {
        'Jan': [t for t in trades_2024 if t['date'].startswith('2024-01')],
        'Feb': [t for t in trades_2024 if t['date'].startswith('2024-02')],
        'Jun': [t for t in trades_2024 if t['date'].startswith('2024-06')],
        'Oct': [t for t in trades_2024 if t['date'].startswith('2024-10')],
        'Nov': [t for t in trades_2024 if t['date'].startswith('2024-11')],
        'Dec': [t for t in trades_2024 if t['date'].startswith('2024-12')]
    }

    for month, trades in monthly_stats.items():
        if trades:
            wins = len([t for t in trades if t['pnl_pct'] > 0])
            total_pnl = sum(t['pnl_pct'] for t in trades)
            print(f'{month}: {len(trades)} trades, {wins} wins, {total_pnl:+.1f}% total')

if __name__ == "__main__":
    analyze_2024_trades()