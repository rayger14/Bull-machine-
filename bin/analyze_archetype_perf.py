#!/usr/bin/env python3
"""
Analyze per-archetype performance from backtest summary
"""
import re
import sys

def parse_trade_log(log_path):
    """Extract trades by archetype with PNL"""
    archetypes = {}

    with open(log_path, 'r') as f:
        content = f.read()

    # Find all trades in format "Trade N: archetype_X ... PNL: $Y (Z%)"
    trade_pattern = r'Trade \d+: (archetype_\w+).*?PNL: \$([+-]?\d+\.?\d*) \(([+-]?\d+\.?\d*)%\)'

    for match in re.finditer(trade_pattern, content, re.DOTALL):
        archetype = match.group(1).replace('archetype_', '')
        pnl = float(match.group(2))
        pnl_pct = float(match.group(3))

        if archetype not in archetypes:
            archetypes[archetype] = {'trades': [], 'wins': 0, 'losses': 0}

        archetypes[archetype]['trades'].append(pnl)
        if pnl > 0:
            archetypes[archetype]['wins'] += 1
        else:
            archetypes[archetype]['losses'] += 1

    return archetypes

def calculate_metrics(trades):
    """Calculate performance metrics for trade list"""
    total = len(trades)
    wins = [t for t in trades if t > 0]
    losses = [t for t in trades if t <= 0]

    win_count = len(wins)
    loss_count = len(losses)

    total_pnl = sum(trades)
    avg_win = sum(wins) / len(wins) if wins else 0
    avg_loss = sum(losses) / len(losses) if losses else 0
    win_rate = (win_count / total * 100) if total > 0 else 0

    gross_profit = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0
    profit_factor = (gross_profit / gross_loss) if gross_loss > 0 else float('inf')

    return {
        'total': total,
        'wins': win_count,
        'losses': loss_count,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'profit_factor': profit_factor
    }

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 analyze_archetype_perf.py <log_file>")
        sys.exit(1)

    log_path = sys.argv[1]
    archetypes = parse_trade_log(log_path)

    print("\n" + "="*80)
    print("PER-ARCHETYPE PERFORMANCE BREAKDOWN")
    print("="*80)

    # Calculate overall metrics
    all_trades = []
    for arch_data in archetypes.values():
        all_trades.extend(arch_data['trades'])

    overall = calculate_metrics(all_trades)

    print(f"\nOVERALL:")
    print(f"  Trades: {overall['total']}")
    print(f"  Win Rate: {overall['win_rate']:.1f}%")
    print(f"  Total PNL: ${overall['total_pnl']:,.2f}")
    print(f"  Profit Factor: {overall['profit_factor']:.2f}")
    print(f"  Avg Win: ${overall['avg_win']:.2f}")
    print(f"  Avg Loss: ${overall['avg_loss']:.2f}")

    print("\n" + "-"*80)
    print("BY ARCHETYPE:")
    print("-"*80)

    # Sort by trade count
    sorted_archetypes = sorted(archetypes.items(), key=lambda x: len(x[1]['trades']), reverse=True)

    for archetype, data in sorted_archetypes:
        metrics = calculate_metrics(data['trades'])
        pct_of_total = (metrics['total'] / overall['total'] * 100)

        print(f"\n{archetype.upper()}:")
        print(f"  Trades: {metrics['total']} ({pct_of_total:.1f}% of total)")
        print(f"  Win Rate: {metrics['win_rate']:.1f}% ({metrics['wins']}W / {metrics['losses']}L)")
        print(f"  Total PNL: ${metrics['total_pnl']:,.2f}")
        print(f"  Profit Factor: {metrics['profit_factor']:.2f}")
        print(f"  Avg Win: ${metrics['avg_win']:.2f}")
        print(f"  Avg Loss: ${metrics['avg_loss']:.2f}")

        # Contribution to overall PNL
        pnl_contribution = (metrics['total_pnl'] / overall['total_pnl'] * 100) if overall['total_pnl'] != 0 else 0
        print(f"  PNL Contribution: {pnl_contribution:.1f}% of total")

    print("\n" + "="*80)

if __name__ == '__main__':
    main()
