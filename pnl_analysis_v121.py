#!/usr/bin/env python3
"""
PnL Analysis for Bull Machine v1.2.1 Production Test Results
Simulates trade outcomes based on risk/reward ratios
"""

import sys
import os
import pandas as pd
from datetime import datetime
import numpy as np

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bull_machine.app.main import run_bull_machine_v1_2_1
from bull_machine.io.feeders import load_csv_to_series

def simulate_trade_outcome(entry_price, stop_price, side, future_bars,
                          r_targets=[1.0, 2.5, 4.5], tp_percentages=[0.25, 0.35, 0.40]):
    """
    Simulate trade outcome with partial take profits
    Returns: (outcome, exit_price, r_achieved, bars_held)
    """
    if not future_bars or len(future_bars) == 0:
        return 'no_data', entry_price, 0.0, 0

    risk_per_unit = abs(entry_price - stop_price)
    if risk_per_unit == 0:
        return 'invalid', entry_price, 0.0, 0

    # Calculate TP levels
    tp_levels = []
    for r in r_targets:
        if side == 'long':
            tp_levels.append(entry_price + risk_per_unit * r)
        else:
            tp_levels.append(entry_price - risk_per_unit * r)

    # Track position and exits
    remaining_position = 1.0
    weighted_exit_price = 0.0
    total_r = 0.0

    for i, bar in enumerate(future_bars):
        # Check stop loss first
        if side == 'long' and bar['low'] <= stop_price:
            # Stop hit - exit remaining position
            weighted_exit_price += stop_price * remaining_position
            total_r += -1.0 * remaining_position  # -1R on stop
            return 'stop_loss', weighted_exit_price, total_r, i + 1

        elif side == 'short' and bar['high'] >= stop_price:
            # Stop hit - exit remaining position
            weighted_exit_price += stop_price * remaining_position
            total_r += -1.0 * remaining_position  # -1R on stop
            return 'stop_loss', weighted_exit_price, total_r, i + 1

        # Check take profit levels
        for tp_idx, (tp_price, tp_pct) in enumerate(zip(tp_levels, tp_percentages)):
            if remaining_position <= 0:
                break

            if side == 'long' and bar['high'] >= tp_price:
                # TP hit - partial exit
                exit_size = min(tp_pct, remaining_position)
                weighted_exit_price += tp_price * exit_size
                total_r += r_targets[tp_idx] * exit_size
                remaining_position -= exit_size

                if remaining_position <= 0.01:  # Fully exited
                    return f'tp{tp_idx+1}_full', weighted_exit_price, total_r, i + 1

            elif side == 'short' and bar['low'] <= tp_price:
                # TP hit - partial exit
                exit_size = min(tp_pct, remaining_position)
                weighted_exit_price += tp_price * exit_size
                total_r += r_targets[tp_idx] * exit_size
                remaining_position -= exit_size

                if remaining_position <= 0.01:  # Fully exited
                    return f'tp{tp_idx+1}_full', weighted_exit_price, total_r, i + 1

    # Time exit (TTL reached)
    if remaining_position > 0:
        final_price = future_bars[-1]['close']
        weighted_exit_price += final_price * remaining_position

        # Calculate R for remaining position
        if side == 'long':
            final_r = (final_price - entry_price) / risk_per_unit
        else:
            final_r = (entry_price - final_price) / risk_per_unit

        total_r += final_r * remaining_position

    return 'time_exit', weighted_exit_price, total_r, len(future_bars)

def run_pnl_analysis(csv_path, symbol, threshold=0.35, test_interval=10, max_tests=50):
    """
    Run PnL analysis on historical data
    """
    print(f"\n{'='*80}")
    print(f"PNL ANALYSIS: {symbol} @ threshold {threshold}")
    print(f"{'='*80}")

    # Load full dataset
    df = pd.read_csv(csv_path)

    # Convert time column if needed
    if 'time' in df.columns:
        df = df.rename(columns={'time': 'timestamp'})

    trades = []
    temp_files = []

    # Test range
    start_idx = 100
    end_idx = min(len(df) - 30, start_idx + test_interval * max_tests)

    for i in range(start_idx, end_idx, test_interval):
        # Create subset CSV
        temp_filename = f'temp_pnl_{symbol}_{i}.csv'
        subset = df.iloc[:i+1].copy()
        subset.to_csv(temp_filename, index=False)
        temp_files.append(temp_filename)

        try:
            # Run v1.2.1 analysis
            result = run_bull_machine_v1_2_1(
                temp_filename,
                account_balance=10000,
                override_signals={'enter_threshold': threshold}
            )

            if result['action'] == 'enter_trade':
                signal = result.get('signal')
                plan = result.get('risk_plan')

                if signal and plan:
                    # Get future bars for outcome simulation (max 20 bars for TTL)
                    future_start = i + 1
                    future_end = min(i + 21, len(df))
                    future_bars = df.iloc[future_start:future_end].to_dict('records')

                    # Simulate trade outcome
                    outcome, exit_price, r_achieved, bars_held = simulate_trade_outcome(
                        plan.entry,
                        plan.stop,
                        signal.side,
                        future_bars
                    )

                    # Calculate dollar PnL
                    position_value = plan.entry * plan.size
                    risk_amount = abs(plan.entry - plan.stop) * plan.size
                    dollar_pnl = r_achieved * risk_amount

                    # Calculate percentage PnL
                    if signal.side == 'long':
                        pct_pnl = ((exit_price - plan.entry) / plan.entry) * 100
                    else:
                        pct_pnl = ((plan.entry - exit_price) / plan.entry) * 100

                    trade = {
                        'date': datetime.utcfromtimestamp(int(df.iloc[i]['timestamp'])).strftime('%Y-%m-%d %H:%M'),
                        'side': signal.side,
                        'entry': plan.entry,
                        'stop': plan.stop,
                        'exit': exit_price,
                        'size': plan.size,
                        'confidence': signal.confidence,
                        'outcome': outcome,
                        'r_achieved': r_achieved,
                        'dollar_pnl': dollar_pnl,
                        'pct_pnl': pct_pnl,
                        'bars_held': bars_held,
                        'position_value': position_value
                    }
                    trades.append(trade)

        except Exception as e:
            continue

    # Cleanup temp files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return trades

def calculate_statistics(trades, initial_balance=10000):
    """Calculate comprehensive trading statistics"""
    if not trades:
        return {}

    # Basic stats
    total_trades = len(trades)
    winners = [t for t in trades if t['r_achieved'] > 0]
    losers = [t for t in trades if t['r_achieved'] <= 0]

    win_rate = len(winners) / total_trades * 100

    # R-based statistics
    total_r = sum(t['r_achieved'] for t in trades)
    avg_r = total_r / total_trades

    avg_winner_r = sum(t['r_achieved'] for t in winners) / len(winners) if winners else 0
    avg_loser_r = sum(t['r_achieved'] for t in losers) / len(losers) if losers else 0

    # Dollar PnL
    total_dollar_pnl = sum(t['dollar_pnl'] for t in trades)
    avg_dollar_pnl = total_dollar_pnl / total_trades

    # Percentage returns
    cumulative_return = 0
    balance = initial_balance
    returns = []
    equity_curve = [initial_balance]

    for trade in trades:
        trade_return = trade['dollar_pnl'] / balance * 100
        returns.append(trade_return)
        balance += trade['dollar_pnl']
        equity_curve.append(balance)
        cumulative_return = (balance - initial_balance) / initial_balance * 100

    # Risk metrics
    max_drawdown = 0
    peak = initial_balance
    for equity in equity_curve:
        if equity > peak:
            peak = equity
        drawdown = (peak - equity) / peak * 100
        max_drawdown = max(max_drawdown, drawdown)

    # Profit factor
    gross_profit = sum(t['dollar_pnl'] for t in winners) if winners else 0
    gross_loss = abs(sum(t['dollar_pnl'] for t in losers)) if losers else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf') if gross_profit > 0 else 0

    # Sharpe ratio (simplified)
    if returns and len(returns) > 1:
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    # Expectancy
    expectancy = (win_rate/100 * avg_winner_r) + ((1 - win_rate/100) * avg_loser_r)

    return {
        'total_trades': total_trades,
        'winners': len(winners),
        'losers': len(losers),
        'win_rate': win_rate,
        'total_r': total_r,
        'avg_r_per_trade': avg_r,
        'avg_winner_r': avg_winner_r,
        'avg_loser_r': avg_loser_r,
        'total_dollar_pnl': total_dollar_pnl,
        'avg_dollar_pnl': avg_dollar_pnl,
        'cumulative_return_pct': cumulative_return,
        'final_balance': balance,
        'max_drawdown_pct': max_drawdown,
        'profit_factor': profit_factor,
        'sharpe_ratio': sharpe,
        'expectancy_r': expectancy
    }

def print_detailed_report(btc_trades, eth_trades, btc_stats, eth_stats):
    """Print detailed PnL report"""

    print("\n" + "="*100)
    print("DETAILED PNL ANALYSIS REPORT - Bull Machine v1.2.1")
    print("="*100)

    # Individual trade details
    print("\n📊 BTC TRADES (First 10)")
    print("-"*80)
    print(f"{'Date':<20} {'Side':<6} {'Entry':<10} {'Exit':<10} {'R':<8} {'PnL%':<8} {'Outcome':<15}")
    print("-"*80)

    for trade in btc_trades[:10]:
        print(f"{trade['date']:<20} {trade['side'].upper():<6} "
              f"${trade['entry']:<10.0f} ${trade['exit']:<10.0f} "
              f"{trade['r_achieved']:<8.2f} {trade['pct_pnl']:<8.2f}% "
              f"{trade['outcome']:<15}")

    print("\n📊 ETH TRADES (First 10)")
    print("-"*80)
    print(f"{'Date':<20} {'Side':<6} {'Entry':<10} {'Exit':<10} {'R':<8} {'PnL%':<8} {'Outcome':<15}")
    print("-"*80)

    for trade in eth_trades[:10]:
        print(f"{trade['date']:<20} {trade['side'].upper():<6} "
              f"${trade['entry']:<10.2f} ${trade['exit']:<10.2f} "
              f"{trade['r_achieved']:<8.2f} {trade['pct_pnl']:<8.2f}% "
              f"{trade['outcome']:<15}")

    # Performance comparison
    print("\n" + "="*80)
    print("PERFORMANCE METRICS COMPARISON")
    print("="*80)
    print(f"{'Metric':<30} {'BTC':<20} {'ETH':<20}")
    print("-"*70)

    metrics = [
        ('Total Trades', 'total_trades', '{:.0f}'),
        ('Win Rate', 'win_rate', '{:.1f}%'),
        ('Avg R per Trade', 'avg_r_per_trade', '{:.3f}R'),
        ('Total R', 'total_r', '{:.1f}R'),
        ('Avg Winner R', 'avg_winner_r', '{:.2f}R'),
        ('Avg Loser R', 'avg_loser_r', '{:.2f}R'),
        ('Expectancy', 'expectancy_r', '{:.3f}R'),
        ('', '', ''),  # Blank line
        ('Total Dollar PnL', 'total_dollar_pnl', '${:.2f}'),
        ('Avg Dollar PnL', 'avg_dollar_pnl', '${:.2f}'),
        ('Cumulative Return', 'cumulative_return_pct', '{:.2f}%'),
        ('Final Balance', 'final_balance', '${:.2f}'),
        ('', '', ''),  # Blank line
        ('Max Drawdown', 'max_drawdown_pct', '{:.2f}%'),
        ('Profit Factor', 'profit_factor', '{:.2f}'),
        ('Sharpe Ratio', 'sharpe_ratio', '{:.2f}'),
    ]

    for label, key, fmt in metrics:
        if label == '':
            print()
            continue

        btc_val = btc_stats.get(key, 0)
        eth_val = eth_stats.get(key, 0)

        if '%' in fmt or 'R' in fmt or '$' in fmt:
            btc_str = fmt.format(btc_val)
            eth_str = fmt.format(eth_val)
        else:
            btc_str = fmt.format(btc_val)
            eth_str = fmt.format(eth_val)

        print(f"{label:<30} {btc_str:<20} {eth_str:<20}")

def main():
    # Configuration
    btc_path = '/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_BTCUSD, 1D_2f7fe.csv'
    eth_path = '/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 720_ffc2d.csv'

    initial_balance = 10000
    threshold = 0.35  # Recommended optimal threshold

    print("="*100)
    print("BULL MACHINE v1.2.1 - PNL ANALYSIS")
    print("="*100)
    print(f"Initial Balance: ${initial_balance:,.2f}")
    print(f"Risk per Trade: $60 (0.6% of initial balance)")
    print(f"Signal Threshold: {threshold}")
    print(f"Risk Management: 1R = Stop Loss, TP1 = 1R (25%), TP2 = 2.5R (35%), TP3 = 4.5R (40%)")
    print("="*100)

    # Run analysis
    print("\n⏳ Analyzing BTC trades...")
    btc_trades = run_pnl_analysis(btc_path, 'BTCUSD', threshold=threshold, max_tests=40)
    btc_stats = calculate_statistics(btc_trades, initial_balance)

    print("\n⏳ Analyzing ETH trades...")
    eth_trades = run_pnl_analysis(eth_path, 'ETHUSD', threshold=threshold, max_tests=50)
    eth_stats = calculate_statistics(eth_trades, initial_balance)

    # Print detailed report
    print_detailed_report(btc_trades, eth_trades, btc_stats, eth_stats)

    # Executive summary
    print("\n" + "="*100)
    print("EXECUTIVE SUMMARY")
    print("="*100)

    combined_trades = len(btc_trades) + len(eth_trades)
    combined_pnl = btc_stats.get('total_dollar_pnl', 0) + eth_stats.get('total_dollar_pnl', 0)
    combined_return = combined_pnl / initial_balance * 100

    print(f"\n📈 COMBINED RESULTS:")
    print(f"   Total Trades: {combined_trades}")
    print(f"   Total PnL: ${combined_pnl:,.2f}")
    print(f"   Return on Capital: {combined_return:.2f}%")

    avg_win_rate = (btc_stats.get('win_rate', 0) + eth_stats.get('win_rate', 0)) / 2
    avg_expectancy = (btc_stats.get('expectancy_r', 0) + eth_stats.get('expectancy_r', 0)) / 2

    print(f"\n📊 SYSTEM METRICS:")
    print(f"   Average Win Rate: {avg_win_rate:.1f}%")
    print(f"   Average Expectancy: {avg_expectancy:.3f}R per trade")
    print(f"   System Edge: {'POSITIVE' if avg_expectancy > 0 else 'NEGATIVE'}")

    if avg_expectancy > 0:
        print(f"\n✅ SYSTEM VERDICT: PROFITABLE")
        print(f"   The v1.2.1 system shows positive expectancy with proper risk management")
    else:
        print(f"\n⚠️ SYSTEM VERDICT: NEEDS OPTIMIZATION")
        print(f"   Consider adjusting thresholds or improving confluence weights")

    print("\n" + "="*100)

if __name__ == "__main__":
    main()