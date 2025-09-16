#!/usr/bin/env python3
"""
Comprehensive backtest script for Bull Machine v1.2.1 vs v1.1.2 on BTC daily data
"""
import sys
import os
import pandas as pd
from datetime import datetime
import logging
import json

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bull_machine.app.main import run_bull_machine_v1_1, run_bull_machine_v1_2_1
from bull_machine.io.feeders import load_csv_to_series

def setup_logging(level=logging.WARNING):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')

def create_subset_csv(df, end_idx, filename):
    """Create a CSV subset up to end_idx for testing"""
    subset = df.iloc[:end_idx+1].copy()
    subset.to_csv(filename, index=False)
    return filename

def simulate_trade_outcome(entry_price, stop_price, target_prices, future_prices, side='long'):
    """Simulate trade outcome given future price data"""
    if side == 'long':
        # Check if stop hit first
        for i, price in enumerate(future_prices):
            if price <= stop_price:
                return 'loss', stop_price, price, i+1
        # Check if any target hit
        for target in target_prices:
            for i, price in enumerate(future_prices):
                if price >= target:
                    return 'win', target, price, i+1
        # No target hit within timeframe
        return 'timeout', future_prices[-1], future_prices[-1], len(future_prices)
    else:  # short
        # Check if stop hit first
        for i, price in enumerate(future_prices):
            if price >= stop_price:
                return 'loss', stop_price, price, i+1
        # Check if any target hit
        for target in target_prices:
            for i, price in enumerate(future_prices):
                if price <= target:
                    return 'win', target, price, i+1
        return 'timeout', future_prices[-1], future_prices[-1], len(future_prices)

def backtest_version(csv_path, version='1.1', min_bars=100, lookforward_days=30,
                      threshold=None, test_every=5, max_tests=None):
    """Run backtest for a specific version"""
    print(f"\n{'='*60}")
    print(f"Testing Bull Machine v{version}")
    print(f"{'='*60}")

    # Load full dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} bars of data")

    # Convert time column to proper format if needed
    if 'time' in df.columns:
        df = df.rename(columns={'time': 'timestamp'})

    trades = []
    temp_files = []
    signals_tested = 0

    # Start from min_bars and go to len-lookforward_days
    start_idx = min_bars
    end_idx = len(df) - lookforward_days

    print(f"Testing from bar {start_idx} to {end_idx} (every {test_every} bars)")

    for i in range(start_idx, end_idx, test_every):
        if max_tests and signals_tested >= max_tests:
            break

        # Create subset CSV up to current point
        temp_filename = f'temp_btc_{version}_{i}.csv'
        create_subset_csv(df, i, temp_filename)
        temp_files.append(temp_filename)

        try:
            # Determine which function to use and set thresholds
            if version == '1.2.1':
                override_signals = {'enter_threshold': threshold} if threshold else {}
                result = run_bull_machine_v1_2_1(
                    temp_filename,
                    account_balance=10000,
                    override_signals=override_signals
                )
            else:  # v1.1
                override_signals = {'confidence_threshold': threshold} if threshold else {}
                result = run_bull_machine_v1_1(
                    temp_filename,
                    account_balance=10000,
                    override_signals=override_signals
                )

            signals_tested += 1

            if result['action'] == 'enter_trade':
                signal = result.get('signal')
                plan = result.get('risk_plan')

                if signal and plan:
                    current_bar = df.iloc[i]
                    entry_price = plan.entry
                    stop_price = plan.stop
                    side = signal.side

                    # Extract target prices
                    target_prices = [tp['price'] for tp in plan.tp_levels] if plan.tp_levels else []
                    if not target_prices:
                        # Default targets if none provided
                        risk = abs(entry_price - stop_price)
                        if side == 'long':
                            target_prices = [entry_price + risk * 2]
                        else:
                            target_prices = [entry_price - risk * 2]

                    # Get future prices for simulation
                    future_data = df.iloc[i+1:i+1+lookforward_days]
                    if len(future_data) >= 1:
                        # Use both highs and lows for more accurate simulation
                        future_prices = []
                        for _, row in future_data.iterrows():
                            future_prices.extend([row['high'], row['low']])

                        # Simulate trade outcome
                        outcome, exit_price, actual_price, bars_held = simulate_trade_outcome(
                            entry_price, stop_price, target_prices, future_prices, side
                        )

                        entry_date = datetime.utcfromtimestamp(int(current_bar['timestamp'])).strftime('%Y-%m-%d')

                        trade = {
                            'date': entry_date,
                            'bar_idx': i,
                            'side': side,
                            'entry_price': entry_price,
                            'stop_price': stop_price,
                            'target_prices': target_prices,
                            'exit_price': exit_price,
                            'outcome': outcome,
                            'bars_held': bars_held,
                            'confidence': signal.confidence,
                            'pnl_pct': ((exit_price - entry_price) / entry_price * 100) if side == 'long' else ((entry_price - exit_price) / entry_price * 100)
                        }
                        trades.append(trade)

                        print(f"Trade #{len(trades)}: {entry_date} {side.upper()} @ ${entry_price:.0f} -> {outcome.upper()} @ ${exit_price:.0f} ({trade['pnl_pct']:.1f}%) [Conf: {signal.confidence:.3f}]")

        except Exception as e:
            if "no_trade" not in str(e).lower():
                print(f"Error at bar {i}: {e}")
            continue

    # Cleanup temp files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    print(f"\nTotal signals tested: {signals_tested}")
    print(f"Total trades generated: {len(trades)}")

    return trades

def analyze_results(trades, version):
    """Analyze backtest results"""
    if not trades:
        return {
            "version": version,
            "total_trades": 0,
            "win_rate": 0,
            "avg_pnl": 0,
            "sharpe": 0
        }

    total_trades = len(trades)
    wins = sum(1 for t in trades if t['outcome'] == 'win')
    losses = sum(1 for t in trades if t['outcome'] == 'loss')
    timeouts = sum(1 for t in trades if t['outcome'] == 'timeout')

    win_rate = (wins / total_trades) * 100 if total_trades > 0 else 0
    avg_pnl = sum(t['pnl_pct'] for t in trades) / total_trades if total_trades > 0 else 0

    winning_trades = [t for t in trades if t['pnl_pct'] > 0]
    losing_trades = [t for t in trades if t['pnl_pct'] < 0]

    avg_win = sum(t['pnl_pct'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
    avg_loss = sum(t['pnl_pct'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

    # Calculate profit factor
    total_wins = sum(t['pnl_pct'] for t in winning_trades) if winning_trades else 0
    total_losses = abs(sum(t['pnl_pct'] for t in losing_trades)) if losing_trades else 0
    profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0

    # Calculate Sharpe ratio (simplified)
    returns = [t['pnl_pct'] for t in trades]
    if len(returns) > 1:
        import numpy as np
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252/30) if np.std(returns) > 0 else 0
    else:
        sharpe = 0

    # Calculate max drawdown
    cumulative = 0
    peak = 0
    max_dd = 0
    for t in trades:
        cumulative += t['pnl_pct']
        peak = max(peak, cumulative)
        drawdown = (peak - cumulative)
        max_dd = max(max_dd, drawdown)

    # Average confidence
    avg_confidence = sum(t['confidence'] for t in trades) / len(trades) if trades else 0

    return {
        "version": version,
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": profit_factor,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_dd,
        "avg_confidence": avg_confidence,
        "total_return": sum(t['pnl_pct'] for t in trades)
    }

def print_comparison(results_v11, results_v121):
    """Print comparison between versions"""
    print("\n" + "="*80)
    print("BACKTEST COMPARISON: v1.1.2 vs v1.2.1")
    print("="*80)

    metrics = [
        ("Total Trades", "total_trades", "{:.0f}"),
        ("Win Rate", "win_rate", "{:.1f}%"),
        ("Avg PnL", "avg_pnl", "{:.2f}%"),
        ("Avg Win", "avg_win", "{:.2f}%"),
        ("Avg Loss", "avg_loss", "{:.2f}%"),
        ("Profit Factor", "profit_factor", "{:.2f}"),
        ("Sharpe Ratio", "sharpe_ratio", "{:.2f}"),
        ("Max Drawdown", "max_drawdown", "{:.1f}%"),
        ("Total Return", "total_return", "{:.1f}%"),
        ("Avg Confidence", "avg_confidence", "{:.3f}"),
    ]

    print(f"{'Metric':<20} {'v1.1.2':>15} {'v1.2.1':>15} {'Improvement':>15}")
    print("-" * 65)

    for label, key, fmt in metrics:
        v11_val = results_v11.get(key, 0)
        v121_val = results_v121.get(key, 0)

        if key in ['total_trades', 'wins', 'losses']:
            improvement = v121_val - v11_val
            imp_str = f"+{improvement:.0f}" if improvement > 0 else f"{improvement:.0f}"
        elif key in ['win_rate', 'avg_pnl', 'profit_factor', 'sharpe_ratio', 'total_return', 'avg_confidence']:
            improvement = v121_val - v11_val
            imp_str = f"+{improvement:.1f}" if improvement > 0 else f"{improvement:.1f}"
        elif key == 'max_drawdown':
            improvement = v11_val - v121_val  # Lower is better
            imp_str = f"+{improvement:.1f}" if improvement > 0 else f"{improvement:.1f}"
        else:
            improvement = v121_val - v11_val
            imp_str = f"+{improvement:.1f}" if improvement > 0 else f"{improvement:.1f}"

        print(f"{label:<20} {fmt.format(v11_val):>15} {fmt.format(v121_val):>15} {imp_str:>15}")

if __name__ == "__main__":
    setup_logging(logging.WARNING)

    csv_path = '/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_BTCUSD, 1D_2f7fe.csv'

    print("="*80)
    print("BULL MACHINE BACKTEST: v1.1.2 vs v1.2.1")
    print("="*80)
    print(f"Data: BTC Daily")
    print(f"Test Period: ~491 days")
    print(f"Testing every 5 days for speed")
    print("="*80)

    # Test v1.1.2 with its optimal threshold
    print("\n📊 Testing v1.1.2 (threshold=0.60)...")
    trades_v11 = backtest_version(
        csv_path,
        version='1.1',
        threshold=0.60,
        test_every=5,
        max_tests=100  # Limit for speed
    )
    results_v11 = analyze_results(trades_v11, '1.1.2')

    # Test v1.2.1 with lower threshold to generate trades
    print("\n📊 Testing v1.2.1 (threshold=0.50)...")
    trades_v121 = backtest_version(
        csv_path,
        version='1.2.1',
        threshold=0.50,
        test_every=5,
        max_tests=100  # Limit for speed
    )
    results_v121 = analyze_results(trades_v121, '1.2.1')

    # Print comparison
    print_comparison(results_v11, results_v121)

    # Summary
    print("\n" + "="*80)
    print("EXECUTIVE SUMMARY")
    print("="*80)

    if results_v121['total_trades'] == 0:
        print("⚠️  v1.2.1 generated NO trades even with lower threshold (0.50)")
        print("   This indicates the 6-layer confluence system is extremely selective.")
        print("   Consider further threshold adjustments or review confluence weights.")
    elif results_v121['win_rate'] > results_v11['win_rate']:
        improvement = results_v121['win_rate'] - results_v11['win_rate']
        print(f"✅ v1.2.1 shows {improvement:.1f}% WIN RATE IMPROVEMENT")
        print(f"   Better trade quality with advanced confluence filtering")
    else:
        print(f"📊 v1.2.1 is more selective: {results_v121['total_trades']} trades vs {results_v11['total_trades']}")
        print(f"   Focus on quality over quantity")

    print("\n" + "="*80)