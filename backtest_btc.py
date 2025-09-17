#!/usr/bin/env python3
"""
Backtest script for Bull Machine v1.1.2 on BTC daily data
"""
import sys
import os
import pandas as pd
from datetime import datetime
import logging

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bull_machine.app.main import run_bull_machine_v1_1
from bull_machine.io.feeders import load_csv_to_series

def setup_logging():
    logging.basicConfig(level=logging.WARNING, format='%(asctime)s - %(levelname)s - %(message)s')

def create_subset_csv(df, end_idx, filename):
    """Create a CSV subset up to end_idx for testing"""
    subset = df.iloc[:end_idx+1].copy()
    subset.to_csv(filename, index=False)
    return filename

def simulate_trade_outcome(entry_price, stop_price, target_prices, future_prices, side='long'):
    """Simulate trade outcome given future price data"""
    if side == 'long':
        # Check if stop hit first
        for price in future_prices:
            if price <= stop_price:
                return 'loss', stop_price, price
        # Check if any target hit
        for target in target_prices:
            for price in future_prices:
                if price >= target:
                    return 'win', target, price
        # No target hit within timeframe
        return 'timeout', future_prices[-1], future_prices[-1]
    else:  # short
        # Check if stop hit first
        for price in future_prices:
            if price >= stop_price:
                return 'loss', stop_price, price
        # Check if any target hit
        for target in target_prices:
            for price in future_prices:
                if price <= target:
                    return 'win', target, price
        return 'timeout', future_prices[-1], future_prices[-1]

def backtest_btc(csv_path, min_bars=100, lookforward_days=30, threshold=0.60):
    """Run backtest on BTC data"""
    print(f"Loading BTC data from: {csv_path}")

    # Load full dataset
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} bars of data")

    # Convert time column to proper format if needed
    if 'time' in df.columns:
        df = df.rename(columns={'time': 'timestamp'})

    trades = []
    temp_files = []

    # Start from min_bars and go to len-lookforward_days to ensure we have future data
    start_idx = min_bars
    end_idx = len(df) - lookforward_days

    print(f"Testing from bar {start_idx} to {end_idx} (testing {end_idx - start_idx + 1} points)")

    for i in range(start_idx, end_idx, 5):  # Test every 5 days to speed up
        # Create subset CSV up to current point
        temp_filename = f'temp_btc_{i}.csv'
        create_subset_csv(df, i, temp_filename)
        temp_files.append(temp_filename)

        try:
            # Run Bull Machine on this subset
            result = run_bull_machine_v1_1(
                temp_filename,
                account_balance=10000,
                override_signals={'confidence_threshold': threshold}
            )

            if result['action'] == 'enter_trade':
                current_bar = df.iloc[i]
                entry_price = current_bar['close']

                # Extract trade details (simplified)
                side = 'long'  # Assume long for now, could extract from logs
                stop_price = entry_price * 0.95 if side == 'long' else entry_price * 1.05
                target_price = entry_price * 1.10 if side == 'long' else entry_price * 0.90

                # Get future prices for simulation
                future_data = df.iloc[i+1:i+1+lookforward_days]
                if len(future_data) >= lookforward_days:
                    future_prices = future_data['high'].tolist() + future_data['low'].tolist()

                    # Simulate trade outcome
                    outcome, exit_price, actual_price = simulate_trade_outcome(
                        entry_price, stop_price, [target_price], future_prices, side
                    )

                    entry_date = datetime.utcfromtimestamp(int(current_bar['timestamp'])).strftime('%Y-%m-%d')

                    trade = {
                        'date': entry_date,
                        'bar_idx': i,
                        'side': side,
                        'entry_price': entry_price,
                        'stop_price': stop_price,
                        'target_price': target_price,
                        'exit_price': exit_price,
                        'outcome': outcome,
                        'pnl_pct': ((exit_price - entry_price) / entry_price * 100) if side == 'long' else ((entry_price - exit_price) / entry_price * 100)
                    }
                    trades.append(trade)

                    print(f"Trade #{len(trades)}: {entry_date} {side.upper()} @ {entry_price:.0f} -> {outcome.upper()} @ {exit_price:.0f} ({trade['pnl_pct']:.1f}%)")

        except Exception as e:
            print(f"Error at bar {i}: {e}")
            continue

    # Cleanup temp files
    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    return trades

def analyze_results(trades):
    """Analyze backtest results"""
    if not trades:
        return {"total_trades": 0, "win_rate": 0, "avg_pnl": 0}

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

    return {
        "total_trades": total_trades,
        "wins": wins,
        "losses": losses,
        "timeouts": timeouts,
        "win_rate": win_rate,
        "avg_pnl": avg_pnl,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "profit_factor": abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    }

if __name__ == "__main__":
    setup_logging()

    csv_path = '/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_BTCUSD, 1D_2f7fe.csv'

    print("=== Bull Machine v1.1.2 BTC Backtest ===")
    print("Testing with threshold=0.60 (lowered from default 0.72)")

    trades = backtest_btc(csv_path, threshold=0.60)
    results = analyze_results(trades)

    print("\n=== BACKTEST RESULTS ===")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Wins: {results['wins']}")
    print(f"Losses: {results['losses']}")
    print(f"Timeouts: {results['timeouts']}")
    print(f"Win Rate: {results['win_rate']:.1f}%")
    print(f"Average PnL: {results['avg_pnl']:.2f}%")
    print(f"Average Win: +{results['avg_win']:.2f}%")
    print(f"Average Loss: {results['avg_loss']:.2f}%")
    print(f"Profit Factor: {results['profit_factor']:.2f}")