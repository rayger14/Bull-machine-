#!/usr/bin/env python3
"""
Simple ETH Chart Logs 2 Test
Direct PnL simulation with v1.6.1 features using simple moving average crossover
"""

import sys
import pandas as pd
import numpy as np
import json
from datetime import datetime

# Add bull_machine to path
sys.path.append('.')

from bull_machine.modules.orderflow.lca import analyze_market_structure, orderflow_lca
from bull_machine.strategy.hidden_fibs import detect_price_time_confluence
from bull_machine.oracle import trigger_whisper

def load_eth_data():
    """Load ETH 4H data from Chart Logs 2"""
    filepath = '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv'

    try:
        df = pd.read_csv(filepath)
        df.columns = [col.lower().strip() for col in df.columns]

        # Fix timestamp and set index
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)

        # Handle volume
        if 'buy+sell v' in df.columns:
            df['volume'] = df['buy+sell v']
        else:
            df['volume'] = df['close'] * 1000  # Fallback

        df = df[['open', 'high', 'low', 'close', 'volume']].copy()
        df = df.dropna()

        return df

    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def simple_trading_strategy():
    """Simple trading strategy with v1.6.1 enhancements"""

    print("🔮 ETH v1.6.1 Chart Logs 2 Simple Test")
    print("=" * 50)

    df = load_eth_data()
    if df is None or len(df) < 100:
        print("❌ Failed to load sufficient data")
        return

    print(f"✅ Loaded {len(df)} bars")
    print(f"   Period: {df.index[0]} to {df.index[-1]}")

    # Simple moving averages for entry signals
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_cross'] = df['sma_20'] > df['sma_50']
    df['sma_cross_signal'] = df['sma_cross'].astype(int).diff()

    # v1.6.1 Configuration
    config = {
        'features': {
            'temporal_fib': True,
            'fib_clusters': True,
            'orderflow_lca': True
        }
    }

    # Trading simulation
    equity = 10000.0
    position = 0
    entry_price = 0
    trades = []

    print(f"💰 Starting equity: ${equity:,.2f}")
    print("🔍 Finding trades...")

    for i in range(100, len(df)):
        current_bar = df.iloc[i]
        current_data = df.iloc[:i+1]

        # Entry signal: SMA crossover + v1.6.1 confirmation
        if position == 0 and df.iloc[i]['sma_cross_signal'] == 1:

            try:
                # v1.6.1 Enhanced Analysis
                confluence_data = detect_price_time_confluence(current_data, config, i)
                structure_analysis = analyze_market_structure(current_data.tail(30), config)
                orderflow_score = orderflow_lca(current_data.tail(30), config)

                # Oracle whisper test
                test_scores = {
                    'cluster_tags': confluence_data.get('tags', []),
                    'confluence_strength': confluence_data.get('confluence_strength', 0),
                    'fib_retracement': 0.45,
                    'wyckoff_phase': 'D',
                    'cvd_delta': structure_analysis['cvd_analysis']['delta'],
                    'cvd_slope': structure_analysis['cvd_analysis']['slope']
                }

                whispers = trigger_whisper(test_scores, phase='D')

                # Enhanced entry criteria
                confluence_strength = confluence_data.get('confluence_strength', 0)
                has_confluence = confluence_data.get('confluence_detected', False)

                # Entry if we have some confluence or strong orderflow
                if confluence_strength > 0.3 or orderflow_score > 0.6 or has_confluence:

                    entry_price = current_bar['close']
                    position_size = (equity * 0.02) / entry_price  # 2% risk
                    position = position_size

                    print(f"  📈 ENTRY: {current_bar.name.strftime('%Y-%m-%d')} @ ${entry_price:.2f}")
                    print(f"       Confluence: {confluence_strength:.3f}, Orderflow: {orderflow_score:.3f}")
                    print(f"       Oracle whispers: {len(whispers) if whispers else 0}")

            except Exception as e:
                continue

        # Exit signal: SMA cross down or time stop
        elif position > 0:

            # Exit conditions
            exit_trigger = False
            exit_reason = ""

            # SMA cross down
            if df.iloc[i]['sma_cross_signal'] == -1:
                exit_trigger = True
                exit_reason = "sma_cross_down"

            # Stop loss (5%)
            elif current_bar['close'] < entry_price * 0.95:
                exit_trigger = True
                exit_reason = "stop_loss"

            # Take profit (10%)
            elif current_bar['close'] > entry_price * 1.10:
                exit_trigger = True
                exit_reason = "take_profit"

            # Time stop (20 bars)
            elif len([t for t in trades if 'exit_date' not in t]) > 0:
                entry_date = [t for t in trades if 'exit_date' not in t][-1]['entry_date']
                if (current_bar.name - entry_date).days > 20:
                    exit_trigger = True
                    exit_reason = "time_stop"

            if exit_trigger:
                exit_price = current_bar['close']
                pnl_pct = (exit_price - entry_price) / entry_price
                pnl_dollars = position * (exit_price - entry_price)

                equity += pnl_dollars

                trade_record = {
                    'entry_date': df.index[i-1],  # Approximate
                    'exit_date': current_bar.name,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'pnl_dollars': pnl_dollars,
                    'exit_reason': exit_reason,
                    'position_size': position
                }

                trades.append(trade_record)

                print(f"  📉 EXIT:  {current_bar.name.strftime('%Y-%m-%d')} @ ${exit_price:.2f}")
                print(f"       PnL: {pnl_pct:.2%} (${pnl_dollars:.2f}) - {exit_reason}")
                print(f"       Equity: ${equity:.2f}")

                position = 0
                entry_price = 0

    # Results Analysis
    print(f"\n📊 Final Results:")
    print(f"  💰 Final equity: ${equity:,.2f}")
    print(f"  💼 Total trades: {len(trades)}")

    if trades:
        total_return = (equity - 10000) / 10000
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        win_rate = len(winning_trades) / len(trades)
        avg_return = np.mean([t['pnl_pct'] for t in trades])

        print(f"  📈 Total return: {total_return:.1%}")
        print(f"  🎯 Win rate: {win_rate:.1%} ({len(winning_trades)}/{len(trades)})")
        print(f"  💹 Avg return per trade: {avg_return:.2%}")

        if winning_trades:
            avg_win = np.mean([t['pnl_pct'] for t in winning_trades])
            print(f"  🟢 Avg winning trade: {avg_win:.2%}")

        losing_trades = [t for t in trades if t['pnl_pct'] < 0]
        if losing_trades:
            avg_loss = np.mean([t['pnl_pct'] for t in losing_trades])
            print(f"  🔴 Avg losing trade: {avg_loss:.2%}")

        # Exit analysis
        exit_reasons = {}
        for trade in trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        print(f"\n🚪 Exit Reasons:")
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count} trades")

        # Save results
        results = {
            'metadata': {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'strategy': 'Simple SMA + v1.6.1',
                'asset': 'ETH-USD',
                'data_source': 'Chart Logs 2'
            },
            'performance': {
                'starting_equity': 10000,
                'final_equity': equity,
                'total_return': total_return,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'avg_return': avg_return
            },
            'trades': trades
        }

        filename = f"eth_simple_chart_logs2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Results saved to: {filename}")

    else:
        print("  ⚠️ No trades executed")

if __name__ == "__main__":
    simple_trading_strategy()