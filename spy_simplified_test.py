#!/usr/bin/env python3
"""
Simplified SPY Orderflow Test - Focus on CVD and BOS Detection
Tests basic orderflow functionality with relaxed criteria to generate actual trades
"""

import sys
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf

# Add bull_machine to path
sys.path.append('.')

from bull_machine.modules.orderflow.lca import orderflow_lca, analyze_market_structure, detect_bos, calculate_intent_nudge
from bull_machine.core.config_loader import load_config

warnings.filterwarnings('ignore')

def fetch_spy_daily_data(start_date="2023-01-01", end_date="2024-09-01"):
    """Fetch SPY daily data"""
    print(f"📊 Fetching SPY daily data from {start_date} to {end_date}...")

    spy = yf.Ticker("SPY")
    data = spy.history(start=start_date, end=end_date, interval="1d")

    # Normalize columns
    data.columns = [col.lower() for col in data.columns]
    data = data[['open', 'high', 'low', 'close', 'volume']].copy()

    print(f"✅ Fetched {len(data)} daily bars")
    return data

def test_orderflow_components(data, config):
    """Test individual orderflow components"""
    print("\n🔧 Testing Orderflow Components...")

    # Test on recent 100 bars
    test_data = data.tail(100).copy()

    print(f"📊 Testing on last {len(test_data)} bars")

    # Test CVD calculation
    try:
        from bull_machine.modules.orderflow.lca import calculate_cvd
        cvd = calculate_cvd(test_data)
        print(f"  ✅ CVD calculation: {cvd:.2f}")
    except Exception as e:
        print(f"  ❌ CVD error: {e}")
        return False

    # Test BOS detection
    try:
        bos_result = detect_bos(test_data)
        print(f"  ✅ BOS detected: {bos_result['detected']}, Direction: {bos_result['direction']}")
    except Exception as e:
        print(f"  ❌ BOS error: {e}")
        return False

    # Test intent nudge
    try:
        intent_result = calculate_intent_nudge(test_data)
        print(f"  ✅ Intent conviction: {intent_result['conviction']}, Score: {intent_result['nudge_score']:.2f}")
    except Exception as e:
        print(f"  ❌ Intent error: {e}")
        return False

    # Test full orderflow system
    try:
        orderflow_score = orderflow_lca(test_data, config)
        print(f"  ✅ Orderflow LCA score: {orderflow_score:.2f}")
    except Exception as e:
        print(f"  ❌ Orderflow LCA error: {e}")
        return False

    return True

def generate_simple_signals(data, config):
    """Generate signals with relaxed criteria"""
    print("\n📈 Generating Trading Signals...")

    signals = []
    lookback = 50

    for i in range(lookback, len(data) - 5):
        window = data.iloc[max(0, i-lookback):i+1].copy()

        try:
            # Get orderflow analysis
            orderflow_score = orderflow_lca(window, config)
            market_structure = analyze_market_structure(window, config)

            current_close = window['close'].iloc[-1]
            current_open = window['open'].iloc[-1]

            # Simple trend filter
            sma_20 = window['close'].rolling(20).mean().iloc[-1]
            trend_bullish = current_close > sma_20
            trend_bearish = current_close < sma_20

            # Relaxed signal criteria
            bos_detected = market_structure['bos_analysis']['detected']
            intent_conviction = market_structure['intent_analysis']['conviction']

            # Long signals (relaxed criteria)
            if (orderflow_score > 0.6 and  # Lower threshold
                bos_detected and
                market_structure['bos_analysis']['direction'] == 'bullish' and
                trend_bullish):

                signals.append({
                    'date': window.index[-1],
                    'side': 'long',
                    'price': current_close,
                    'orderflow_score': orderflow_score,
                    'bos_strength': market_structure['bos_analysis']['strength'],
                    'intent_conviction': intent_conviction,
                    'reason': 'bullish_bos_orderflow'
                })

            # Short signals (relaxed criteria)
            elif (orderflow_score < 0.4 and  # Higher threshold for shorts
                  bos_detected and
                  market_structure['bos_analysis']['direction'] == 'bearish' and
                  trend_bearish):

                signals.append({
                    'date': window.index[-1],
                    'side': 'short',
                    'price': current_close,
                    'orderflow_score': orderflow_score,
                    'bos_strength': market_structure['bos_analysis']['strength'],
                    'intent_conviction': intent_conviction,
                    'reason': 'bearish_bos_orderflow'
                })

        except Exception as e:
            # Skip problematic bars
            continue

    print(f"  🎯 Generated {len(signals)} signals")
    return signals

def simulate_simple_trades(signals, data):
    """Simulate trades with simple exit logic"""
    print("\n💰 Simulating Trades...")

    trades = []

    for signal in signals:
        signal_date = signal['date']
        entry_price = signal['price']
        side = signal['side']

        # Find entry bar index
        try:
            entry_idx = data.index.get_loc(signal_date)
        except:
            continue

        # Look ahead 10 days for exit
        exit_found = False
        for j in range(1, min(11, len(data) - entry_idx)):
            future_bar = data.iloc[entry_idx + j]

            if side == 'long':
                # Simple profit target (2% gain) or stop loss (1% loss)
                if future_bar['high'] >= entry_price * 1.02:  # 2% profit
                    exit_price = entry_price * 1.02
                    exit_reason = 'profit_target'
                    exit_found = True
                elif future_bar['low'] <= entry_price * 0.99:  # 1% loss
                    exit_price = entry_price * 0.99
                    exit_reason = 'stop_loss'
                    exit_found = True

            else:  # short
                # Simple profit target (2% gain) or stop loss (1% loss)
                if future_bar['low'] <= entry_price * 0.98:  # 2% profit on short
                    exit_price = entry_price * 0.98
                    exit_reason = 'profit_target'
                    exit_found = True
                elif future_bar['high'] >= entry_price * 1.01:  # 1% loss on short
                    exit_price = entry_price * 1.01
                    exit_reason = 'stop_loss'
                    exit_found = True

            if exit_found:
                # Calculate PnL
                if side == 'long':
                    pnl = (exit_price - entry_price) / entry_price
                    r_multiple = pnl / 0.01  # Risk was 1%
                else:
                    pnl = (entry_price - exit_price) / entry_price
                    r_multiple = pnl / 0.01  # Risk was 1%

                trades.append({
                    'entry_date': signal_date,
                    'exit_date': data.index[entry_idx + j],
                    'side': side,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl,
                    'r_multiple': r_multiple,
                    'days_held': j,
                    'exit_reason': exit_reason,
                    'orderflow_score': signal['orderflow_score'],
                    'bos_strength': signal['bos_strength'],
                    'intent_conviction': signal['intent_conviction']
                })
                break

        # If no exit found in 10 days, exit at market
        if not exit_found and entry_idx + 10 < len(data):
            exit_price = data.iloc[entry_idx + 10]['close']
            if side == 'long':
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price
            r_multiple = pnl / 0.01

            trades.append({
                'entry_date': signal_date,
                'exit_date': data.index[entry_idx + 10],
                'side': side,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl_pct': pnl,
                'r_multiple': r_multiple,
                'days_held': 10,
                'exit_reason': 'time_exit',
                'orderflow_score': signal['orderflow_score'],
                'bos_strength': signal['bos_strength'],
                'intent_conviction': signal['intent_conviction']
            })

    print(f"  💼 Executed {len(trades)} trades")
    return trades

def analyze_results(trades):
    """Analyze trading results"""
    print("\n📊 Performance Analysis...")

    if not trades:
        print("  ❌ No trades to analyze")
        return

    # Basic metrics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['pnl_pct'] > 0]
    losing_trades = [t for t in trades if t['pnl_pct'] <= 0]

    win_rate = len(winning_trades) / total_trades
    avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0

    total_return = sum(t['pnl_pct'] for t in trades)
    avg_return_per_trade = total_return / total_trades

    # R multiples
    avg_r = np.mean([t['r_multiple'] for t in trades])
    total_r = sum(t['r_multiple'] for t in trades)

    # Print results
    print(f"  🎯 Total Trades: {total_trades}")
    print(f"  📈 Win Rate: {win_rate:.1%}")
    print(f"  💰 Total Return: {total_return:.1%}")
    print(f"  📊 Avg Return/Trade: {avg_return_per_trade:.2%}")
    print(f"  🏆 Average R: {avg_r:.2f}")
    print(f"  🚀 Total R: {total_r:.2f}")
    print(f"  ✅ Avg Win: {avg_win:.2%}")
    print(f"  ❌ Avg Loss: {avg_loss:.2%}")

    # Orderflow insights
    long_trades = [t for t in trades if t['side'] == 'long']
    short_trades = [t for t in trades if t['side'] == 'short']

    if long_trades:
        long_win_rate = len([t for t in long_trades if t['pnl_pct'] > 0]) / len(long_trades)
        print(f"  📈 Long Win Rate: {long_win_rate:.1%} ({len(long_trades)} trades)")

    if short_trades:
        short_win_rate = len([t for t in short_trades if t['pnl_pct'] > 0]) / len(short_trades)
        print(f"  📉 Short Win Rate: {short_win_rate:.1%} ({len(short_trades)} trades)")

    # BOS strength analysis
    high_bos_trades = [t for t in trades if t['bos_strength'] > 0.05]
    if high_bos_trades:
        high_bos_win_rate = len([t for t in high_bos_trades if t['pnl_pct'] > 0]) / len(high_bos_trades)
        print(f"  💪 High BOS Strength Win Rate: {high_bos_win_rate:.1%} ({len(high_bos_trades)} trades)")

    # Intent conviction analysis
    high_conviction_trades = [t for t in trades if t['intent_conviction'] in ['high', 'very_high']]
    if high_conviction_trades:
        high_conviction_win_rate = len([t for t in high_conviction_trades if t['pnl_pct'] > 0]) / len(high_conviction_trades)
        print(f"  🎯 High Conviction Win Rate: {high_conviction_win_rate:.1%} ({len(high_conviction_trades)} trades)")

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_return': total_return,
        'avg_r': avg_r,
        'total_r': total_r
    }

def main():
    """Main execution"""
    print("🚀 SPY Simplified Orderflow Test")
    print("Testing CVD, BOS Detection, and Intent Analysis")
    print("=" * 50)

    # Fetch data
    data = fetch_spy_daily_data()

    # Load config with orderflow features
    config = load_config()
    config['features']['orderflow_lca'] = True
    config['features']['negative_vip'] = True

    # Test components
    if not test_orderflow_components(data, config):
        print("❌ Component test failed")
        return

    print("✅ All orderflow components working")

    # Generate signals
    signals = generate_simple_signals(data, config)

    if not signals:
        print("❌ No signals generated")
        return

    # Show sample signals
    print(f"\n📋 Sample Signals (showing first 5 of {len(signals)}):")
    for i, signal in enumerate(signals[:5]):
        print(f"  {i+1}. {signal['date'].strftime('%Y-%m-%d')} | {signal['side']:<5} | "
              f"${signal['price']:.2f} | Score: {signal['orderflow_score']:.2f} | "
              f"BOS: {signal['bos_strength']:.3f} | {signal['reason']}")

    # Simulate trades
    trades = simulate_simple_trades(signals, data)

    if not trades:
        print("❌ No trades executed")
        return

    # Show sample trades
    print(f"\n💼 Sample Trades (showing first 5 of {len(trades)}):")
    for i, trade in enumerate(trades[:5]):
        print(f"  {i+1}. {trade['entry_date'].strftime('%Y-%m-%d')} | {trade['side']:<5} | "
              f"${trade['entry_price']:.2f} → ${trade['exit_price']:.2f} | "
              f"R: {trade['r_multiple']:.2f} | {trade['exit_reason']}")

    # Analyze results
    results = analyze_results(trades)

    # Save detailed results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"spy_orderflow_test_{timestamp}.csv"

    trade_df = pd.DataFrame(trades)
    trade_df.to_csv(filename, index=False)

    print(f"\n📄 Detailed results saved to: {filename}")

    return results

if __name__ == "__main__":
    results = main()