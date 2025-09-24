#!/usr/bin/env python3
"""
Bull Machine v1.4.2 - ETH Simple Backtest
Using actual working modules from the codebase
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime
import subprocess
import sys
import os

# Add Bull Machine to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from bull_machine.version import __version__, get_version_banner

# ETH data files with clear timeframe labels
ETH_DATA = {
    "1D": [
        ("/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 1D_fa116.csv", "Chart_Logs_1D"),
        ("/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv", "Downloads_1D")
    ],
    "4H (240min)": [
        ("/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 240_ab8a9.csv", "Chart_Logs_4H"),
        ("/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv", "Downloads_4H")
    ],
    "6H (360min)": [
        ("/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 360_3c856.csv", "Chart_Logs_6H")
    ],
    "12H (720min)": [
        ("/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 720_ffc2d.csv", "Chart_Logs_12H")
    ],
    "22H (1320min)": [
        ("/Users/raymondghandchi/Desktop/Chart Logs/COINBASE_ETHUSD, 1320_f91da.csv", "Chart_Logs_22H")
    ],
    "1H (60min)": [
        ("/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv", "Downloads_1H")
    ]
}

def analyze_eth_data(file_path: str) -> dict:
    """Analyze ETH data file and extract key statistics."""
    try:
        df = pd.read_csv(file_path)

        # Basic stats
        stats = {
            'bars': len(df),
            'has_ohlc': all(col in df.columns for col in ['open', 'high', 'low', 'close']),
            'has_volume': 'volume' in df.columns or 'BUY+SELL V' in df.columns,
            'price_min': df['low'].min() if 'low' in df.columns else 0,
            'price_max': df['high'].max() if 'high' in df.columns else 0,
            'price_current': df['close'].iloc[-1] if 'close' in df.columns else 0
        }

        # Date range
        if 'time' in df.columns:
            df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            stats['date_start'] = df['timestamp'].min()
            stats['date_end'] = df['timestamp'].max()
            stats['days_covered'] = (stats['date_end'] - stats['date_start']).days
        else:
            stats['date_start'] = 'Unknown'
            stats['date_end'] = 'Unknown'
            stats['days_covered'] = 0

        # Calculate volatility and trends
        if 'close' in df.columns:
            returns = df['close'].pct_change().dropna()
            stats['volatility'] = returns.std() * np.sqrt(252) * 100  # Annualized
            stats['total_return'] = ((df['close'].iloc[-1] - df['close'].iloc[0]) / df['close'].iloc[0]) * 100

            # Trend analysis
            sma_20 = df['close'].rolling(20).mean().iloc[-1]
            sma_50 = df['close'].rolling(50).mean().iloc[-1] if len(df) > 50 else sma_20
            current = df['close'].iloc[-1]

            if current > sma_20 > sma_50:
                stats['trend'] = 'Strong Uptrend'
            elif current > sma_20:
                stats['trend'] = 'Uptrend'
            elif current < sma_20 < sma_50:
                stats['trend'] = 'Strong Downtrend'
            elif current < sma_20:
                stats['trend'] = 'Downtrend'
            else:
                stats['trend'] = 'Sideways'

        return stats

    except Exception as e:
        return {'error': str(e)}

def simulate_simple_trades(file_path: str, timeframe: str) -> dict:
    """Run a simple trading simulation on the data."""
    try:
        df = pd.read_csv(file_path)

        # Ensure we have required columns
        if not all(col in df.columns for col in ['open', 'high', 'low', 'close']):
            return {'error': 'Missing OHLC data'}

        # Simple strategy parameters
        results = {
            'timeframe': timeframe,
            'trades': [],
            'total_return': 0,
            'win_rate': 0,
            'max_drawdown': 0
        }

        # Calculate indicators
        df['sma_20'] = df['close'].rolling(20).mean()
        df['sma_50'] = df['close'].rolling(50).mean()
        df['rsi'] = calculate_rsi(df['close'])
        df['atr'] = calculate_atr(df)

        # Simple trading logic
        balance = 10000
        peak_balance = balance
        position = None
        trades = []

        for i in range(50, len(df)):
            current = df.iloc[i]

            # Exit logic
            if position:
                # Check stop loss
                if position['type'] == 'long' and current['low'] <= position['stop']:
                    exit_price = position['stop']
                    pnl = (exit_price - position['entry']) / position['entry']
                    balance *= (1 + pnl)
                    position['exit'] = exit_price
                    position['pnl'] = pnl * 100
                    trades.append(position)
                    position = None

                # Check take profit (2:1 RR)
                elif position['type'] == 'long':
                    target = position['entry'] + 2 * (position['entry'] - position['stop'])
                    if current['high'] >= target:
                        exit_price = target
                        pnl = (exit_price - position['entry']) / position['entry']
                        balance *= (1 + pnl)
                        position['exit'] = exit_price
                        position['pnl'] = pnl * 100
                        trades.append(position)
                        position = None

            # Entry logic (only if no position)
            if not position and i > 0:
                prev = df.iloc[i-1]

                # Long entry conditions (simplified v1.4.2 logic)
                if (current['sma_20'] > current['sma_50'] and  # Uptrend
                    prev['sma_20'] <= prev['sma_50'] and  # Golden cross
                    current['rsi'] < 70):  # Not overbought

                    entry_price = current['close']
                    stop_price = entry_price - (current['atr'] * 1.5)  # v1.4.2 ATR stop

                    position = {
                        'type': 'long',
                        'entry': entry_price,
                        'stop': stop_price,
                        'entry_date': i,
                        'atr': current['atr']
                    }

            # Track drawdown
            if balance > peak_balance:
                peak_balance = balance
            else:
                drawdown = (peak_balance - balance) / peak_balance * 100
                results['max_drawdown'] = max(results['max_drawdown'], drawdown)

        # Close any open position
        if position:
            position['exit'] = df.iloc[-1]['close']
            position['pnl'] = ((position['exit'] - position['entry']) / position['entry']) * 100
            trades.append(position)

        # Calculate results
        results['trades'] = trades
        results['total_trades'] = len(trades)

        if trades:
            wins = [t for t in trades if t['pnl'] > 0]
            results['winning_trades'] = len(wins)
            results['losing_trades'] = len(trades) - len(wins)
            results['win_rate'] = (len(wins) / len(trades)) * 100
            results['avg_win'] = np.mean([t['pnl'] for t in wins]) if wins else 0
            results['avg_loss'] = np.mean([t['pnl'] for t in trades if t['pnl'] <= 0]) if results['losing_trades'] > 0 else 0

        results['final_balance'] = balance
        results['total_return'] = ((balance - 10000) / 10000) * 100

        return results

    except Exception as e:
        return {'error': str(e)}

def calculate_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """Calculate RSI."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate ATR."""
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    return true_range.rolling(period).mean()

def main():
    """Run comprehensive ETH analysis and backtest."""
    print("=" * 80)
    print(f"🚀 {get_version_banner()} - ETH Comprehensive Analysis & Backtest")
    print("=" * 80)
    print(f"📅 Analysis Date: {datetime.now()}")
    print("=" * 80)

    all_results = {}
    trade_summary = []

    # Process each timeframe
    for timeframe, files in ETH_DATA.items():
        print(f"\n{'='*70}")
        print(f"📊 Analyzing ETH {timeframe} Data")
        print(f"{'='*70}")

        for file_path, label in files:
            if not Path(file_path).exists():
                print(f"\n⚠️  File not found: {file_path}")
                continue

            print(f"\n📂 File: {Path(file_path).name}")
            print(f"📍 Label: {label}")

            # Analyze data
            stats = analyze_eth_data(file_path)

            if 'error' not in stats:
                print(f"\n📈 Data Statistics:")
                print(f"  Bars: {stats['bars']}")
                print(f"  Date Range: {stats['date_start']} to {stats['date_end']}")
                print(f"  Days Covered: {stats['days_covered']}")
                print(f"  Price Range: ${stats['price_min']:.2f} - ${stats['price_max']:.2f}")
                print(f"  Current Price: ${stats['price_current']:.2f}")
                print(f"  Total Return: {stats.get('total_return', 0):+.2f}%")
                print(f"  Annualized Vol: {stats.get('volatility', 0):.1f}%")
                print(f"  Current Trend: {stats.get('trend', 'Unknown')}")

                # Run simple backtest
                print(f"\n🔄 Running Trading Simulation...")
                backtest = simulate_simple_trades(file_path, timeframe)

                if 'error' not in backtest:
                    print(f"\n💰 Trading Results:")
                    print(f"  Total Trades: {backtest['total_trades']}")

                    if backtest['total_trades'] > 0:
                        print(f"  Winning Trades: {backtest.get('winning_trades', 0)}")
                        print(f"  Losing Trades: {backtest.get('losing_trades', 0)}")
                        print(f"  Win Rate: {backtest['win_rate']:.1f}%")
                        print(f"  Avg Win: {backtest.get('avg_win', 0):+.2f}%")
                        print(f"  Avg Loss: {backtest.get('avg_loss', 0):.2f}%")
                        print(f"  Max Drawdown: {backtest['max_drawdown']:.1f}%")
                        print(f"  Final Balance: ${backtest['final_balance']:,.2f}")
                        print(f"  Total Return: {backtest['total_return']:+.2f}%")

                        # Show sample trades
                        if backtest['trades']:
                            print(f"\n  Sample Trades (First 3):")
                            for i, trade in enumerate(backtest['trades'][:3], 1):
                                print(f"    {i}. LONG | Entry: ${trade['entry']:.2f} | "
                                      f"Exit: ${trade.get('exit', 0):.2f} | "
                                      f"PnL: {trade.get('pnl', 0):+.2f}%")

                        # Track for summary
                        trade_summary.append({
                            'timeframe': timeframe,
                            'label': label,
                            'trades': backtest['total_trades'],
                            'win_rate': backtest['win_rate'],
                            'return': backtest['total_return']
                        })
                    else:
                        print(f"  No trades generated with current criteria")
                else:
                    print(f"  Error in backtest: {backtest['error']}")

                # Store results
                key = f"{timeframe}_{label}"
                all_results[key] = {
                    'stats': stats,
                    'backtest': backtest
                }
            else:
                print(f"  Error analyzing data: {stats['error']}")

    # Overall Summary
    print(f"\n{'='*80}")
    print(f"📊 COMPREHENSIVE SUMMARY - ALL TIMEFRAMES")
    print(f"{'='*80}")

    if trade_summary:
        print(f"\n🎯 Trading Performance by Timeframe:")
        print(f"{'Timeframe':<20} {'Label':<20} {'Trades':<10} {'Win Rate':<12} {'Return':<12}")
        print("-" * 74)

        for result in trade_summary:
            print(f"{result['timeframe']:<20} {result['label']:<20} "
                  f"{result['trades']:<10} {result['win_rate']:<12.1f}% "
                  f"{result['return']:+12.2f}%")

        # Best performer
        if trade_summary:
            best = max(trade_summary, key=lambda x: x['return'])
            print(f"\n🏆 Best Performance:")
            print(f"  Timeframe: {best['timeframe']} - {best['label']}")
            print(f"  Return: {best['return']:+.2f}%")
            print(f"  Win Rate: {best['win_rate']:.1f}%")
            print(f"  Total Trades: {best['trades']}")

    print(f"\n📝 Analysis Notes:")
    print(f"  • This analysis uses simplified trading logic for demonstration")
    print(f"  • Full v1.4.2 engine includes:")
    print(f"    - 7-layer confluence system")
    print(f"    - MTF synchronization")
    print(f"    - Advanced exit rules")
    print(f"    - Phase-aware position sizing")
    print(f"    - Quality floor enforcement")

    # Save results
    output_file = f"eth_analysis_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'version': __version__,
            'timestamp': datetime.now().isoformat(),
            'results': all_results,
            'summary': trade_summary
        }, f, indent=2, default=str)

    print(f"\n💾 Detailed results saved to: {output_file}")

    print(f"\n{'='*80}")
    print(f"✅ ETH Analysis & Backtest Complete!")
    print(f"   {get_version_banner()}")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()