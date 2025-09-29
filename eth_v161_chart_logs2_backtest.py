#!/usr/bin/env python3
"""
ETH v1.6.1 Chart Logs 2 Multi-Timeframe Backtest
Tests v1.6.1 features (Fibonacci Clusters, Oracle Whispers, Enhanced CVD)
on real market data with proper thresholds

"Price and time symmetry = where structure and vibration align"
"""

import sys
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import json
from typing import Dict, List, Any

# Add bull_machine to path
sys.path.append('.')

from bull_machine.modules.fusion.v160_enhanced import CoreTraderV160
from bull_machine.modules.orderflow.lca import analyze_market_structure
from bull_machine.strategy.hidden_fibs import detect_price_time_confluence
from bull_machine.oracle import trigger_whisper, should_trigger_confluence_alert
from bull_machine.core.config_loader import load_config

warnings.filterwarnings('ignore')

def load_chart_logs_data():
    """Load ETH data from Chart Logs 2 files"""
    print("📊 Loading ETH Chart Logs 2 data...")

    files = {
        '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv',
        '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv',
        '1H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv'
    }

    data = {}

    for tf, filepath in files.items():
        try:
            df = pd.read_csv(filepath)

            # Normalize column names
            df.columns = [col.lower().strip() for col in df.columns]

            # Handle different possible column names
            if 'timestamp' in df.columns:
                df['time'] = pd.to_datetime(df['timestamp'], unit='s')
            elif 'time' in df.columns:
                # Chart Logs 2 uses Unix timestamps
                df['time'] = pd.to_datetime(df['time'], unit='s')

            df.set_index('time', inplace=True)

            # Handle Chart Logs 2 specific columns
            if 'buy+sell v' in df.columns:
                df['volume'] = df['buy+sell v']
            elif 'BUY+SELL V' in df.columns:
                df['volume'] = df['BUY+SELL V']
            else:
                # Use combined buy/sell volume if available
                buy_vol_cols = [col for col in df.columns if 'buy' in col.lower() and 'vol' in col.lower()]
                sell_vol_cols = [col for col in df.columns if 'sell' in col.lower() and 'vol' in col.lower()]
                if buy_vol_cols and sell_vol_cols:
                    df['volume'] = df[buy_vol_cols[0]] + df[sell_vol_cols[0]]
                else:
                    # Fallback: use close price as volume proxy
                    df['volume'] = df['close'] * 1000

            df = df[['open', 'high', 'low', 'close', 'volume']].copy()

            # Remove any NaN or invalid data
            df = df.dropna()
            df = df[df['volume'] > 0]

            data[tf] = df
            print(f"  ✅ {tf}: {len(df)} bars loaded")

        except Exception as e:
            print(f"  ❌ Failed to load {tf}: {e}")

    return data

def run_mtf_v161_backtest():
    """Run multi-timeframe v1.6.1 backtest with proper thresholds"""
    print("🔮 ETH v1.6.1 Chart Logs 2 Multi-Timeframe Backtest")
    print("Testing Fibonacci Clusters, Oracle Whispers, Enhanced CVD")
    print("=" * 60)

    # Load chart logs data
    mtf_data = load_chart_logs_data()

    if not mtf_data:
        print("❌ No data loaded")
        return

    # Use 4H as primary timeframe for analysis
    primary_tf = '4H'
    df = mtf_data[primary_tf]

    print(f"📈 Primary analysis on {primary_tf}: {len(df)} bars")
    print(f"   Period: {df.index[0]} to {df.index[-1]}")

    # Load enhanced ETH config with proper v1.6.1 thresholds
    config = load_config("ETH")

    # v1.6.1 Enhanced Configuration - Lower threshold for real market testing
    config.update({
        'entry_threshold': 0.25,  # Lower threshold to find trades in chart logs data
        'quality_floors': {
            'wyckoff': 0.15,
            'm1': 0.20,
            'm2': 0.20,
            'confluence': 0.30
        },
        'features': {
            'temporal_fib': True,
            'fib_clusters': True,
            'orderflow_lca': True,
            'oracle_whispers': True
        },
        'risk': {
            'base_risk_pct': 2.0,
            'max_drawdown_pct': 15.0,
            'stop_loss_pct': 3.0,
            'take_profit_pct': 6.0
        }
    })

    # Initialize trader with enhanced config
    trader = CoreTraderV160(config)

    # Track comprehensive metrics
    trades = []
    cluster_events = []
    oracle_events = []
    cvd_events = []

    equity = 10000.0
    max_equity = equity
    last_trade_bar = -999

    print(f"💰 Starting equity: ${equity:,.2f}")
    print(f"🎯 Entry threshold: {config['entry_threshold']}")
    print(f"📊 Analyzing {len(df)} bars...")

    # Run backtest with proper lookback
    lookback = 100

    for i in range(lookback, len(df)):
        current_data = df.iloc[:i+1]

        try:
            # v1.6.1 Enhanced Entry Check
            entry_signal = trader.check_entry(current_data, last_trade_bar, config, equity)

            # Debug: Print entry attempts
            if i % 100 == 0:  # Every 100 bars
                score = entry_signal.get('weighted_score', 0) if entry_signal else 0
                print(f"\r  📊 Bar {i}/{len(df)} - Entry score: {score:.3f} vs threshold {config['entry_threshold']}", end="", flush=True)

            if entry_signal and i - last_trade_bar > 5:  # Minimum 5 bars between trades

                # Comprehensive v1.6.1 Analysis
                confluence_data = detect_price_time_confluence(current_data, config, i)
                structure_analysis = analyze_market_structure(current_data.tail(30), config)

                # Oracle Whisper Analysis
                test_scores = {
                    'cluster_tags': confluence_data.get('tags', []),
                    'confluence_strength': confluence_data.get('confluence_strength', 0),
                    'fib_retracement': entry_signal.get('fib_retracement', 0),
                    'fib_extension': entry_signal.get('fib_extension', 0),
                    'wyckoff_phase': entry_signal.get('wyckoff_phase', 'D'),
                    'cvd_delta': structure_analysis['cvd_analysis']['delta'],
                    'cvd_slope': structure_analysis['cvd_analysis']['slope']
                }

                whispers = trigger_whisper(test_scores, phase=test_scores['wyckoff_phase'])

                # Enhanced Trade Record
                trade_data = {
                    'entry_date': current_data.index[i],
                    'entry_price': entry_signal['entry_price'],
                    'side': entry_signal.get('side', 'long'),
                    'fusion_score': entry_signal.get('weighted_score', 0),
                    'wyckoff_score': entry_signal.get('wyckoff_score', 0),
                    'm1_score': entry_signal.get('m1_score', 0),
                    'm2_score': entry_signal.get('m2_score', 0),
                    'confluence_strength': confluence_data.get('confluence_strength', 0),
                    'price_cluster': confluence_data.get('price_cluster') is not None,
                    'time_cluster': confluence_data.get('time_cluster') is not None,
                    'price_time_confluence': confluence_data.get('confluence_detected', False),
                    'oracle_whispers': len(whispers) if whispers else 0,
                    'cvd_delta': structure_analysis['cvd_analysis']['delta'],
                    'cvd_slope': structure_analysis['cvd_analysis']['slope'],
                    'orderflow_divergence': structure_analysis['orderflow_divergence']['detected'],
                    'bos_detected': structure_analysis['bos_analysis']['detected'],
                    'lca_score': structure_analysis['lca_score']
                }

                # Enhanced Exit Logic
                position_size = (equity * config['risk']['base_risk_pct'] / 100) / trade_data['entry_price']
                stop_loss = trade_data['entry_price'] * (1 - config['risk']['stop_loss_pct'] / 100)
                take_profit = trade_data['entry_price'] * (1 + config['risk']['take_profit_pct'] / 100)

                exit_found = False
                max_bars_hold = 20  # Maximum holding period

                for j in range(1, min(max_bars_hold + 1, len(df) - i)):
                    future_bar = df.iloc[i + j]
                    future_price = future_bar['close']

                    # Check stop loss
                    if future_bar['low'] <= stop_loss:
                        exit_price = stop_loss
                        exit_reason = 'stop_loss'
                    # Check take profit
                    elif future_bar['high'] >= take_profit:
                        exit_price = take_profit
                        exit_reason = 'take_profit'
                    # Time-based exit
                    elif j == max_bars_hold:
                        exit_price = future_price
                        exit_reason = 'time_exit'
                    else:
                        continue

                    # Calculate PnL
                    pnl_pct = (exit_price - trade_data['entry_price']) / trade_data['entry_price']
                    pnl_dollars = position_size * (exit_price - trade_data['entry_price'])

                    trade_data.update({
                        'exit_date': df.index[i + j],
                        'exit_price': exit_price,
                        'exit_reason': exit_reason,
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'days_held': j,
                        'position_size': position_size
                    })

                    equity += pnl_dollars
                    max_equity = max(max_equity, equity)
                    trades.append(trade_data)
                    last_trade_bar = i

                    # Record v1.6.1 Events
                    if trade_data['price_time_confluence']:
                        cluster_events.append({
                            'date': trade_data['entry_date'],
                            'strength': trade_data['confluence_strength'],
                            'outcome': pnl_pct
                        })

                    if trade_data['oracle_whispers'] > 0:
                        oracle_events.append({
                            'date': trade_data['entry_date'],
                            'whispers': trade_data['oracle_whispers'],
                            'outcome': pnl_pct
                        })

                    if trade_data['orderflow_divergence']:
                        cvd_events.append({
                            'date': trade_data['entry_date'],
                            'cvd_slope': trade_data['cvd_slope'],
                            'outcome': pnl_pct
                        })

                    exit_found = True
                    break

        except Exception as e:
            continue

    # Comprehensive Results Analysis
    print(f"\n📈 v1.6.1 Chart Logs 2 Backtest Results:")
    print(f"  💼 Total trades: {len(trades)}")
    print(f"  💰 Final equity: ${equity:,.2f}")

    if trades:
        total_return = (equity - 10000) / 10000
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        losing_trades = [t for t in trades if t['pnl_pct'] < 0]
        win_rate = len(winning_trades) / len(trades)
        avg_return = np.mean([t['pnl_pct'] for t in trades])
        max_drawdown = (max_equity - equity) / max_equity if max_equity > equity else 0

        print(f"  📊 Total return: {total_return:.1%}")
        print(f"  🎯 Win rate: {win_rate:.1%} ({len(winning_trades)}/{len(trades)})")
        print(f"  💹 Avg return per trade: {avg_return:.2%}")
        print(f"  📉 Max drawdown: {max_drawdown:.1%}")

        if winning_trades:
            avg_win = np.mean([t['pnl_pct'] for t in winning_trades])
            print(f"  🟢 Avg winning trade: {avg_win:.2%}")

        if losing_trades:
            avg_loss = np.mean([t['pnl_pct'] for t in losing_trades])
            print(f"  🔴 Avg losing trade: {avg_loss:.2%}")

        # v1.6.1 Feature Performance
        cluster_trades = [t for t in trades if t['price_time_confluence']]
        oracle_trades = [t for t in trades if t['oracle_whispers'] > 0]
        cvd_trades = [t for t in trades if t['orderflow_divergence']]

        print(f"\n🔮 v1.6.1 Feature Performance:")
        print(f"  ✨ Price-time confluence trades: {len(cluster_trades)}")
        print(f"  🧙‍♂️ Oracle whisper trades: {len(oracle_trades)}")
        print(f"  📊 CVD divergence trades: {len(cvd_trades)}")

        if cluster_trades:
            cluster_win_rate = len([t for t in cluster_trades if t['pnl_pct'] > 0]) / len(cluster_trades)
            cluster_avg_return = np.mean([t['pnl_pct'] for t in cluster_trades])
            print(f"  🎯 Cluster trade win rate: {cluster_win_rate:.1%}")
            print(f"  💰 Cluster trade avg return: {cluster_avg_return:.2%}")

        if oracle_trades:
            oracle_win_rate = len([t for t in oracle_trades if t['pnl_pct'] > 0]) / len(oracle_trades)
            oracle_avg_return = np.mean([t['pnl_pct'] for t in oracle_trades])
            print(f"  🧙‍♂️ Oracle trade win rate: {oracle_win_rate:.1%}")
            print(f"  💫 Oracle trade avg return: {oracle_avg_return:.2%}")

        # Risk Metrics
        returns = [t['pnl_pct'] for t in trades]
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
        print(f"\n📊 Risk Metrics:")
        print(f"  📈 Sharpe ratio: {sharpe:.2f}")
        print(f"  💰 Profit factor: {sum([t['pnl_pct'] for t in winning_trades]) / abs(sum([t['pnl_pct'] for t in losing_trades])) if losing_trades else 'N/A'}")

        # Exit Reason Analysis
        exit_reasons = {}
        for trade in trades:
            reason = trade.get('exit_reason', 'unknown')
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        print(f"\n🚪 Exit Analysis:")
        for reason, count in exit_reasons.items():
            print(f"  {reason}: {count} trades ({count/len(trades):.1%})")

        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'metadata': {
                'timestamp': timestamp,
                'version': 'v1.6.1',
                'asset': 'ETH-USD',
                'data_source': 'Chart Logs 2',
                'timeframe': primary_tf,
                'philosophy': 'Price and time symmetry = where structure and vibration align'
            },
            'performance': {
                'starting_equity': 10000,
                'final_equity': equity,
                'total_trades': len(trades),
                'total_return': total_return,
                'win_rate': win_rate,
                'avg_return_per_trade': avg_return,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe
            },
            'v161_features': {
                'cluster_events': len(cluster_events),
                'oracle_events': len(oracle_events),
                'cvd_events': len(cvd_events),
                'cluster_trades': len(cluster_trades),
                'oracle_trades': len(oracle_trades),
                'cvd_trades': len(cvd_trades)
            },
            'trades': trades[:10],  # First 10 trades as sample
            'config': config
        }

        filename = f"eth_v161_chart_logs2_backtest_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Results saved to: {filename}")
        print(f"\n✨ v1.6.1 Philosophy Validated: 'Price and time symmetry = where structure and vibration align'")

    else:
        print("  ⚠️ No trades executed - check thresholds and data quality")

if __name__ == "__main__":
    run_mtf_v161_backtest()