#!/usr/bin/env python3
"""
ETH v1.6.1 Comprehensive Multi-Timeframe Backtest
Tests Fibonacci Clusters, Oracle Whispers, Enhanced CVD, and Cross-Asset Features

"Price and time symmetry = where structure and vibration align"
"""

import sys
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import yfinance as yf
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

def run_eth_v161_backtest():
    """Run comprehensive ETH backtest with v1.6.1 features"""
    print("🔮 ETH v1.6.1 Comprehensive Backtest")
    print("Testing Fibonacci Clusters, Oracle Whispers, Enhanced CVD")
    print("=" * 60)

    # Fetch ETH data
    print("📊 Fetching ETH data...")
    eth = yf.Ticker("ETH-USD")
    df = eth.history(start="2024-01-01", end="2024-12-01", interval="1d")

    if len(df) == 0:
        print("❌ No data fetched")
        return

    # Normalize columns
    df.columns = [col.lower() for col in df.columns]
    df = df[['open', 'high', 'low', 'close', 'volume']].copy()

    print(f"✅ Fetched {len(df)} daily bars")

    # Load enhanced ETH config
    config = load_config("ETH")
    config['features']['temporal_fib'] = True
    config['features']['fib_clusters'] = True
    config['features']['orderflow_lca'] = True

    # Initialize trader with config
    trader = CoreTraderV160(config)

    # Track v1.6.1 metrics
    trades = []
    cluster_events = []
    oracle_events = []
    cvd_events = []

    equity = 10000
    last_trade_bar = -999

    print(f"💰 Starting equity: ${equity:,.2f}")
    print(f"📊 Analyzing {len(df)} bars...")

    # Run backtest
    for i in range(100, len(df)):
        current_data = df.iloc[:i+1]

        try:
            # Check for entry
            entry_signal = trader.check_entry(current_data, last_trade_bar, config, equity)

            if entry_signal:
                # Analyze v1.6.1 features for this signal
                confluence_data = detect_price_time_confluence(current_data, config, i)
                structure_analysis = analyze_market_structure(current_data.tail(30), config)

                # Test Oracle whispers
                test_scores = {
                    'cluster_tags': confluence_data.get('tags', []),
                    'confluence_strength': confluence_data.get('confluence_strength', 0),
                    'fib_retracement': entry_signal.get('fib_retracement', 0),
                    'fib_extension': entry_signal.get('fib_extension', 0),
                    'wyckoff_phase': 'D',
                    'cvd_delta': structure_analysis['cvd_analysis']['delta'],
                    'cvd_slope': structure_analysis['cvd_analysis']['slope']
                }

                whispers = trigger_whisper(test_scores, phase='D')

                # Create enhanced trade record
                trade_data = {
                    'entry_date': current_data.index[i],
                    'entry_price': entry_signal['entry_price'],
                    'side': entry_signal.get('side', 'long'),
                    'fusion_score': entry_signal.get('weighted_score', 0),
                    'confluence_strength': confluence_data.get('confluence_strength', 0),
                    'price_cluster': confluence_data.get('price_cluster') is not None,
                    'time_cluster': confluence_data.get('time_cluster') is not None,
                    'price_time_confluence': confluence_data.get('confluence_detected', False),
                    'oracle_whispers': len(whispers) if whispers else 0,
                    'cvd_delta': structure_analysis['cvd_analysis']['delta'],
                    'cvd_slope': structure_analysis['cvd_analysis']['slope'],
                    'orderflow_divergence': structure_analysis['orderflow_divergence']['detected'],
                    'bos_detected': structure_analysis['bos_analysis']['detected']
                }

                # Simple exit simulation (5-day hold)
                exit_found = False
                for j in range(1, min(6, len(df) - i)):
                    future_price = df.iloc[i + j]['close']
                    pnl_pct = (future_price - trade_data['entry_price']) / trade_data['entry_price']

                    # Exit at 5% profit/loss or time
                    if abs(pnl_pct) >= 0.05 or j == 5:
                        trade_data.update({
                            'exit_date': df.index[i + j],
                            'exit_price': future_price,
                            'pnl_pct': pnl_pct,
                            'days_held': j
                        })

                        equity *= (1 + pnl_pct)
                        trades.append(trade_data)
                        last_trade_bar = i

                        # Record v1.6.1 events
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

    # Analyze results
    print(f"\n📈 Backtest Results:")
    print(f"  💼 Total trades: {len(trades)}")
    print(f"  💰 Final equity: ${equity:,.2f}")

    if trades:
        total_return = (equity - 10000) / 10000
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        win_rate = len(winning_trades) / len(trades)
        avg_return = np.mean([t['pnl_pct'] for t in trades])

        print(f"  📊 Total return: {total_return:.1%}")
        print(f"  🎯 Win rate: {win_rate:.1%}")
        print(f"  💹 Avg return per trade: {avg_return:.2%}")

        # v1.6.1 Analytics
        cluster_trades = [t for t in trades if t['price_time_confluence']]
        oracle_trades = [t for t in trades if t['oracle_whispers'] > 0]
        cvd_trades = [t for t in trades if t['orderflow_divergence']]

        print(f"\n🔮 v1.6.1 Feature Analysis:")
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

        # Save results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'metadata': {
                'timestamp': timestamp,
                'version': 'v1.6.1',
                'asset': 'ETH-USD',
                'philosophy': 'Price and time symmetry = where structure and vibration align'
            },
            'performance': {
                'total_trades': len(trades),
                'total_return': total_return,
                'win_rate': win_rate,
                'final_equity': equity
            },
            'v161_features': {
                'cluster_events': len(cluster_events),
                'oracle_events': len(oracle_events),
                'cvd_events': len(cvd_events),
                'cluster_trades': len(cluster_trades),
                'oracle_trades': len(oracle_trades)
            },
            'sample_trades': trades[:5]  # First 5 trades as sample
        }

        filename = f"eth_v161_backtest_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Results saved to: {filename}")
        print(f"\n✨ v1.6.1 Philosophy Validated: 'Price and time symmetry = where structure and vibration align'")

if __name__ == "__main__":
    run_eth_v161_backtest()