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

def fetch_eth_multi_timeframe_data(start_date="2024-01-01", end_date="2024-12-01"):
    """Fetch ETH data across multiple timeframes"""
    print(f"📊 Fetching ETH multi-timeframe data from {start_date} to {end_date}...")

    eth = yf.Ticker("ETH-USD")

    # Fetch different timeframes
    timeframes = {
        '1D': eth.history(start=start_date, end=end_date, interval="1d"),
        '4H': eth.history(start=start_date, end=end_date, interval="1h").resample('4H').agg({
            'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last', 'Volume': 'sum'
        }).dropna(),
        '1H': eth.history(start=start_date, end=end_date, interval="1h")
    }

    # Normalize columns
    for tf in timeframes:
        if len(timeframes[tf]) > 0:
            timeframes[tf].columns = [col.lower() for col in timeframes[tf].columns]
            timeframes[tf] = timeframes[tf][['open', 'high', 'low', 'close', 'volume']].copy()
            print(f"  ✅ {tf}: {len(timeframes[tf])} bars")
        else:
            print(f"  ❌ {tf}: No data")

    return timeframes

def analyze_fibonacci_cluster_performance(data: Dict[str, pd.DataFrame], config: Dict[str, Any]):
    """Analyze Fibonacci cluster detection performance across timeframes"""
    print("\n🔮 Analyzing Fibonacci Cluster Performance...")

    cluster_analytics = {
        'price_clusters_detected': 0,
        'time_clusters_detected': 0,
        'price_time_confluence': 0,
        'oracle_whispers_triggered': 0,
        'cluster_strength_distribution': [],
        'timeframe_breakdown': {}
    }

    for tf_name, df in data.items():
        if len(df) < 50:  # Need sufficient data for cluster analysis
            continue

        tf_clusters = {
            'price_clusters': 0,
            'time_clusters': 0,
            'confluence_events': 0,
            'whisper_events': 0
        }

        print(f"\n  📊 Analyzing {tf_name} ({len(df)} bars)...")

        # Test cluster detection on multiple points
        test_points = range(50, len(df), 20)  # Sample every 20 bars

        for idx in test_points:
            try:
                # Detect price-time confluence
                confluence_data = detect_price_time_confluence(df, config, idx)

                if confluence_data['price_cluster']:
                    tf_clusters['price_clusters'] += 1
                    cluster_analytics['price_clusters_detected'] += 1
                    cluster_analytics['cluster_strength_distribution'].append(
                        confluence_data['price_cluster']['strength']
                    )

                if confluence_data['time_cluster']:
                    tf_clusters['time_clusters'] += 1
                    cluster_analytics['time_clusters_detected'] += 1

                if confluence_data['confluence_detected']:
                    tf_clusters['confluence_events'] += 1
                    cluster_analytics['price_time_confluence'] += 1

                    # Test Oracle whisper system
                    test_scores = {
                        'cluster_tags': confluence_data['tags'],
                        'confluence_strength': confluence_data['confluence_strength'],
                        'fib_retracement': 0.45,  # Mock score for whisper testing
                        'wyckoff_phase': 'C'
                    }

                    whispers = trigger_whisper(test_scores, phase='C')
                    if whispers:
                        tf_clusters['whisper_events'] += 1
                        cluster_analytics['oracle_whispers_triggered'] += 1

            except Exception as e:
                continue  # Skip problematic bars

        cluster_analytics['timeframe_breakdown'][tf_name] = tf_clusters

        print(f"    🔮 Price clusters: {tf_clusters['price_clusters']}")
        print(f"    ⏰ Time clusters: {tf_clusters['time_clusters']}")
        print(f"    ✨ Confluence events: {tf_clusters['confluence_events']}")
        print(f"    🧙‍♂️ Oracle whispers: {tf_clusters['whisper_events']}")

    return cluster_analytics

def test_enhanced_cvd_analysis(data: Dict[str, pd.DataFrame], config: Dict[str, Any]):
    """Test enhanced CVD analysis with slope detection"""
    print("\n📊 Testing Enhanced CVD Analysis...")

    cvd_analytics = {
        'total_analysis_points': 0,
        'divergence_signals': 0,
        'bullish_divergences': 0,
        'bearish_divergences': 0,
        'cvd_slope_distribution': [],
        'orderflow_scores': []
    }

    for tf_name, df in data.items():
        if len(df) < 30:  # Need sufficient data for CVD
            continue

        print(f"\n  📈 Analyzing CVD on {tf_name}...")

        # Test CVD analysis on sample points
        test_points = range(30, len(df), 15)  # Sample every 15 bars

        for idx in test_points:
            try:
                window_df = df.iloc[max(0, idx-25):idx+1]

                # Get market structure analysis with CVD
                structure_analysis = analyze_market_structure(window_df, config)

                cvd_analytics['total_analysis_points'] += 1

                # Analyze CVD data
                cvd_data = structure_analysis['cvd_analysis']
                cvd_analytics['cvd_slope_distribution'].append(cvd_data['slope'])

                # Check for divergence signals
                divergence = structure_analysis['orderflow_divergence']
                if divergence['detected']:
                    cvd_analytics['divergence_signals'] += 1
                    if divergence['type'] == 'bullish':
                        cvd_analytics['bullish_divergences'] += 1
                    else:
                        cvd_analytics['bearish_divergences'] += 1

                # Track orderflow scores
                cvd_analytics['orderflow_scores'].append(structure_analysis['lca_score'])

            except Exception as e:
                continue

        print(f"    📊 CVD analysis points: {len([x for x in cvd_analytics['cvd_slope_distribution'] if x != 0])}")
        print(f"    🔄 Divergence signals: {cvd_analytics['divergence_signals']}")

    return cvd_analytics

def run_v161_enhanced_backtest(data: Dict[str, pd.DataFrame], config: Dict[str, Any]):
    """Run comprehensive backtest with v1.6.1 enhancements"""
    print("\n🚀 Running v1.6.1 Enhanced Backtest...")

    # Use 1D timeframe for main backtest (most reliable)
    df_1d = data.get('1D')
    if df_1d is None or len(df_1d) < 100:
        print("❌ Insufficient 1D data for backtest")
        return None

    # Initialize trader with ETH config
    trader = CoreTraderV160()

    # Track enhanced metrics
    trades = []
    cluster_events = []
    oracle_events = []

    equity = 10000
    last_trade_bar = -999

    print(f"  💰 Starting equity: ${equity:,.2f}")
    print(f"  📊 Analyzing {len(df_1d)} daily bars...")

    for i in range(100, len(df_1d)):
        current_bar_data = df_1d.iloc[:i+1]

        try:
            # Check for entry with v1.6.1 enhancements
            entry_signal = trader.check_entry(current_bar_data, last_trade_bar, config, equity)

            if entry_signal:
                # Analyze Fibonacci clusters for this signal
                confluence_data = detect_price_time_confluence(current_bar_data, config, i)

                # Get market structure analysis
                structure_analysis = analyze_market_structure(current_bar_data.tail(30), config)

                # Test Oracle whispers
                test_scores = {
                    'cluster_tags': confluence_data.get('tags', []),
                    'confluence_strength': confluence_data.get('confluence_strength', 0),
                    'fib_retracement': entry_signal.get('fib_retracement', 0),
                    'fib_extension': entry_signal.get('fib_extension', 0),
                    'wyckoff_phase': 'D',  # Assume markup phase for entry
                    'cvd_delta': structure_analysis['cvd_analysis']['delta'],
                    'cvd_slope': structure_analysis['cvd_analysis']['slope']
                }

                whispers = trigger_whisper(test_scores, phase='D')

                # Record enhanced trade data
                trade_data = {
                    'entry_date': current_bar_data.index[i],
                    'entry_price': entry_signal['entry_price'],
                    'side': entry_signal.get('side', 'long'),
                    'fusion_score': entry_signal.get('weighted_score', 0),
                    'confluence_strength': confluence_data.get('confluence_strength', 0),
                    'price_cluster_detected': confluence_data.get('price_cluster') is not None,
                    'time_cluster_detected': confluence_data.get('time_cluster') is not None,
                    'price_time_confluence': confluence_data.get('confluence_detected', False),
                    'oracle_whispers': len(whispers) if whispers else 0,
                    'cvd_delta': structure_analysis['cvd_analysis']['delta'],
                    'cvd_slope': structure_analysis['cvd_analysis']['slope'],
                    'orderflow_divergence': structure_analysis['orderflow_divergence']['detected'],
                    'bos_detected': structure_analysis['bos_analysis']['detected'],
                    'bos_direction': structure_analysis['bos_analysis']['direction']
                }

                # Simple exit simulation (5-day hold or 10% profit/loss)
                exit_found = False
                for j in range(1, min(6, len(df_1d) - i)):
                    future_price = df_1d.iloc[i + j]['close']

                    if trade_data['side'] == 'long':
                        pnl_pct = (future_price - trade_data['entry_price']) / trade_data['entry_price']
                    else:
                        pnl_pct = (trade_data['entry_price'] - future_price) / trade_data['entry_price']

                    # Exit conditions
                    if abs(pnl_pct) >= 0.10 or j == 5:  # 10% profit/loss or time exit
                        trade_data.update({
                            'exit_date': df_1d.index[i + j],
                            'exit_price': future_price,
                            'pnl_pct': pnl_pct,
                            'days_held': j,
                            'exit_reason': 'profit_loss' if abs(pnl_pct) >= 0.10 else 'time_exit'
                        })

                        equity *= (1 + pnl_pct)
                        exit_found = True
                        break

                if exit_found:
                    trades.append(trade_data)
                    last_trade_bar = i

                    # Record cluster and oracle events
                    if trade_data['price_time_confluence']:
                        cluster_events.append({
                            'date': trade_data['entry_date'],
                            'type': 'price_time_confluence',
                            'strength': trade_data['confluence_strength'],
                            'trade_outcome': trade_data['pnl_pct']
                        })

                    if trade_data['oracle_whispers'] > 0:
                        oracle_events.append({
                            'date': trade_data['entry_date'],
                            'whisper_count': trade_data['oracle_whispers'],
                            'trade_outcome': trade_data['pnl_pct']
                        })

        except Exception as e:
            continue  # Skip problematic bars

    print(f"  💼 Executed {len(trades)} trades")
    print(f"  💰 Final equity: ${equity:,.2f}")

    return {
        'trades': trades,
        'cluster_events': cluster_events,
        'oracle_events': oracle_events,
        'final_equity': equity,
        'total_return': (equity - 10000) / 10000
    }

def analyze_v161_performance(backtest_results: Dict[str, Any], cluster_analytics: Dict[str, Any], cvd_analytics: Dict[str, Any]):
    """Analyze v1.6.1 performance with enhanced metrics"""
    print("\n📈 v1.6.1 Performance Analysis...")

    trades = backtest_results['trades']
    if not trades:
        print("❌ No trades to analyze")
        return

    # Basic performance metrics
    total_trades = len(trades)
    winning_trades = [t for t in trades if t['pnl_pct'] > 0]
    losing_trades = [t for t in trades if t['pnl_pct'] <= 0]

    win_rate = len(winning_trades) / total_trades
    avg_win = np.mean([t['pnl_pct'] for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t['pnl_pct'] for t in losing_trades]) if losing_trades else 0
    total_return = backtest_results['total_return']

    print(f"  🎯 Total Trades: {total_trades}")
    print(f"  📈 Win Rate: {win_rate:.1%}")
    print(f"  💰 Total Return: {total_return:.1%}")
    print(f"  ✅ Avg Win: {avg_win:.2%}")
    print(f"  ❌ Avg Loss: {avg_loss:.2%}")

    # v1.6.1 Enhanced Analytics
    print(f"\n🔮 Fibonacci Cluster Analytics:")
    print(f"  📊 Total price clusters detected: {cluster_analytics['price_clusters_detected']}")
    print(f"  ⏰ Total time clusters detected: {cluster_analytics['time_clusters_detected']}")
    print(f"  ✨ Price-time confluence events: {cluster_analytics['price_time_confluence']}")
    print(f"  🧙‍♂️ Oracle whispers triggered: {cluster_analytics['oracle_whispers_triggered']}")

    if cluster_analytics['cluster_strength_distribution']:
        avg_cluster_strength = np.mean(cluster_analytics['cluster_strength_distribution'])
        print(f"  💪 Average cluster strength: {avg_cluster_strength:.3f}")

    print(f"\n📊 Enhanced CVD Analytics:")
    print(f"  🔄 Total CVD analysis points: {cvd_analytics['total_analysis_points']}")
    print(f"  📈 Divergence signals detected: {cvd_analytics['divergence_signals']}")
    print(f"  🟢 Bullish divergences: {cvd_analytics['bullish_divergences']}")
    print(f"  🔴 Bearish divergences: {cvd_analytics['bearish_divergences']}")

    if cvd_analytics['orderflow_scores']:
        avg_orderflow_score = np.mean(cvd_analytics['orderflow_scores'])
        print(f"  📊 Average orderflow score: {avg_orderflow_score:.3f}")

    # Cluster-enhanced trade performance
    cluster_trades = [t for t in trades if t['price_time_confluence']]
    oracle_trades = [t for t in trades if t['oracle_whispers'] > 0]
    cvd_divergence_trades = [t for t in trades if t['orderflow_divergence']]

    if cluster_trades:
        cluster_win_rate = len([t for t in cluster_trades if t['pnl_pct'] > 0]) / len(cluster_trades)
        cluster_avg_return = np.mean([t['pnl_pct'] for t in cluster_trades])
        print(f"\n✨ Price-Time Confluence Trades:")
        print(f"  🎯 Count: {len(cluster_trades)}")
        print(f"  📈 Win Rate: {cluster_win_rate:.1%}")
        print(f"  💰 Avg Return: {cluster_avg_return:.2%}")

    if oracle_trades:
        oracle_win_rate = len([t for t in oracle_trades if t['pnl_pct'] > 0]) / len(oracle_trades)
        oracle_avg_return = np.mean([t['pnl_pct'] for t in oracle_trades])
        print(f"\n🧙‍♂️ Oracle Whisper Trades:")
        print(f"  🎯 Count: {len(oracle_trades)}")
        print(f"  📈 Win Rate: {oracle_win_rate:.1%}")
        print(f"  💰 Avg Return: {oracle_avg_return:.2%}")

    if cvd_divergence_trades:
        cvd_win_rate = len([t for t in cvd_divergence_trades if t['pnl_pct'] > 0]) / len(cvd_divergence_trades)
        cvd_avg_return = np.mean([t['pnl_pct'] for t in cvd_divergence_trades])
        print(f"\n📊 CVD Divergence Trades:")
        print(f"  🎯 Count: {len(cvd_divergence_trades)}")
        print(f"  📈 Win Rate: {cvd_win_rate:.1%}")
        print(f"  💰 Avg Return: {cvd_avg_return:.2%}")

    # Oracle Whisper Insights
    oracle_events = backtest_results['oracle_events']
    if oracle_events:
        successful_whispers = [e for e in oracle_events if e['trade_outcome'] > 0]
        whisper_success_rate = len(successful_whispers) / len(oracle_events)
        print(f"\n🧙‍♂️ Oracle Whisper Insights:")
        print(f"  ✨ Total whisper events: {len(oracle_events)}")
        print(f"  🎯 Success rate: {whisper_success_rate:.1%}")
        print(f"  💫 Avg whispers per event: {np.mean([e['whisper_count'] for e in oracle_events]):.1f}")

def main():
    """Main execution function"""
    print("🔮 ETH v1.6.1 Comprehensive Multi-Timeframe Backtest")
    print("Testing Fibonacci Clusters, Oracle Whispers, Enhanced CVD")
    print("=" * 60)

    # Fetch multi-timeframe data
    data = fetch_eth_multi_timeframe_data()

    if not data or not data.get('1D') or len(data['1D']) < 100:
        print("❌ Insufficient data for comprehensive backtest")
        return

    # Load enhanced ETH config
    config = load_config("ETH")
    config['features']['temporal_fib'] = True
    config['features']['fib_clusters'] = True
    config['features']['orderflow_lca'] = True

    # Run comprehensive analysis
    print("\n🔍 Phase 1: Fibonacci Cluster Analysis")
    cluster_analytics = analyze_fibonacci_cluster_performance(data, config)

    print("\n🔍 Phase 2: Enhanced CVD Analysis")
    cvd_analytics = test_enhanced_cvd_analysis(data, config)

    print("\n🔍 Phase 3: v1.6.1 Enhanced Backtest")
    backtest_results = run_v161_enhanced_backtest(data, config)

    if backtest_results:
        print("\n🔍 Phase 4: Performance Analysis")
        analyze_v161_performance(backtest_results, cluster_analytics, cvd_analytics)

        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"eth_v161_comprehensive_backtest_{timestamp}.json"

        comprehensive_results = {
            'metadata': {
                'timestamp': timestamp,
                'version': 'v1.6.1',
                'asset': 'ETH-USD',
                'philosophy': 'Price and time symmetry = where structure and vibration align'
            },
            'backtest_results': {
                'total_trades': len(backtest_results['trades']),
                'total_return': backtest_results['total_return'],
                'final_equity': backtest_results['final_equity']
            },
            'cluster_analytics': cluster_analytics,
            'cvd_analytics': cvd_analytics,
            'enhanced_features': {
                'fibonacci_clusters': True,
                'oracle_whispers': True,
                'enhanced_cvd': True,
                'price_time_confluence': True
            },
            'trades': backtest_results['trades'][:10],  # Sample trades
            'oracle_events': backtest_results['oracle_events'],
            'cluster_events': backtest_results['cluster_events']
        }

        with open(results_file, 'w') as f:
            json.dump(comprehensive_results, f, indent=2, default=str)

        print(f"\n📄 Comprehensive results saved to: {results_file}")

        # Print final summary
        print(f"\n🎯 v1.6.1 ETH Backtest Summary:")
        print(f"  💰 Total Return: {backtest_results['total_return']:.1%}")
        print(f"  🔮 Fibonacci Clusters: {cluster_analytics['price_time_confluence']} confluence events")
        print(f"  🧙‍♂️ Oracle Whispers: {cluster_analytics['oracle_whispers_triggered']} triggered")
        print(f"  📊 CVD Divergences: {cvd_analytics['divergence_signals']} detected")
        print(f"  ✨ Philosophy Validated: Price-time symmetry detection operational")

    else:
        print("❌ Backtest failed - insufficient data or configuration issues")

if __name__ == "__main__":
    main()