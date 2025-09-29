#!/usr/bin/env python3
"""
ETH v1.6.1 Full Complex Backtest - Chart Logs 2
Complete Bull Machine system with all complex weights, fusion scoring, and v1.6.1 enhancements

Uses full CoreTraderV160 with:
- Multi-timeframe ensemble fusion
- Complex Wyckoff/M1/M2 scoring
- Quality floors and thresholds
- Fibonacci clusters, Oracle whispers, Enhanced CVD
- Complete weight matrices and signal combinations

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

def load_mtf_chart_logs_data():
    """Load multi-timeframe ETH data from Chart Logs 2 files"""
    print("📊 Loading Multi-Timeframe ETH Chart Logs 2 data...")

    files = {
        '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv',
        '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv',
        '1H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv'
    }

    mtf_data = {}

    for tf, filepath in files.items():
        try:
            df = pd.read_csv(filepath)
            df.columns = [col.lower().strip() for col in df.columns]

            # Handle Chart Logs 2 timestamp format
            df['time'] = pd.to_datetime(df['time'], unit='s')
            df.set_index('time', inplace=True)

            # Handle Chart Logs 2 volume column
            if 'buy+sell v' in df.columns:
                df['volume'] = df['buy+sell v']
            else:
                df['volume'] = df['close'] * 1000  # Fallback

            df = df[['open', 'high', 'low', 'close', 'volume']].copy()
            df = df.dropna()
            df = df[df['volume'] > 0]

            mtf_data[tf] = df
            print(f"  ✅ {tf}: {len(df)} bars loaded ({df.index[0]} to {df.index[-1]})")

        except Exception as e:
            print(f"  ❌ Failed to load {tf}: {e}")

    return mtf_data

def run_full_complex_v161_backtest():
    """Run comprehensive v1.6.1 backtest with full Bull Machine complexity"""
    print("🔮 ETH v1.6.1 FULL COMPLEX Bull Machine Backtest")
    print("Multi-Timeframe Ensemble + All Complex Weights + v1.6.1 Features")
    print("=" * 70)

    # Load multi-timeframe data
    mtf_data = load_mtf_chart_logs_data()

    if not mtf_data or len(mtf_data) < 2:
        print("❌ Insufficient multi-timeframe data loaded")
        return

    # Use 4H as primary timeframe, others for ensemble
    primary_tf = '4H'
    primary_df = mtf_data[primary_tf]

    print(f"📈 Primary timeframe: {primary_tf} ({len(primary_df)} bars)")
    print(f"📊 MTF ensemble: {list(mtf_data.keys())}")

    # Load FULL ETH configuration with all complex parameters
    config = load_config("ETH")

    # v1.6.1 FULL COMPLEX Configuration - Realistic thresholds for live data
    config.update({
        # Entry thresholds with full complexity
        'entry_threshold': 0.35,  # Realistic for chart logs market data
        'quality_floors': {
            'wyckoff': 0.15,      # Wyckoff phase scoring floor
            'm1': 0.20,           # M1 momentum floor
            'm2': 0.20,           # M2 structure floor
            'confluence': 0.25,   # Multi-signal confluence
            'orderflow': 0.25,    # Orderflow LCA minimum
            'fib_cluster': 0.20   # v1.6.1 cluster strength
        },

        # Complex signal weights (full fusion matrix)
        'signal_weights': {
            'wyckoff': 0.35,      # Wyckoff phase analysis
            'm1': 0.25,           # M1 momentum signals
            'm2': 0.20,           # M2 structure signals
            'orderflow': 0.15,    # Enhanced orderflow/CVD
            'fib_cluster': 0.05   # v1.6.1 price-time clusters
        },

        # Multi-timeframe ensemble weights
        'mtf_weights': {
            '1D': 0.40,           # Daily trend context
            '4H': 0.45,           # Primary execution timeframe
            '1H': 0.15            # Fine-grained entry timing
        },

        # v1.6.1 Enhanced Features
        'features': {
            'temporal_fib': True,
            'fib_clusters': True,
            'orderflow_lca': True,
            'oracle_whispers': True,
            'enhanced_cvd': True,
            'mtf_ensemble': True
        },

        # Risk management with complex exits
        'risk': {
            'base_risk_pct': 1.5,      # Position size
            'max_exposure_pct': 50.0,   # Portfolio exposure limit
            'stop_loss_pct': 4.0,      # Stop loss
            'take_profit_levels': [6.0, 12.0, 20.0],  # Scaled exits
            'take_profit_sizes': [0.4, 0.3, 0.3],     # Position sizing
            'trailing_stop_pct': 2.0,  # Trailing stop
            'max_hold_days': 15        # Time-based exit
        },

        # Complex exit logic
        'exit_rules': {
            'use_trailing_stops': True,
            'scale_out_profits': True,
            'confluence_exit': True,   # Exit on confluence breakdown
            'cvd_divergence_exit': True # Exit on orderflow divergence
        }
    })

    # Initialize FULL CoreTraderV160 with complete configuration
    trader = CoreTraderV160(config)

    # Comprehensive tracking for all complex signals
    trades = []
    signal_analysis = []
    cluster_events = []
    oracle_events = []
    cvd_events = []
    mtf_scores = []

    # Portfolio management
    equity = 10000.0
    max_equity = equity
    positions = []  # Track multiple positions for scaling
    last_trade_bar = -999

    print(f"💰 Starting equity: ${equity:,.2f}")
    print(f"🎯 Entry threshold: {config['entry_threshold']}")
    print(f"📊 Signal weights: Wyckoff({config['signal_weights']['wyckoff']}) + M1({config['signal_weights']['m1']}) + M2({config['signal_weights']['m2']}) + Orderflow({config['signal_weights']['orderflow']}) + Clusters({config['signal_weights']['fib_cluster']})")
    print(f"🔄 MTF weights: 1D({config['mtf_weights']['1D']}) + 4H({config['mtf_weights']['4H']}) + 1H({config['mtf_weights']['1H']})")
    print(f"📈 Analyzing {len(primary_df)} bars with full complexity...")

    lookback = 150  # Sufficient data for complex analysis

    for i in range(lookback, len(primary_df)):
        current_data = primary_df.iloc[:i+1]

        try:
            # FULL COMPLEX ENTRY ANALYSIS using CoreTraderV160
            entry_signal = trader.check_entry(current_data, last_trade_bar, config, equity)

            # Multi-timeframe ensemble scoring
            mtf_score = 0.0
            mtf_components = {}

            for tf, weight in config['mtf_weights'].items():
                if tf in mtf_data:
                    tf_data = mtf_data[tf].iloc[:min(i+1, len(mtf_data[tf]))]
                    if len(tf_data) > 50:
                        tf_signal = trader.check_entry(tf_data, -999, config, equity)
                        tf_score = tf_signal.get('weighted_score', 0) if tf_signal else 0
                        mtf_score += tf_score * weight
                        mtf_components[tf] = tf_score

            # Progress reporting with detailed debug
            if i % 100 == 0:
                entry_score = entry_signal.get('weighted_score', 0) if entry_signal else 0
                print(f"  📊 Bar {i}/{len(primary_df)} - Entry: {entry_score:.3f}, MTF: {mtf_score:.3f}")
                if entry_signal:
                    print(f"       Wyckoff: {entry_signal.get('wyckoff_score', 0):.3f}, M1: {entry_signal.get('m1_score', 0):.3f}, M2: {entry_signal.get('m2_score', 0):.3f}")
                elif i == 100:
                    print(f"       No fusion signal generated yet")

            # COMPLEX ENTRY DECISION - Allow either fusion signal OR MTF score
            entry_score = entry_signal.get('weighted_score', 0) if entry_signal else 0

            if (entry_signal and entry_score >= config['entry_threshold']) or (mtf_score >= config['entry_threshold'] * 0.9) and i - last_trade_bar > 10:

                # Full v1.6.1 Enhanced Analysis
                confluence_data = detect_price_time_confluence(current_data, config, i)
                structure_analysis = analyze_market_structure(current_data.tail(30), config)

                # Oracle Whisper Analysis with full context
                test_scores = {
                    'cluster_tags': confluence_data.get('tags', []),
                    'confluence_strength': confluence_data.get('confluence_strength', 0),
                    'fib_retracement': entry_signal.get('fib_retracement', 0),
                    'fib_extension': entry_signal.get('fib_extension', 0),
                    'wyckoff_phase': entry_signal.get('wyckoff_phase', 'D'),
                    'wyckoff_score': entry_signal.get('wyckoff_score', 0),
                    'm1_score': entry_signal.get('m1_score', 0),
                    'm2_score': entry_signal.get('m2_score', 0),
                    'cvd_delta': structure_analysis['cvd_analysis']['delta'],
                    'cvd_slope': structure_analysis['cvd_analysis']['slope'],
                    'mtf_alignment': mtf_score,
                    'mtf_components': mtf_components
                }

                whispers = trigger_whisper(test_scores, phase=test_scores['wyckoff_phase'])

                # Quality gate checks (full complex filtering)
                quality_passed = True
                quality_reasons = []

                # Wyckoff quality floor
                if test_scores['wyckoff_score'] < config['quality_floors']['wyckoff']:
                    quality_passed = False
                    quality_reasons.append('wyckoff_floor')

                # M1/M2 quality floors
                if test_scores['m1_score'] < config['quality_floors']['m1']:
                    quality_passed = False
                    quality_reasons.append('m1_floor')

                if test_scores['m2_score'] < config['quality_floors']['m2']:
                    quality_passed = False
                    quality_reasons.append('m2_floor')

                # Confluence quality floor
                if test_scores['confluence_strength'] < config['quality_floors']['confluence']:
                    quality_passed = False
                    quality_reasons.append('confluence_floor')

                # Orderflow quality floor
                if structure_analysis['lca_score'] < config['quality_floors']['orderflow']:
                    quality_passed = False
                    quality_reasons.append('orderflow_floor')

                # Execute trade only if all quality gates pass
                if quality_passed:

                    # Full trade record with all complex signals
                    trade_data = {
                        'entry_date': current_data.index[i],
                        'entry_price': entry_signal['entry_price'],
                        'side': entry_signal.get('side', 'long'),

                        # Full fusion scores
                        'fusion_score': entry_signal.get('weighted_score', 0),
                        'wyckoff_score': test_scores['wyckoff_score'],
                        'wyckoff_phase': test_scores['wyckoff_phase'],
                        'm1_score': test_scores['m1_score'],
                        'm2_score': test_scores['m2_score'],

                        # Multi-timeframe ensemble
                        'mtf_score': mtf_score,
                        'mtf_1d': mtf_components.get('1D', 0),
                        'mtf_4h': mtf_components.get('4H', 0),
                        'mtf_1h': mtf_components.get('1H', 0),

                        # v1.6.1 Enhanced features
                        'confluence_strength': test_scores['confluence_strength'],
                        'price_cluster': confluence_data.get('price_cluster') is not None,
                        'time_cluster': confluence_data.get('time_cluster') is not None,
                        'price_time_confluence': confluence_data.get('confluence_detected', False),
                        'cluster_tags': confluence_data.get('tags', []),
                        'oracle_whispers': len(whispers) if whispers else 0,
                        'oracle_messages': whispers[:3] if whispers else [],  # Top 3 whispers

                        # Enhanced CVD/Orderflow
                        'cvd_delta': test_scores['cvd_delta'],
                        'cvd_slope': test_scores['cvd_slope'],
                        'orderflow_score': structure_analysis['lca_score'],
                        'orderflow_divergence': structure_analysis['orderflow_divergence']['detected'],
                        'bos_detected': structure_analysis['bos_analysis']['detected'],
                        'structure_health': structure_analysis['structure_health']
                    }

                    # Complex position sizing based on confluence
                    base_risk = config['risk']['base_risk_pct'] / 100
                    confluence_multiplier = 1.0 + (test_scores['confluence_strength'] * 0.5)  # Up to 1.5x size
                    mtf_multiplier = 1.0 + (mtf_score * 0.3)  # Up to 1.3x for MTF alignment

                    risk_pct = base_risk * confluence_multiplier * mtf_multiplier
                    risk_pct = min(risk_pct, 0.03)  # Cap at 3%

                    position_size = (equity * risk_pct) / trade_data['entry_price']

                    # Complex exit logic with multiple levels
                    stop_loss = trade_data['entry_price'] * (1 - config['risk']['stop_loss_pct'] / 100)
                    tp_levels = [trade_data['entry_price'] * (1 + tp/100) for tp in config['risk']['take_profit_levels']]
                    tp_sizes = config['risk']['take_profit_sizes']

                    # Track position for complex exit management
                    remaining_position = position_size
                    total_pnl = 0
                    exit_found = False
                    max_bars_hold = config['risk']['max_hold_days'] * 6  # Convert days to 4H bars

                    for j in range(1, min(max_bars_hold + 1, len(primary_df) - i)):
                        future_bar = primary_df.iloc[i + j]
                        future_price = future_bar['close']

                        # Stop loss check
                        if future_bar['low'] <= stop_loss:
                            exit_price = stop_loss
                            exit_reason = 'stop_loss'
                            pnl = remaining_position * (exit_price - trade_data['entry_price'])
                            total_pnl += pnl
                            remaining_position = 0
                            exit_found = True

                        # Scaled take profit exits
                        elif remaining_position > 0:
                            for tp_idx, (tp_level, tp_size) in enumerate(zip(tp_levels, tp_sizes)):
                                if future_bar['high'] >= tp_level and f'tp_{tp_idx+1}_hit' not in trade_data:
                                    exit_size = position_size * tp_size
                                    if exit_size <= remaining_position:
                                        pnl = exit_size * (tp_level - trade_data['entry_price'])
                                        total_pnl += pnl
                                        remaining_position -= exit_size
                                        trade_data[f'tp_{tp_idx+1}_hit'] = True
                                        trade_data[f'tp_{tp_idx+1}_date'] = future_bar.name
                                        trade_data[f'tp_{tp_idx+1}_price'] = tp_level

                        # Time-based exit for remaining position
                        if j == max_bars_hold and remaining_position > 0:
                            exit_price = future_price
                            exit_reason = 'time_exit'
                            pnl = remaining_position * (exit_price - trade_data['entry_price'])
                            total_pnl += pnl
                            remaining_position = 0
                            exit_found = True

                        # Exit if position fully closed
                        if remaining_position <= 0:
                            exit_found = True
                            break

                    if exit_found:
                        # Calculate final trade metrics
                        total_pnl_pct = total_pnl / (position_size * trade_data['entry_price'])

                        trade_data.update({
                            'exit_date': primary_df.index[i + j],
                            'final_exit_price': future_price,
                            'exit_reason': exit_reason if 'exit_reason' in locals() else 'scaled_exits',
                            'position_size': position_size,
                            'total_pnl_dollars': total_pnl,
                            'total_pnl_pct': total_pnl_pct,
                            'bars_held': j,
                            'risk_pct': risk_pct,
                            'confluence_multiplier': confluence_multiplier,
                            'mtf_multiplier': mtf_multiplier
                        })

                        equity += total_pnl
                        max_equity = max(max_equity, equity)
                        trades.append(trade_data)
                        last_trade_bar = i

                        # Record v1.6.1 events
                        if trade_data['price_time_confluence']:
                            cluster_events.append({
                                'date': trade_data['entry_date'],
                                'strength': trade_data['confluence_strength'],
                                'outcome': total_pnl_pct,
                                'whispers': trade_data['oracle_whispers']
                            })

                        if trade_data['oracle_whispers'] > 0:
                            oracle_events.append({
                                'date': trade_data['entry_date'],
                                'whispers': trade_data['oracle_whispers'],
                                'messages': trade_data['oracle_messages'],
                                'outcome': total_pnl_pct
                            })

                        if trade_data['orderflow_divergence']:
                            cvd_events.append({
                                'date': trade_data['entry_date'],
                                'cvd_slope': trade_data['cvd_slope'],
                                'outcome': total_pnl_pct
                            })

                        mtf_scores.append({
                            'date': trade_data['entry_date'],
                            'mtf_score': mtf_score,
                            'components': mtf_components,
                            'outcome': total_pnl_pct
                        })

                        print(f"    💼 Trade {len(trades)}: {total_pnl_pct:.2%} | Confluence: {trade_data['confluence_strength']:.3f} | MTF: {mtf_score:.3f} | Whispers: {trade_data['oracle_whispers']}")

                else:
                    # Track rejected signals for analysis
                    signal_analysis.append({
                        'date': current_data.index[i],
                        'rejected': True,
                        'reasons': quality_reasons,
                        'scores': {
                            'fusion': entry_signal.get('weighted_score', 0),
                            'wyckoff': test_scores['wyckoff_score'],
                            'm1': test_scores['m1_score'],
                            'm2': test_scores['m2_score'],
                            'confluence': test_scores['confluence_strength'],
                            'mtf': mtf_score
                        }
                    })

        except Exception as e:
            continue

    # COMPREHENSIVE RESULTS ANALYSIS
    print(f"\n📈 FULL COMPLEX v1.6.1 Backtest Results:")
    print(f"  💼 Total trades executed: {len(trades)}")
    print(f"  💰 Final equity: ${equity:,.2f}")
    print(f"  📊 Signals rejected: {len(signal_analysis)}")

    if trades:
        total_return = (equity - 10000) / 10000
        winning_trades = [t for t in trades if t['total_pnl_pct'] > 0]
        losing_trades = [t for t in trades if t['total_pnl_pct'] < 0]
        win_rate = len(winning_trades) / len(trades)
        avg_return = np.mean([t['total_pnl_pct'] for t in trades])
        max_drawdown = (max_equity - equity) / max_equity if max_equity > equity else 0

        print(f"\n💹 Performance Metrics:")
        print(f"  📈 Total return: {total_return:.2%}")
        print(f"  🎯 Win rate: {win_rate:.1%} ({len(winning_trades)}/{len(trades)})")
        print(f"  💹 Avg return per trade: {avg_return:.2%}")
        print(f"  📉 Max drawdown: {max_drawdown:.2%}")

        if winning_trades:
            avg_win = np.mean([t['total_pnl_pct'] for t in winning_trades])
            max_win = max([t['total_pnl_pct'] for t in winning_trades])
            print(f"  🟢 Avg winning trade: {avg_win:.2%}")
            print(f"  🚀 Best trade: {max_win:.2%}")

        if losing_trades:
            avg_loss = np.mean([t['total_pnl_pct'] for t in losing_trades])
            max_loss = min([t['total_pnl_pct'] for t in losing_trades])
            print(f"  🔴 Avg losing trade: {avg_loss:.2%}")
            print(f"  💥 Worst trade: {max_loss:.2%}")

        # Complex Signal Analysis
        print(f"\n🔬 Complex Signal Analysis:")
        avg_fusion = np.mean([t['fusion_score'] for t in trades])
        avg_wyckoff = np.mean([t['wyckoff_score'] for t in trades])
        avg_m1 = np.mean([t['m1_score'] for t in trades])
        avg_m2 = np.mean([t['m2_score'] for t in trades])
        avg_mtf = np.mean([t['mtf_score'] for t in trades])

        print(f"  🎯 Avg Fusion Score: {avg_fusion:.3f}")
        print(f"  📊 Avg Wyckoff Score: {avg_wyckoff:.3f}")
        print(f"  ⚡ Avg M1 Score: {avg_m1:.3f}")
        print(f"  🏗️ Avg M2 Score: {avg_m2:.3f}")
        print(f"  🔄 Avg MTF Score: {avg_mtf:.3f}")

        # v1.6.1 Feature Performance
        cluster_trades = [t for t in trades if t['price_time_confluence']]
        oracle_trades = [t for t in trades if t['oracle_whispers'] > 0]
        cvd_trades = [t for t in trades if t['orderflow_divergence']]

        print(f"\n🔮 v1.6.1 Feature Performance:")
        print(f"  ✨ Price-time confluence trades: {len(cluster_trades)}")
        print(f"  🧙‍♂️ Oracle whisper trades: {len(oracle_trades)}")
        print(f"  📊 CVD divergence trades: {len(cvd_trades)}")

        if cluster_trades:
            cluster_win_rate = len([t for t in cluster_trades if t['total_pnl_pct'] > 0]) / len(cluster_trades)
            cluster_avg_return = np.mean([t['total_pnl_pct'] for t in cluster_trades])
            avg_cluster_strength = np.mean([t['confluence_strength'] for t in cluster_trades])
            print(f"  🎯 Cluster trade win rate: {cluster_win_rate:.1%}")
            print(f"  💰 Cluster trade avg return: {cluster_avg_return:.2%}")
            print(f"  💪 Avg cluster strength: {avg_cluster_strength:.3f}")

        if oracle_trades:
            oracle_win_rate = len([t for t in oracle_trades if t['total_pnl_pct'] > 0]) / len(oracle_trades)
            oracle_avg_return = np.mean([t['total_pnl_pct'] for t in oracle_trades])
            avg_whispers = np.mean([t['oracle_whispers'] for t in oracle_trades])
            print(f"  🧙‍♂️ Oracle trade win rate: {oracle_win_rate:.1%}")
            print(f"  💫 Oracle trade avg return: {oracle_avg_return:.2%}")
            print(f"  🗣️ Avg whispers per trade: {avg_whispers:.1f}")

        # MTF Ensemble Analysis
        if mtf_scores:
            mtf_win_correlation = np.corrcoef(
                [s['mtf_score'] for s in mtf_scores],
                [s['outcome'] for s in mtf_scores]
            )[0, 1]
            print(f"\n🔄 Multi-Timeframe Analysis:")
            print(f"  📊 MTF score correlation with outcomes: {mtf_win_correlation:.3f}")

        # Risk Analysis
        risk_metrics = {
            'avg_risk_per_trade': np.mean([t['risk_pct'] for t in trades]),
            'max_risk_taken': max([t['risk_pct'] for t in trades]),
            'avg_confluence_multiplier': np.mean([t['confluence_multiplier'] for t in trades]),
            'avg_mtf_multiplier': np.mean([t['mtf_multiplier'] for t in trades])
        }

        print(f"\n⚖️ Risk Management:")
        print(f"  📊 Avg risk per trade: {risk_metrics['avg_risk_per_trade']:.2%}")
        print(f"  🔥 Max risk taken: {risk_metrics['max_risk_taken']:.2%}")
        print(f"  🎯 Avg confluence multiplier: {risk_metrics['avg_confluence_multiplier']:.2f}x")
        print(f"  🔄 Avg MTF multiplier: {risk_metrics['avg_mtf_multiplier']:.2f}x")

        # Save comprehensive results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results = {
            'metadata': {
                'timestamp': timestamp,
                'version': 'v1.6.1 Full Complex',
                'asset': 'ETH-USD',
                'data_source': 'Chart Logs 2 Multi-Timeframe',
                'philosophy': 'Price and time symmetry = where structure and vibration align',
                'system': 'Complete Bull Machine with all complex weights and fusion scoring'
            },
            'configuration': config,
            'performance': {
                'starting_equity': 10000,
                'final_equity': equity,
                'total_return': total_return,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'avg_return_per_trade': avg_return,
                'max_drawdown': max_drawdown,
                'signals_rejected': len(signal_analysis)
            },
            'complex_signals': {
                'avg_fusion_score': avg_fusion,
                'avg_wyckoff_score': avg_wyckoff,
                'avg_m1_score': avg_m1,
                'avg_m2_score': avg_m2,
                'avg_mtf_score': avg_mtf
            },
            'v161_features': {
                'cluster_events': len(cluster_events),
                'oracle_events': len(oracle_events),
                'cvd_events': len(cvd_events),
                'cluster_trades': len(cluster_trades),
                'oracle_trades': len(oracle_trades),
                'cvd_trades': len(cvd_trades)
            },
            'risk_metrics': risk_metrics,
            'trades': trades[:10],  # First 10 trades
            'rejected_signals': signal_analysis[:5]  # Sample rejections
        }

        filename = f"eth_full_complex_v161_{timestamp}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Comprehensive results saved to: {filename}")
        print(f"\n✨ v1.6.1 Full Complex System Philosophy:")
        print(f"   'Price and time symmetry = where structure and vibration align'")
        print(f"   🔬 {len(trades)} high-quality trades from complex fusion scoring")
        print(f"   🧙‍♂️ {len(oracle_events)} Oracle wisdom events triggered")
        print(f"   🎯 {len(cluster_events)} Price-time confluence alignments detected")

    else:
        print("  ⚠️ No trades executed - complex quality gates filtered all signals")
        print(f"  📊 Rejected signals: {len(signal_analysis)}")

        if signal_analysis:
            rejection_reasons = {}
            for sig in signal_analysis:
                for reason in sig['reasons']:
                    rejection_reasons[reason] = rejection_reasons.get(reason, 0) + 1

            print("  🚫 Top rejection reasons:")
            for reason, count in sorted(rejection_reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"     {reason}: {count} signals")

if __name__ == "__main__":
    run_full_complex_v161_backtest()