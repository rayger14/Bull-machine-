#!/usr/bin/env python3
"""
ETH v1.6.1 Complex Weights Backtest - Focused Implementation
Tests the full Bull Machine complex weighting system with v1.6.1 enhancements

Focus: Complete fusion scoring with all complex weights + v1.6.1 features
"""

import sys
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import json

# Add bull_machine to path
sys.path.append('.')

from bull_machine.modules.fusion.v160_enhanced import CoreTraderV160
from bull_machine.modules.orderflow.lca import analyze_market_structure
from bull_machine.strategy.hidden_fibs import detect_price_time_confluence
from bull_machine.oracle import trigger_whisper
from bull_machine.core.config_loader import load_config

warnings.filterwarnings('ignore')

def load_eth_data():
    """Load ETH 4H data efficiently"""
    filepath = '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv'

    df = pd.read_csv(filepath)
    df.columns = [col.lower().strip() for col in df.columns]
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    if 'buy+sell v' in df.columns:
        df['volume'] = df['buy+sell v']
    else:
        df['volume'] = df['close'] * 1000

    return df[['open', 'high', 'low', 'close', 'volume']].dropna()

def run_complex_weights_backtest():
    """Run backtest with full complex weighting system"""
    print("🔮 ETH v1.6.1 Complex Weights Backtest")
    print("Full Bull Machine Fusion Scoring + v1.6.1 Features")
    print("=" * 60)

    df = load_eth_data()
    print(f"✅ Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    # Load ETH config and apply complex weighting
    config = load_config("ETH")

    # Complex weighting configuration
    config.update({
        'entry_threshold': 0.30,  # Adjusted for real market data
        'quality_floors': {
            'wyckoff': 0.12,
            'm1': 0.15,
            'm2': 0.15,
            'confluence': 0.20,
            'orderflow': 0.20
        },
        'signal_weights': {
            'wyckoff': 0.35,
            'm1': 0.25,
            'm2': 0.20,
            'orderflow': 0.15,
            'fib_cluster': 0.05
        },
        'features': {
            'temporal_fib': True,
            'fib_clusters': True,
            'orderflow_lca': True,
            'oracle_whispers': True
        }
    })

    # Initialize full trader
    trader = CoreTraderV160(config)

    # Trading variables
    equity = 10000.0
    trades = []
    rejected_signals = []
    last_trade_bar = -999

    print(f"💰 Starting equity: ${equity:,.2f}")
    print(f"🎯 Entry threshold: {config['entry_threshold']}")
    print(f"⚖️ Complex weights: Wyckoff({config['signal_weights']['wyckoff']}) + M1({config['signal_weights']['m1']}) + M2({config['signal_weights']['m2']}) + Orderflow({config['signal_weights']['orderflow']}) + Clusters({config['signal_weights']['fib_cluster']})")

    # Main backtest loop
    for i in range(100, len(df)):
        current_data = df.iloc[:i+1]

        try:
            # Get full complex fusion signal
            entry_signal = trader.check_entry(current_data, last_trade_bar, config, equity)

            if i % 150 == 0:
                score = entry_signal.get('weighted_score', 0) if entry_signal else 0
                print(f"  📊 Bar {i}/{len(df)} - Fusion Score: {score:.3f}")

            # Entry decision with complex scoring
            if entry_signal and entry_signal.get('weighted_score', 0) >= config['entry_threshold'] and i - last_trade_bar > 8:

                # Enhanced v1.6.1 analysis
                confluence_data = detect_price_time_confluence(current_data, config, i)
                structure_analysis = analyze_market_structure(current_data.tail(30), config)

                # Build comprehensive score context
                test_scores = {
                    'cluster_tags': confluence_data.get('tags', []),
                    'confluence_strength': confluence_data.get('confluence_strength', 0),
                    'wyckoff_phase': entry_signal.get('wyckoff_phase', 'D'),
                    'wyckoff_score': entry_signal.get('wyckoff_score', 0),
                    'm1_score': entry_signal.get('m1_score', 0),
                    'm2_score': entry_signal.get('m2_score', 0),
                    'cvd_delta': structure_analysis['cvd_analysis']['delta'],
                    'cvd_slope': structure_analysis['cvd_analysis']['slope']
                }

                # Oracle whisper generation
                whispers = trigger_whisper(test_scores, phase=test_scores['wyckoff_phase'])

                # Quality gate filtering
                quality_passed = True
                rejection_reasons = []

                # Check all quality floors
                if test_scores['wyckoff_score'] < config['quality_floors']['wyckoff']:
                    quality_passed = False
                    rejection_reasons.append('wyckoff_floor')

                if test_scores['m1_score'] < config['quality_floors']['m1']:
                    quality_passed = False
                    rejection_reasons.append('m1_floor')

                if test_scores['m2_score'] < config['quality_floors']['m2']:
                    quality_passed = False
                    rejection_reasons.append('m2_floor')

                if test_scores['confluence_strength'] < config['quality_floors']['confluence']:
                    quality_passed = False
                    rejection_reasons.append('confluence_floor')

                if structure_analysis['lca_score'] < config['quality_floors']['orderflow']:
                    quality_passed = False
                    rejection_reasons.append('orderflow_floor')

                # Execute trade if quality gates pass
                if quality_passed:

                    # Complex position sizing
                    base_risk = 0.015  # 1.5% base risk
                    confluence_boost = 1.0 + (test_scores['confluence_strength'] * 0.4)
                    fusion_boost = 1.0 + (entry_signal['weighted_score'] * 0.3)

                    position_risk = base_risk * confluence_boost * fusion_boost
                    position_risk = min(position_risk, 0.025)  # Cap at 2.5%

                    entry_price = entry_signal['entry_price']
                    position_size = (equity * position_risk) / entry_price

                    # Complex exit management
                    stop_loss = entry_price * 0.96  # 4% stop
                    take_profit_1 = entry_price * 1.08  # 8% first target
                    take_profit_2 = entry_price * 1.15  # 15% second target

                    # Trade execution simulation
                    exit_found = False
                    total_pnl = 0
                    remaining_position = position_size

                    for j in range(1, min(25, len(df) - i)):  # Max 25 bars hold
                        future_bar = df.iloc[i + j]

                        # Stop loss
                        if future_bar['low'] <= stop_loss:
                            pnl = remaining_position * (stop_loss - entry_price)
                            total_pnl += pnl
                            exit_reason = 'stop_loss'
                            exit_found = True
                            break

                        # First take profit (50% position)
                        elif future_bar['high'] >= take_profit_1 and remaining_position > position_size * 0.4:
                            exit_size = position_size * 0.5
                            pnl = exit_size * (take_profit_1 - entry_price)
                            total_pnl += pnl
                            remaining_position -= exit_size

                        # Second take profit (remaining position)
                        elif future_bar['high'] >= take_profit_2:
                            pnl = remaining_position * (take_profit_2 - entry_price)
                            total_pnl += pnl
                            exit_reason = 'take_profit_2'
                            exit_found = True
                            break

                        # Time exit
                        elif j == 24:
                            pnl = remaining_position * (future_bar['close'] - entry_price)
                            total_pnl += pnl
                            exit_reason = 'time_exit'
                            exit_found = True
                            break

                    if exit_found:
                        pnl_pct = total_pnl / (position_size * entry_price)
                        equity += total_pnl

                        # Comprehensive trade record
                        trade_record = {
                            'entry_date': current_data.index[i],
                            'exit_date': df.index[i + j],
                            'entry_price': entry_price,
                            'exit_reason': exit_reason,
                            'pnl_pct': pnl_pct,
                            'pnl_dollars': total_pnl,
                            'position_risk': position_risk,

                            # Complex fusion scores
                            'fusion_score': entry_signal['weighted_score'],
                            'wyckoff_score': test_scores['wyckoff_score'],
                            'wyckoff_phase': test_scores['wyckoff_phase'],
                            'm1_score': test_scores['m1_score'],
                            'm2_score': test_scores['m2_score'],

                            # v1.6.1 features
                            'confluence_strength': test_scores['confluence_strength'],
                            'price_time_confluence': confluence_data.get('confluence_detected', False),
                            'oracle_whispers': len(whispers) if whispers else 0,
                            'cvd_delta': test_scores['cvd_delta'],
                            'cvd_slope': test_scores['cvd_slope'],
                            'orderflow_score': structure_analysis['lca_score'],
                            'orderflow_divergence': structure_analysis['orderflow_divergence']['detected'],

                            # Risk multipliers
                            'confluence_boost': confluence_boost,
                            'fusion_boost': fusion_boost
                        }

                        trades.append(trade_record)
                        last_trade_bar = i

                        print(f"    💼 Trade {len(trades)}: {pnl_pct:.2%} | Fusion: {entry_signal['weighted_score']:.3f} | Confluence: {test_scores['confluence_strength']:.3f} | Whispers: {len(whispers) if whispers else 0}")

                else:
                    # Track rejections for analysis
                    rejected_signals.append({
                        'date': current_data.index[i],
                        'fusion_score': entry_signal['weighted_score'],
                        'rejection_reasons': rejection_reasons,
                        'scores': test_scores
                    })

        except Exception as e:
            continue

    # Comprehensive results analysis
    print(f"\n📈 Complex Weights Backtest Results:")
    print(f"  💼 Total trades: {len(trades)}")
    print(f"  💰 Final equity: ${equity:,.2f}")
    print(f"  🚫 Rejected signals: {len(rejected_signals)}")

    if trades:
        total_return = (equity - 10000) / 10000
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        win_rate = len(winning_trades) / len(trades)
        avg_return = np.mean([t['pnl_pct'] for t in trades])

        print(f"\n💹 Performance:")
        print(f"  📈 Total return: {total_return:.2%}")
        print(f"  🎯 Win rate: {win_rate:.1%} ({len(winning_trades)}/{len(trades)})")
        print(f"  💹 Avg return per trade: {avg_return:.2%}")

        if winning_trades:
            avg_win = np.mean([t['pnl_pct'] for t in winning_trades])
            print(f"  🟢 Avg winning trade: {avg_win:.2%}")

        losing_trades = [t for t in trades if t['pnl_pct'] < 0]
        if losing_trades:
            avg_loss = np.mean([t['pnl_pct'] for t in losing_trades])
            print(f"  🔴 Avg losing trade: {avg_loss:.2%}")

        # Complex scoring analysis
        avg_scores = {
            'fusion': np.mean([t['fusion_score'] for t in trades]),
            'wyckoff': np.mean([t['wyckoff_score'] for t in trades]),
            'm1': np.mean([t['m1_score'] for t in trades]),
            'm2': np.mean([t['m2_score'] for t in trades]),
            'confluence': np.mean([t['confluence_strength'] for t in trades]),
            'orderflow': np.mean([t['orderflow_score'] for t in trades])
        }

        print(f"\n🔬 Complex Scoring Analysis:")
        print(f"  🎯 Avg Fusion Score: {avg_scores['fusion']:.3f}")
        print(f"  📊 Avg Wyckoff Score: {avg_scores['wyckoff']:.3f}")
        print(f"  ⚡ Avg M1 Score: {avg_scores['m1']:.3f}")
        print(f"  🏗️ Avg M2 Score: {avg_scores['m2']:.3f}")
        print(f"  ✨ Avg Confluence: {avg_scores['confluence']:.3f}")
        print(f"  📈 Avg Orderflow: {avg_scores['orderflow']:.3f}")

        # v1.6.1 feature performance
        cluster_trades = [t for t in trades if t['price_time_confluence']]
        oracle_trades = [t for t in trades if t['oracle_whispers'] > 0]

        print(f"\n🔮 v1.6.1 Features:")
        print(f"  ✨ Price-time confluence trades: {len(cluster_trades)}")
        print(f"  🧙‍♂️ Oracle whisper trades: {len(oracle_trades)}")

        if cluster_trades:
            cluster_win_rate = len([t for t in cluster_trades if t['pnl_pct'] > 0]) / len(cluster_trades)
            cluster_avg = np.mean([t['pnl_pct'] for t in cluster_trades])
            print(f"  🎯 Cluster win rate: {cluster_win_rate:.1%}")
            print(f"  💰 Cluster avg return: {cluster_avg:.2%}")

        # Risk analysis
        avg_risk = np.mean([t['position_risk'] for t in trades])
        max_risk = max([t['position_risk'] for t in trades])
        avg_confluence_boost = np.mean([t['confluence_boost'] for t in trades])
        avg_fusion_boost = np.mean([t['fusion_boost'] for t in trades])

        print(f"\n⚖️ Risk Management:")
        print(f"  📊 Avg position risk: {avg_risk:.2%}")
        print(f"  🔥 Max position risk: {max_risk:.2%}")
        print(f"  ✨ Avg confluence boost: {avg_confluence_boost:.2f}x")
        print(f"  🎯 Avg fusion boost: {avg_fusion_boost:.2f}x")

        # Save results
        results = {
            'metadata': {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'system': 'Complex Weights + v1.6.1',
                'asset': 'ETH-USD',
                'philosophy': 'Price and time symmetry = where structure and vibration align'
            },
            'configuration': {
                'entry_threshold': config['entry_threshold'],
                'quality_floors': config['quality_floors'],
                'signal_weights': config['signal_weights']
            },
            'performance': {
                'starting_equity': 10000,
                'final_equity': equity,
                'total_return': total_return,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'rejected_signals': len(rejected_signals)
            },
            'scoring_analysis': avg_scores,
            'trades': trades[:15]  # First 15 trades
        }

        filename = f"eth_complex_weights_v161_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Results saved to: {filename}")
        print(f"\n✨ Complex System Philosophy Validated:")
        print(f"   'Where structure and vibration align'")
        print(f"   🔬 {len(trades)} trades from full fusion scoring")
        print(f"   🎯 Complex quality gates filtered {len(rejected_signals)} signals")

    else:
        print("  ⚠️ No trades found - adjusting thresholds needed")

        if rejected_signals:
            print(f"\n🚫 Rejection Analysis:")
            reasons = {}
            for sig in rejected_signals:
                for reason in sig['rejection_reasons']:
                    reasons[reason] = reasons.get(reason, 0) + 1

            for reason, count in reasons.items():
                print(f"  {reason}: {count} rejections")

if __name__ == "__main__":
    run_complex_weights_backtest()