#!/usr/bin/env python3
"""
ETH Working Complex Backtest - Chart Logs 2
Implements complex weighting system manually to ensure it works with real data

Tests ALL the complex components:
- Wyckoff phase analysis (35% weight)
- M1 momentum signals (25% weight)
- M2 structure signals (20% weight)
- Enhanced orderflow/CVD (15% weight)
- v1.6.1 Fibonacci clusters (5% weight)
"""

import sys
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import json

# Add bull_machine to path
sys.path.append('.')

from bull_machine.modules.orderflow.lca import analyze_market_structure, orderflow_lca
from bull_machine.strategy.hidden_fibs import detect_price_time_confluence
from bull_machine.oracle import trigger_whisper
from bull_machine.core.config_loader import load_config

warnings.filterwarnings('ignore')

def calculate_wyckoff_score(df):
    """Calculate Wyckoff phase scoring"""
    if len(df) < 50:
        return 0, 'N'

    # Volume analysis
    vol_ma = df['volume'].rolling(20).mean().iloc[-1]
    current_vol = df['volume'].iloc[-1]
    vol_ratio = current_vol / vol_ma if vol_ma > 0 else 1

    # Price action analysis
    close = df['close'].iloc[-1]
    high = df['high'].iloc[-10:].max()
    low = df['low'].iloc[-10:].min()
    range_pos = (close - low) / (high - low) if high != low else 0.5

    # Trend context
    sma_20 = df['close'].rolling(20).mean().iloc[-1]
    sma_50 = df['close'].rolling(50).mean().iloc[-1]
    trend_strength = (sma_20 - sma_50) / sma_50 if sma_50 > 0 else 0

    # Phase determination
    if vol_ratio > 1.5 and range_pos > 0.7 and trend_strength > 0.02:
        phase = 'D'  # Distribution/Markup
        score = min(1.0, 0.4 + vol_ratio * 0.3 + range_pos * 0.2 + abs(trend_strength) * 5)
    elif vol_ratio > 1.2 and range_pos < 0.3 and trend_strength < -0.02:
        phase = 'A'  # Accumulation
        score = min(1.0, 0.3 + vol_ratio * 0.25 + (1-range_pos) * 0.25 + abs(trend_strength) * 5)
    elif vol_ratio > 1.0 and abs(trend_strength) < 0.01:
        phase = 'C'  # Consolidation
        score = min(1.0, 0.25 + vol_ratio * 0.2 + (0.5 - abs(range_pos - 0.5)) * 0.3)
    else:
        phase = 'B'  # Background/Transition
        score = min(1.0, 0.15 + vol_ratio * 0.15)

    return score, phase

def calculate_m1_momentum(df):
    """Calculate M1 momentum signals"""
    if len(df) < 20:
        return 0

    # RSI
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    rsi_score = 0

    current_rsi = rsi.iloc[-1]
    if 30 <= current_rsi <= 70:
        rsi_score = 0.5
    elif current_rsi < 30:
        rsi_score = 0.8  # Oversold
    elif current_rsi > 70:
        rsi_score = 0.3  # Overbought

    # MACD
    ema_12 = df['close'].ewm(span=12).mean()
    ema_26 = df['close'].ewm(span=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9).mean()
    histogram = macd - signal

    macd_score = 0
    if histogram.iloc[-1] > histogram.iloc[-2] and macd.iloc[-1] > signal.iloc[-1]:
        macd_score = 0.7
    elif histogram.iloc[-1] > 0:
        macd_score = 0.4
    else:
        macd_score = 0.2

    # Price momentum
    returns_5 = (df['close'].iloc[-1] / df['close'].iloc[-6] - 1) if len(df) > 5 else 0
    momentum_score = min(1.0, 0.5 + abs(returns_5) * 10)

    return min(1.0, (rsi_score * 0.4 + macd_score * 0.4 + momentum_score * 0.2))

def calculate_m2_structure(df):
    """Calculate M2 structure signals"""
    if len(df) < 50:
        return 0

    # Support/Resistance
    highs = df['high'].rolling(10, center=True).max()
    lows = df['low'].rolling(10, center=True).min()

    current_price = df['close'].iloc[-1]
    recent_high = df['high'].iloc[-20:].max()
    recent_low = df['low'].iloc[-20:].min()

    # Structure scoring
    structure_score = 0

    # Breakout detection
    if current_price > recent_high * 1.01:
        structure_score += 0.4  # Bullish breakout
    elif current_price < recent_low * 0.99:
        structure_score += 0.3  # Bearish breakdown

    # Trend structure
    sma_20 = df['close'].rolling(20).mean().iloc[-1]
    sma_50 = df['close'].rolling(50).mean().iloc[-1]

    if sma_20 > sma_50 and current_price > sma_20:
        structure_score += 0.3  # Uptrend structure
    elif sma_20 < sma_50 and current_price < sma_20:
        structure_score += 0.25  # Downtrend structure

    # Volatility structure
    atr = df['high'].subtract(df['low']).rolling(14).mean().iloc[-1]
    current_range = df['high'].iloc[-1] - df['low'].iloc[-1]

    if current_range > atr * 1.2:
        structure_score += 0.2  # Expansion

    return min(1.0, structure_score)

def run_working_complex_backtest():
    """Run working complex backtest with manual signal implementation"""
    print("🔮 ETH Working Complex Backtest - Chart Logs 2")
    print("Manual Implementation of All Complex Weights + v1.6.1")
    print("=" * 65)

    # Load data
    filepath = '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv'
    df = pd.read_csv(filepath)
    df.columns = [col.lower().strip() for col in df.columns]
    df['time'] = pd.to_datetime(df['time'], unit='s')
    df.set_index('time', inplace=True)

    if 'buy+sell v' in df.columns:
        df['volume'] = df['buy+sell v']
    else:
        df['volume'] = df['close'] * 1000

    df = df[['open', 'high', 'low', 'close', 'volume']].dropna()

    print(f"✅ Loaded {len(df)} bars ({df.index[0]} to {df.index[-1]})")

    # Configuration
    config = {
        'features': {
            'temporal_fib': True,
            'fib_clusters': True,
            'orderflow_lca': True
        }
    }

    # Complex weight configuration
    signal_weights = {
        'wyckoff': 0.35,
        'm1': 0.25,
        'm2': 0.20,
        'orderflow': 0.15,
        'fib_cluster': 0.05
    }

    quality_floors = {
        'wyckoff': 0.10,
        'm1': 0.15,
        'm2': 0.15,
        'confluence': 0.15,
        'orderflow': 0.15
    }

    entry_threshold = 0.25

    print(f"💰 Starting equity: $10,000.00")
    print(f"🎯 Entry threshold: {entry_threshold}")
    print(f"⚖️ Signal weights: {signal_weights}")
    print(f"🚪 Quality floors: {quality_floors}")

    # Trading variables
    equity = 10000.0
    trades = []
    rejected_signals = []
    last_trade_bar = -999

    print(f"📊 Analyzing {len(df)} bars...")

    # Main backtest loop
    for i in range(100, len(df)):
        current_data = df.iloc[:i+1]

        try:
            # Calculate all complex signals
            wyckoff_score, wyckoff_phase = calculate_wyckoff_score(current_data)
            m1_score = calculate_m1_momentum(current_data)
            m2_score = calculate_m2_structure(current_data)

            # Enhanced orderflow and v1.6.1 features
            orderflow_score = orderflow_lca(current_data.tail(30), config)
            confluence_data = detect_price_time_confluence(current_data, config, i)
            confluence_strength = confluence_data.get('confluence_strength', 0)

            # Calculate fusion score
            fusion_score = (
                wyckoff_score * signal_weights['wyckoff'] +
                m1_score * signal_weights['m1'] +
                m2_score * signal_weights['m2'] +
                orderflow_score * signal_weights['orderflow'] +
                confluence_strength * signal_weights['fib_cluster']
            )

            # Progress reporting
            if i % 200 == 0:
                print(f"  📊 Bar {i}/{len(df)} - Fusion: {fusion_score:.3f} | W:{wyckoff_score:.2f} M1:{m1_score:.2f} M2:{m2_score:.2f} OF:{orderflow_score:.2f}")

            # Entry decision
            if fusion_score >= entry_threshold and i - last_trade_bar > 10:

                # Quality gate checks
                quality_passed = True
                rejection_reasons = []

                if wyckoff_score < quality_floors['wyckoff']:
                    quality_passed = False
                    rejection_reasons.append('wyckoff_floor')

                if m1_score < quality_floors['m1']:
                    quality_passed = False
                    rejection_reasons.append('m1_floor')

                if m2_score < quality_floors['m2']:
                    quality_passed = False
                    rejection_reasons.append('m2_floor')

                if confluence_strength < quality_floors['confluence']:
                    quality_passed = False
                    rejection_reasons.append('confluence_floor')

                if orderflow_score < quality_floors['orderflow']:
                    quality_passed = False
                    rejection_reasons.append('orderflow_floor')

                if quality_passed:

                    # Enhanced analysis for trade record
                    structure_analysis = analyze_market_structure(current_data.tail(30), config)

                    # Oracle whisper system
                    test_scores = {
                        'cluster_tags': confluence_data.get('tags', []),
                        'confluence_strength': confluence_strength,
                        'wyckoff_phase': wyckoff_phase,
                        'wyckoff_score': wyckoff_score,
                        'm1_score': m1_score,
                        'm2_score': m2_score,
                        'cvd_delta': structure_analysis['cvd_analysis']['delta'],
                        'cvd_slope': structure_analysis['cvd_analysis']['slope']
                    }

                    whispers = trigger_whisper(test_scores, phase=wyckoff_phase)

                    # Position sizing based on fusion strength
                    base_risk = 0.015  # 1.5%
                    fusion_multiplier = 1.0 + (fusion_score * 0.5)
                    confluence_multiplier = 1.0 + (confluence_strength * 0.3)

                    position_risk = base_risk * fusion_multiplier * confluence_multiplier
                    position_risk = min(position_risk, 0.03)  # Cap at 3%

                    entry_price = current_data['close'].iloc[-1]
                    position_size = (equity * position_risk) / entry_price

                    # Exit management
                    stop_loss = entry_price * 0.95  # 5% stop
                    take_profit_1 = entry_price * 1.08  # 8% target
                    take_profit_2 = entry_price * 1.16  # 16% target

                    # Trade simulation
                    exit_found = False
                    total_pnl = 0
                    remaining_position = position_size

                    for j in range(1, min(30, len(df) - i)):
                        future_bar = df.iloc[i + j]

                        # Stop loss
                        if future_bar['low'] <= stop_loss:
                            pnl = remaining_position * (stop_loss - entry_price)
                            total_pnl += pnl
                            exit_reason = 'stop_loss'
                            exit_found = True
                            break

                        # First take profit (60% position)
                        elif future_bar['high'] >= take_profit_1 and remaining_position > position_size * 0.3:
                            exit_size = position_size * 0.6
                            pnl = exit_size * (take_profit_1 - entry_price)
                            total_pnl += pnl
                            remaining_position -= exit_size

                        # Second take profit (remaining)
                        elif future_bar['high'] >= take_profit_2:
                            pnl = remaining_position * (take_profit_2 - entry_price)
                            total_pnl += pnl
                            exit_reason = 'take_profit_2'
                            exit_found = True
                            break

                        # Time exit
                        elif j == 29:
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
                            'bars_held': j,

                            # Complex signal scores
                            'fusion_score': fusion_score,
                            'wyckoff_score': wyckoff_score,
                            'wyckoff_phase': wyckoff_phase,
                            'm1_score': m1_score,
                            'm2_score': m2_score,
                            'orderflow_score': orderflow_score,

                            # v1.6.1 features
                            'confluence_strength': confluence_strength,
                            'price_time_confluence': confluence_data.get('confluence_detected', False),
                            'cluster_tags': confluence_data.get('tags', []),
                            'oracle_whispers': len(whispers) if whispers else 0,
                            'oracle_messages': whispers[:2] if whispers else [],

                            # CVD analysis
                            'cvd_delta': test_scores['cvd_delta'],
                            'cvd_slope': test_scores['cvd_slope'],
                            'orderflow_divergence': structure_analysis['orderflow_divergence']['detected'],
                            'bos_detected': structure_analysis['bos_analysis']['detected'],

                            # Risk multipliers
                            'fusion_multiplier': fusion_multiplier,
                            'confluence_multiplier': confluence_multiplier
                        }

                        trades.append(trade_record)
                        last_trade_bar = i

                        print(f"    💼 Trade {len(trades)}: {pnl_pct:.2%} | Fusion: {fusion_score:.3f} | Phase: {wyckoff_phase} | Whispers: {len(whispers) if whispers else 0}")

                else:
                    # Track rejected signals
                    rejected_signals.append({
                        'date': current_data.index[i],
                        'fusion_score': fusion_score,
                        'rejection_reasons': rejection_reasons,
                        'component_scores': {
                            'wyckoff': wyckoff_score,
                            'm1': m1_score,
                            'm2': m2_score,
                            'orderflow': orderflow_score,
                            'confluence': confluence_strength
                        }
                    })

        except Exception as e:
            continue

    # COMPREHENSIVE RESULTS ANALYSIS
    print(f"\n📈 Working Complex Backtest Results:")
    print(f"  💼 Total trades executed: {len(trades)}")
    print(f"  💰 Final equity: ${equity:,.2f}")
    print(f"  🚫 Signals rejected: {len(rejected_signals)}")

    if trades:
        total_return = (equity - 10000) / 10000
        winning_trades = [t for t in trades if t['pnl_pct'] > 0]
        losing_trades = [t for t in trades if t['pnl_pct'] < 0]
        win_rate = len(winning_trades) / len(trades)
        avg_return = np.mean([t['pnl_pct'] for t in trades])

        print(f"\n💹 Performance Metrics:")
        print(f"  📈 Total return: {total_return:.2%}")
        print(f"  🎯 Win rate: {win_rate:.1%} ({len(winning_trades)}/{len(trades)})")
        print(f"  💹 Avg return per trade: {avg_return:.2%}")

        if winning_trades:
            avg_win = np.mean([t['pnl_pct'] for t in winning_trades])
            max_win = max([t['pnl_pct'] for t in winning_trades])
            print(f"  🟢 Avg winning trade: {avg_win:.2%}")
            print(f"  🚀 Best trade: {max_win:.2%}")

        if losing_trades:
            avg_loss = np.mean([t['pnl_pct'] for t in losing_trades])
            max_loss = min([t['pnl_pct'] for t in losing_trades])
            print(f"  🔴 Avg losing trade: {avg_loss:.2%}")
            print(f"  💥 Worst trade: {max_loss:.2%}")

        # Complex signal analysis
        avg_scores = {
            'fusion': np.mean([t['fusion_score'] for t in trades]),
            'wyckoff': np.mean([t['wyckoff_score'] for t in trades]),
            'm1': np.mean([t['m1_score'] for t in trades]),
            'm2': np.mean([t['m2_score'] for t in trades]),
            'orderflow': np.mean([t['orderflow_score'] for t in trades]),
            'confluence': np.mean([t['confluence_strength'] for t in trades])
        }

        print(f"\n🔬 Complex Signal Analysis:")
        print(f"  🎯 Avg Fusion Score: {avg_scores['fusion']:.3f}")
        print(f"  📊 Avg Wyckoff Score: {avg_scores['wyckoff']:.3f}")
        print(f"  ⚡ Avg M1 Score: {avg_scores['m1']:.3f}")
        print(f"  🏗️ Avg M2 Score: {avg_scores['m2']:.3f}")
        print(f"  📈 Avg Orderflow: {avg_scores['orderflow']:.3f}")
        print(f"  ✨ Avg Confluence: {avg_scores['confluence']:.3f}")

        # Wyckoff phase analysis
        phase_trades = {}
        for trade in trades:
            phase = trade['wyckoff_phase']
            if phase not in phase_trades:
                phase_trades[phase] = []
            phase_trades[phase].append(trade['pnl_pct'])

        print(f"\n📊 Wyckoff Phase Analysis:")
        for phase, returns in phase_trades.items():
            avg_return = np.mean(returns)
            win_rate = len([r for r in returns if r > 0]) / len(returns)
            print(f"  Phase {phase}: {len(returns)} trades, {avg_return:.2%} avg, {win_rate:.1%} win rate")

        # v1.6.1 feature performance
        cluster_trades = [t for t in trades if t['price_time_confluence']]
        oracle_trades = [t for t in trades if t['oracle_whispers'] > 0]
        cvd_trades = [t for t in trades if t['orderflow_divergence']]

        print(f"\n🔮 v1.6.1 Feature Performance:")
        print(f"  ✨ Price-time confluence trades: {len(cluster_trades)}")
        print(f"  🧙‍♂️ Oracle whisper trades: {len(oracle_trades)}")
        print(f"  📊 CVD divergence trades: {len(cvd_trades)}")

        if cluster_trades:
            cluster_win_rate = len([t for t in cluster_trades if t['pnl_pct'] > 0]) / len(cluster_trades)
            cluster_avg = np.mean([t['pnl_pct'] for t in cluster_trades])
            avg_cluster_strength = np.mean([t['confluence_strength'] for t in cluster_trades])
            print(f"  🎯 Cluster win rate: {cluster_win_rate:.1%}")
            print(f"  💰 Cluster avg return: {cluster_avg:.2%}")
            print(f"  💪 Avg cluster strength: {avg_cluster_strength:.3f}")

        if oracle_trades:
            oracle_win_rate = len([t for t in oracle_trades if t['pnl_pct'] > 0]) / len(oracle_trades)
            oracle_avg = np.mean([t['pnl_pct'] for t in oracle_trades])
            avg_whispers = np.mean([t['oracle_whispers'] for t in oracle_trades])
            print(f"  🧙‍♂️ Oracle win rate: {oracle_win_rate:.1%}")
            print(f"  💫 Oracle avg return: {oracle_avg:.2%}")
            print(f"  🗣️ Avg whispers per trade: {avg_whispers:.1f}")

        # Risk analysis
        avg_risk = np.mean([t['position_risk'] for t in trades])
        max_risk = max([t['position_risk'] for t in trades])
        avg_fusion_mult = np.mean([t['fusion_multiplier'] for t in trades])
        avg_conf_mult = np.mean([t['confluence_multiplier'] for t in trades])

        print(f"\n⚖️ Risk Management:")
        print(f"  📊 Avg position risk: {avg_risk:.2%}")
        print(f"  🔥 Max position risk: {max_risk:.2%}")
        print(f"  🎯 Avg fusion multiplier: {avg_fusion_mult:.2f}x")
        print(f"  ✨ Avg confluence multiplier: {avg_conf_mult:.2f}x")

        # Exit analysis
        exit_reasons = {}
        for trade in trades:
            reason = trade['exit_reason']
            exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

        print(f"\n🚪 Exit Analysis:")
        for reason, count in exit_reasons.items():
            pct = count / len(trades)
            print(f"  {reason}: {count} trades ({pct:.1%})")

        # Save comprehensive results
        results = {
            'metadata': {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'system': 'Working Complex Weights + v1.6.1',
                'asset': 'ETH-USD',
                'data_source': 'Chart Logs 2',
                'philosophy': 'Price and time symmetry = where structure and vibration align'
            },
            'configuration': {
                'entry_threshold': entry_threshold,
                'signal_weights': signal_weights,
                'quality_floors': quality_floors
            },
            'performance': {
                'starting_equity': 10000,
                'final_equity': equity,
                'total_return': total_return,
                'total_trades': len(trades),
                'win_rate': win_rate,
                'avg_return': avg_return,
                'rejected_signals': len(rejected_signals)
            },
            'signal_analysis': avg_scores,
            'wyckoff_phases': {phase: {'trades': len(returns), 'avg_return': np.mean(returns), 'win_rate': len([r for r in returns if r > 0]) / len(returns)} for phase, returns in phase_trades.items()},
            'v161_features': {
                'cluster_trades': len(cluster_trades),
                'oracle_trades': len(oracle_trades),
                'cvd_trades': len(cvd_trades)
            },
            'trades': trades[:20]  # First 20 trades
        }

        filename = f"eth_working_complex_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\n📄 Comprehensive results saved to: {filename}")
        print(f"\n✨ Working Complex System Success:")
        print(f"   'Price and time symmetry = where structure and vibration align'")
        print(f"   🔬 {len(trades)} trades using full complex fusion scoring")
        print(f"   🧙‍♂️ {len(oracle_trades)} Oracle wisdom events triggered")
        print(f"   🎯 {len(cluster_trades)} Price-time confluence alignments")
        print(f"   📊 Complex quality gates filtered {len(rejected_signals)} signals")

    else:
        print("  ⚠️ No trades executed")

        if rejected_signals:
            print(f"\n🚫 Rejection Analysis:")
            reasons = {}
            for sig in rejected_signals:
                for reason in sig['rejection_reasons']:
                    reasons[reason] = reasons.get(reason, 0) + 1

            for reason, count in sorted(reasons.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count} rejections")

if __name__ == "__main__":
    run_working_complex_backtest()