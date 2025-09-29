#!/usr/bin/env python3
"""
ETH v1.6.2 Comprehensive Backtest - Bull Machine
Features: PO3 + Bojan microstructure + Wyckoff M1/M2 + Fibonacci clusters
Data: Chart Logs 2 multi-timeframe analysis

Key Enhancements in v1.6.2:
- Bojan wick magnets (unfinished business detection)
- Trap reset patterns (sweep + flip + large body commitment)
- pHOB zones (hidden order blocks behind FVGs)
- Fibonacci .705/.786 prime zone confluences
- Enhanced PO3 with Bojan microstructure confluence
- Anti-double-counting logic for overlapping signals
"""

import sys
import os
import warnings
from datetime import datetime
import pandas as pd
import numpy as np
import json

# Add bull_machine to path
sys.path.append('.')

from bull_machine.modules.orderflow.lca import analyze_market_structure, orderflow_lca
from bull_machine.strategy.hidden_fibs import detect_price_time_confluence
from bull_machine.strategy.po3_detection import detect_po3_with_bojan_confluence
from bull_machine.modules.bojan.bojan import compute_bojan_score
from bull_machine.oracle import trigger_whisper
from bull_machine.core.config_loader import load_config

warnings.filterwarnings('ignore')


def load_chart_logs_data():
    """Load Chart Logs 2 data files for ETH multi-timeframe analysis."""
    print("Loading Chart Logs 2 data...")

    # File paths as specified by user
    files = {
        '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 1D_64942.csv',
        '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 240_1d04a.csv',
        '1H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_ETHUSD, 60_2f4ab.csv'
    }

    data = {}

    for timeframe, filepath in files.items():
        if os.path.exists(filepath):
            print(f"Loading {timeframe} data from {filepath}")
            df = pd.read_csv(filepath)

            # Handle Chart Logs 2 format
            if 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            elif 'Date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['Date'])

            # Standardize column names
            df.columns = df.columns.str.lower()
            if 'buy+sell v' in df.columns:
                df['volume'] = df['buy+sell v']

            df = df.set_index('timestamp').sort_index()
            data[timeframe] = df
            print(f"Loaded {len(df)} {timeframe} bars from {df.index[0]} to {df.index[-1]}")
        else:
            print(f"Warning: File not found: {filepath}")

    return data


def calculate_wyckoff_score(df):
    """Calculate Wyckoff phase scoring with v1.6.2 enhancements"""
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

    # Phase detection logic
    score = 0
    phase = 'N'

    if vol_ratio > 1.3 and abs(trend_strength) > 0.02:
        if trend_strength > 0 and range_pos > 0.6:
            score = 0.45  # Phase C/D accumulation
            phase = 'C'
        elif trend_strength < 0 and range_pos < 0.4:
            score = 0.40  # Phase A distribution
            phase = 'A'

    return score, phase


def calculate_m1_score(df):
    """Calculate M1 momentum scoring"""
    if len(df) < 20:
        return 0

    # RSI momentum
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]

    # Price momentum
    close = df['close'].iloc[-1]
    sma_10 = df['close'].rolling(10).mean().iloc[-1]
    momentum = (close - sma_10) / sma_10 if sma_10 > 0 else 0

    score = 0
    if 30 < current_rsi < 70 and abs(momentum) > 0.01:
        score = min(0.35 + abs(momentum) * 5, 0.45)

    return score


def calculate_m2_score(df):
    """Calculate M2 structure scoring"""
    if len(df) < 30:
        return 0

    # Support/Resistance levels
    highs = df['high'].rolling(5).max()
    lows = df['low'].rolling(5).min()

    current_price = df['close'].iloc[-1]
    recent_high = highs.iloc[-10:].max()
    recent_low = lows.iloc[-10:].min()

    # Structure break detection
    structure_strength = 0
    if abs(current_price - recent_high) / recent_high < 0.02:
        structure_strength = 0.35  # Near resistance
    elif abs(current_price - recent_low) / recent_low < 0.02:
        structure_strength = 0.35  # Near support

    return structure_strength


def calculate_enhanced_fusion_score(df, i):
    """
    v1.6.2 Enhanced fusion scoring with PO3 + Bojan microstructure integration

    Features:
    - Wyckoff M1/M2 (60% weight)
    - PO3 + Bojan confluence (25% weight)
    - Fibonacci clusters (10% weight)
    - Orderflow (5% weight)
    """
    if i < 100:  # Need sufficient history
        return 0, {}

    window_data = df.iloc[max(0, i-100):i+1]

    # Core components
    wyckoff_score, wyckoff_phase = calculate_wyckoff_score(window_data)
    m1_score = calculate_m1_score(window_data)
    m2_score = calculate_m2_score(window_data)

    # v1.6.2 Enhancement: PO3 + Bojan Integration
    po3_bojan_score = 0.0
    po3_bojan_tags = []

    try:
        # Define IRH/IRL from recent range
        range_data = window_data.iloc[-25:-5] if len(window_data) > 25 else window_data.iloc[:-5] if len(window_data) > 5 else window_data
        if len(range_data) >= 5:
            irh = range_data['high'].max()
            irl = range_data['low'].min()

            # Enhanced PO3 with Bojan confluence
            po3_result = detect_po3_with_bojan_confluence(window_data.tail(20), irh, irl, vol_spike_threshold=1.4)
            if po3_result and po3_result['strength'] > 0.5:
                po3_bojan_score += po3_result['strength'] * 0.15

                # Bojan confluence bonuses
                if po3_result.get('bojan_confluence', False):
                    confluence_boost = min(po3_result.get('bojan_score', 0) * 0.10, 0.15)
                    po3_bojan_score += confluence_boost
                    po3_bojan_tags.extend(po3_result.get('confluence_tags', []))

            # Standalone Bojan analysis
            bojan_analysis = compute_bojan_score(window_data.tail(20))
            bojan_score = bojan_analysis.get('bojan_score', 0.0)

            if bojan_score > 0.3:
                # Anti-double-counting logic
                max_bojan_boost = 0.08
                if po3_result and po3_result.get('bojan_confluence', False):
                    max_bojan_boost = 0.04  # Reduce if already counted in PO3

                standalone_bojan = min(bojan_score * 0.06, max_bojan_boost)
                po3_bojan_score += standalone_bojan

                # Tag significant Bojan signals
                bojan_signals = bojan_analysis.get('signals', {})
                if bojan_signals.get('wick_magnet', {}).get('is_magnet', False):
                    po3_bojan_tags.append('bojan_wick_magnet')
                if bojan_signals.get('trap_reset', {}).get('is_trap_reset', False):
                    po3_bojan_tags.append('bojan_trap_reset')

    except Exception as e:
        print(f"Warning: PO3/Bojan analysis failed: {e}")
        po3_bojan_score = 0.0

    # Orderflow component (simplified)
    orderflow_score = 0
    try:
        if len(window_data) >= 20:
            vol_profile = window_data['volume'].rolling(10).mean().iloc[-1]
            vol_current = window_data['volume'].iloc[-1]
            if vol_profile > 0:
                orderflow_score = min((vol_current / vol_profile - 1) * 0.15, 0.10)
    except:
        orderflow_score = 0

    # Fibonacci clusters
    fib_score = 0
    try:
        confluence_result = detect_price_time_confluence(window_data)
        if confluence_result and confluence_result.get('strength', 0) > 0.3:
            fib_score = min(confluence_result['strength'] * 0.08, 0.05)
    except:
        fib_score = 0

    # v1.6.2 Enhanced weighting system
    components = {
        'wyckoff': wyckoff_score * 0.35,          # 35%
        'm1': m1_score * 0.25,                   # 25%
        'm2': m2_score * 0.20,                   # 20%
        'po3_bojan': po3_bojan_score,            # Variable weight (10-25%)
        'orderflow': orderflow_score * 0.05,     # 5%
        'fib_clusters': fib_score * 0.05,        # 5%
    }

    total_score = sum(components.values())

    # v1.6.2 Confluence bonuses
    if po3_bojan_tags:
        components['confluence_tags'] = po3_bojan_tags

        # Specific confluence bonuses
        if 'bojan_trap_reset' in po3_bojan_tags and wyckoff_phase in ['C', 'D']:
            total_score += 0.08  # Trap reset in Wyckoff manipulation phase

        if 'bojan_fib_prime' in po3_bojan_tags and fib_score > 0.02:
            total_score += 0.12  # Bojan + Fibonacci prime zone confluence

    return total_score, components


def run_v162_comprehensive_backtest():
    """Run v1.6.2 comprehensive backtest with PO3 + Bojan enhancements"""
    print("=== ETH v1.6.2 Comprehensive Backtest - Bull Machine ===")
    print("Features: PO3 + Bojan microstructure + Wyckoff M1/M2 + Fibonacci clusters")
    print("Data: Chart Logs 2 multi-timeframe analysis")

    # Load data
    data = load_chart_logs_data()

    if not data or '1D' not in data:
        print("ERROR: Missing required 1D data")
        return

    df = data['1D'].copy()
    print(f"\\nAnalyzing {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # v1.6.2 Enhanced backtest settings
    entry_threshold = 0.48  # Slightly higher due to enhanced signals
    cooldown_bars = 120     # 5 days (reduced from 7 due to better quality)
    risk_per_trade = 0.005  # 0.5%
    initial_capital = 10000

    print(f"Entry threshold: {entry_threshold}")
    print(f"PO3 + Bojan enhanced fusion scoring enabled")
    print(f"Risk per trade: {risk_per_trade*100}%")
    print(f"Cooldown: {cooldown_bars} bars")

    # Run enhanced backtest
    print("\\nRunning v1.6.2 enhanced backtest...")

    trades = []
    in_trade = False
    last_trade_bar = -999
    capital = initial_capital

    for i in range(100, len(df) - 1):
        current_bar = df.iloc[i]

        if in_trade:
            continue

        if i - last_trade_bar < cooldown_bars:
            continue

        # Calculate v1.6.2 enhanced fusion score
        score, components = calculate_enhanced_fusion_score(df, i)

        if score >= entry_threshold:
            # Entry conditions met
            entry_price = current_bar['close']
            entry_time = current_bar.name

            # Enhanced position sizing
            position_size = capital * risk_per_trade / entry_price

            # Enhanced exit logic (5-8 bars based on signal strength)
            exit_bars = 5 if score < 0.60 else 8
            exit_bar_idx = min(i + exit_bars, len(df) - 1)
            exit_price = df.iloc[exit_bar_idx]['close']

            # Calculate PnL
            pnl = position_size * (exit_price - entry_price)
            capital += pnl

            # Generate enhanced tags for analysis
            tags = []
            if 'confluence_tags' in components:
                tags.extend(components['confluence_tags'])
            if components.get('po3_bojan', 0) > 0.10:
                tags.append('po3_bojan_signal')
            if components.get('wyckoff', 0) > 0.30:
                tags.append('wyckoff_signal')
            if components.get('fib_clusters', 0) > 0.02:
                tags.append('fib_confluence')

            trade = {
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'score': score,
                'components': components,
                'tags': tags,
                'side': 'long',
                'exit_bars': exit_bars
            }

            trades.append(trade)
            last_trade_bar = i

    results = {'trades': trades, 'final_capital': capital}

    # Display enhanced results
    print("\\n" + "="*70)
    print("V1.6.2 ENHANCED BACKTEST RESULTS")
    print("="*70)

    if results['trades']:
        total_trades = len(results['trades'])
        winning_trades = len([t for t in results['trades'] if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        total_pnl = sum([t['pnl'] for t in results['trades']])
        total_return_pct = (total_pnl / initial_capital) * 100

        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Total Return: {total_return_pct:.2f}%")

        if total_trades > 0:
            avg_pnl = total_pnl / total_trades
            print(f"Average PnL per trade: ${avg_pnl:.2f}")

        # v1.6.2 Enhanced analysis
        po3_bojan_trades = [t for t in results['trades'] if 'po3_bojan_signal' in t.get('tags', [])]
        if po3_bojan_trades:
            po3_pnl = sum([t['pnl'] for t in po3_bojan_trades])
            po3_win_rate = (len([t for t in po3_bojan_trades if t['pnl'] > 0]) / len(po3_bojan_trades)) * 100
            print(f"\\nPO3 + Bojan enhanced trades: {len(po3_bojan_trades)}")
            print(f"PO3 + Bojan Win Rate: {po3_win_rate:.1f}%")
            print(f"PO3 + Bojan Total PnL: ${po3_pnl:.2f}")

        # Bojan-specific analysis
        bojan_trap_trades = [t for t in results['trades'] if 'bojan_trap_reset' in t.get('tags', [])]
        bojan_wick_trades = [t for t in results['trades'] if 'bojan_wick_magnet' in t.get('tags', [])]

        if bojan_trap_trades:
            print(f"\\nBojan trap reset trades: {len(bojan_trap_trades)}")
            trap_pnl = sum([t['pnl'] for t in bojan_trap_trades])
            print(f"Trap reset total PnL: ${trap_pnl:.2f}")

        if bojan_wick_trades:
            print(f"Bojan wick magnet trades: {len(bojan_wick_trades)}")
            wick_pnl = sum([t['pnl'] for t in bojan_wick_trades])
            print(f"Wick magnet total PnL: ${wick_pnl:.2f}")

        # Show sample trades with enhanced details
        print("\\nSample v1.6.2 enhanced trades:")
        for i, trade in enumerate(results['trades'][:7]):
            entry_date = trade['entry_time'].strftime('%Y-%m-%d') if hasattr(trade['entry_time'], 'strftime') else trade['entry_time']
            print(f"  {i+1}. {entry_date}: ${trade['pnl']:.2f} (score: {trade['score']:.3f}, exit: {trade['exit_bars']} bars)")
            if trade['tags']:
                print(f"     Tags: {', '.join(trade['tags'])}")
            components = trade['components']
            print(f"     PO3+Bojan: {components.get('po3_bojan', 0):.3f}, Wyckoff: {components.get('wyckoff', 0):.3f}")

    else:
        print("No trades generated")
        print("Possible adjustments needed:")
        print("- Lower entry threshold from 0.48")
        print("- Adjust PO3/Bojan sensitivity")
        print("- Check volume thresholds")

    print("\\n" + "="*70)

    # v1.6.2 Enhanced PO3 + Bojan pattern analysis
    print("\\nv1.6.2 PO3 + Bojan Pattern Analysis (Recent 100 bars):")
    recent_data = df.tail(100)

    pattern_detections = []
    for i in range(20, len(recent_data)):
        lookback_data = recent_data.iloc[i-20:i+1]

        # IRH/IRL definition
        irh = lookback_data['high'].iloc[:-5].max()
        irl = lookback_data['low'].iloc[:-5].min()

        # PO3 + Bojan confluence detection
        try:
            po3_result = detect_po3_with_bojan_confluence(lookback_data, irh, irl, vol_spike_threshold=1.4)
            if po3_result:
                pattern_detections.append({
                    'date': recent_data.index[i],
                    'po3_type': po3_result['po3_type'],
                    'strength': po3_result['strength'],
                    'bojan_confluence': po3_result.get('bojan_confluence', False),
                    'confluence_tags': po3_result.get('confluence_tags', [])
                })
        except Exception as e:
            continue

    if pattern_detections:
        print(f"Found {len(pattern_detections)} enhanced PO3 + Bojan patterns:")
        for detection in pattern_detections[-5:]:  # Show last 5
            date_str = detection['date'].strftime('%Y-%m-%d') if hasattr(detection['date'], 'strftime') else str(detection['date'])
            bojan_status = "✓ Bojan" if detection['bojan_confluence'] else "○ PO3 only"
            print(f"  {date_str}: {detection['po3_type']} (str: {detection['strength']:.2f}) {bojan_status}")
            if detection['confluence_tags']:
                print(f"    └─ {', '.join(detection['confluence_tags'])}")
    else:
        print("No enhanced PO3 + Bojan patterns detected in recent data")

    return results


if __name__ == '__main__':
    results = run_v162_comprehensive_backtest()