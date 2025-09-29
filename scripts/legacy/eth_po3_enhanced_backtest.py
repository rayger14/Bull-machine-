#!/usr/bin/env python3
"""
ETH PO3 Enhanced Backtest - Bull Machine v1.6.1
Comprehensive backtest with PO3 detection, complex fusion scoring, and Chart Logs 2 data
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
from bull_machine.strategy.po3_detection import detect_po3
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
                # Unix timestamp format
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            elif 'Date' in df.columns:
                # Date string format
                df['timestamp'] = pd.to_datetime(df['Date'])

            # Standardize column names
            df.columns = df.columns.str.lower()
            if 'buy+sell v' in df.columns:
                df['volume'] = df['buy+sell v']

            # Ensure required columns exist
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            for col in required_cols:
                if col not in df.columns:
                    print(f"Warning: Missing {col} column in {timeframe} data")

            df = df.set_index('timestamp').sort_index()
            data[timeframe] = df
            print(f"Loaded {len(df)} {timeframe} bars from {df.index[0]} to {df.index[-1]}")
        else:
            print(f"Warning: File not found: {filepath}")

    return data

def create_enhanced_config():
    """Create enhanced config with PO3 and complex fusion scoring."""
    config = {
        'entry_threshold': 0.44,
        'quality_floors': {
            'wyckoff': 0.25,
            'm1': 0.30,
            'm2': 0.30,
            'liquidity': 0.25,
            'structure': 0.25,
            'momentum': 0.27,
            'volume': 0.25,
            'context': 0.25,
            'mtf': 0.27,
            'fib_retracement': 0.25,
            'fib_extension': 0.25
        },
        'features': {
            'mtf_dl2': True,
            'six_candle_leg': True,
            'orderflow_lca': False,
            'negative_vip': False,
            'atr_exits': True,
            'atr_sizing': True,
            'regime_filter': True,
            'ensemble_htf_bias': True,
            'wyckoff_phase': True,
            'wyckoff_m1m2': True,
            'liquidity_sweep': False,
            'order_blocks': True,
            'wick_magnet': True,
            'hidden_fibs': True,
            'temporal_fib': True,
            'fib_clusters': True,
            'po3': True,  # Enable PO3 detection
            'live_data': False,
            'use_asset_profiles': True
        },
        'cooldown_bars': 168,
        'timeframe': '1D',
        'risk': {
            'risk_pct': 0.005,
            'atr_window': 14,
            'sl_atr': 1.8,
            'tp_atr': 3.0,
            'trail_atr': 1.2,
            'profit_ladders': [
                {'ratio': 1.5, 'percent': 0.25},
                {'ratio': 2.5, 'percent': 0.50},
                {'ratio': 4.0, 'percent': 0.25}
            ]
        },
        'regime': {
            'vol_ratio_min': 1.0,
            'atr_pct_max': 0.08
        },
        'ensemble': {
            'enabled': True,
            'min_consensus': 2,
            'consensus_penalty': 0.02,
            'rolling_k': 4,
            'rolling_n': 5,
            'lead_lag_window': 3,
            'dynamic_thresholds': True,
            'confluence_mode': 'adaptive',
            'm1m2_backoff_bars': 30,
            'm1m2_epsilon': 0.05,
            'solo_fib_min_entry': 0.44,
            'vol_override_atr_pct': 0.05
        },
        'weights': {
            'temporal': 0.10,
            'po3': 0.10  # PO3 weight in fusion scoring
        },
        'fib': {
            'tolerance': 0.02
        },
        'temporal': {
            'fib_nums': [21, 34, 55, 89, 144],
            'tolerance_bars': 3,
            'pivot_window': 5
        },
        'max_bars_held': 8,
        'profile_name': 'ETH_PO3_Enhanced_v1.6.1',
        'version': '1.6.1',
        'description': 'ETH with PO3 detection, complex fusion scoring, and Chart Logs 2 data'
    }
    return config

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

def calculate_po3_score(df, lookback=20):
    """Calculate PO3 pattern scoring with enhanced detection"""
    if len(df) < lookback:
        return 0, []

    # Define IRH/IRL from recent range (excluding last few bars)
    range_data = df.iloc[-(lookback+5):-5] if len(df) > lookback+5 else df.iloc[:-5]
    if len(range_data) < 10:
        return 0, []

    irh = range_data['high'].max()
    irl = range_data['low'].min()

    # Test with multiple window sizes for robust detection
    po3_signals = []
    max_score = 0

    for window in [10, 15, 20]:
        if len(df) >= window:
            test_data = df.tail(window)
            po3_result = detect_po3(test_data, irh, irl, vol_spike_threshold=1.4)

            if po3_result:
                po3_signals.append(po3_result)
                max_score = max(max_score, po3_result['strength'])

    # Additional boost for multiple timeframe confluence
    if len(po3_signals) > 1:
        max_score += 0.05

    return max_score, po3_signals

def calculate_fusion_score(df, i):
    """Calculate comprehensive fusion score with PO3 integration"""
    if i < 100:  # Need sufficient history
        return 0, {}

    window_data = df.iloc[max(0, i-100):i+1]

    # Core components
    wyckoff_score, wyckoff_phase = calculate_wyckoff_score(window_data)
    m1_score = calculate_m1_score(window_data)
    m2_score = calculate_m2_score(window_data)

    # PO3 Enhancement
    po3_score, po3_signals = calculate_po3_score(window_data)

    # Orderflow component (simplified)
    orderflow_score = 0
    try:
        if len(window_data) >= 20:
            vol_profile = window_data['volume'].rolling(10).mean().iloc[-1]
            vol_current = window_data['volume'].iloc[-1]
            if vol_profile > 0:
                orderflow_score = min((vol_current / vol_profile - 1) * 0.2, 0.15)
    except:
        orderflow_score = 0

    # Fibonacci clusters (simplified)
    fib_score = 0
    try:
        confluence_result = detect_price_time_confluence(window_data)
        if confluence_result and confluence_result.get('strength', 0) > 0.3:
            fib_score = min(confluence_result['strength'] * 0.1, 0.05)
    except:
        fib_score = 0

    # Apply complex weighting system
    components = {
        'wyckoff': wyckoff_score * 0.35,      # 35%
        'm1': m1_score * 0.25,               # 25%
        'm2': m2_score * 0.20,               # 20%
        'orderflow': orderflow_score * 0.15,  # 15%
        'fib_clusters': fib_score * 0.05,     # 5%
        'po3': po3_score * 0.10               # 10% PO3 boost
    }

    total_score = sum(components.values())

    # PO3 confluence boost
    if po3_signals:
        po3_types = [signal['po3_type'] for signal in po3_signals]
        confluence_tags = []

        # Check for PO3 + Fibonacci alignment
        if fib_score > 0.02 and po3_score > 0.5:
            total_score += 0.15  # Significant boost for PO3 + Fib confluence
            confluence_tags.append('po3_fib_confluence')

        # Check for PO3 + Wyckoff phase alignment
        if wyckoff_phase in ['C', 'D'] and po3_score > 0.5:
            total_score += 0.10  # Boost for PO3 + Wyckoff phase
            confluence_tags.append('po3_wyckoff_confluence')

        components['confluence_tags'] = confluence_tags

    return total_score, components

def run_po3_enhanced_backtest():
    """Run comprehensive backtest with PO3 enhancements."""
    print("=== ETH PO3 Enhanced Backtest - Bull Machine v1.6.1 ===")
    print("Features: PO3 detection, Wyckoff M1/M2, Fibonacci clusters, Complex fusion scoring")
    print("Data: Chart Logs 2 multi-timeframe (1D/4H/1H)")

    # Load data
    data = load_chart_logs_data()

    if not data:
        print("ERROR: No data loaded. Check file paths.")
        return

    # Use 1D data as primary timeframe
    if '1D' not in data:
        print("ERROR: Missing 1D data for primary analysis")
        return

    df = data['1D'].copy()
    print(f"\\nAnalyzing {len(df)} bars from {df.index[0]} to {df.index[-1]}")

    # Enhanced backtest settings
    entry_threshold = 0.44
    cooldown_bars = 168  # 1 week
    risk_per_trade = 0.005  # 0.5%
    initial_capital = 10000

    print(f"Entry threshold: {entry_threshold}")
    print(f"PO3 enhanced fusion scoring enabled")
    print(f"Risk per trade: {risk_per_trade*100}%")

    # Run enhanced backtest
    print("\\nRunning PO3 enhanced backtest...")

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

        # Calculate enhanced fusion score with PO3
        score, components = calculate_fusion_score(df, i)

        if score >= entry_threshold:
            # Entry conditions met
            entry_price = current_bar['close']
            entry_time = current_bar.name

            # Determine position size
            position_size = capital * risk_per_trade / entry_price

            # Simple exit after 5 bars (can be enhanced)
            exit_bar_idx = min(i + 5, len(df) - 1)
            exit_price = df.iloc[exit_bar_idx]['close']

            # Calculate PnL
            pnl = position_size * (exit_price - entry_price)
            capital += pnl

            # Generate tags for analysis
            tags = []
            if 'confluence_tags' in components:
                tags.extend(components['confluence_tags'])
            if components.get('po3', 0) > 0.3:
                tags.append('po3_signal')
            if components.get('wyckoff', 0) > 0.3:
                tags.append('wyckoff_signal')

            trade = {
                'entry_time': entry_time,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'score': score,
                'components': components,
                'tags': tags,
                'side': 'long'
            }

            trades.append(trade)
            last_trade_bar = i

    results = {'trades': trades, 'final_capital': capital}

    # Display results
    print("\\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    if results['trades']:
        total_trades = len(results['trades'])
        winning_trades = len([t for t in results['trades'] if t['pnl'] > 0])
        losing_trades = total_trades - winning_trades
        win_rate = (winning_trades / total_trades) * 100 if total_trades > 0 else 0

        total_pnl = sum([t['pnl'] for t in results['trades']])
        total_return_pct = (total_pnl / 10000) * 100  # Assuming $10k starting capital

        print(f"Total Trades: {total_trades}")
        print(f"Winning Trades: {winning_trades}")
        print(f"Losing Trades: {losing_trades}")
        print(f"Win Rate: {win_rate:.1f}%")
        print(f"Total PnL: ${total_pnl:.2f}")
        print(f"Total Return: {total_return_pct:.2f}%")

        if total_trades > 0:
            avg_pnl = total_pnl / total_trades
            print(f"Average PnL per trade: ${avg_pnl:.2f}")

        # PO3-specific analysis
        po3_trades = [t for t in results['trades'] if 'po3' in t.get('tags', [])]
        if po3_trades:
            po3_pnl = sum([t['pnl'] for t in po3_trades])
            po3_win_rate = (len([t for t in po3_trades if t['pnl'] > 0]) / len(po3_trades)) * 100
            print(f"\\nPO3-enhanced trades: {len(po3_trades)}")
            print(f"PO3 Win Rate: {po3_win_rate:.1f}%")
            print(f"PO3 Total PnL: ${po3_pnl:.2f}")

        # Show sample trades
        print("\\nSample trades:")
        for i, trade in enumerate(results['trades'][:5]):
            entry_date = trade['entry_time'].strftime('%Y-%m-%d') if hasattr(trade['entry_time'], 'strftime') else trade['entry_time']
            print(f"  {i+1}. {entry_date}: ${trade['pnl']:.2f} ({trade.get('side', 'unknown')})")
            if 'scores' in trade:
                print(f"     Entry score: {trade['scores'].get('total_score', 'N/A'):.3f}")
            if 'tags' in trade:
                print(f"     Tags: {', '.join(trade['tags'])}")

    else:
        print("No trades generated")
        print("Possible issues:")
        print("- Entry threshold too high")
        print("- Quality floors too strict")
        print("- Insufficient PO3 patterns detected")
        print("- Volume/regime filters too restrictive")

    print("\\n" + "="*60)

    # Test PO3 detection on recent data
    print("\\nTesting PO3 detection on recent data...")
    recent_data = df.tail(100)  # Last 100 bars

    po3_detections = []
    for i in range(20, len(recent_data)):
        lookback_data = recent_data.iloc[i-20:i+1]

        # Define IRH/IRL from recent range
        irh = lookback_data['high'].iloc[:-5].max()  # Exclude last 5 bars
        irl = lookback_data['low'].iloc[:-5].min()

        po3_result = detect_po3(lookback_data, irh, irl, vol_spike_threshold=1.4)
        if po3_result:
            po3_detections.append({
                'date': recent_data.index[i],
                'po3_type': po3_result['po3_type'],
                'strength': po3_result['strength'],
                'irh': irh,
                'irl': irl
            })

    if po3_detections:
        print(f"Found {len(po3_detections)} PO3 patterns in recent data:")
        for detection in po3_detections[-5:]:  # Show last 5
            date_str = detection['date'].strftime('%Y-%m-%d') if hasattr(detection['date'], 'strftime') else str(detection['date'])
            print(f"  {date_str}: {detection['po3_type']} (strength: {detection['strength']:.2f})")
    else:
        print("No PO3 patterns detected in recent data")

    return results

if __name__ == '__main__':
    results = run_po3_enhanced_backtest()