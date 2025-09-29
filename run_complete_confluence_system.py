#!/usr/bin/env python3
"""
Bull Machine v1.6.2 - COMPLETE 5-Domain Confluence System
The definitive test of ALL confluence layers working as a stacking filter

🔹 Bull Machine Confluence Domains:
1. Wyckoff & Structural Layer (market psychology backbone)
2. Liquidity Layer (MM engineering detection)
3. Momentum & Volume Layer (strength validation)
4. Temporal & Fibonacci Layer (time/vibration context)
5. Fusion & Psychological Layer (high-confidence filter)

Only trades when story + liquidity + momentum + time + alignment all sing the same note.
"""

import sys
import os
import json
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import math

sys.path.append('.')
warnings.filterwarnings('ignore')


def load_multi_timeframe_data(asset, data_paths):
    """Load complete multi-timeframe dataset"""
    print(f"\n=== Loading {asset} Complete MTF Dataset ===")

    data = {}
    for timeframe, filepath in data_paths.items():
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)

            # Handle Chart Logs 2 format
            if 'time' in df.columns:
                df['timestamp'] = pd.to_datetime(df['time'], unit='s')
            elif 'Date' in df.columns:
                df['timestamp'] = pd.to_datetime(df['Date'])

            df.columns = df.columns.str.lower()
            if 'buy+sell v' in df.columns:
                df['volume'] = df['buy+sell v']

            # 2-year period for SOL/XRP analysis
            df = df.set_index('timestamp').sort_index()
            start_date = '2023-01-01'  # 2-year period
            end_date = '2025-01-01'

            if len(df) > 0:
                actual_start = max(df.index[0], pd.Timestamp(start_date))
                actual_end = min(df.index[-1], pd.Timestamp(end_date))
                df = df[actual_start:actual_end]

            # Only process if we have data after filtering
            if len(df) > 0:
                # Add comprehensive technical analysis
                df = add_complete_technical_suite(df, timeframe)
                data[timeframe] = df
                print(f"  {timeframe}: {len(df)} bars from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
            else:
                print(f"  {timeframe}: No data in specified date range")
                data[timeframe] = None

    return data


def add_complete_technical_suite(df, timeframe):
    """Complete technical analysis suite for all confluence layers"""

    # === CORE PRICE ACTION ===
    df['sma_20'] = df['close'].rolling(20).mean()
    df['sma_50'] = df['close'].rolling(50).mean()
    df['sma_200'] = df['close'].rolling(200).mean()
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()

    # === VOLUME ANALYSIS ===
    df['vol_sma_20'] = df['volume'].rolling(20).mean()
    df['vol_ratio'] = df['volume'] / df['vol_sma_20']
    df['vol_spike'] = df['vol_ratio'] > 2.0
    df['vol_absorption'] = (df['vol_ratio'] > 1.5) & (df['vol_ratio'] <= 2.0)

    # === ATR & VOLATILITY ===
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = np.maximum(high_low, np.maximum(high_close, low_close))
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close']

    # === MOMENTUM INDICATORS ===
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['macd'] = df['ema_12'] - df['ema_26']
    df['macd_signal'] = df['macd'].ewm(span=9).mean()
    df['momentum_1'] = df['close'].pct_change(1)
    df['momentum_5'] = df['close'].pct_change(5)
    df['momentum_20'] = df['close'].pct_change(20)

    # === RANGE & STRUCTURE ANALYSIS ===
    df['hh_20'] = df['high'].rolling(20).max()
    df['ll_20'] = df['low'].rolling(20).min()
    df['range_position'] = (df['close'] - df['ll_20']) / (df['hh_20'] - df['ll_20'])
    df['range_position'] = df['range_position'].fillna(0.5)

    # === WYCKOFF SPECIFIC ===
    df['higher_high'] = df['high'] > df['high'].shift(1)
    df['lower_low'] = df['low'] < df['low'].shift(1)
    df['break_of_structure'] = df['close'] > df['hh_20'].shift(1)
    df['change_of_character'] = (df['close'] < df['ll_20'].shift(1)) & (df['close'].shift(1) >= df['ll_20'].shift(2))

    # === LIQUIDITY CONCEPTS ===
    # Fair Value Gaps (simplified)
    df['gap_up'] = (df['low'] > df['high'].shift(2)) & pd.notna(df['high'].shift(2))
    df['gap_down'] = (df['high'] < df['low'].shift(2)) & pd.notna(df['low'].shift(2))
    df['has_gap'] = df['gap_up'] | df['gap_down']

    # Wick Analysis (Bojan style)
    df['body_size'] = np.abs(df['close'] - df['open'])
    df['upper_wick'] = df['high'] - np.maximum(df['open'], df['close'])
    df['lower_wick'] = np.minimum(df['open'], df['close']) - df['low']
    df['total_wick'] = df['upper_wick'] + df['lower_wick']
    df['wick_dominance'] = df['total_wick'] / (df['high'] - df['low'])
    df['wick_dominance'] = df['wick_dominance'].fillna(0)

    # === FIBONACCI ZONES ===
    # Calculate Fibonacci retracement levels from recent swings
    swing_window = 50 if timeframe == '1D' else 20
    df['swing_high'] = df['high'].rolling(swing_window).max()
    df['swing_low'] = df['low'].rolling(swing_window).min()
    df['fib_range'] = df['swing_high'] - df['swing_low']

    # Key Fib levels
    df['fib_382'] = df['swing_low'] + 0.382 * df['fib_range']
    df['fib_500'] = df['swing_low'] + 0.500 * df['fib_range']
    df['fib_618'] = df['swing_low'] + 0.618 * df['fib_range']
    df['fib_786'] = df['swing_low'] + 0.786 * df['fib_range']

    # Distance to key Fib levels
    df['dist_to_382'] = np.abs(df['close'] - df['fib_382']) / df['close']
    df['dist_to_618'] = np.abs(df['close'] - df['fib_618']) / df['close']
    df['near_fib'] = (df['dist_to_382'] < 0.02) | (df['dist_to_618'] < 0.02)

    # === TEMPORAL ANALYSIS ===
    # Fibonacci time sequences (only if we have data)
    if len(df) > 0:
        fib_numbers = [21, 34, 55, 89, 144]
        for fib_num in fib_numbers:
            df[f'fib_time_{fib_num}'] = (df.index - df.index[0]).days % fib_num == 0

        # Sacred geometry (simplified)
        df['cycle_360'] = (df.index - df.index[0]).days % 360 == 0
    else:
        # Handle empty dataframe
        fib_numbers = [21, 34, 55, 89, 144]
        for fib_num in fib_numbers:
            df[f'fib_time_{fib_num}'] = False
        df['cycle_360'] = False

    return df


def calculate_rsi(prices, window=14):
    """Calculate RSI"""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def sync_timeframes_for_confluence(data, target_time):
    """Synchronize all timeframes for confluence analysis"""
    synced = {}

    for tf, df in data.items():
        if df is not None and len(df) > 0:
            # Get the most recent data point at or before target time
            mask = df.index <= target_time
            if mask.any():
                synced[tf] = df.loc[df.index[mask][-1]]
            else:
                synced[tf] = None

    return synced


# ========================================================================
# CONFLUENCE DOMAIN 1: WYCKOFF & STRUCTURAL LAYER
# ========================================================================

def analyze_wyckoff_structural_domain(synced_data, lookback_data):
    """
    Domain 1: Wyckoff & Structural Layer
    - M1/M2 Phases, Range Logic, BOS/CHOCH, Trap Resets, Unfinished Candles
    Returns: (score, signals, veto_reason)
    """
    score = 0
    signals = []
    veto_reason = None

    try:
        # === M1/M2 PHASE DETECTION ===
        if '1D' in synced_data and synced_data['1D'] is not None:
            d1_data = synced_data['1D']

            # M1 Phase: Accumulation/Spring detection
            range_pos = d1_data.get('range_position', 0.5)
            vol_ratio = d1_data.get('vol_ratio', 1.0)
            rsi = d1_data.get('rsi', 50)

            # Spring setup (Wyckoff accumulation)
            if range_pos < 0.3 and vol_ratio > 1.5 and 25 <= rsi <= 40:
                score += 0.25
                signals.append('wyckoff_spring')

            # M2 Phase: Markup detection
            bos = d1_data.get('break_of_structure', False)
            momentum = d1_data.get('momentum_5', 0)

            if bos and momentum > 0.02 and range_pos > 0.7:
                score += 0.20
                signals.append('wyckoff_markup')

        # === RANGE LOGIC (SC, AR, ST patterns) ===
        if '4H' in synced_data and synced_data['4H'] is not None:
            h4_data = synced_data['4H']

            # Supply Test (ST) - testing resistance with low volume
            range_pos_4h = h4_data.get('range_position', 0.5)
            vol_ratio_4h = h4_data.get('vol_ratio', 1.0)

            if range_pos_4h > 0.8 and vol_ratio_4h < 1.2:  # High range, low volume
                score += 0.15
                signals.append('supply_test')

            # Automatic Rally (AR) - strong move with volume
            if range_pos_4h > 0.6 and vol_ratio_4h > 1.8:
                score += 0.15
                signals.append('automatic_rally')

        # === TRAP RESETS (Bojan candle flips) ===
        if '1H' in synced_data and synced_data['1H'] is not None:
            h1_data = synced_data['1H']

            wick_dominance = h1_data.get('wick_dominance', 0)
            body_size = h1_data.get('body_size', 0)
            total_range = h1_data.get('high', 0) - h1_data.get('low', 0)

            # Unfinished candle (large wicks, small body)
            if wick_dominance > 0.6 and total_range > 0:
                score += 0.10
                signals.append('unfinished_candle')

            # Trap reset (wick rejection + reversal)
            momentum_1h = h1_data.get('momentum_1', 0)
            if wick_dominance > 0.7 and momentum_1h > 0.01:
                score += 0.15
                signals.append('trap_reset')

        # === STRUCTURAL VETOS ===
        # Veto if we're in obvious distribution phase
        if '1D' in synced_data and synced_data['1D'] is not None:
            d1_rsi = synced_data['1D'].get('rsi', 50)
            d1_range = synced_data['1D'].get('range_position', 0.5)

            if d1_rsi > 80 and d1_range > 0.9:  # Extreme overbought at range highs
                veto_reason = "Distribution phase detected"
                score *= 0.2  # Heavy penalty

    except Exception as e:
        print(f"Wyckoff domain error: {e}")

    return score, signals, veto_reason


# ========================================================================
# CONFLUENCE DOMAIN 2: LIQUIDITY LAYER
# ========================================================================

def analyze_liquidity_domain(synced_data, lookback_data):
    """
    Domain 2: Liquidity Layer
    - Sweeps, FVGs, HOBs, Wick Magnets, Liquidity Clusters
    Returns: (score, signals, veto_reason)
    """
    score = 0
    signals = []
    veto_reason = None

    try:
        # === LIQUIDITY SWEEPS ===
        if '1H' in synced_data and synced_data['1H'] is not None:
            h1_data = synced_data['1H']

            # Stop hunt above recent highs
            current_high = h1_data.get('high', 0)
            recent_high = h1_data.get('hh_20', 0)

            if current_high > recent_high * 1.005:  # Sweep above highs
                score += 0.20
                signals.append('liquidity_sweep_high')

            # Sweep below recent lows (spring setup)
            current_low = h1_data.get('low', 0)
            recent_low = h1_data.get('ll_20', 0)

            if current_low < recent_low * 0.995:  # Sweep below lows
                score += 0.25
                signals.append('liquidity_sweep_low')

        # === FAIR VALUE GAPS ===
        for tf in ['4H', '1H']:
            if tf in synced_data and synced_data[tf] is not None:
                tf_data = synced_data[tf]

                has_gap = tf_data.get('has_gap', False)
                if has_gap:
                    score += 0.10
                    signals.append(f'fvg_{tf.lower()}')

        # === WICK MAGNETS (Bojan high/low) ===
        if '1D' in synced_data and synced_data['1D'] is not None:
            d1_data = synced_data['1D']

            upper_wick = d1_data.get('upper_wick', 0)
            lower_wick = d1_data.get('lower_wick', 0)
            total_range = d1_data.get('high', 0) - d1_data.get('low', 0)

            if total_range > 0:
                # Significant upper wick = liquidity above
                if upper_wick / total_range > 0.4:
                    score += 0.15
                    signals.append('wick_magnet_above')

                # Significant lower wick = liquidity below
                if lower_wick / total_range > 0.4:
                    score += 0.15
                    signals.append('wick_magnet_below')

        # === LIQUIDITY CLUSTERS ===
        # Multi-timeframe volume spike confluence
        volume_signals = 0
        for tf in ['1D', '4H', '1H']:
            if tf in synced_data and synced_data[tf] is not None:
                vol_spike = synced_data[tf].get('vol_spike', False)
                if vol_spike:
                    volume_signals += 1

        if volume_signals >= 2:
            score += 0.20
            signals.append('liquidity_cluster')

        # === LIQUIDITY VETOS ===
        # Veto if liquidity is completely dried up
        total_volume_signals = sum([1 for tf in ['1D', '4H', '1H']
                                  if tf in synced_data and synced_data[tf] is not None
                                  and synced_data[tf].get('vol_ratio', 1.0) > 1.1])

        if total_volume_signals == 0:
            veto_reason = "No volume/liquidity signals"
            score *= 0.3

    except Exception as e:
        print(f"Liquidity domain error: {e}")

    return score, signals, veto_reason


# ========================================================================
# CONFLUENCE DOMAIN 3: MOMENTUM & VOLUME LAYER
# ========================================================================

def analyze_momentum_volume_domain(synced_data, lookback_data):
    """
    Domain 3: Momentum & Volume Layer
    - CVD Divergence, Volume Absorption, Body/Wick Dominance, Orderflow BOS
    Returns: (score, signals, veto_reason)
    """
    score = 0
    signals = []
    veto_reason = None

    try:
        # === VOLUME EXPANSION ANALYSIS ===
        volume_strength = 0

        for tf in ['1D', '4H', '1H']:
            if tf in synced_data and synced_data[tf] is not None:
                tf_data = synced_data[tf]
                vol_ratio = tf_data.get('vol_ratio', 1.0)

                if vol_ratio > 2.5:
                    volume_strength += 0.25
                elif vol_ratio > 2.0:
                    volume_strength += 0.20
                elif vol_ratio > 1.5:
                    volume_strength += 0.15
                elif vol_ratio > 1.2:
                    volume_strength += 0.10

        score += min(volume_strength, 0.35)
        if volume_strength > 0.20:
            signals.append('volume_expansion')

        # === MOMENTUM CONFLUENCE ===
        momentum_signals = 0

        if '1D' in synced_data and synced_data['1D'] is not None:
            d1_data = synced_data['1D']
            rsi_1d = d1_data.get('rsi', 50)
            momentum_1d = d1_data.get('momentum_5', 0)
            macd_1d = d1_data.get('macd', 0)

            # Momentum building (oversold + positive momentum)
            if 30 <= rsi_1d <= 45 and momentum_1d > 0.015:
                momentum_signals += 1
                score += 0.15

            # MACD turning positive
            if macd_1d > 0:
                momentum_signals += 1
                score += 0.10

        if '4H' in synced_data and synced_data['4H'] is not None:
            h4_data = synced_data['4H']
            momentum_4h = h4_data.get('momentum_5', 0)

            if momentum_4h > 0.02:
                momentum_signals += 1
                score += 0.10

        if momentum_signals >= 2:
            signals.append('momentum_confluence')

        # === BODY/WICK DOMINANCE (Bojan Intent) ===
        for tf in ['4H', '1H']:
            if tf in synced_data and synced_data[tf] is not None:
                tf_data = synced_data[tf]
                wick_dominance = tf_data.get('wick_dominance', 0)
                body_size = tf_data.get('body_size', 0)
                momentum = tf_data.get('momentum_1', 0)

                # Strong body dominance + momentum = intent
                if wick_dominance < 0.3 and momentum > 0.01:
                    score += 0.10
                    signals.append(f'body_dominance_{tf.lower()}')

        # === ORDERFLOW BOS ===
        bos_signals = 0
        for tf in ['1D', '4H']:
            if tf in synced_data and synced_data[tf] is not None:
                bos = synced_data[tf].get('break_of_structure', False)
                if bos:
                    bos_signals += 1
                    score += 0.15

        if bos_signals >= 1:
            signals.append('orderflow_bos')

        # === MOMENTUM VETOS ===
        # Veto if momentum is clearly against us
        if '1D' in synced_data and synced_data['1D'] is not None:
            d1_momentum = synced_data['1D'].get('momentum_20', 0)
            d1_rsi = synced_data['1D'].get('rsi', 50)

            if d1_momentum < -0.05 and d1_rsi < 30:  # Strong bearish momentum
                veto_reason = "Strong bearish momentum"
                score *= 0.2

        # Veto if no volume confirmation
        if volume_strength < 0.05:
            veto_reason = "No volume confirmation"
            score *= 0.4

    except Exception as e:
        print(f"Momentum domain error: {e}")

    return score, signals, veto_reason


# ========================================================================
# CONFLUENCE DOMAIN 4: TEMPORAL & FIBONACCI LAYER
# ========================================================================

def analyze_temporal_fibonacci_domain(synced_data, lookback_data):
    """
    Domain 4: Temporal & Fibonacci Layer
    - Fib Price Zones, Hidden Fibs, Time Clusters, Sacred Geometry
    Returns: (score, signals, veto_reason)
    """
    score = 0
    signals = []
    veto_reason = None

    try:
        # === FIBONACCI PRICE ZONES ===
        fib_confluence = 0

        for tf in ['1D', '4H']:
            if tf in synced_data and synced_data[tf] is not None:
                tf_data = synced_data[tf]

                # Check distance to key Fib levels
                near_fib = tf_data.get('near_fib', False)
                dist_382 = tf_data.get('dist_to_382', 1.0)
                dist_618 = tf_data.get('dist_to_618', 1.0)

                if near_fib:
                    fib_confluence += 0.15
                    signals.append(f'fib_zone_{tf.lower()}')

                # Golden ratio zones (.618, .786)
                if dist_618 < 0.015:  # Very close to 61.8%
                    fib_confluence += 0.20
                    signals.append(f'golden_ratio_{tf.lower()}')

        score += min(fib_confluence, 0.30)

        # === FIBONACCI TIME CLUSTERS ===
        time_signals = 0

        if '1D' in synced_data and synced_data['1D'] is not None:
            d1_data = synced_data['1D']

            # Check Fibonacci time sequences
            fib_times = ['fib_time_21', 'fib_time_34', 'fib_time_55', 'fib_time_89']
            for fib_time in fib_times:
                if d1_data.get(fib_time, False):
                    time_signals += 1
                    score += 0.05

            # Sacred time (360-day cycles)
            if d1_data.get('cycle_360', False):
                score += 0.10
                signals.append('sacred_time')

        if time_signals >= 2:
            signals.append('fib_time_cluster')

        # === HIDDEN FIBS (Internal swing levels) ===
        # Multi-timeframe Fib alignment
        fib_alignment = 0
        for tf in ['1D', '4H', '1H']:
            if tf in synced_data and synced_data[tf] is not None:
                tf_data = synced_data[tf]
                range_pos = tf_data.get('range_position', 0.5)

                # Check if we're at key internal Fib levels
                key_levels = [0.236, 0.382, 0.500, 0.618, 0.786]
                min_distance = min([abs(range_pos - level) for level in key_levels])

                if min_distance < 0.05:  # Within 5% of key level
                    fib_alignment += 1

        if fib_alignment >= 2:
            score += 0.15
            signals.append('hidden_fib_alignment')

        # === GANN/SACRED GEOMETRY ===
        # Simplified sacred number confluence
        if '1D' in synced_data and synced_data['1D'] is not None:
            d1_data = synced_data['1D']
            close = d1_data.get('close', 0)

            # Check if price is near round numbers (psychological levels)
            if close > 0:
                # Check proximity to round thousands, hundreds
                round_1000 = close % 1000
                round_100 = close % 100

                if round_1000 < 50 or round_1000 > 950:  # Near thousand
                    score += 0.08
                    signals.append('round_number_1000')
                elif round_100 < 10 or round_100 > 90:  # Near hundred
                    score += 0.05
                    signals.append('round_number_100')

        # === TEMPORAL VETOS ===
        # No major temporal/Fib vetos - this domain is supportive

    except Exception as e:
        print(f"Temporal domain error: {e}")

    return score, signals, veto_reason


# ========================================================================
# CONFLUENCE DOMAIN 5: FUSION & PSYCHOLOGICAL LAYER
# ========================================================================

def analyze_fusion_psychological_domain(synced_data, lookback_data, domain_scores):
    """
    Domain 5: Fusion & Psychological Layer
    - Ensemble Alignment, Plus-One Scoring, Oracle Whispers, Narrative Radar
    Returns: (score, signals, veto_reason)
    """
    score = 0
    signals = []
    veto_reason = None

    try:
        # === ENSEMBLE ALIGNMENT (1D bias → 4H structure → 1H entry) ===
        alignment_score = 0

        # 1D Bias Check
        d1_bias_bullish = False
        if '1D' in synced_data and synced_data['1D'] is not None:
            d1_data = synced_data['1D']
            sma_20 = d1_data.get('sma_20', 0)
            sma_50 = d1_data.get('sma_50', 0)
            close = d1_data.get('close', 0)

            if close > sma_20 > sma_50:  # Bullish bias
                d1_bias_bullish = True
                alignment_score += 0.10
                signals.append('d1_bias_bullish')

        # 4H Structure Check
        h4_structure_bullish = False
        if '4H' in synced_data and synced_data['4H'] is not None:
            h4_data = synced_data['4H']
            bos = h4_data.get('break_of_structure', False)
            range_pos = h4_data.get('range_position', 0.5)

            if bos or range_pos > 0.6:  # Bullish structure
                h4_structure_bullish = True
                alignment_score += 0.10
                signals.append('h4_structure_bullish')

        # 1H Entry Check
        h1_entry_signal = False
        if '1H' in synced_data and synced_data['1H'] is not None:
            h1_data = synced_data['1H']
            momentum = h1_data.get('momentum_1', 0)
            vol_ratio = h1_data.get('vol_ratio', 1.0)

            if momentum > 0.005 and vol_ratio > 1.2:  # Entry signal
                h1_entry_signal = True
                alignment_score += 0.10
                signals.append('h1_entry_signal')

        # Full alignment bonus
        if d1_bias_bullish and h4_structure_bullish and h1_entry_signal:
            alignment_score += 0.20
            signals.append('full_mtf_alignment')

        score += alignment_score

        # === PLUS-ONE SCORING (Domain stacking) ===
        active_domains = sum([1 for domain_score in domain_scores if domain_score > 0.15])

        if active_domains >= 4:
            score += 0.25
            signals.append('plus_four_domains')
        elif active_domains >= 3:
            score += 0.20
            signals.append('plus_three_domains')
        elif active_domains >= 2:
            score += 0.15
            signals.append('plus_two_domains')

        # === ORACLE WHISPERS (Narrative/Sentiment) ===
        # Simplified sentiment analysis based on market structure
        narrative_score = 0

        # "Spring whisper" - oversold with volume
        if '1D' in synced_data and synced_data['1D'] is not None:
            d1_data = synced_data['1D']
            rsi = d1_data.get('rsi', 50)
            vol_ratio = d1_data.get('vol_ratio', 1.0)
            range_pos = d1_data.get('range_position', 0.5)

            if rsi < 35 and vol_ratio > 1.5 and range_pos < 0.4:
                narrative_score += 0.15
                signals.append('spring_whisper')

        # "Breakout whisper" - accumulation complete
        if active_domains >= 3 and alignment_score > 0.25:
            narrative_score += 0.10
            signals.append('breakout_whisper')

        score += narrative_score

        # === PSYCHOLOGICAL CONFLUENCE ===
        # Fear/Greed balance based on RSI + momentum across timeframes
        psychological_balance = 0

        for tf in ['1D', '4H']:
            if tf in synced_data and synced_data[tf] is not None:
                tf_data = synced_data[tf]
                rsi = tf_data.get('rsi', 50)
                momentum = tf_data.get('momentum_5', 0)

                # Sweet spot: Slight fear but building momentum
                if 30 <= rsi <= 50 and momentum > 0.01:
                    psychological_balance += 0.08

        score += psychological_balance

        # === FUSION VETOS ===
        # Veto if domains are conflicting
        if len([s for s in domain_scores if s > 0.15]) < 2:
            veto_reason = "Insufficient domain confluence"
            score *= 0.3

        # Veto if MTF alignment is completely off
        if not (d1_bias_bullish or h4_structure_bullish):
            veto_reason = "No higher timeframe support"
            score *= 0.5

    except Exception as e:
        print(f"Fusion domain error: {e}")

    return score, signals, veto_reason


# ========================================================================
# COMPLETE CONFLUENCE ORCHESTRATOR
# ========================================================================

def calculate_complete_confluence_score(data, current_time, config):
    """
    Complete Bull Machine v1.6.2 Confluence Analysis
    Orchestrates all 5 domains and calculates Plus-One stacking score
    """

    # Sync all timeframes
    synced_data = sync_timeframes_for_confluence(data, current_time)

    # Get lookback data for context
    lookback_data = {}
    for tf, df in data.items():
        if df is not None and len(df) > 0:
            mask = df.index <= current_time
            if mask.any():
                recent_idx = df.index[mask][-1]
                start_idx = max(0, df.index.get_loc(recent_idx) - 50)
                lookback_data[tf] = df.iloc[start_idx:df.index.get_loc(recent_idx)+1]

    # Initialize confluence analysis
    domain_scores = []
    domain_signals = []
    domain_vetos = []
    domain_names = ['Wyckoff', 'Liquidity', 'Momentum', 'Temporal', 'Fusion']

    # Domain 1: Wyckoff & Structural
    score_1, signals_1, veto_1 = analyze_wyckoff_structural_domain(synced_data, lookback_data)
    domain_scores.append(score_1)
    domain_signals.append(signals_1)
    domain_vetos.append(veto_1)

    # Domain 2: Liquidity
    score_2, signals_2, veto_2 = analyze_liquidity_domain(synced_data, lookback_data)
    domain_scores.append(score_2)
    domain_signals.append(signals_2)
    domain_vetos.append(veto_2)

    # Domain 3: Momentum & Volume
    score_3, signals_3, veto_3 = analyze_momentum_volume_domain(synced_data, lookback_data)
    domain_scores.append(score_3)
    domain_signals.append(signals_3)
    domain_vetos.append(veto_3)

    # Domain 4: Temporal & Fibonacci
    score_4, signals_4, veto_4 = analyze_temporal_fibonacci_domain(synced_data, lookback_data)
    domain_scores.append(score_4)
    domain_signals.append(signals_4)
    domain_vetos.append(veto_4)

    # Domain 5: Fusion & Psychological (needs other domain scores)
    score_5, signals_5, veto_5 = analyze_fusion_psychological_domain(synced_data, lookback_data, domain_scores[:4])
    domain_scores.append(score_5)
    domain_signals.append(signals_5)
    domain_vetos.append(veto_5)

    # Calculate final confluence
    base_score = sum(domain_scores)

    # Plus-One scoring bonus
    active_domains = sum([1 for score in domain_scores if score > 0.15])
    plus_one_bonus = 0

    if active_domains >= 5:
        plus_one_bonus = 0.30  # All domains singing
    elif active_domains >= 4:
        plus_one_bonus = 0.20  # Strong confluence
    elif active_domains >= 3:
        plus_one_bonus = 0.15  # Good confluence
    elif active_domains >= 2:
        plus_one_bonus = 0.10  # Minimum confluence

    final_score = base_score + plus_one_bonus

    # Apply veto penalties
    major_vetos = [v for v in domain_vetos if v is not None]
    if major_vetos:
        final_score *= 0.5  # Significant penalty for vetos

    # Compile complete analysis
    complete_analysis = {
        'final_score': final_score,
        'base_score': base_score,
        'plus_one_bonus': plus_one_bonus,
        'active_domains': active_domains,
        'domain_breakdown': {
            domain_names[i]: {
                'score': domain_scores[i],
                'signals': domain_signals[i],
                'veto': domain_vetos[i]
            } for i in range(5)
        },
        'total_signals': sum([len(signals) for signals in domain_signals]),
        'confluence_grade': get_confluence_grade(active_domains, final_score),
        'major_vetos': major_vetos
    }

    return final_score, complete_analysis


def get_confluence_grade(active_domains, final_score):
    """Assign confluence grade based on domains and score"""
    if active_domains >= 5 and final_score > 1.0:
        return "A+ (Rare High-Signal)"
    elif active_domains >= 4 and final_score > 0.8:
        return "A (Excellent Confluence)"
    elif active_domains >= 3 and final_score > 0.6:
        return "B+ (Good Confluence)"
    elif active_domains >= 2 and final_score > 0.4:
        return "B (Adequate Confluence)"
    else:
        return "C (Weak Confluence)"


# ========================================================================
# COMPLETE SYSTEM BACKTEST
# ========================================================================

def run_complete_confluence_backtest(asset, data, config):
    """Run complete 5-domain confluence backtest"""
    print(f"\n=== Running {asset} Complete 5-Domain Confluence Backtest ===")

    # Use 1D timeframe for extended 24-month validation (better data coverage)
    # Allow shorter periods for walk-forward tuning (minimum 100 bars)
    if '1D' not in data or data['1D'] is None or len(data['1D']) < 100:
        print(f"Insufficient {asset} 1D data")
        return {}

    # Apply date filtering if specified in config
    filtered_data = {}
    for timeframe, df in data.items():
        if df is not None and len(df) > 0:
            # Apply date range filtering if specified in config
            start_date = config.get('start_date')
            end_date = config.get('end_date')

            if start_date and end_date:
                # Filter data to specified date range
                start_ts = pd.Timestamp(start_date)
                end_ts = pd.Timestamp(end_date)
                df_filtered = df[(df.index >= start_ts) & (df.index <= end_ts)]
                filtered_data[timeframe] = df_filtered if len(df_filtered) > 0 else df
            else:
                filtered_data[timeframe] = df
        else:
            filtered_data[timeframe] = df

    # Use filtered data
    data = filtered_data
    primary_df = data['1D']

    if len(primary_df) < 100:
        print(f"Insufficient {asset} 1D data after filtering")
        return {}

    print(f"Backtesting {len(primary_df)} daily bars with 5-domain confluence validation")
    print(f"Available timeframes: {list(data.keys())}")
    print(f"Period: {primary_df.index[0].strftime('%Y-%m-%d')} to {primary_df.index[-1].strftime('%Y-%m-%d')}")

    # Extended production parameters for 24-month validation
    entry_threshold = config.get('entry_threshold', 0.30)    # Optimized for 24-month frequency
    min_active_domains = config.get('min_active_domains', 3) # Maintain quality filter
    cooldown_days = config.get('cooldown_days', 7)         # 1 week cooldown for daily bars
    risk_per_trade = config.get('risk_pct', 0.025)          # 2.5% risk per trade
    initial_capital = 10000

    print(f"Entry threshold: {entry_threshold}")
    print(f"Min active domains: {min_active_domains}")
    print(f"Risk per trade: {risk_per_trade*100}%")
    print(f"Cooldown: {cooldown_days} days")

    # Run complete backtest
    trades = []
    capital = initial_capital
    last_trade_time = None

    # Start analysis from bar 100 to ensure sufficient lookback
    for i in range(100, len(primary_df) - 1):
        current_time = primary_df.index[i]

        # Cooldown check
        if last_trade_time and (current_time - last_trade_time).days < cooldown_days:
            continue

        # Calculate complete confluence score
        score, analysis = calculate_complete_confluence_score(data, current_time, config)

        # Entry criteria: High score + sufficient domain activation
        if score >= entry_threshold and analysis['active_domains'] >= min_active_domains:
            current_bar = primary_df.iloc[i]
            entry_price = current_bar['close']

            # Position sizing
            position_size = capital * risk_per_trade / entry_price

            # Dynamic exit based on confluence grade (daily bars)
            confluence_grade = analysis['confluence_grade']
            if 'A+' in confluence_grade:
                exit_days = 10  # Hold rare signals longer
            elif 'A' in confluence_grade:
                exit_days = 7
            elif 'B+' in confluence_grade:
                exit_days = 5
            else:
                exit_days = 3

            # Find exit point
            exit_time_target = current_time + timedelta(days=exit_days)
            exit_mask = primary_df.index >= exit_time_target

            if exit_mask.any():
                exit_idx = primary_df.index[exit_mask][0]
                exit_bar = primary_df.loc[exit_idx]
                exit_price = exit_bar['close']
            else:
                exit_price = primary_df.iloc[-1]['close']
                exit_idx = primary_df.index[-1]

            # Calculate PnL with fees/slippage
            fee_bps = 5
            slip_bps = 2
            entry_cost = entry_price * (1 + (fee_bps + slip_bps) / 10000)
            exit_proceeds = exit_price * (1 - (fee_bps + slip_bps) / 10000)
            pnl = position_size * (exit_proceeds - entry_cost)
            capital += pnl

            # Comprehensive trade tagging
            tags = []

            # Domain-specific tags
            for domain_name, domain_data in analysis['domain_breakdown'].items():
                if domain_data['score'] > 0.15:
                    tags.append(f"{domain_name.lower()}_active")

                # Specific signal tags
                for signal in domain_data['signals']:
                    tags.append(signal)

            # Confluence quality tags
            tags.append(confluence_grade.split()[0].lower())
            if analysis['active_domains'] >= 4:
                tags.append('multi_domain_confluence')
            if analysis['total_signals'] >= 8:
                tags.append('high_signal_density')

            # Create comprehensive trade record
            trade = {
                'entry_time': current_time,
                'exit_time': exit_idx,
                'entry_price': entry_price,
                'exit_price': exit_price,
                'pnl': pnl,
                'score': score,
                'confluence_analysis': analysis,
                'tags': tags,
                'side': 'long',
                'exit_days': exit_days,
                'confluence_grade': confluence_grade
            }

            trades.append(trade)
            last_trade_time = current_time

            # Real-time trade logging
            print(f"🎯 TRADE #{len(trades)}: {current_time.strftime('%Y-%m-%d %H:%M')}")
            print(f"   Score: {score:.3f} | Domains: {analysis['active_domains']}/5 | Grade: {confluence_grade}")
            print(f"   Signals: {analysis['total_signals']} | PnL: ${pnl:.2f}")
            print(f"   Active: {[d for d, data in analysis['domain_breakdown'].items() if data['score'] > 0.15]}")
            if analysis['major_vetos']:
                print(f"   Vetos: {analysis['major_vetos']}")
            print()

    # Calculate comprehensive metrics
    metrics = calculate_comprehensive_metrics(trades, initial_capital)

    # Add confluence-specific metrics
    if trades:
        confluence_grades = [t['confluence_grade'] for t in trades]
        grade_distribution = {}
        for grade in ['A+', 'A', 'B+', 'B', 'C']:
            grade_distribution[grade] = len([g for g in confluence_grades if grade in g])

        metrics['confluence_distribution'] = grade_distribution
        metrics['avg_domains_per_trade'] = np.mean([t['confluence_analysis']['active_domains'] for t in trades])
        metrics['avg_signals_per_trade'] = np.mean([t['confluence_analysis']['total_signals'] for t in trades])

    return {
        'asset': asset,
        'trades': trades,
        'metrics': metrics,
        'final_capital': capital,
        'config': config
    }


def calculate_comprehensive_metrics(trades, initial_capital=10000):
    """Calculate comprehensive trading metrics"""
    if not trades:
        return {
            'total_trades': 0, 'win_rate': 0, 'total_pnl_pct': 0,
            'max_drawdown_pct': 0, 'profit_factor': 0, 'sharpe_ratio': 0,
            'avg_trade': 0, 'trades_per_month': 0, 'gross_profit': 0, 'gross_loss': 0
        }

    total_trades = len(trades)
    winning_trades = len([t for t in trades if t['pnl'] > 0])
    win_rate = (winning_trades / total_trades) * 100

    pnl_series = [t['pnl'] for t in trades]
    total_pnl = sum(pnl_series)
    total_pnl_pct = (total_pnl / initial_capital) * 100
    avg_trade = total_pnl / total_trades

    # Drawdown calculation
    cumulative_pnl = np.cumsum([0] + pnl_series)
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdowns = (cumulative_pnl - running_max)
    max_drawdown = abs(min(drawdowns))
    max_drawdown_pct = (max_drawdown / initial_capital) * 100

    # Profit factor
    gross_profit = sum([p for p in pnl_series if p > 0])
    gross_loss = abs(sum([p for p in pnl_series if p < 0]))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    # Sharpe ratio
    returns = np.array(pnl_series) / initial_capital
    sharpe_ratio = np.mean(returns) / np.std(returns) * np.sqrt(252*24) if np.std(returns) > 0 else 0

    # Frequency
    if len(trades) > 1:
        date_range_hours = (trades[-1]['entry_time'] - trades[0]['entry_time']).total_seconds() / 3600
        months = date_range_hours / (24 * 30.44)
        trades_per_month = total_trades / months if months > 0 else 0
    else:
        trades_per_month = 0

    return {
        'total_trades': total_trades, 'win_rate': win_rate, 'total_pnl_pct': total_pnl_pct,
        'max_drawdown_pct': max_drawdown_pct, 'profit_factor': profit_factor, 'sharpe_ratio': sharpe_ratio,
        'avg_trade': avg_trade, 'trades_per_month': trades_per_month,
        'gross_profit': gross_profit, 'gross_loss': gross_loss
    }


def load_asset_config(asset):
    """Load asset-specific configuration for extended production backtest"""
    config_path = f"configs/v160/assets/{asset}.json"
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)

        # Extended backtest parameters with optimal risk allocation
        config.update({
            'entry_threshold': 0.30,  # Slightly lower for more opportunities over 24 months
            'min_active_domains': 3,  # Maintain quality filter
            'cooldown_days': 7,       # 1 week cooldown for daily bars
            'risk_pct': 0.025         # 2.5% risk per trade for production-ready sizing
        })

        return config
    except FileNotFoundError:
        return {
            'entry_threshold': 0.30,
            'min_active_domains': 3,
            'cooldown_days': 7,
            'risk_pct': 0.025
        }


def run_complete_bull_machine_system():
    """Run the complete Bull Machine v1.6.2 5-domain confluence system"""
    print("="*100)
    print("🎯 BULL MACHINE v1.6.2 - SOL & XRP 2-YEAR BACKTEST")
    print("Complete 5-Domain Confluence | 2.5% Risk | 2023-2025 | $10k Starting Stack")
    print("Story + Liquidity + Momentum + Time + Alignment → Alt-Coin Validation")
    print("="*100)

    # Multi-timeframe data paths
    data_paths = {
        'SOL': {
            '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_SOLUSD, 1D_8ffaf.csv',
            '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_SOLUSD, 240_f9a53.csv',
            '1H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_SOLUSD, 60_07764.csv'
        },
        'XRP': {
            '1D': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_XRPUSD, 1D_c20d3.csv',
            '4H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_XRPUSD, 240_a43c1.csv',
            '1H': '/Users/raymondghandchi/Downloads/Chart logs 2/COINBASE_XRPUSD, 60_689bb.csv'
        }
    }

    results = {}

    # Run complete system for each asset
    for asset, paths in data_paths.items():
        try:
            # Load multi-timeframe data
            data = load_multi_timeframe_data(asset, paths)

            # Verify we have sufficient MTF data
            required_tfs = ['1D', '4H', '1H']
            available_tfs = [tf for tf in required_tfs if tf in data and data[tf] is not None and len(data[tf]) > 200]

            if len(available_tfs) >= 2:  # Need at least 2 timeframes
                # Load configuration
                config = load_asset_config(asset)

                # Run complete confluence backtest
                result = run_complete_confluence_backtest(asset, data, config)
                results[asset] = result
            else:
                print(f"\n❌ Skipping {asset} - insufficient multi-timeframe data")
                print(f"   Available: {available_tfs}, Required: {required_tfs}")

        except Exception as e:
            print(f"\n❌ ERROR running {asset} complete system: {e}")
            import traceback
            traceback.print_exc()
            continue

    # ========================================================================
    # COMPREHENSIVE ANALYSIS & RESULTS
    # ========================================================================

    print("\n" + "="*100)
    print("🎯 COMPLETE 5-DOMAIN CONFLUENCE RESULTS")
    print("="*100)

    # Results summary table
    print(f"{'Asset':<6} {'Trades':<8} {'Win %':<8} {'PnL %':<10} {'DD %':<8} {'PF':<8} {'Sharpe':<8} {'Avg Dom':<8} {'Grade Dist':<20}")
    print("-" * 100)

    combined_trades = []

    for asset, result in results.items():
        if result and result['trades']:
            metrics = result['metrics']
            combined_trades.extend(result['trades'])

            # Grade distribution summary
            grade_dist = metrics.get('confluence_distribution', {})
            grade_summary = f"A+:{grade_dist.get('A+', 0)} A:{grade_dist.get('A', 0)} B+:{grade_dist.get('B+', 0)}"

            print(f"{asset:<6} {metrics['total_trades']:<8} {metrics['win_rate']:<8.1f} "
                  f"{metrics['total_pnl_pct']:<10.2f} {metrics['max_drawdown_pct']:<8.2f} "
                  f"{metrics['profit_factor']:<8.2f} {metrics['sharpe_ratio']:<8.2f} "
                  f"{metrics.get('avg_domains_per_trade', 0):<8.1f} {grade_summary:<20}")

    # Combined portfolio analysis
    if combined_trades:
        combined_metrics = calculate_comprehensive_metrics(combined_trades, 20000)
        combined_grades = {}
        for trade in combined_trades:
            grade = trade['confluence_grade'].split()[0]
            combined_grades[grade] = combined_grades.get(grade, 0) + 1

        grade_summary = f"A+:{combined_grades.get('A+', 0)} A:{combined_grades.get('A', 0)} B+:{combined_grades.get('B+', 0)}"

        print("-" * 100)
        print(f"{'TOTAL':<6} {combined_metrics['total_trades']:<8} {combined_metrics['win_rate']:<8.1f} "
              f"{combined_metrics['total_pnl_pct']:<10.2f} {combined_metrics['max_drawdown_pct']:<8.2f} "
              f"{combined_metrics['profit_factor']:<8.2f} {combined_metrics['sharpe_ratio']:<8.2f} "
              f"{np.mean([t['confluence_analysis']['active_domains'] for t in combined_trades]):<8.1f} {grade_summary:<20}")

    # ========================================================================
    # DETAILED CONFLUENCE ANALYSIS
    # ========================================================================

    print("\n" + "="*100)
    print("🔍 DETAILED CONFLUENCE DOMAIN ANALYSIS")
    print("="*100)

    for asset, result in results.items():
        if result and result['trades']:
            trades = result['trades']
            print(f"\n🎯 {asset} Complete Analysis ({len(trades)} trades):")

            # Domain activation frequency
            domain_activations = {
                'Wyckoff': 0, 'Liquidity': 0, 'Momentum': 0, 'Temporal': 0, 'Fusion': 0
            }

            signal_frequency = {}
            total_signals = 0

            for trade in trades:
                analysis = trade['confluence_analysis']

                # Count domain activations
                for domain_name, domain_data in analysis['domain_breakdown'].items():
                    if domain_data['score'] > 0.15:
                        domain_activations[domain_name] += 1

                    # Count signal types
                    for signal in domain_data['signals']:
                        signal_frequency[signal] = signal_frequency.get(signal, 0) + 1
                        total_signals += 1

            print(f"   Domain Activation Frequency:")
            for domain, count in domain_activations.items():
                percentage = (count / len(trades)) * 100
                print(f"     {domain}: {count}/{len(trades)} ({percentage:.1f}%)")

            print(f"   Top Signal Types:")
            top_signals = sorted(signal_frequency.items(), key=lambda x: x[1], reverse=True)[:8]
            for signal, count in top_signals:
                percentage = (count / total_signals) * 100 if total_signals > 0 else 0
                print(f"     {signal}: {count} ({percentage:.1f}%)")

            # Performance by confluence grade
            print(f"   Performance by Confluence Grade:")
            grade_performance = {}
            for trade in trades:
                grade = trade['confluence_grade'].split()[0]
                if grade not in grade_performance:
                    grade_performance[grade] = {'trades': [], 'pnl': 0}
                grade_performance[grade]['trades'].append(trade)
                grade_performance[grade]['pnl'] += trade['pnl']

            for grade in ['A+', 'A', 'B+', 'B']:
                if grade in grade_performance:
                    perf = grade_performance[grade]
                    count = len(perf['trades'])
                    avg_pnl = perf['pnl'] / count
                    win_rate = len([t for t in perf['trades'] if t['pnl'] > 0]) / count * 100
                    print(f"     {grade}: {count} trades, ${avg_pnl:.2f} avg PnL, {win_rate:.1f}% WR")

    # ========================================================================
    # FINAL RC-READY ASSESSMENT
    # ========================================================================

    print("\n" + "="*100)
    print("🏆 FINAL RC-READY ASSESSMENT (Complete 5-Domain System)")
    print("="*100)

    # Updated RC criteria for SOL/XRP 2-year validation
    rc_criteria = {
        'SOL': {'min_pnl': 50, 'min_pf': 2.0, 'max_dd': 25, 'min_trades': 10, 'min_avg_domains': 3.0},
        'XRP': {'min_pnl': 30, 'min_pf': 2.0, 'max_dd': 20, 'min_trades': 10, 'min_avg_domains': 3.0}
    }

    overall_ready = True
    ready_count = 0

    for asset, result in results.items():
        if result and result['trades'] and asset in rc_criteria:
            metrics = result['metrics']
            criteria = rc_criteria[asset]

            pnl_ok = metrics['total_pnl_pct'] >= criteria['min_pnl']
            pf_ok = metrics['profit_factor'] >= criteria['min_pf']
            dd_ok = metrics['max_drawdown_pct'] <= criteria['max_dd']
            trades_ok = metrics['total_trades'] >= criteria['min_trades']
            domains_ok = metrics.get('avg_domains_per_trade', 0) >= criteria['min_avg_domains']

            asset_ready = pnl_ok and pf_ok and dd_ok and trades_ok and domains_ok
            if asset_ready:
                ready_count += 1
            else:
                overall_ready = False

            status = "✅" if asset_ready else "❌"
            print(f"{asset} RC-Ready (Complete System): {status}")
            print(f"  PnL: {metrics['total_pnl_pct']:.1f}% (need ≥{criteria['min_pnl']}%) {'✅' if pnl_ok else '❌'}")
            print(f"  PF: {metrics['profit_factor']:.2f} (need ≥{criteria['min_pf']}) {'✅' if pf_ok else '❌'}")
            print(f"  DD: {metrics['max_drawdown_pct']:.1f}% (need ≤{criteria['max_dd']}%) {'✅' if dd_ok else '❌'}")
            print(f"  Trades: {metrics['total_trades']} (need ≥{criteria['min_trades']}) {'✅' if trades_ok else '❌'}")
            print(f"  Avg Domains: {metrics.get('avg_domains_per_trade', 0):.1f} (need ≥{criteria['min_avg_domains']}) {'✅' if domains_ok else '❌'}")
            print(f"  Quality: {metrics.get('confluence_distribution', {}).get('A+', 0)} A+ trades")

    # Final system verdict
    if ready_count >= 1:
        final_status = f"✅ PRODUCTION READY ({ready_count}/2 assets qualified)"
    else:
        final_status = "❌ REQUIRES FURTHER OPTIMIZATION"

    print(f"\n🎯 FINAL BULL MACHINE v1.6.2 STATUS: {final_status}")

    if combined_trades:
        total_a_plus = len([t for t in combined_trades if 'A+' in t['confluence_grade']])
        total_a_grade = len([t for t in combined_trades if 'A' in t['confluence_grade']])

        # Calculate annualized performance for 2-year period
        months_tested = 24
        annual_return = (combined_metrics['total_pnl_pct'] / months_tested) * 12
        compound_return = ((1 + combined_metrics['total_pnl_pct']/100) ** (12/months_tested) - 1) * 100

        print(f"📊 SOL & XRP 2-Year Performance Summary:")
        print(f"   Total Trades: {len(combined_trades)} ({len(combined_trades)/months_tested:.1f}/month)")
        print(f"   Quality Distribution: {total_a_plus} A+, {total_a_grade} A-grade signals")
        print(f"   Combined Return: {combined_metrics['total_pnl_pct']:.2f}% over 24 months")
        print(f"   Annualized Return: {annual_return:.1f}% (simple) | {compound_return:.1f}% (compound)")
        print(f"   Max Drawdown: {combined_metrics['max_drawdown_pct']:.2f}%")
        print(f"   Risk-Adjusted: {combined_metrics['total_pnl_pct'] / max(combined_metrics['max_drawdown_pct'], 0.1):.1f}x return/risk ratio")

    # Save comprehensive results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"reports/v162_complete_confluence_{timestamp}.json"

    os.makedirs('reports', exist_ok=True)
    with open(results_file, 'w') as f:
        # Serialize results with datetime conversion
        serializable_results = {}
        for asset, result in results.items():
            if result:
                serializable_result = dict(result)
                if 'trades' in serializable_result:
                    for trade in serializable_result['trades']:
                        if 'entry_time' in trade:
                            trade['entry_time'] = trade['entry_time'].isoformat()
                        if 'exit_time' in trade:
                            trade['exit_time'] = trade['exit_time'].isoformat()
                        # Clean up numpy types in confluence analysis
                        if 'confluence_analysis' in trade:
                            analysis = trade['confluence_analysis']
                            for key, value in analysis.items():
                                if isinstance(value, (np.integer, np.floating)):
                                    analysis[key] = float(value)
                serializable_results[asset] = serializable_result

        json.dump(serializable_results, f, indent=2, default=str)

    print(f"\n📁 Complete system results saved to: {results_file}")

    return results


if __name__ == '__main__':
    results = run_complete_bull_machine_system()