#!/usr/bin/env python3
"""
Multi-Timeframe Feature Store Builder v2.0 - MVP Architecture

Implements 3-level MTF hierarchy with down-casting:
- 1D: Governor features (Wyckoff, BOMS, Range, FRVP, PTI, Macro Echo)
- 4H: Structure features (Squiggle, OB/FVG, Internal/External, CHOCH)
- 1H: Execution features (Micro PTI, FRVP local, Kelly inputs)

Column Naming Convention:
  tf1d_wyckoff_phase, tf1d_boms_strength
  tf4h_squiggle_stage, tf4h_choch_flag
  tf1h_pti_score, tf1h_kelly_hint
  mtf_alignment_ok, mtf_conflict_score
  k2_threshold_delta, k2_score_delta
  macro_regime, macro_dxy_trend

Asset Support:
- Crypto (BTC, ETH): 24/7 data
- Equities (SPY, TSLA): RTH-only (09:30-16:00 ET)

Output: Parquet file ready for fast_backtest_v2.py (vectorized operations)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime, time
import argparse
import json

from engine.io.tradingview_loader import load_tv
from engine.context.loader import load_macro_data, fetch_macro_snapshot
from engine.context.macro_engine import analyze_macro, create_default_macro_config
from engine.fusion.domain_fusion import analyze_fusion

# Wyckoff (for precompute)
from engine.wyckoff.wyckoff_engine import detect_wyckoff_phase

# Advanced Wyckoff M1/M2 (with spring/markup detection)
try:
    from bull_machine.strategy.wyckoff_m1m2 import compute_m1m2_scores
    HAS_M1M2 = True
except ImportError:
    HAS_M1M2 = False
    print("⚠️  Advanced Wyckoff M1/M2 module not available, using basic detector only")

# Week 1: Structure
from engine.structure.internal_external import detect_structure_state
from engine.structure.boms_detector import detect_boms
from engine.structure.squiggle_pattern import detect_squiggle_123
from engine.structure.range_classifier import classify_range_outcome

# Week 2: Psychology & Volume
from engine.psychology.pti import (
    detect_rsi_divergence, detect_volume_exhaustion,
    detect_wick_trap, detect_failed_breakout
)
from engine.volume.frvp import calculate_frvp
from engine.psychology.fakeout_intensity import detect_fakeout_intensity

# Week 4: Macro Echo
from engine.exits.macro_echo import analyze_macro_echo

# SMC Structure Levels (for Phase 1 exit invalidation)
from engine.smc.smc_engine import SMCEngine

# Contract validation
from engine.fusion.knowledge_hooks import assert_feature_contract


# ============================================================================
# RTH Filtering for Equities
# ============================================================================

def filter_rth_only(df: pd.DataFrame, asset: str) -> pd.DataFrame:
    """
    Filter to Regular Trading Hours (09:30-16:00 ET) for equities.

    Crypto (BTC, ETH) passes through unchanged.
    Equities (SPY, TSLA) are filtered to RTH bars only.
    """
    if asset in ['BTC', 'ETH', 'SOL']:
        # Crypto trades 24/7, no filtering
        return df

    # Equities: filter to RTH (09:30-16:00 ET)
    # Convert index to ET timezone
    df_et = df.copy()
    df_et.index = df_et.index.tz_convert('America/New_York')

    # Keep only bars during RTH
    rth_start = time(9, 30)
    rth_end = time(16, 0)

    mask = df_et.index.time >= rth_start
    mask &= df_et.index.time < rth_end

    # Convert back to UTC
    df_rth = df_et[mask].copy()
    df_rth.index = df_rth.index.tz_convert('UTC')

    return df_rth


# ============================================================================
# Technical Indicators (Base Layer)
# ============================================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average True Range."""
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)

    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(period).mean()

    return atr


def calculate_adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Average Directional Index."""
    high = df['high']
    low = df['low']
    close = df['close']

    plus_dm = high.diff()
    minus_dm = -low.diff()

    plus_dm[plus_dm < 0] = 0
    minus_dm[minus_dm < 0] = 0

    tr = calculate_atr(df, 1) * period  # Un-smoothed TR

    plus_di = 100 * (plus_dm.rolling(period).mean() / tr)
    minus_di = 100 * (minus_dm.rolling(period).mean() / tr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()

    return adx


def calculate_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = df['close'].diff()

    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    avg_gain = gain.rolling(period).mean()
    avg_loss = loss.rolling(period).mean()

    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))

    return rsi


# ============================================================================
# TF1D Features (Governor - Daily Timeframe)
# ============================================================================

def compute_tf1d_features(df_1d: pd.DataFrame, df_4h: pd.DataFrame,
                         df_1h: pd.DataFrame, macro_data: dict,
                         timestamp: pd.Timestamp, config: dict,
                         precomputed: pd.DataFrame = None) -> dict:
    """
    Compute 1D pack: Wyckoff, BOMS, Range, FRVP, PTI, Macro Echo

    Args:
        precomputed: Optional DataFrame with precomputed full-series detector results
                     (Wyckoff, BOMS). If provided, use instead of re-running detectors.

    Returns dict with tf1d_* prefixed columns.
    """
    features = {}

    try:
        # Get causal windows (only past data)
        window_1d = df_1d[df_1d.index <= timestamp].tail(50)
        window_4h = df_4h[df_4h.index <= timestamp].tail(100)
        window_1h = df_1h[df_1h.index <= timestamp].tail(200)

        if len(window_1d) < 20:
            return get_default_tf1d_features()

        # Use precomputed Wyckoff if available (MUCH faster + better context)
        if precomputed is not None and timestamp in precomputed.index:
            # Use precomputed Wyckoff
            features['tf1d_wyckoff_score'] = precomputed.loc[timestamp, 'tf1d_wyckoff_score']
            features['tf1d_wyckoff_phase'] = precomputed.loc[timestamp, 'tf1d_wyckoff_phase']

            # Still need to compute BOMS (not precomputed yet)
            boms_1d = detect_boms(window_1d, timeframe='1D', config=config)
            features['tf1d_boms_detected'] = boms_1d.boms_detected

            # Normalize BOMS strength: displacement / (2.0 × ATR), capped at 1.0
            # Rationale: 2× ATR displacement = very strong, > 2× ATR = maximum strength
            atr_1d = window_1d['close'].pct_change().abs().rolling(14).mean().iloc[-1] * window_1d['close'].iloc[-1]
            if atr_1d > 0 and boms_1d.displacement > 0:
                features['tf1d_boms_strength'] = min(boms_1d.displacement / (2.0 * atr_1d), 1.0)
            else:
                features['tf1d_boms_strength'] = 0.0

            features['tf1d_boms_direction'] = boms_1d.direction if boms_1d.boms_detected else 'none'
        else:
            # Fallback: Run detectors on window (will return neutral for early timestamps)
            fusion_result = analyze_fusion(
                window_1h, window_4h, window_1d,
                config={'fusion': {'weights': {'wyckoff': 0.30, 'smc': 0.15, 'liquidity': 0.25, 'momentum': 0.30}}}
            )
            features['tf1d_wyckoff_score'] = fusion_result.wyckoff_score
            features['tf1d_wyckoff_phase'] = fusion_result.wyckoff_phase

            boms_1d = detect_boms(window_1d, timeframe='1D', config=config)
            features['tf1d_boms_detected'] = boms_1d.boms_detected

            # Normalize BOMS strength: displacement / (2.0 × ATR), capped at 1.0
            atr_1d = window_1d['close'].pct_change().abs().rolling(14).mean().iloc[-1] * window_1d['close'].iloc[-1]
            if atr_1d > 0 and boms_1d.displacement > 0:
                features['tf1d_boms_strength'] = min(boms_1d.displacement / (2.0 * atr_1d), 1.0)
            else:
                features['tf1d_boms_strength'] = 0.0

            features['tf1d_boms_direction'] = boms_1d.direction if boms_1d.boms_detected else 'none'

        # Range outcome classification
        range_outcome = classify_range_outcome(window_1d, timeframe='1D', config=config)
        features['tf1d_range_outcome'] = range_outcome.outcome
        features['tf1d_range_confidence'] = range_outcome.confidence
        features['tf1d_range_direction'] = range_outcome.direction

        # FRVP on 1D (high-level support/resistance)
        frvp_1d = calculate_frvp(window_1d, lookback=50, config=config)
        features['tf1d_frvp_poc'] = frvp_1d.poc
        features['tf1d_frvp_va_high'] = frvp_1d.va_high
        features['tf1d_frvp_va_low'] = frvp_1d.va_low
        features['tf1d_frvp_position'] = frvp_1d.current_position

        # PTI on 1D (major reversal signals)
        rsi_div = detect_rsi_divergence(window_1d, lookback=10)
        vol_exh = detect_volume_exhaustion(window_1d, lookback=5)

        features['tf1d_pti_score'] = (
            rsi_div.get('strength', 0.0) * 0.5 +
            vol_exh.get('strength', 0.0) * 0.5
        )
        features['tf1d_pti_reversal'] = features['tf1d_pti_score'] > 0.7

        # Macro Echo - Extract historical series for trend analysis
        ts_naive = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') else timestamp

        # Build Series for each macro indicator (7-day lookback for trends)
        lookback_start = timestamp - pd.Timedelta(days=7)

        def extract_macro_series(symbol: str, lookback_start_ts, end_ts) -> pd.Series:
            """Extract macro series for lookback window (daily bars for trend calculation)."""
            if symbol not in macro_data or macro_data[symbol].empty:
                # Return default single-value series
                defaults = {'DXY': 100.0, 'US10Y': 4.0, 'WTI': 75.0, 'VIX': 18.0}
                return pd.Series([defaults.get(symbol, 50.0)])

            df = macro_data[symbol]

            # Convert timestamps to tz-naive for comparison (macro data is tz-naive)
            end_naive = end_ts.replace(tzinfo=None) if hasattr(end_ts, 'tzinfo') else end_ts

            # FIX: Macro data is daily granularity. To get 7-day trend, we need last 7 DAILY bars,
            # not bars within 7-day timestamp window (which gives 0-1 bars when building hourly features).
            # Floor end timestamp to day, then grab last 7 daily bars.
            end_date = pd.Timestamp(end_naive).normalize()  # Floor to day

            # Filter to bars up to end date
            available = df[df['timestamp'] <= end_date]

            if available.empty:
                defaults = {'DXY': 100.0, 'US10Y': 4.0, 'WTI': 75.0, 'VIX': 18.0}
                return pd.Series([defaults.get(symbol, 50.0)])

            # Grab last 7 daily bars (or however many are available)
            last_n_bars = available.tail(7)

            return last_n_bars['value'].reset_index(drop=True)

        dxy_series = extract_macro_series('DXY', lookback_start, timestamp)
        yields_series = extract_macro_series('US10Y', lookback_start, timestamp)
        oil_series = extract_macro_series('WTI', lookback_start, timestamp)
        vix_series = extract_macro_series('VIX', lookback_start, timestamp)

        macro_echo = analyze_macro_echo({
            'DXY': dxy_series,
            'YIELDS_10Y': yields_series,
            'OIL': oil_series,
            'VIX': vix_series
        }, lookback=7, config=config)

        features['macro_regime'] = macro_echo.regime
        features['macro_dxy_trend'] = macro_echo.dxy_trend
        features['macro_yields_trend'] = macro_echo.yields_trend
        features['macro_oil_trend'] = macro_echo.oil_trend
        features['macro_vix_level'] = macro_echo.vix_level
        features['macro_correlation_score'] = macro_echo.correlation_score
        features['macro_exit_recommended'] = macro_echo.exit_recommended

    except Exception as e:
        print(f"WARNING: Exception in compute_tf1d_features at {timestamp}: {e}")
        import traceback
        traceback.print_exc()
        features = get_default_tf1d_features()

    return features


def get_default_tf1d_features() -> dict:
    """Default neutral values for 1D features."""
    return {
        'tf1d_wyckoff_score': 0.5,
        'tf1d_wyckoff_phase': 'transition',
        'tf1d_boms_detected': False,
        'tf1d_boms_strength': 0.0,
        'tf1d_boms_direction': 'none',
        'tf1d_range_outcome': 'none',
        'tf1d_range_confidence': 0.0,
        'tf1d_range_direction': 'neutral',
        'tf1d_frvp_poc': 0.0,
        'tf1d_frvp_va_high': 0.0,
        'tf1d_frvp_va_low': 0.0,
        'tf1d_frvp_position': 'in_va',
        'tf1d_pti_score': 0.0,
        'tf1d_pti_reversal': False,
        'macro_regime': 'neutral',
        'macro_dxy_trend': 'flat',
        'macro_yields_trend': 'flat',
        'macro_oil_trend': 'flat',
        'macro_vix_level': 'medium',
        'macro_correlation_score': 0.0,
        'macro_exit_recommended': False,
    }


# ============================================================================
# TF4H Features (Structure - 4-Hour Timeframe)
# ============================================================================

def compute_tf4h_features(df_4h: pd.DataFrame, df_1h: pd.DataFrame,
                         timestamp: pd.Timestamp, config: dict) -> dict:
    """
    Compute 4H pack: Structure, Squiggle, OB/FVG, Internal/External

    Returns dict with tf4h_* prefixed columns.
    """
    features = {}

    try:
        # Get causal windows
        window_4h = df_4h[df_4h.index <= timestamp].tail(100)
        window_1h = df_1h[df_1h.index <= timestamp].tail(200)

        if len(window_4h) < 14:
            return get_default_tf4h_features()

        # Internal vs External structure
        window_1d = pd.DataFrame()  # Not needed for 4H structure
        structure_state = detect_structure_state(
            window_1h, window_4h, window_1d, config
        )
        features['tf4h_internal_phase'] = structure_state.internal_phase
        features['tf4h_external_trend'] = structure_state.external_trend
        features['tf4h_structure_alignment'] = structure_state.alignment
        features['tf4h_conflict_score'] = structure_state.conflict_score

        # Squiggle 1-2-3 pattern
        squiggle = detect_squiggle_123(window_4h, timeframe='4H', config=config)
        features['tf4h_squiggle_stage'] = squiggle.stage
        features['tf4h_squiggle_direction'] = squiggle.direction
        features['tf4h_squiggle_entry_window'] = squiggle.entry_window
        features['tf4h_squiggle_confidence'] = squiggle.confidence

        # BOMS on 4H (CHOCH detection)
        boms_4h = detect_boms(window_4h, timeframe='4H', config=config)
        features['tf4h_choch_flag'] = boms_4h.boms_detected  # FIXED: was .detected
        features['tf4h_boms_direction'] = boms_4h.direction if boms_4h.boms_detected else 'none'
        features['tf4h_boms_displacement'] = boms_4h.displacement if boms_4h.boms_detected else 0.0
        features['tf4h_fvg_present'] = boms_4h.fvg_present if boms_4h.boms_detected else False

        # Range outcome on 4H
        range_4h = classify_range_outcome(window_4h, timeframe='4H', config=config)
        features['tf4h_range_outcome'] = range_4h.outcome
        features['tf4h_range_breakout_strength'] = range_4h.breakout_strength if range_4h.outcome != 'none' else 0.0

    except Exception as e:
        print(f"WARNING: Exception in compute_tf4h_features at {timestamp}: {e}")
        import traceback
        traceback.print_exc()
        features = get_default_tf4h_features()

    return features


def get_default_tf4h_features() -> dict:
    """Default neutral values for 4H features."""
    return {
        'tf4h_internal_phase': 'transition',
        'tf4h_external_trend': 'neutral',
        'tf4h_structure_alignment': False,
        'tf4h_conflict_score': 0.0,
        'tf4h_squiggle_stage': 0,
        'tf4h_squiggle_direction': 'none',
        'tf4h_squiggle_entry_window': False,
        'tf4h_squiggle_confidence': 0.0,
        'tf4h_choch_flag': False,
        'tf4h_boms_direction': 'none',
        'tf4h_boms_displacement': 0.0,
        'tf4h_fvg_present': False,
        'tf4h_range_outcome': 'none',
        'tf4h_range_breakout_strength': 0.0,
    }


# ============================================================================
# TF1H Features (Execution - 1-Hour Timeframe)
# ============================================================================

def compute_tf1h_features(df_1h: pd.DataFrame, timestamp: pd.Timestamp,
                         config: dict) -> dict:
    """
    Compute 1H pack: Micro PTI, FRVP local, Kelly inputs

    Returns dict with tf1h_* prefixed columns.
    """
    features = {}

    try:
        # Get causal window
        window_1h = df_1h[df_1h.index <= timestamp].tail(200)

        if len(window_1h) < 50:
            return get_default_tf1h_features()

        # Micro PTI (short-term reversal signals)
        rsi_div = detect_rsi_divergence(window_1h, lookback=20)
        vol_exh = detect_volume_exhaustion(window_1h, lookback=10)
        wick_trap = detect_wick_trap(window_1h, lookback=5)
        failed_bo = detect_failed_breakout(window_1h, lookback=20)

        pti_score = (
            rsi_div.get('strength', 0.0) * 0.30 +
            vol_exh.get('strength', 0.0) * 0.25 +
            wick_trap.get('strength', 0.0) * 0.25 +
            failed_bo.get('strength', 0.0) * 0.20
        )

        features['tf1h_pti_score'] = pti_score

        # Classify trap type based on PTI components
        trap_type = 'none'
        if pti_score > 0.6:
            # Analyze which PTI component triggered
            if rsi_div.get('strength', 0.0) > 0.6:
                # Check if bearish or bullish divergence based on RSI trend vs price
                rsi_trend = rsi_div.get('type', 'none')
                if rsi_trend == 'bearish':
                    trap_type = 'bull_trap'  # Price made high but RSI diverged down
                elif rsi_trend == 'bullish':
                    trap_type = 'bear_trap'  # Price made low but RSI diverged up
            elif vol_exh.get('strength', 0.0) > 0.6:
                # Volume exhaustion suggests end of move - classify as spring or UTAD
                price_direction = window_1h['close'].iloc[-1] > window_1h['close'].iloc[-5]
                if price_direction:
                    trap_type = 'utad'  # Upthrust after distribution
                else:
                    trap_type = 'spring'  # Spring after accumulation
            elif wick_trap.get('strength', 0.0) > 0.5:
                # Wick rejection - check direction
                last_bar = window_1h.iloc[-1]
                upper_wick = last_bar['high'] - max(last_bar['open'], last_bar['close'])
                lower_wick = min(last_bar['open'], last_bar['close']) - last_bar['low']
                if upper_wick > lower_wick * 2:
                    trap_type = 'bull_trap'  # Upper wick rejection
                else:
                    trap_type = 'bear_trap'  # Lower wick rejection

        features['tf1h_pti_trap_type'] = trap_type
        features['tf1h_pti_confidence'] = pti_score
        features['tf1h_pti_reversal_likely'] = pti_score > 0.7

        # FRVP local (1H value areas for entries)
        frvp_1h = calculate_frvp(window_1h, lookback=100, config=config)
        features['tf1h_frvp_poc'] = frvp_1h.poc
        features['tf1h_frvp_va_high'] = frvp_1h.va_high
        features['tf1h_frvp_va_low'] = frvp_1h.va_low
        features['tf1h_frvp_position'] = frvp_1h.current_position
        # FIXED: Calculate distance_to_poc manually
        current_price = window_1h['close'].iloc[-1]
        features['tf1h_frvp_distance_to_poc'] = abs(current_price - frvp_1h.poc) / current_price if frvp_1h.poc > 0 else 0.0

        # Fakeout Intensity
        fakeout = detect_fakeout_intensity(window_1h, lookback=30, config=config)
        features['tf1h_fakeout_detected'] = fakeout.fakeout_detected  # FIXED: was .detected
        features['tf1h_fakeout_intensity'] = fakeout.intensity
        features['tf1h_fakeout_direction'] = fakeout.direction

        # Kelly inputs (ATR, volatility for position sizing)
        atr_14 = calculate_atr(window_1h, 14).iloc[-1]
        atr_20 = calculate_atr(window_1h, 20).iloc[-1]
        close = window_1h['close'].iloc[-1]

        features['tf1h_kelly_atr_pct'] = (atr_14 / close) * 100
        features['tf1h_kelly_volatility_ratio'] = atr_14 / atr_20 if atr_20 > 0 else 1.0
        features['tf1h_kelly_hint'] = 'reduce' if features['tf1h_kelly_volatility_ratio'] > 1.5 else 'normal'

        # SMC Structure Levels (Phase 1: Structure Invalidation Exits)
        # Extract nearest OB/FVG levels and BOS flags for exit invalidation checks
        try:
            # Initialize SMC engine with default config
            smc_config = config.get('smc', {})
            smc_engine = SMCEngine(smc_config)

            # Run SMC analysis on 1H window
            smc_signal = smc_engine.analyze_smc(window_1h)

            # Extract nearest Order Block levels (bullish = support, bearish = resistance)
            # Get the closest OB to current price for each side
            bullish_obs = [ob for ob in smc_signal.order_blocks if ob.ob_type.value == 'bullish' and ob.active]
            bearish_obs = [ob for ob in smc_signal.order_blocks if ob.ob_type.value == 'bearish' and ob.active]

            # Sort by distance to current price
            current_price = window_1h['close'].iloc[-1]
            if bullish_obs:
                nearest_bullish_ob = min(bullish_obs, key=lambda ob: abs(current_price - ob.low))
                features['tf1h_ob_low'] = nearest_bullish_ob.low
            else:
                features['tf1h_ob_low'] = None

            if bearish_obs:
                nearest_bearish_ob = min(bearish_obs, key=lambda ob: abs(current_price - ob.high))
                features['tf1h_ob_high'] = nearest_bearish_ob.high
            else:
                features['tf1h_ob_high'] = None

            # Extract nearest FVG levels
            bullish_fvgs = [fvg for fvg in smc_signal.fair_value_gaps if fvg.fvg_type.value == 'bullish']
            bearish_fvgs = [fvg for fvg in smc_signal.fair_value_gaps if fvg.fvg_type.value == 'bearish']

            if bullish_fvgs:
                nearest_bullish_fvg = min(bullish_fvgs, key=lambda fvg: abs(current_price - fvg.low))
                features['tf1h_fvg_low'] = nearest_bullish_fvg.low
                features['tf1h_fvg_present'] = True
            else:
                features['tf1h_fvg_low'] = None
                features['tf1h_fvg_present'] = False

            if bearish_fvgs:
                nearest_bearish_fvg = min(bearish_fvgs, key=lambda fvg: abs(current_price - fvg.high))
                features['tf1h_fvg_high'] = nearest_bearish_fvg.high
                features['tf1h_fvg_present'] = features['tf1h_fvg_present'] or True
            else:
                features['tf1h_fvg_high'] = None

            # Extract BOS (Break of Structure) flags
            # Recent BOS events (last 3 bars) indicate structure breaks
            if smc_signal.structure_breaks:
                recent_bos = smc_signal.structure_breaks[-3:]  # Last 3 BOS events

                # Check for bearish BOS (structure break to downside)
                bearish_bos_recent = any(bos.bos_type.value == 'bearish' for bos in recent_bos)
                # Check for bullish BOS (structure break to upside)
                bullish_bos_recent = any(bos.bos_type.value == 'bullish' for bos in recent_bos)

                features['tf1h_bos_bearish'] = bearish_bos_recent
                features['tf1h_bos_bullish'] = bullish_bos_recent
            else:
                features['tf1h_bos_bearish'] = False
                features['tf1h_bos_bullish'] = False

            # Breaker Block (BB) levels - Use strongest OB as BB proxy
            # (True BB detection would require mitigation tracking, using OB for now)
            features['tf1h_bb_low'] = features['tf1h_ob_low']  # Proxy: strongest bullish OB
            features['tf1h_bb_high'] = features['tf1h_ob_high']  # Proxy: strongest bearish OB

        except Exception as smc_error:
            # Fallback to None if SMC analysis fails
            features['tf1h_ob_low'] = None
            features['tf1h_ob_high'] = None
            features['tf1h_bb_low'] = None
            features['tf1h_bb_high'] = None
            features['tf1h_fvg_low'] = None
            features['tf1h_fvg_high'] = None
            features['tf1h_fvg_present'] = False
            features['tf1h_bos_bearish'] = False
            features['tf1h_bos_bullish'] = False

    except Exception as e:
        print(f"WARNING: Exception in compute_tf1h_features at {timestamp}: {e}")
        import traceback
        traceback.print_exc()
        features = get_default_tf1h_features()

    return features


def get_default_tf1h_features() -> dict:
    """Default neutral values for 1H features."""
    return {
        'tf1h_pti_score': 0.0,
        'tf1h_pti_trap_type': 'none',
        'tf1h_pti_confidence': 0.0,
        'tf1h_pti_reversal_likely': False,
        'tf1h_frvp_poc': 0.0,
        'tf1h_frvp_va_high': 0.0,
        'tf1h_frvp_va_low': 0.0,
        'tf1h_frvp_position': 'in_va',
        'tf1h_frvp_distance_to_poc': 0.0,
        'tf1h_fakeout_detected': False,
        'tf1h_fakeout_intensity': 0.0,
        'tf1h_fakeout_direction': 'none',
        'tf1h_kelly_atr_pct': 0.0,
        'tf1h_kelly_volatility_ratio': 1.0,
        'tf1h_kelly_hint': 'normal',
        # SMC Structure Levels (Phase 1)
        'tf1h_ob_low': None,
        'tf1h_ob_high': None,
        'tf1h_bb_low': None,
        'tf1h_bb_high': None,
        'tf1h_fvg_low': None,
        'tf1h_fvg_high': None,
        'tf1h_fvg_present': False,
        'tf1h_bos_bearish': False,
        'tf1h_bos_bullish': False,
    }


# ============================================================================
# MTF Down-Casting (Forward-Fill to 1H Resolution)
# ============================================================================

def downcast_to_1h(df_1d_features: pd.DataFrame, df_4h_features: pd.DataFrame,
                   df_1h_index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Forward-fill 1D and 4H features to 1H resolution.

    - 1D features: forward-fill for 24 hours (1 bar = 1 day)
    - 4H features: forward-fill for 4 bars (1 bar = 4 hours)

    Returns DataFrame with 1H index and all MTF features aligned.
    """
    # Create empty dataframe with 1H index
    result = pd.DataFrame(index=df_1h_index)

    # Forward-fill 1D features (24H = 1 day)
    if not df_1d_features.empty:
        df_1d_resampled = df_1d_features.reindex(
            df_1h_index, method='ffill', limit=24  # Forward-fill max 24 bars (1 day)
        )
        result = result.join(df_1d_resampled)

    # Forward-fill 4H features (4 bars = 4 hours)
    if not df_4h_features.empty:
        df_4h_resampled = df_4h_features.reindex(
            df_1h_index, method='ffill', limit=4  # Forward-fill max 4 bars (4 hours)
        )
        result = result.join(df_4h_resampled)

    return result


# ============================================================================
# MTF Alignment & Conflict Detection
# ============================================================================

def compute_mtf_alignment(row: pd.Series) -> dict:
    """
    Compute MTF alignment flags and conflict scores.

    Returns dict with mtf_* prefixed columns.
    """
    features = {}

    # Extract key signals from each timeframe
    tf1d_bullish = row.get('tf1d_wyckoff_score', 0.5) > 0.6
    tf1d_bearish = row.get('tf1d_wyckoff_score', 0.5) < 0.4

    tf4h_bullish = row.get('tf4h_external_trend', 'neutral') == 'bullish'
    tf4h_bearish = row.get('tf4h_external_trend', 'neutral') == 'bearish'

    tf1h_reversal = row.get('tf1h_pti_reversal_likely', False)

    # Check alignment
    all_bullish = tf1d_bullish and tf4h_bullish and not tf1h_reversal
    all_bearish = tf1d_bearish and tf4h_bearish and not tf1h_reversal

    features['mtf_alignment_ok'] = all_bullish or all_bearish

    # Conflict score (0.0 = aligned, 1.0 = maximum conflict)
    conflict = 0.0
    if tf1d_bullish and tf4h_bearish:
        conflict += 0.5
    if tf1d_bearish and tf4h_bullish:
        conflict += 0.5
    if (tf1d_bullish or tf4h_bullish) and tf1h_reversal:
        conflict += 0.3

    features['mtf_conflict_score'] = min(conflict, 1.0)

    # Governor veto (1D overrides lower TFs)
    features['mtf_governor_veto'] = row.get('macro_exit_recommended', False) or \
                                    row.get('tf1d_pti_reversal', False)

    return features


# ============================================================================
# Main Build Pipeline
# ============================================================================

def build_mtf_feature_store(asset: str, start_date: str, end_date: str):
    """
    Build multi-timeframe feature store with 1D → 4H → 1H down-casting.

    Output schema:
    - OHLCV (1H resolution)
    - Technical indicators (ATR, ADX, RSI, SMAs)
    - tf1d_* features (forward-filled from 1D)
    - tf4h_* features (forward-filled from 4H)
    - tf1h_* features (native 1H)
    - mtf_* alignment flags
    - macro_* regime features
    - k2_* Knowledge v2.0 fusion outputs (for future integration)
    """
    print("=" * 80)
    print(f"Multi-Timeframe Feature Store Builder v2.0")
    print("=" * 80)
    print(f"Asset:  {asset}")
    print(f"Period: {start_date} → {end_date}")
    print(f"Output: data/features_mtf/{asset}_1H_{start_date}_to_{end_date}.parquet")
    print("=" * 80)

    # Load raw OHLCV data
    print("\n📊 Loading OHLCV data...")
    df_1h_raw = load_tv(f"{asset}_1H")
    df_4h_raw = load_tv(f"{asset}_4H")
    df_1d_raw = load_tv(f"{asset}_1D")

    # Apply RTH filtering for equities
    print(f"\n🕒 Applying {'RTH filter' if asset in ['SPY', 'TSLA'] else '24/7 schedule'} for {asset}...")
    df_1h = filter_rth_only(df_1h_raw, asset)
    df_4h = filter_rth_only(df_4h_raw, asset)
    df_1d = filter_rth_only(df_1d_raw, asset)

    # CRITICAL FIX: Load extra history BEFORE start_date for Wyckoff warm-up
    # Wyckoff needs 150-300 days of context, so prepend 1 year before start_date
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')

    WARMUP_DAYS = 360  # 1 year of historical context for Wyckoff
    warmup_start = start_ts - pd.Timedelta(days=WARMUP_DAYS)

    print(f"\n🔥 Loading warm-up history from {warmup_start.date()} (Wyckoff needs 300+ days)")

    # Load full history including warm-up period
    df_1h_full = df_1h[df_1h.index >= warmup_start].copy()
    df_4h_full = df_4h[df_4h.index >= warmup_start].copy()
    df_1d_full = df_1d[df_1d.index >= warmup_start].copy()

    # Standardize columns on full datasets
    for df in [df_1h_full, df_4h_full, df_1d_full]:
        df.columns = [c.lower() for c in df.columns]

    print(f"   1D: {len(df_1d_full)} bars (includes {WARMUP_DAYS}-day warm-up)")
    print(f"   4H: {len(df_4h_full)} bars")
    print(f"   1H: {len(df_1h_full)} bars")

    # Store output range indices (what we'll actually save to parquet)
    output_1h_index = df_1h_full[(df_1h_full.index >= start_ts) & (df_1h_full.index <= end_ts)].index
    output_4h_index = df_4h_full[(df_4h_full.index >= start_ts) & (df_4h_full.index <= end_ts)].index
    output_1d_index = df_1d_full[(df_1d_full.index >= start_ts) & (df_1d_full.index <= end_ts)].index

    # Use full series for all detector calls
    df_1h = df_1h_full
    df_4h = df_4h_full
    df_1d = df_1d_full

    # Load macro data
    print("\n📈 Loading macro data...")
    macro_data = load_macro_data()
    config = {}  # Default config for all modules

    # ========================================================================
    # PRECOMPUTE: Run full-series detectors ONCE for long-horizon signals
    # ========================================================================
    print("\n🔮 Precomputing full-series detectors (Wyckoff, BOMS, Structure)...")

    # Wyckoff on full 1D series (needs 200-400 bars of context)
    # Call detector in rolling manner with FULL historical context per timestamp
    print("   Running Wyckoff detector with full historical context...")
    if HAS_M1M2:
        print("      Using ADVANCED M1/M2 Wyckoff implementation")
    else:
        print("      Using basic Wyckoff implementation")

    wyckoff_results = []

    for i, timestamp in enumerate(df_1d.index):
        # Use ALL history up to this timestamp (not just tail(50))
        historical_window = df_1d[df_1d.index <= timestamp]

        if len(historical_window) >= 50:  # Min history requirement
            # Basic Wyckoff phases
            wyck_dict = detect_wyckoff_phase(historical_window, config, usdt_stag_strength=0.5)
            phase = wyck_dict.get('phase', 'transition')
            confidence = wyck_dict.get('confidence', 0.0)
            direction = wyck_dict.get('direction', 'neutral')

            # Advanced M1/M2 scores (if available)
            m1_score = 0.0
            m2_score = 0.0
            m1m2_side = 'neutral'

            if HAS_M1M2 and len(historical_window) >= 20:
                try:
                    m1m2_result = compute_m1m2_scores(
                        df_ltf=historical_window,
                        tf='1D',
                        df_htf=None,  # Using 1D as highest timeframe
                        fib_scores=None
                    )
                    m1_score = m1m2_result.get('m1', 0.0)
                    m2_score = m1m2_result.get('m2', 0.0)
                    m1m2_side = m1m2_result.get('side', 'neutral')

                    # Override basic phase if M1/M2 has strong signal
                    if m1_score > 0.6:
                        phase = 'spring'
                        confidence = max(confidence, m1_score)
                        direction = 'long'
                    elif m2_score > 0.5:
                        phase = 'markup'
                        confidence = max(confidence, m2_score)
                        direction = 'long'
                except Exception as e:
                    # Fall back to basic detector on error
                    pass
        else:
            phase = 'transition'
            confidence = 0.0
            direction = 'neutral'
            m1_score = 0.0
            m2_score = 0.0
            m1m2_side = 'neutral'

        wyckoff_results.append({
            'timestamp': timestamp,
            'tf1d_wyckoff_phase': phase,
            'tf1d_wyckoff_direction': direction,
            'tf1d_wyckoff_confidence': confidence,
            'tf1d_wyckoff_m1': m1_score,
            'tf1d_wyckoff_m2': m2_score,
            'tf1d_wyckoff_m1m2_side': m1m2_side
        })

    # Create precomputed 1D features dataframe
    precomputed_1d = pd.DataFrame(wyckoff_results).set_index('timestamp')

    # Convert Wyckoff phase to score (from domain_fusion.py logic)
    wyckoff_score_map = {
        'accumulation': 0.7,
        'markup': 0.9,
        'distribution': 0.3,
        'markdown': 0.1,
        'transition': 0.5,
        'reaccumulation': 0.6,
        'redistribution': 0.4,
        'B': 0.6,  # Phase B = reaccumulation
        'spring': 0.8,  # Spring = strong buy (M1 pattern)
        'upthrust': 0.2,  # Upthrust = strong sell
        None: 0.5  # Handle None phase
    }
    precomputed_1d['tf1d_wyckoff_score'] = precomputed_1d['tf1d_wyckoff_phase'].map(wyckoff_score_map).fillna(0.5)

    # Enhance score with M1/M2 signals if available
    if HAS_M1M2 and 'tf1d_wyckoff_m1' in precomputed_1d.columns:
        # M1 (spring) boosts long bias
        m1_boost = (precomputed_1d['tf1d_wyckoff_m1'] - 0.5) * 0.3  # ±0.15 max
        # M2 (markup) boosts long bias
        m2_boost = (precomputed_1d['tf1d_wyckoff_m2'] - 0.5) * 0.2  # ±0.10 max
        precomputed_1d['tf1d_wyckoff_score'] = (precomputed_1d['tf1d_wyckoff_score'] + m1_boost + m2_boost).clip(0.0, 1.0)

    print(f"   ✅ Wyckoff: {len(precomputed_1d)} bars, score range [{precomputed_1d['tf1d_wyckoff_score'].min():.2f}, {precomputed_1d['tf1d_wyckoff_score'].max():.2f}]")
    print(f"      Unique phases: {precomputed_1d['tf1d_wyckoff_phase'].unique().tolist()}")

    if HAS_M1M2 and 'tf1d_wyckoff_m1' in precomputed_1d.columns:
        m1_bars = (precomputed_1d['tf1d_wyckoff_m1'] > 0.5).sum()
        m2_bars = (precomputed_1d['tf1d_wyckoff_m2'] > 0.5).sum()
        print(f"      M1 (spring) signals: {m1_bars} bars, M2 (markup) signals: {m2_bars} bars")
        print(f"      M1 score range: [{precomputed_1d['tf1d_wyckoff_m1'].min():.2f}, {precomputed_1d['tf1d_wyckoff_m1'].max():.2f}]")
        print(f"      M2 score range: [{precomputed_1d['tf1d_wyckoff_m2'].min():.2f}, {precomputed_1d['tf1d_wyckoff_m2'].max():.2f}]")

    # BOMS on 1D series - also returns single signal, skip precompute for now
    # TODO: Implement rolling BOMS detection similar to Wyckoff
    print("   Skipping BOMS precompute (will use per-timestamp detection)...")

    # Initialize feature dataframe (1H resolution)
    print("\n🏗️  Building feature store...")
    features = pd.DataFrame(index=df_1h.index)

    # OHLCV baseline
    features['open'] = df_1h['open']
    features['high'] = df_1h['high']
    features['low'] = df_1h['low']
    features['close'] = df_1h['close']
    features['volume'] = df_1h['volume']

    # Technical indicators (1H native)
    print("   Computing technical indicators...")
    features['atr_14'] = calculate_atr(df_1h, 14)
    features['atr_20'] = calculate_atr(df_1h, 20)
    features['adx_14'] = calculate_adx(df_1h, 14)
    features['rsi_14'] = calculate_rsi(df_1h, 14)

    for period in [20, 50, 100, 200]:
        features[f'sma_{period}'] = df_1h['close'].rolling(period).mean()

    # Compute 1D features (governor)
    print("\n🌅 Computing 1D features (governor)...")
    tf1d_features_list = []

    for i, timestamp in enumerate(df_1d.index):
        if i % 10 == 0:
            print(f"   Processing 1D bar {i+1}/{len(df_1d)}...")

        tf1d_feats = compute_tf1d_features(
            df_1d, df_4h, df_1h, macro_data, timestamp, config,
            precomputed=precomputed_1d  # Pass precomputed Wyckoff + BOMS
        )
        tf1d_features_list.append({'timestamp': timestamp, **tf1d_feats})

    df_1d_features = pd.DataFrame(tf1d_features_list).set_index('timestamp')
    print(f"   ✅ Computed {len(df_1d_features)} 1D feature rows")

    # Compute 4H features (structure)
    print("\n🏗️  Computing 4H features (structure)...")
    tf4h_features_list = []

    for i, timestamp in enumerate(df_4h.index):
        if i % 25 == 0:
            print(f"   Processing 4H bar {i+1}/{len(df_4h)}...")

        tf4h_feats = compute_tf4h_features(
            df_4h, df_1h, timestamp, config
        )
        tf4h_features_list.append({'timestamp': timestamp, **tf4h_feats})

    df_4h_features = pd.DataFrame(tf4h_features_list).set_index('timestamp')
    print(f"   ✅ Computed {len(df_4h_features)} 4H feature rows")

    # Compute 1H features (execution)
    print("\n⚡ Computing 1H features (execution)...")
    tf1h_features_list = []

    for i, timestamp in enumerate(df_1h.index):
        if i % 100 == 0:
            print(f"   Processing 1H bar {i+1}/{len(df_1h)}...")

        tf1h_feats = compute_tf1h_features(df_1h, timestamp, config)
        tf1h_features_list.append({'timestamp': timestamp, **tf1h_feats})

    df_1h_features = pd.DataFrame(tf1h_features_list).set_index('timestamp')
    print(f"   ✅ Computed {len(df_1h_features)} 1H feature rows")

    # Down-cast to 1H resolution
    print("\n🔽 Down-casting 1D/4H features to 1H resolution...")
    mtf_downcast = downcast_to_1h(df_1d_features, df_4h_features, df_1h.index)

    # Merge all features
    features = features.join(mtf_downcast)
    features = features.join(df_1h_features)

    # Compute MTF alignment flags
    print("\n🎯 Computing MTF alignment flags...")
    mtf_alignment_list = []
    for idx, row in features.iterrows():
        alignment = compute_mtf_alignment(row)
        mtf_alignment_list.append(alignment)

    alignment_df = pd.DataFrame(mtf_alignment_list, index=features.index)
    features = features.join(alignment_df)
    print(f"DEBUG: After MTF alignment, features has {len(features.columns)} columns")

    # Phase 4 Re-Entry: Add tf4h_fusion_score and volume_zscore
    print("\n🎯 Computing Phase 4 re-entry features...")
    print(f"DEBUG: Starting Phase 4 feature computation...")

    # tf4h_fusion_score: Calculate from available 4H features (structure alignment + squiggle + CHOCH)
    # Use structure alignment + squiggle confidence + CHOCH as proxies for 4H fusion
    tf4h_fusion = 0.0

    # Check each component and add to fusion score
    if 'tf4h_structure_alignment' in features.columns:
        # Internal/external aligned (30% weight)
        features['tf4h_fusion_score'] = features['tf4h_structure_alignment'].astype(float) * 0.30
        tf4h_fusion = features['tf4h_fusion_score'].copy()
    else:
        features['tf4h_fusion_score'] = 0.0
        tf4h_fusion = features['tf4h_fusion_score'].copy()

    if 'tf4h_squiggle_entry_window' in features.columns:
        # Squiggle 1-2-3 entry window (20% weight)
        tf4h_fusion += features['tf4h_squiggle_entry_window'].astype(float) * 0.20

        # Squiggle confidence (20% weight)
        if 'tf4h_squiggle_confidence' in features.columns:
            tf4h_fusion += features['tf4h_squiggle_confidence'] * 0.20

    if 'tf4h_choch_flag' in features.columns:
        # CHOCH detected (30% weight)
        tf4h_fusion += features['tf4h_choch_flag'].astype(float) * 0.30

    # Cap at 1.0 and update features
    features['tf4h_fusion_score'] = tf4h_fusion.clip(upper=1.0)

    # Smooth with 4-bar rolling mean to reduce noise
    features['tf4h_fusion_score'] = features['tf4h_fusion_score'].rolling(4, min_periods=1).mean()

    # volume_zscore: Z-score of volume with 20-bar lookback
    features['volume_zscore'] = (
        (features['volume'] - features['volume'].rolling(20, min_periods=1).mean()) /
        features['volume'].rolling(20, min_periods=1).std()
    ).fillna(0.0)

    print(f"   ✅ Added tf4h_fusion_score (range: {features['tf4h_fusion_score'].min():.3f} to {features['tf4h_fusion_score'].max():.3f})")
    print(f"   ✅ Added volume_zscore (range: {features['volume_zscore'].min():.2f} to {features['volume_zscore'].max():.2f})")

    # Add placeholders for K2 fusion outputs (for future use)
    features['k2_threshold_delta'] = 0.0
    features['k2_score_delta'] = 0.0
    features['k2_fusion_score'] = 0.5

    # Filter to requested output range (exclude warm-up period)
    print(f"\n✂️  Filtering to output range: {start_date} → {end_date}")
    print(f"   Before filter: {len(features)} bars (includes warm-up)")
    features = features.loc[output_1h_index]
    print(f"   After filter:  {len(features)} bars (output only)")

    # Drop initial NaN rows
    features = features.dropna(subset=['atr_20', 'sma_20'])

    # Save to parquet
    output_dir = Path('data/features_mtf')
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f'{asset}_1H_{start_date}_to_{end_date}.parquet'

    # Add metadata
    features.attrs['schema_version'] = 'MTF_2.0'
    features.attrs['asset'] = asset
    features.attrs['start_date'] = start_date
    features.attrs['end_date'] = end_date
    features.attrs['rth_filtered'] = asset in ['SPY', 'TSLA']

    features.to_parquet(output_path, compression='snappy')

    print("\n" + "=" * 80)
    print("✅ MTF Feature Store Complete!")
    print("=" * 80)
    print(f"📁 Output: {output_path}")
    print(f"📊 Shape:  {len(features)} bars × {len(features.columns)} features")
    print(f"💾 Size:   {output_path.stat().st_size / 1024 / 1024:.1f} MB")
    print("=" * 80)

    # Generate schema report
    schema_report = {
        'asset': asset,
        'period': f"{start_date} to {end_date}",
        'bars': len(features),
        'features': len(features.columns),
        'size_mb': round(output_path.stat().st_size / 1024 / 1024, 2),
        'schema_version': 'MTF_2.0',
        'rth_filtered': asset in ['SPY', 'TSLA'],
        'columns': {
            'ohlcv': ['open', 'high', 'low', 'close', 'volume'],
            'technicals': [c for c in features.columns if c.startswith(('atr_', 'adx_', 'rsi_', 'sma_'))],
            'tf1d': [c for c in features.columns if c.startswith('tf1d_')],
            'tf4h': [c for c in features.columns if c.startswith('tf4h_')],
            'tf1h': [c for c in features.columns if c.startswith('tf1h_')],
            'mtf': [c for c in features.columns if c.startswith('mtf_')],
            'macro': [c for c in features.columns if c.startswith('macro_')],
            'k2': [c for c in features.columns if c.startswith('k2_')],
        },
        'column_counts': {
            'tf1d': len([c for c in features.columns if c.startswith('tf1d_')]),
            'tf4h': len([c for c in features.columns if c.startswith('tf4h_')]),
            'tf1h': len([c for c in features.columns if c.startswith('tf1h_')]),
            'mtf': len([c for c in features.columns if c.startswith('mtf_')]),
            'macro': len([c for c in features.columns if c.startswith('macro_')]),
            'total': len(features.columns),
        }
    }

    # Save schema report
    schema_path = output_dir / f'{asset}_schema_report.json'
    with open(schema_path, 'w') as f:
        json.dump(schema_report, f, indent=2)

    print(f"\n📋 Schema Report: {schema_path}")
    print(f"\n   Total Features: {schema_report['column_counts']['total']}")
    print(f"   - 1D Features:  {schema_report['column_counts']['tf1d']}")
    print(f"   - 4H Features:  {schema_report['column_counts']['tf4h']}")
    print(f"   - 1H Features:  {schema_report['column_counts']['tf1h']}")
    print(f"   - MTF Flags:    {schema_report['column_counts']['mtf']}")
    print(f"   - Macro:        {schema_report['column_counts']['macro']}")
    print("=" * 80)

    return features


def main():
    parser = argparse.ArgumentParser(
        description='Build multi-timeframe feature store with 1D→4H→1H down-casting'
    )
    parser.add_argument('--asset', required=True,
                       help='Asset to process (BTC, ETH, SPY, TSLA)')
    parser.add_argument('--start', default='2024-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', default=datetime.now().strftime('%Y-%m-%d'),
                       help='End date (YYYY-MM-DD, default: today)')

    args = parser.parse_args()

    # Validate asset
    valid_assets = ['BTC', 'ETH', 'SPY', 'TSLA', 'SOL']
    if args.asset not in valid_assets:
        print(f"❌ Error: Asset must be one of {valid_assets}")
        return

    build_mtf_feature_store(args.asset, args.start, args.end)


if __name__ == '__main__':
    main()
