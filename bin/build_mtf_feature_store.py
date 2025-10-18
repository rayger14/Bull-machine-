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
                         timestamp: pd.Timestamp, config: dict) -> dict:
    """
    Compute 1D pack: Wyckoff, BOMS, Range, FRVP, PTI, Macro Echo

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

        # Wyckoff phase (from fusion)
        fusion_result = analyze_fusion(
            window_1h, window_4h, window_1d,
            config={'fusion': {'weights': {'wyckoff': 0.30, 'smc': 0.15, 'liquidity': 0.25, 'momentum': 0.30}}}
        )

        features['tf1d_wyckoff_score'] = fusion_result.wyckoff_score
        features['tf1d_wyckoff_phase'] = fusion_result.wyckoff_phase

        # BOMS on 1D
        boms_1d = detect_boms(window_1d, timeframe='1D', config=config)
        features['tf1d_boms_detected'] = boms_1d.detected
        features['tf1d_boms_strength'] = boms_1d.displacement if boms_1d.detected else 0.0
        features['tf1d_boms_direction'] = boms_1d.direction if boms_1d.detected else 'none'

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
            rsi_div.get('divergence_strength', 0.0) * 0.5 +
            vol_exh.get('exhaustion_score', 0.0) * 0.5
        )
        features['tf1d_pti_reversal'] = features['tf1d_pti_score'] > 0.7

        # Macro Echo
        ts_naive = timestamp.replace(tzinfo=None) if hasattr(timestamp, 'tzinfo') else timestamp
        snapshot = fetch_macro_snapshot(macro_data, ts_naive)
        macro_echo = analyze_macro_echo({
            'DXY': snapshot.get('dxy_series', pd.Series([100.0])),
            'YIELDS_10Y': snapshot.get('yields_series', pd.Series([4.0])),
            'OIL': snapshot.get('oil_series', pd.Series([75.0])),
            'VIX': snapshot.get('vix_series', pd.Series([18.0]))
        }, lookback=7, config=config)

        features['macro_regime'] = macro_echo.regime
        features['macro_dxy_trend'] = macro_echo.dxy_trend
        features['macro_yields_trend'] = macro_echo.yields_trend
        features['macro_oil_trend'] = macro_echo.oil_trend
        features['macro_vix_level'] = macro_echo.vix_level
        features['macro_correlation_score'] = macro_echo.correlation_score
        features['macro_exit_recommended'] = macro_echo.exit_recommended

    except Exception as e:
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
        features['tf4h_choch_flag'] = boms_4h.detected
        features['tf4h_boms_direction'] = boms_4h.direction if boms_4h.detected else 'none'
        features['tf4h_boms_displacement'] = boms_4h.displacement if boms_4h.detected else 0.0
        features['tf4h_fvg_present'] = boms_4h.fvg_present if boms_4h.detected else False

        # Range outcome on 4H
        range_4h = classify_range_outcome(window_4h, timeframe='4H', config=config)
        features['tf4h_range_outcome'] = range_4h.outcome
        features['tf4h_range_breakout_strength'] = range_4h.breakout_strength if range_4h.outcome != 'none' else 0.0

    except Exception as e:
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
            rsi_div.get('divergence_strength', 0.0) * 0.30 +
            vol_exh.get('exhaustion_score', 0.0) * 0.25 +
            wick_trap.get('trap_strength', 0.0) * 0.25 +
            failed_bo.get('failure_score', 0.0) * 0.20
        )

        features['tf1h_pti_score'] = pti_score
        features['tf1h_pti_trap_type'] = 'bullish_trap' if pti_score > 0.6 else 'none'
        features['tf1h_pti_confidence'] = pti_score
        features['tf1h_pti_reversal_likely'] = pti_score > 0.7

        # FRVP local (1H value areas for entries)
        frvp_1h = calculate_frvp(window_1h, lookback=100, config=config)
        features['tf1h_frvp_poc'] = frvp_1h.poc
        features['tf1h_frvp_va_high'] = frvp_1h.va_high
        features['tf1h_frvp_va_low'] = frvp_1h.va_low
        features['tf1h_frvp_position'] = frvp_1h.current_position
        features['tf1h_frvp_distance_to_poc'] = frvp_1h.distance_to_poc

        # Fakeout Intensity
        fakeout = detect_fakeout_intensity(window_1h, lookback=30, config=config)
        features['tf1h_fakeout_detected'] = fakeout.detected
        features['tf1h_fakeout_intensity'] = fakeout.intensity
        features['tf1h_fakeout_direction'] = fakeout.direction

        # Kelly inputs (ATR, volatility for position sizing)
        atr_14 = calculate_atr(window_1h, 14).iloc[-1]
        atr_20 = calculate_atr(window_1h, 20).iloc[-1]
        close = window_1h['close'].iloc[-1]

        features['tf1h_kelly_atr_pct'] = (atr_14 / close) * 100
        features['tf1h_kelly_volatility_ratio'] = atr_14 / atr_20 if atr_20 > 0 else 1.0
        features['tf1h_kelly_hint'] = 'reduce' if features['tf1h_kelly_volatility_ratio'] > 1.5 else 'normal'

    except Exception as e:
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

    # Filter to date range
    start_ts = pd.Timestamp(start_date, tz='UTC')
    end_ts = pd.Timestamp(end_date, tz='UTC')
    df_1h = df_1h[(df_1h.index >= start_ts) & (df_1h.index <= end_ts)].copy()
    df_4h = df_4h[(df_4h.index >= start_ts) & (df_4h.index <= end_ts)].copy()
    df_1d = df_1d[(df_1d.index >= start_ts) & (df_1d.index <= end_ts)].copy()

    # Standardize columns
    for df in [df_1h, df_4h, df_1d]:
        df.columns = [c.lower() for c in df.columns]

    print(f"   1H: {len(df_1h)} bars")
    print(f"   4H: {len(df_4h)} bars")
    print(f"   1D: {len(df_1d)} bars")

    # Load macro data
    print("\n📈 Loading macro data...")
    macro_data = load_macro_data()
    config = {}  # Default config for all modules

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
            df_1d, df_4h, df_1h, macro_data, timestamp, config
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

    # Add placeholders for K2 fusion outputs (for future use)
    features['k2_threshold_delta'] = 0.0
    features['k2_score_delta'] = 0.0
    features['k2_fusion_score'] = 0.5

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
