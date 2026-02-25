#!/usr/bin/env python3
"""
ML Feature Engineering for Bull Machine v1.8.6
==============================================

Builds regime feature vectors from macro snapshots for ML-based config suggestion.

Features include:
- VIX, MOVE, DXY (core volatility/dollar)
- Oil, Gold (commodities)
- US2Y, US10Y (yields)
- TOTAL/TOTAL2/TOTAL3 (crypto breadth)
- USDT.D, BTC.D (dominance)
- Derived features (ROC, z-score, EMA)
"""

import numpy as np
from typing import Dict, List, Optional
import pandas as pd


def build_regime_vector(macro_snapshot: Dict, lookback_window: Optional[pd.DataFrame] = None) -> Dict:
    """
    Build feature vector from macro snapshot for ML model.

    Args:
        macro_snapshot: Current macro data dict with all indicators
            Format: {
                'VIX': {'value': 24.5, 'stale': False},
                'MOVE': {'value': 85.0, 'stale': False},
                ...
            }
        lookback_window: Optional DataFrame with historical macro data for derived features
            Columns: timestamp, vix, move, dxy, oil, gold, us2y, us10y, etc.

    Returns:
        Dict with:
            - 'features': Dict of feature_name -> value
            - 'feature_vector': np.ndarray ready for model input
            - 'feature_names': List of feature names in order
    """
    features = {}

    # Helper to safely extract values
    def get_value(symbol: str, default: float = np.nan) -> float:
        data = macro_snapshot.get(symbol, {})
        val = data.get('value')
        is_stale = data.get('stale', True)
        return val if (val is not None and not is_stale) else default

    # ========================================================================
    # CORE FEATURES (Level values)
    # ========================================================================

    features['vix'] = get_value('VIX', 20.0)
    features['move'] = get_value('MOVE', 80.0)
    features['dxy'] = get_value('DXY', 100.0)
    features['oil'] = get_value('WTI', 70.0)
    features['gold'] = get_value('GOLD', 2500.0)
    features['us2y'] = get_value('US2Y', 4.0)
    features['us10y'] = get_value('US10Y', 4.0)

    # Crypto breadth
    features['total_mc'] = get_value('TOTAL', np.nan)
    features['total2_mc'] = get_value('TOTAL2', np.nan)
    features['total3_mc'] = get_value('TOTAL3', np.nan)

    # Dominance
    features['usdt_d'] = get_value('USDT.D', np.nan)
    features['btc_d'] = get_value('BTC.D', 0.55)

    # ========================================================================
    # DERIVED FEATURES
    # ========================================================================

    # Yield spread (curve)
    if not np.isnan(features['us10y']) and not np.isnan(features['us2y']):
        features['yield_spread'] = features['us10y'] - features['us2y']
    else:
        features['yield_spread'] = np.nan

    # BTC dominance from TOTAL/TOTAL2
    if not np.isnan(features['total_mc']) and not np.isnan(features['total2_mc']) and features['total_mc'] > 0:
        features['btc_dominance_calc'] = 1.0 - (features['total2_mc'] / features['total_mc'])
    else:
        features['btc_dominance_calc'] = features['btc_d']  # Fallback

    # ========================================================================
    # TIME-SERIES FEATURES (if lookback_window provided)
    # ========================================================================

    if lookback_window is not None and len(lookback_window) > 0:
        # Compute ROC (rate of change) over last N periods
        for col, periods in [('vix', 5), ('move', 5), ('dxy', 10), ('oil', 10), ('gold', 10)]:
            if col in lookback_window.columns and len(lookback_window) >= periods + 1:
                current = lookback_window[col].iloc[-1]
                past = lookback_window[col].iloc[-periods-1]

                if not np.isnan(current) and not np.isnan(past) and past != 0:
                    features[f'{col}_roc_{periods}'] = ((current - past) / past) * 100.0
                else:
                    features[f'{col}_roc_{periods}'] = 0.0
            else:
                features[f'{col}_roc_{periods}'] = 0.0

        # Compute z-scores (standardized distance from recent mean)
        for col, lookback in [('vix', 60), ('move', 60), ('dxy', 60)]:
            if col in lookback_window.columns and len(lookback_window) >= lookback:
                window = lookback_window[col].iloc[-lookback:]
                valid = window[~np.isnan(window)]

                if len(valid) >= lookback // 2:
                    mean = np.mean(valid)
                    std = np.std(valid)
                    current = lookback_window[col].iloc[-1]

                    if std > 0 and not np.isnan(current):
                        features[f'{col}_zscore'] = (current - mean) / std
                    else:
                        features[f'{col}_zscore'] = 0.0
                else:
                    features[f'{col}_zscore'] = 0.0
            else:
                features[f'{col}_zscore'] = 0.0

        # EMA features (trend direction)
        for col, period in [('dxy', 10), ('dxy', 50), ('oil', 20), ('gold', 20)]:
            if col in lookback_window.columns and len(lookback_window) >= period:
                # Simple EMA approximation using exponential weights
                alpha = 2.0 / (period + 1)
                values = lookback_window[col].iloc[-period:].values
                valid_values = values[~np.isnan(values)]

                if len(valid_values) > 0:
                    ema = valid_values[0]
                    for v in valid_values[1:]:
                        ema = alpha * v + (1 - alpha) * ema
                    features[f'{col}_ema_{period}'] = ema
                else:
                    features[f'{col}_ema_{period}'] = features[col]
            else:
                features[f'{col}_ema_{period}'] = features[col]

    else:
        # No lookback window - fill with zeros/defaults
        for col in ['vix', 'move', 'dxy', 'oil', 'gold']:
            features[f'{col}_roc_5'] = 0.0 if col in ['vix', 'move'] else 0.0
            features[f'{col}_roc_10'] = 0.0 if col in ['dxy', 'oil', 'gold'] else 0.0
            features[f'{col}_zscore'] = 0.0 if col in ['vix', 'move', 'dxy'] else 0.0

        for col in ['dxy', 'oil', 'gold']:
            features[f'{col}_ema_10'] = features.get(col, 0.0)
            features[f'{col}_ema_20'] = features.get(col, 0.0)
        features['dxy_ema_50'] = features.get('dxy', 100.0)

    # ========================================================================
    # REGIME CLASSIFICATION FEATURES
    # ========================================================================

    # VIX regime (calm/elevated/panic)
    vix_val = features['vix']
    if vix_val < 18:
        features['vix_regime'] = 0  # Calm
    elif vix_val < 30:
        features['vix_regime'] = 1  # Elevated
    else:
        features['vix_regime'] = 2  # Panic

    # DXY regime (weak/neutral/strong)
    dxy_val = features['dxy']
    if dxy_val < 100:
        features['dxy_regime'] = 0  # Weak (crypto bullish)
    elif dxy_val < 105:
        features['dxy_regime'] = 1  # Neutral
    else:
        features['dxy_regime'] = 2  # Strong (crypto bearish)

    # Yield curve regime (inverted/flat/steep)
    spread = features['yield_spread']
    if np.isnan(spread):
        features['curve_regime'] = 1  # Unknown -> neutral
    elif spread < -0.2:
        features['curve_regime'] = 0  # Inverted (recession risk)
    elif spread < 0.5:
        features['curve_regime'] = 1  # Flat
    else:
        features['curve_regime'] = 2  # Steep (growth expectations)

    # ========================================================================
    # BUILD FEATURE VECTOR
    # ========================================================================

    # Define feature order (must be consistent across all calls)
    feature_names = [
        # Level features
        'vix', 'move', 'dxy', 'oil', 'gold', 'us2y', 'us10y',
        'total_mc', 'total2_mc', 'total3_mc', 'usdt_d', 'btc_d',
        'yield_spread', 'btc_dominance_calc',

        # ROC features
        'vix_roc_5', 'move_roc_5', 'dxy_roc_10', 'oil_roc_10', 'gold_roc_10',

        # Z-score features
        'vix_zscore', 'move_zscore', 'dxy_zscore',

        # EMA features
        'dxy_ema_10', 'dxy_ema_50', 'oil_ema_20', 'gold_ema_20',

        # Regime classification
        'vix_regime', 'dxy_regime', 'curve_regime'
    ]

    # Build vector (replace NaN with 0.0 for model input)
    feature_vector = np.array([features.get(name, 0.0) for name in feature_names], dtype=np.float32)
    feature_vector = np.nan_to_num(feature_vector, nan=0.0, posinf=0.0, neginf=0.0)

    return {
        'features': features,
        'feature_vector': feature_vector,
        'feature_names': feature_names
    }


def build_training_row(
    regime_vector: Dict,
    config: Dict,
    metrics: Dict,
    metadata: Dict = None
) -> Dict:
    """
    Build training row combining regime features + config params + performance metrics.

    Args:
        regime_vector: Output from build_regime_vector()
        config: Config dict used for backtest
        metrics: Performance metrics (PF, Sharpe, MaxDD, trades, etc.)
        metadata: Optional metadata (asset, date range, etc.)

    Returns:
        Dict ready to append to training dataset
    """
    row = {}

    # Add regime features
    row.update(regime_vector['features'])

    # Add config parameters
    row['config_fusion_threshold'] = config.get('fusion', {}).get('entry_threshold_confidence', 0.70)
    row['config_wyckoff_weight'] = config.get('fusion', {}).get('weights', {}).get('wyckoff', 0.30)
    row['config_smc_weight'] = config.get('fusion', {}).get('weights', {}).get('smc', 0.15)
    row['config_hob_weight'] = config.get('fusion', {}).get('weights', {}).get('liquidity', 0.25)
    row['config_momentum_weight'] = config.get('fusion', {}).get('weights', {}).get('momentum', 0.30)

    row['config_stop_atr'] = config.get('exits', {}).get('atr_k', 1.0)
    row['config_trail_atr'] = config.get('exits', {}).get('trail_atr_k', 1.2)
    row['config_tp1_r'] = config.get('exits', {}).get('tp1_r', 1.0)

    row['config_base_risk_pct'] = config.get('risk', {}).get('base_risk_pct', 0.0075)
    row['config_adx_threshold'] = config.get('fast_signals', {}).get('adx_threshold', 20)

    # Add performance metrics (targets)
    row['pf'] = metrics.get('profit_factor', 0.0)
    row['sharpe'] = metrics.get('sharpe', 0.0)
    row['max_dd'] = metrics.get('max_drawdown', 0.0)
    row['total_trades'] = metrics.get('total_trades', 0)
    row['win_rate'] = metrics.get('win_rate', 0.0)
    row['total_return_pct'] = metrics.get('total_return_pct', 0.0)
    row['avg_r'] = metrics.get('avg_r_multiple', 0.0)

    # Add metadata
    if metadata:
        row['asset'] = metadata.get('asset', 'BTC')
        row['start_date'] = str(metadata.get('start_date', ''))
        row['end_date'] = str(metadata.get('end_date', ''))

    return row


def normalize_features(feature_vectors: np.ndarray, method: str = 'standardize') -> tuple:
    """
    Normalize feature vectors for ML model input.

    Args:
        feature_vectors: (N, F) array of features
        method: 'standardize' (z-score) or 'minmax' (0-1 scaling)

    Returns:
        (normalized_vectors, normalization_params)
    """
    if method == 'standardize':
        mean = np.mean(feature_vectors, axis=0)
        std = np.std(feature_vectors, axis=0)
        std = np.where(std == 0, 1.0, std)  # Avoid division by zero

        normalized = (feature_vectors - mean) / std
        params = {'method': 'standardize', 'mean': mean, 'std': std}

    elif method == 'minmax':
        min_val = np.min(feature_vectors, axis=0)
        max_val = np.max(feature_vectors, axis=0)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1.0, range_val)  # Avoid division by zero

        normalized = (feature_vectors - min_val) / range_val
        params = {'method': 'minmax', 'min': min_val, 'max': max_val, 'range': range_val}

    else:
        raise ValueError(f"Unknown normalization method: {method}")

    return normalized, params


def apply_normalization(feature_vectors: np.ndarray, normalization_params: Dict) -> np.ndarray:
    """
    Apply saved normalization parameters to new feature vectors.

    Args:
        feature_vectors: (N, F) array of features
        normalization_params: Dict from normalize_features()

    Returns:
        Normalized feature vectors
    """
    method = normalization_params['method']

    if method == 'standardize':
        mean = normalization_params['mean']
        std = normalization_params['std']
        return (feature_vectors - mean) / std

    elif method == 'minmax':
        min_val = normalization_params['min']
        range_val = normalization_params['range']
        return (feature_vectors - min_val) / range_val

    else:
        raise ValueError(f"Unknown normalization method: {method}")
