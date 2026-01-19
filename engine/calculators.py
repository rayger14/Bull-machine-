#!/usr/bin/env python3
"""
Feature Calculator Functions

Modular calculator functions for feature store columns.
Used by both:
- bin/build_mtf_feature_store.py (production feature store builder)
- bin/patch_feature_columns.py (column patching tool)

Architecture:
- Each calculator is a pure function: takes df_1h + OHLCV data → returns computed values
- All calculators use engine/utils_align.py for HTF alignment
- No side effects (no in-place modifications)
- Returns Series or dict of Series (multiple columns)

P0 Calculators:
1. calc_boms_displacement() - BOMS displacement on 4H/1D (absolute price)
2. calc_boms_strength() - BOMS strength normalized to [0, 1] using ATR
3. calc_tf4h_fusion() - Fusion score from 4H structure indicators

Usage:
    from engine.calculators import calc_boms_displacement, calc_boms_strength

    # Calculate BOMS displacement on 4H
    df_1h['tf4h_boms_displacement'] = calc_boms_displacement(
        df_1h=df_1h,
        timeframe='4H'
    )

    # Calculate BOMS strength on 1D
    df_1h['tf1d_boms_strength'] = calc_boms_strength(
        df_1h=df_1h,
        timeframe='1D'
    )
"""

import pandas as pd
import numpy as np
from typing import Optional
import logging

from engine.utils_align import resample_to_timeframe, align_htf_to_1h
from engine.structure.boms_detector import detect_boms

logger = logging.getLogger(__name__)


def calc_boms_displacement(
    df_1h: pd.DataFrame,
    timeframe: str = '4H',
    config: Optional[dict] = None
) -> pd.Series:
    """
    Calculate BOMS displacement on higher timeframe.

    Args:
        df_1h: 1H DataFrame with OHLCV data
        timeframe: Higher timeframe ('4H', '1D', etc.)
        config: Optional BOMS detector config

    Returns:
        Series of displacement values (absolute price) aligned to 1H

    Notes:
        - Displacement is in absolute price terms (not percentage)
        - Example: displacement=850 means price broke swing by $850
        - Uses detect_boms() from engine/structure/boms_detector.py
        - Zero values are valid (no structure break occurred)

    Expected Non-Zero Rate:
        - 4H: > 5% (structure breaks somewhat rare)
        - 1D: > 5% (daily structure breaks)

    Example:
        >>> df_1h['tf4h_boms_displacement'] = calc_boms_displacement(df_1h, '4H')
        >>> print(f"Non-zero: {(df_1h['tf4h_boms_displacement'] > 0).sum() / len(df_1h):.1%}")
        Non-zero: 8.3%
    """
    if config is None:
        config = {}

    logger.info(f"Calculating BOMS displacement on {timeframe}...")

    # Resample to higher timeframe
    df_htf = resample_to_timeframe(df_1h, timeframe)

    # Calculate displacement for each HTF bar
    displacements = []
    for idx in range(len(df_htf)):
        # Get window up to current bar (for point-in-time correctness)
        window_htf = df_htf.iloc[:idx + 1].tail(100)

        if len(window_htf) >= 30:
            boms_signal = detect_boms(window_htf, timeframe=timeframe, config=config)
            displacement = boms_signal.displacement if boms_signal else 0.0
        else:
            displacement = 0.0

        displacements.append(displacement)

    # Add displacement column to HTF dataframe
    df_htf['boms_displacement'] = displacements

    # Align back to 1H
    df_1h_aligned = align_htf_to_1h(
        df_1h=df_1h.copy(),
        df_htf=df_htf[['boms_displacement']],
        htf=timeframe,
        columns=['boms_displacement'],
        prefix=''
    )

    result = df_1h_aligned['boms_displacement']

    non_zero_pct = (result > 0).sum() / len(result) * 100
    logger.info(f"BOMS displacement {timeframe}: {non_zero_pct:.1f}% non-zero")

    return result


def calc_boms_strength(
    df_1h: pd.DataFrame,
    timeframe: str = '1D',
    config: Optional[dict] = None
) -> pd.Series:
    """
    Calculate BOMS strength normalized to [0, 1] range.

    Args:
        df_1h: 1H DataFrame with OHLCV data
        timeframe: Higher timeframe ('1D', etc.)
        config: Optional BOMS detector config

    Returns:
        Series of strength values [0, 1] aligned to 1H

    Notes:
        - Strength = displacement / (2.0 × ATR), capped at 1.0
        - Rationale: 2× ATR displacement = very strong move
        - Zero values are valid (no structure break)

    Expected Non-Zero Rate:
        - 1D: > 5% (daily BOMS events)

    Example:
        >>> df_1h['tf1d_boms_strength'] = calc_boms_strength(df_1h, '1D')
        >>> print(f"Non-zero: {(df_1h['tf1d_boms_strength'] > 0).sum() / len(df_1h):.1%}")
        Non-zero: 6.2%
    """
    if config is None:
        config = {}

    logger.info(f"Calculating BOMS strength on {timeframe}...")

    # Resample to higher timeframe
    df_htf = resample_to_timeframe(df_1h, timeframe)

    # Calculate ATR on HTF (14-period)
    df_htf['tr'] = np.maximum(
        df_htf['high'] - df_htf['low'],
        np.maximum(
            abs(df_htf['high'] - df_htf['close'].shift(1)),
            abs(df_htf['low'] - df_htf['close'].shift(1))
        )
    )
    df_htf['atr'] = df_htf['tr'].rolling(14).mean()

    # Calculate displacement and strength for each HTF bar
    strengths = []
    for idx in range(len(df_htf)):
        # Get window up to current bar
        window_htf = df_htf.iloc[:idx + 1].tail(100)

        if len(window_htf) >= 30:
            boms_signal = detect_boms(window_htf, timeframe=timeframe, config=config)
            displacement = boms_signal.displacement if boms_signal else 0.0

            # Get ATR at current bar
            atr = df_htf.iloc[idx]['atr']

            # Normalize: strength = displacement / (2.0 × ATR), capped at 1.0
            if atr > 0 and displacement > 0:
                strength = min(displacement / (2.0 * atr), 1.0)
            else:
                strength = 0.0
        else:
            strength = 0.0

        strengths.append(strength)

    # Add strength column to HTF dataframe
    df_htf['boms_strength'] = strengths

    # Align back to 1H
    df_1h_aligned = align_htf_to_1h(
        df_1h=df_1h.copy(),
        df_htf=df_htf[['boms_strength']],
        htf=timeframe,
        columns=['boms_strength'],
        prefix=''
    )

    result = df_1h_aligned['boms_strength']

    non_zero_pct = (result > 0).sum() / len(result) * 100
    logger.info(f"BOMS strength {timeframe}: {non_zero_pct:.1f}% non-zero")

    return result


def calc_tf4h_fusion(
    df_1h: pd.DataFrame,
    tf4h_features: pd.DataFrame
) -> pd.Series:
    """
    Calculate fusion score from 4H structure indicators.

    Args:
        df_1h: 1H DataFrame (for alignment)
        tf4h_features: 4H DataFrame with structure features:
            - tf4h_structure_alignment (bool)
            - tf4h_squiggle_entry_window (bool)
            - tf4h_squiggle_confidence (float 0-1)
            - tf4h_choch_flag (bool)

    Returns:
        Series of fusion scores [0, 1] aligned to 1H

    Notes:
        - Fusion score combines multiple 4H structure signals
        - Components:
          * Structure alignment: 0.30 (internal/external aligned)
          * Squiggle entry window: 0.20 (1-2-3 entry detected)
          * Squiggle confidence: 0.20 (scaled by confidence)
          * CHOCH flag: 0.30 (change of character detected)
        - Total max: 1.0

    Expected Non-Zero Rate:
        - > 15% (structure setups moderately common)

    Example:
        >>> df_1h['tf4h_fusion_score'] = calc_tf4h_fusion(df_1h, df_4h)
        >>> print(f"Non-zero: {(df_1h['tf4h_fusion_score'] > 0).sum() / len(df_1h):.1%}")
        Non-zero: 18.7%
    """
    logger.info("Calculating tf4h_fusion_score...")

    # Calculate fusion score for each 4H bar
    fusion_scores = []

    for idx in range(len(tf4h_features)):
        score = 0.0

        # Structure alignment: 0.30
        if tf4h_features.iloc[idx].get('tf4h_structure_alignment', False):
            score += 0.30

        # Squiggle entry window: 0.20
        if tf4h_features.iloc[idx].get('tf4h_squiggle_entry_window', False):
            score += 0.20

            # Squiggle confidence: 0.20 (scaled)
            confidence = tf4h_features.iloc[idx].get('tf4h_squiggle_confidence', 0.0)
            score += confidence * 0.20

        # CHOCH flag: 0.30
        if tf4h_features.iloc[idx].get('tf4h_choch_flag', False):
            score += 0.30

        # Cap at 1.0
        fusion_scores.append(min(score, 1.0))

    # Add to 4H dataframe
    tf4h_features = tf4h_features.copy()
    tf4h_features['fusion_score'] = fusion_scores

    # Align to 1H
    df_1h_aligned = align_htf_to_1h(
        df_1h=df_1h.copy(),
        df_htf=tf4h_features[['fusion_score']],
        htf='4H',
        columns=['fusion_score'],
        prefix=''
    )

    result = df_1h_aligned['fusion_score']

    non_zero_pct = (result > 0).sum() / len(result) * 100
    logger.info(f"tf4h_fusion_score: {non_zero_pct:.1f}% non-zero")

    return result


# Calculator registry for dynamic lookup
CALCULATORS = {
    'tf4h_boms_displacement': lambda df_1h, config=None: calc_boms_displacement(df_1h, '4H', config),
    'tf1d_boms_displacement': lambda df_1h, config=None: calc_boms_displacement(df_1h, '1D', config),
    'tf1d_boms_strength': lambda df_1h, config=None: calc_boms_strength(df_1h, '1D', config),
    # tf4h_fusion requires additional features - handled separately
}


# Example usage
if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)

    # Create sample 1H data
    dates_1h = pd.date_range('2024-01-01', periods=1000, freq='1H')
    df_1h = pd.DataFrame({
        'open': np.random.randn(1000).cumsum() + 60000,
        'high': np.random.randn(1000).cumsum() + 60200,
        'low': np.random.randn(1000).cumsum() + 59800,
        'close': np.random.randn(1000).cumsum() + 60000,
        'volume': np.random.randint(100, 1000, 1000)
    }, index=dates_1h)

    # Test BOMS displacement
    print("\n=== Testing BOMS Displacement ===")
    df_1h['tf4h_boms_displacement'] = calc_boms_displacement(df_1h, '4H')
    print(f"Non-zero: {(df_1h['tf4h_boms_displacement'] > 0).sum()} / {len(df_1h)}")
    print(f"Range: {df_1h['tf4h_boms_displacement'].min():.2f} - {df_1h['tf4h_boms_displacement'].max():.2f}")

    # Test BOMS strength
    print("\n=== Testing BOMS Strength ===")
    df_1h['tf1d_boms_strength'] = calc_boms_strength(df_1h, '1D')
    print(f"Non-zero: {(df_1h['tf1d_boms_strength'] > 0).sum()} / {len(df_1h)}")
    print(f"Range: {df_1h['tf1d_boms_strength'].min():.3f} - {df_1h['tf1d_boms_strength'].max():.3f}")

    # Test fusion (requires 4H features)
    print("\n=== Testing Fusion Score ===")
    df_4h = resample_to_timeframe(df_1h, '4H')
    df_4h['tf4h_structure_alignment'] = np.random.rand(len(df_4h)) > 0.7
    df_4h['tf4h_squiggle_entry_window'] = np.random.rand(len(df_4h)) > 0.8
    df_4h['tf4h_squiggle_confidence'] = np.random.rand(len(df_4h))
    df_4h['tf4h_choch_flag'] = np.random.rand(len(df_4h)) > 0.75

    df_1h['tf4h_fusion_score'] = calc_tf4h_fusion(df_1h, df_4h)
    print(f"Non-zero: {(df_1h['tf4h_fusion_score'] > 0).sum()} / {len(df_1h)}")
    print(f"Range: {df_1h['tf4h_fusion_score'].min():.3f} - {df_1h['tf4h_fusion_score'].max():.3f}")
