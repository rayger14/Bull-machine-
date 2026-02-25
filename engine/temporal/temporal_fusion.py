#!/usr/bin/env python3
"""
Temporal Fusion Engine - The Bull Machine's Sense of Time

This module implements the Wisdom Time Layer that adjusts fusion weights based on
temporal confluence from 4 time-based systems:
    1. Fibonacci Time Clusters (40%) - Harmonic time projections from pivots
    2. Gann Cycles (30%) - Sacred vibration numbers (144, 90, 72, 45, etc.)
    3. Volatility Cycles (20%) - Compression/expansion phase detection
    4. Emotional Cycles (10%) - Wall Street Cheat Sheet psychology mapping

Philosophy:
    "Time is pressure, not prediction."

    This layer does NOT predict when moves will happen. It detects when multiple
    time cycles align to create temporal confluence - pressure zones where signals
    deserve higher conviction.

Design Principles:
    - Soft adjustments only (±5-15%), no hard vetoes
    - Uses existing Wyckoff events for pivot timing
    - Integrates existing Gann cycles module
    - Observable: All component scores logged
    - Feature-flagged: Can be disabled via config

Integration:
    Called by ArchetypeLogic.evaluate() to adjust fusion weights before threshold check.

    base_fusion (0.65) → temporal_confluence (0.85) → adjusted_fusion (0.75)

Author: Bull Machine v2.0 - Temporal Intelligence Layer
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from engine.runtime.context import RuntimeContext

logger = logging.getLogger(__name__)


class TemporalFusionEngine:
    """
    Wisdom Time Layer - Adjusts fusion based on temporal confluence.

    The sense of time. Determines WHEN signals deserve trust by detecting
    alignment across 4 temporal systems.
    """

    def __init__(self, config: Dict):
        """
        Initialize temporal fusion engine.

        Args:
            config: Temporal configuration dict with:
                - temporal_weights: Component weights (fib/gann/vol/emotional)
                - enable_temporal_fusion: Master enable flag
                - temporal_adjustment_range: [min_multiplier, max_multiplier]
                - fib_levels: Fibonacci bar counts to detect
                - gann_vibrations: Gann cycle numbers
                - vol_compression_threshold: ATR ratio for compression
                - emotional_rsi_thresholds: RSI levels for psychology mapping
        """
        self.config = config

        # Component weights (must sum to 1.0)
        self.weights = config.get('temporal_weights', {
            'fib_time': 0.40,
            'gann_cycles': 0.30,
            'volatility_cycles': 0.20,
            'emotional_cycles': 0.10
        })

        # Master enable flag
        self.enabled = config.get('enable_temporal_fusion', True)

        # Adjustment range (soft bounds)
        adj_range = config.get('temporal_adjustment_range', [0.85, 1.15])
        self.min_multiplier = adj_range[0]  # -15% max penalty
        self.max_multiplier = adj_range[1]  # +15% max boost

        # Fibonacci levels (bar counts)
        self.fib_levels = config.get('fib_levels', [13, 21, 34, 55, 89, 144])

        # Gann vibrations (sacred numbers)
        self.gann_vibrations = config.get('gann_vibrations', [3, 7, 9, 12, 21, 36, 45, 72, 90, 144])

        # Volatility cycle parameters
        self.vol_compression_threshold = config.get('vol_compression_threshold', 0.75)
        self.vol_expansion_threshold = config.get('vol_expansion_threshold', 1.25)

        # Emotional cycle parameters (RSI thresholds)
        self.emotional_rsi_thresholds = config.get('emotional_rsi_thresholds', {
            'extreme_fear': 25,
            'hope_lower': 35,
            'hope_upper': 45,
            'greed': 65,
            'extreme_greed': 75
        })

        # Tolerances
        self.fib_tolerance = config.get('fib_tolerance_bars', 3)
        self.gann_tolerance = config.get('gann_tolerance_bars', 2)

        logger.info(f"[TemporalFusion] Initialized - enabled={self.enabled}, weights={self.weights}")

    def compute_temporal_confluence(self, context: RuntimeContext) -> float:
        """
        Compute temporal confluence score [0-1].

        Returns:
            0.0-0.3: Out of phase (reduce fusion)
            0.3-0.7: Normal (no adjustment)
            0.7-1.0: High confluence (boost fusion)

        This is the main entry point. Calls all 4 component scorers and combines
        via weighted average.
        """
        if not self.enabled:
            return 0.5  # Neutral - no adjustment

        try:
            # Component 1: Fibonacci time clusters (40%)
            fib_score = self._compute_fib_cluster_score(context)

            # Component 2: Gann cycles (30%)
            gann_score = self._compute_gann_cycle_score(context)

            # Component 3: Volatility cycles (20%)
            vol_score = self._compute_volatility_cycle_score(context)

            # Component 4: Emotional cycles (10%)
            emotional_score = self._compute_emotional_cycle_score(context)

            # Weighted confluence
            confluence = (
                fib_score * self.weights['fib_time'] +
                gann_score * self.weights['gann_cycles'] +
                vol_score * self.weights['volatility_cycles'] +
                emotional_score * self.weights['emotional_cycles']
            )

            # Clip to [0, 1]
            confluence = np.clip(confluence, 0.0, 1.0)

            # Log component breakdown (debug level)
            logger.debug(
                f"[TemporalFusion] confluence={confluence:.3f} | "
                f"fib={fib_score:.3f} gann={gann_score:.3f} "
                f"vol={vol_score:.3f} emotional={emotional_score:.3f}"
            )

            return confluence

        except Exception as e:
            logger.error(f"[TemporalFusion] Error computing confluence: {e}")
            return 0.5  # Neutral on error

    def adjust_fusion_weight(self, base_fusion: float, confluence: float) -> float:
        """
        Adjust fusion weight based on temporal confluence.

        Soft adjustments only (±5-15%), no hard vetoes.

        Adjustment zones:
            confluence >= 0.85: +15% boost (strong alignment)
            confluence >= 0.70: +10% boost (moderate alignment)
            confluence <= 0.15: -15% penalty (strong misalignment)
            confluence <= 0.30: -5% penalty (light misalignment)
            else: no adjustment (neutral zone 0.30-0.70)

        Args:
            base_fusion: Original fusion score [0-1]
            confluence: Temporal confluence [0-1]

        Returns:
            Adjusted fusion score [0-1]
        """
        if not self.enabled or confluence < 0 or confluence > 1:
            return base_fusion

        # Determine multiplier based on confluence zones
        if confluence >= 0.85:
            multiplier = 1.15  # Strong boost
        elif confluence >= 0.70:
            multiplier = 1.10  # Moderate boost
        elif confluence <= 0.15:
            multiplier = 0.85  # Strong penalty
        elif confluence <= 0.30:
            multiplier = 0.95  # Light penalty
        else:
            multiplier = 1.0  # Neutral range (0.30-0.70)

        # Apply adjustment
        adjusted = base_fusion * multiplier

        # Clip to valid range
        adjusted = np.clip(adjusted, 0.0, 1.0)

        # Log significant adjustments
        if abs(adjusted - base_fusion) > 0.01:
            logger.debug(
                f"[TemporalFusion] Fusion adjusted: {base_fusion:.3f} → {adjusted:.3f} "
                f"(confluence={confluence:.3f}, multiplier={multiplier:.2f})"
            )

        return adjusted

    # ========================================================================
    # Component Scorers
    # ========================================================================

    def _compute_fib_cluster_score(self, context: RuntimeContext) -> float:
        """
        Score based on bars since Wyckoff events aligned with Fib levels.

        Uses existing features:
            - bars_since_sc (Selling Climax)
            - bars_since_ar (Automatic Rally)
            - bars_since_st (Secondary Test)
            - bars_since_sos_long (Sign of Strength)
            - bars_since_sos_short (Sign of Weakness)

        Algorithm:
            1. Get bars since each event
            2. Check if any align with Fib levels (13, 21, 34, 55, 89, 144)
            3. Count hits (within ±3 bars tolerance)
            4. Score: 3+ hits=0.80, 2 hits=0.60, 1 hit=0.40, 0 hits=0.20

        Returns:
            Score [0.2-0.8]
        """
        try:
            row = context.row

            # Get bars since key Wyckoff events
            event_keys = [
                'bars_since_sc',
                'bars_since_ar',
                'bars_since_st',
                'bars_since_sos_long',
                'bars_since_sos_short'
            ]

            events = []
            for key in event_keys:
                bars = row.get(key, 999)
                if pd.notna(bars) and bars < 999:
                    events.append(int(bars))

            if not events:
                return 0.20  # No recent events

            # Count Fibonacci hits
            fib_hits = 0
            for event_bars in events:
                for fib in self.fib_levels:
                    if abs(event_bars - fib) <= self.fib_tolerance:
                        fib_hits += 1
                        break  # Only count one hit per event

            # Score mapping
            if fib_hits >= 3:
                return 0.80  # Strong confluence
            elif fib_hits == 2:
                return 0.60  # Moderate confluence
            elif fib_hits == 1:
                return 0.40  # Weak confluence
            else:
                return 0.20  # No confluence

        except Exception as e:
            logger.error(f"[TemporalFusion] Error in fib cluster score: {e}")
            return 0.20

    def _compute_gann_cycle_score(self, context: RuntimeContext) -> float:
        """
        Score based on Gann vibration hits (3, 7, 9, 12, 21, 36, 45, 72, 90, 144).

        Uses: bars_since_sc (primary reference point)

        Algorithm:
            1. Get bars since Selling Climax (or AR if SC not available)
            2. Check if current bar aligns with Gann vibration (±2 bars)
            3. Score based on vibration strength:
                - Major vibrations (90, 144): 0.90
                - Strong vibrations (45, 72): 0.75
                - Medium vibrations (21, 36): 0.60
                - Minor vibrations (3, 7, 9, 12): 0.45
                - No hit: 0.20

        Returns:
            Score [0.2-0.9]
        """
        try:
            row = context.row

            # Get bars since SC (primary) or AR (secondary)
            bars_since_sc = row.get('bars_since_sc', 999)
            if pd.isna(bars_since_sc) or bars_since_sc >= 999:
                bars_since_sc = row.get('bars_since_ar', 999)

            if pd.isna(bars_since_sc) or bars_since_sc >= 999:
                return 0.20  # No reference point

            bars_since_sc = int(bars_since_sc)

            # Check for Gann vibration hits
            for vib in self.gann_vibrations:
                if abs(bars_since_sc - vib) <= self.gann_tolerance:
                    # Score based on vibration magnitude
                    if vib in [90, 144]:
                        return 0.90  # Major vibrations
                    elif vib in [45, 72]:
                        return 0.75  # Strong vibrations
                    elif vib in [21, 36]:
                        return 0.60  # Medium vibrations
                    else:
                        return 0.45  # Minor vibrations (3, 7, 9, 12)

            return 0.20  # No vibration hit

        except Exception as e:
            logger.error(f"[TemporalFusion] Error in Gann cycle score: {e}")
            return 0.20

    def _compute_volatility_cycle_score(self, context: RuntimeContext) -> float:
        """
        Score based on compression/expansion phase.

        Uses: atr, atr_ma_20 (or compute from context)

        Algorithm:
            1. Calculate ATR ratio: atr / atr_ma_20
            2. Compression (ATR < 0.75 * MA): High score (0.90) - coiled spring
            3. Expansion (ATR > 1.25 * MA): Low score (0.10) - climax, avoid
            4. Normal: Neutral (0.50)

        Returns:
            Score [0.1-0.9]
        """
        try:
            row = context.row

            # Get ATR and ATR MA
            atr = row.get('atr', None)
            atr_ma = row.get('atr_ma_20', None)

            # Try alternate column names
            if pd.isna(atr):
                atr = row.get('atr_14', None)
            if pd.isna(atr_ma):
                atr_ma = row.get('atr_ma', None)

            # If still missing, try to compute from context
            if pd.isna(atr) or pd.isna(atr_ma):
                return 0.50  # Neutral if data missing

            # Calculate ratio
            atr_ratio = atr / atr_ma if atr_ma > 0 else 1.0

            # Score based on phase
            if atr_ratio < self.vol_compression_threshold:
                return 0.90  # Compression - coiled spring, high score
            elif atr_ratio > self.vol_expansion_threshold:
                return 0.10  # Expansion climax - avoid
            else:
                return 0.50  # Normal volatility

        except Exception as e:
            logger.error(f"[TemporalFusion] Error in volatility cycle score: {e}")
            return 0.50

    def _compute_emotional_cycle_score(self, context: RuntimeContext) -> float:
        """
        Score based on market psychology (Wall Street Cheat Sheet).

        Uses: rsi, funding (funding rate as sentiment proxy)

        Algorithm:
            Maps RSI + funding to psychological phases:
            - Capitulation (RSI < 25, funding < -0.02): 0.95 (extreme fear, best entry)
            - Hope/Relief (RSI 35-45): 0.70 (recovery phase)
            - Euphoria (RSI > 75, funding > 0.03): 0.05 (extreme greed, avoid)
            - Greed (RSI > 65): 0.20 (late stage)
            - Neutral: 0.50

        Returns:
            Score [0.05-0.95]
        """
        try:
            row = context.row

            # Get RSI
            rsi = row.get('rsi', None)
            if pd.isna(rsi):
                rsi = row.get('rsi_14', 50)  # Default to neutral

            # Get funding rate (sentiment proxy)
            funding = row.get('funding', 0)
            if pd.isna(funding):
                funding = 0

            # Map to Wall Street Cheat Sheet phases
            thresholds = self.emotional_rsi_thresholds

            # Capitulation zone (extreme fear)
            if rsi < thresholds['extreme_fear'] and funding < -0.02:
                return 0.95  # Best entry - blood in streets

            # Hope/Relief zone (recovery)
            elif thresholds['hope_lower'] <= rsi <= thresholds['hope_upper']:
                return 0.70  # Good entry - early recovery

            # Euphoria zone (extreme greed)
            elif rsi > thresholds['extreme_greed'] and funding > 0.03:
                return 0.05  # Avoid - everyone is long

            # Greed zone (late stage)
            elif rsi > thresholds['greed']:
                return 0.20  # Low confidence - topping

            # Neutral zone
            else:
                return 0.50

        except Exception as e:
            logger.error(f"[TemporalFusion] Error in emotional cycle score: {e}")
            return 0.50

    def get_component_scores(self, context: RuntimeContext) -> Dict[str, float]:
        """
        Get all component scores for debugging/observability.

        Returns:
            Dict with individual component scores
        """
        return {
            'fib_cluster_score': self._compute_fib_cluster_score(context),
            'gann_cycle_score': self._compute_gann_cycle_score(context),
            'volatility_cycle_score': self._compute_volatility_cycle_score(context),
            'emotional_cycle_score': self._compute_emotional_cycle_score(context),
            'confluence': self.compute_temporal_confluence(context)
        }


# ============================================================================
# Standalone Functions for Feature Pipeline
# ============================================================================

def compute_temporal_features_batch(
    df: pd.DataFrame,
    config: Dict,
    add_to_df: bool = True
) -> pd.DataFrame:
    """
    Compute temporal fusion features for entire dataframe.

    This is for feature store / backtest pipelines.

    Args:
        df: OHLCV dataframe with Wyckoff events and technical indicators
        config: Temporal fusion configuration
        add_to_df: If True, add columns to df. If False, return new df.

    Returns:
        DataFrame with temporal features added:
            - temporal_fib_score
            - temporal_gann_score
            - temporal_vol_score
            - temporal_emotional_score
            - temporal_confluence
    """
    engine = TemporalFusionEngine(config)

    # Initialize columns
    df['temporal_fib_score'] = 0.0
    df['temporal_gann_score'] = 0.0
    df['temporal_vol_score'] = 0.0
    df['temporal_emotional_score'] = 0.0
    df['temporal_confluence'] = 0.0

    # Compute for each row (vectorized where possible)
    # Note: This is simplified - in production, vectorize further
    for idx in range(len(df)):
        try:
            # Create minimal context (only needs row)
            row = df.iloc[idx]

            # Mock context (enough for scoring)
            from dataclasses import dataclass
            @dataclass
            class MockContext:
                row: pd.Series

            context = MockContext(row=row)

            # Compute scores
            scores = engine.get_component_scores(context)

            df.at[df.index[idx], 'temporal_fib_score'] = scores['fib_cluster_score']
            df.at[df.index[idx], 'temporal_gann_score'] = scores['gann_cycle_score']
            df.at[df.index[idx], 'temporal_vol_score'] = scores['volatility_cycle_score']
            df.at[df.index[idx], 'temporal_emotional_score'] = scores['emotional_cycle_score']
            df.at[df.index[idx], 'temporal_confluence'] = scores['confluence']

        except Exception as e:
            logger.error(f"[TemporalFusion] Error at row {idx}: {e}")
            continue

    logger.info(f"[TemporalFusion] Computed temporal features for {len(df)} bars")

    return df


def compute_bars_since_wyckoff_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute bars_since_* features for all Wyckoff events.

    This should be called during feature computation if not already present.

    Args:
        df: DataFrame with Wyckoff event boolean columns

    Returns:
        DataFrame with bars_since_* columns added
    """
    event_mappings = {
        'wyckoff_sc': 'bars_since_sc',
        'wyckoff_ar': 'bars_since_ar',
        'wyckoff_st': 'bars_since_st',
        'wyckoff_sos': 'bars_since_sos_long',
        'wyckoff_sow': 'bars_since_sos_short'
    }

    for event_col, bars_col in event_mappings.items():
        if event_col in df.columns:
            # Find last occurrence of event
            df[bars_col] = 999  # Default large value

            for idx in range(len(df)):
                # Find most recent event before this bar
                recent_events = df.iloc[:idx+1][df[event_col]]
                if len(recent_events) > 0:
                    last_event_idx = recent_events.index[-1]
                    df.at[df.index[idx], bars_col] = idx - df.index.get_loc(last_event_idx)

    logger.info(f"[TemporalFusion] Computed bars_since features for {len(event_mappings)} events")

    return df
