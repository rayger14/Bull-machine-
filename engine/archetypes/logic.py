#!/usr/bin/env python3
"""
PR#6A: Rule-Based Archetype Expansion Logic

Implements 11 distinct market archetypes (A-H + K, L, M) using rule-based
heuristics. These create clean labeled data for future PyTorch training.

Archetypes:
- A: Trap Reversal (PTI-based spring/UTAD)
- B: Order Block Retest (BOMS + Wyckoff + BOS proximity)
- C: FVG Continuation (Displacement + momentum + recent BOS)
- D: Failed Continuation (FVG present + weak RSI + falling ADX)
- E: Liquidity Compression (Low ATR + narrow range + stable book)
- F: Expansion Exhaustion (Extreme RSI + high ATR + volume spike)
- G: Re-Accumulate Base (BOMS strength + rising RSI from sub-40)
- H: Trap Within Trend (HTF trend + liquidity drop + wick against trend)
- K: Wick Trap (Moneytaur) (Wick anomaly + ADX > 25 + BOS context)
- L: Volume Exhaustion (Zeroika) (Volume spike + extreme RSI + falling momentum)
- M: Ratio Coil Break (Wyckoff Insider) (Low ATR + near POC + BOMS strength)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class ArchetypeLogic:
    """
    Rule-based archetype detection for PR#6A.

    Returns archetype name + scores when pattern matches, otherwise (None, 0, 0).
    All archetypes require global liquidity_score >= min_liquidity threshold.
    """

    def __init__(self, config: dict):
        """
        Initialize with archetype config section.

        Args:
            config: Dict containing 'use_archetypes', enable flags, and thresholds
        """
        self.config = config
        self.use_archetypes = config.get('use_archetypes', False)

        # Extract thresholds
        thresholds = config.get('thresholds', {})
        self.min_liquidity = thresholds.get('min_liquidity', 0.30)

        # Per-archetype thresholds
        self.thresh_A = thresholds.get('A', {})
        self.thresh_B = thresholds.get('B', {})
        self.thresh_C = thresholds.get('C', {})
        self.thresh_D = thresholds.get('D', {})
        self.thresh_E = thresholds.get('E', {})
        self.thresh_F = thresholds.get('F', {})
        self.thresh_G = thresholds.get('G', {})
        self.thresh_H = thresholds.get('H', {})
        self.thresh_K = thresholds.get('K', {})
        self.thresh_L = thresholds.get('L', {})
        self.thresh_M = thresholds.get('M', {})

        # Bear archetype thresholds
        self.thresh_S1 = thresholds.get('S1', {})  # Breakdown
        self.thresh_S2 = thresholds.get('S2', {})  # Rejection
        self.thresh_S3 = thresholds.get('S3', {})  # Whipsaw
        self.thresh_S4 = thresholds.get('S4', {})  # Distribution
        self.thresh_S5 = thresholds.get('S5', {})  # Short Squeeze
        self.thresh_S6 = thresholds.get('S6', {})  # Alt Rotation Down
        self.thresh_S7 = thresholds.get('S7', {})  # Curve Inversion
        self.thresh_S8 = thresholds.get('S8', {})  # Volume Fade Chop

        # Enable flags
        self.enabled = {
            'A': config.get('enable_A', True),
            'B': config.get('enable_B', True),
            'C': config.get('enable_C', True),
            'D': config.get('enable_D', True),
            'E': config.get('enable_E', True),
            'F': config.get('enable_F', True),
            'G': config.get('enable_G', True),
            'H': config.get('enable_H', True),
            'K': config.get('enable_K', True),
            'L': config.get('enable_L', True),
            'M': config.get('enable_M', True),
            # Bear archetypes
            'S1': config.get('enable_S1', False),  # Disabled by default
            'S2': config.get('enable_S2', False),
            'S3': config.get('enable_S3', False),
            'S4': config.get('enable_S4', False),
            'S5': config.get('enable_S5', False),
            'S6': config.get('enable_S6', False),
            'S7': config.get('enable_S7', False),
            'S8': config.get('enable_S8', False),
        }

        # Fusion weights (from config or defaults)
        self.fusion_weights = {
            'wyckoff': 0.265,
            'liquidity': 0.313,
            'momentum': 0.164,
            'smc': 0.258  # PATCH: Added SMC weight for fusion calculation
        }
        self.fakeout_penalty = 0.075

        # Rolling window for ATR percentile computation
        self.atr_percentile_window = 500

    def _get_liquidity_score(self, row: pd.Series) -> float:
        """
        PATCH: Compute liquidity_score from available features.

        Original design expected pre-computed 'liquidity_score', but actual
        feature store doesn't have this. Compute from available liquidity-related features.
        """
        # Try direct column first (in case it exists)
        if 'liquidity_score' in row.index:
            return row.get('liquidity_score', 0.0)

        # Compute composite from available features
        # Use: BOMS strength, volume zscore, spread indicators
        boms_strength = row.get('tf1d_boms_strength', 0.0)
        volume_z = max(0.0, min(row.get('volume_zscore', 0.0) / 2.0, 1.0))  # Normalize

        # Weighted composite (BOMS is primary liquidity indicator)
        liquidity = 0.70 * boms_strength + 0.30 * volume_z
        return max(0.0, min(liquidity, 1.0))

    def _get_wyckoff_score(self, row: pd.Series) -> float:
        """PATCH: Map to actual tf1d_wyckoff_score column."""
        return row.get('tf1d_wyckoff_score', 0.0)

    def _get_momentum_score(self, row: pd.Series) -> float:
        """
        PATCH: Compute momentum_score from available indicators.

        Use RSI, volume zscore, and displacement as momentum proxies.
        """
        if 'momentum_score' in row.index:
            return row.get('momentum_score', 0.0)

        # RSI-based momentum (normalize to 0-1, centered at 50)
        rsi = row.get('rsi_14', 50.0)
        rsi_momentum = abs(rsi - 50.0) / 50.0  # 0 at neutral, 1 at extremes

        # Volume momentum
        volume_z = row.get('volume_zscore', 0.0)
        vol_momentum = max(0.0, min(volume_z / 2.0, 1.0))

        # Displacement momentum (use 4h BOMS displacement as proxy)
        atr = row.get('atr_20', 1.0)
        disp = abs(row.get('tf4h_boms_displacement', 0.0))
        disp_momentum = max(0.0, min(disp / (2.0 * atr), 1.0)) if atr > 0 else 0.0

        # Weighted composite
        momentum = 0.40 * rsi_momentum + 0.30 * vol_momentum + 0.30 * disp_momentum
        return max(0.0, min(momentum, 1.0))

    def _get_smc_score(self, row: pd.Series) -> float:
        """
        PATCH: Compute SMC score from BOS, FVG, and order block features.
        """
        if 'smc_score' in row.index:
            return row.get('smc_score', 0.0)

        # BOS indicators (binary, so convert to 0/1)
        bos_bullish = 1.0 if row.get('tf1h_bos_bullish', False) else 0.0
        bos_bearish = 1.0 if row.get('tf1h_bos_bearish', False) else 0.0
        bos_score = max(bos_bullish, bos_bearish)

        # FVG indicators
        fvg_1h = 1.0 if row.get('tf1h_fvg_present', False) else 0.0
        fvg_4h = 1.0 if row.get('tf4h_fvg_present', False) else 0.0
        fvg_score = max(fvg_1h, fvg_4h * 1.2)  # 4h FVG weighted higher
        fvg_score = min(fvg_score, 1.0)

        # Composite SMC score
        smc = 0.60 * bos_score + 0.40 * fvg_score
        return max(0.0, min(smc, 1.0))

    def _compute_atr_percentile(self, row: pd.Series, df: pd.DataFrame, index: int) -> float:
        """
        PATCH: Compute ATR percentile from rolling window.

        Not available as pre-computed column, so calculate on-the-fly.
        """
        if 'atr_percentile' in row.index:
            return row.get('atr_percentile', 0.5)

        # Get current ATR
        current_atr = row.get('atr_20', 0.0)

        # Get historical ATR values for percentile calculation
        lookback = min(self.atr_percentile_window, index + 1)
        if lookback < 20:  # Not enough history
            return 0.5  # Default to median

        historical_atrs = df.iloc[max(0, index-lookback):index+1]['atr_20'].values

        # Calculate percentile rank
        if len(historical_atrs) == 0:
            return 0.5

        percentile = (historical_atrs < current_atr).sum() / len(historical_atrs)
        return max(0.0, min(percentile, 1.0))

    def _get_bos_flag(self, row: pd.Series) -> int:
        """
        PATCH: Synthesize BOS flag from tf1h_bos_bullish and tf1h_bos_bearish.

        Returns: 1 for bullish BOS, -1 for bearish BOS, 0 for no BOS
        """
        if 'tf1h_bos_flag' in row.index:
            return int(row.get('tf1h_bos_flag', 0))

        bullish = row.get('tf1h_bos_bullish', False)
        bearish = row.get('tf1h_bos_bearish', False)

        if bullish and not bearish:
            return 1
        elif bearish and not bullish:
            return -1
        elif bullish and bearish:
            return 1  # Prefer bullish if both (shouldn't happen but handle edge case)
        else:
            return 0

    def _get_wick_anomaly(self, row: pd.Series) -> bool:
        """
        PATCH: Detect wick anomaly from OHLC data.

        Wick anomaly = wick length > 2× body length
        """
        if 'wick_anomaly' in row.index:
            return bool(row.get('wick_anomaly', False))

        close = row.get('close', 0.0)
        open_price = row.get('open', close)
        high = row.get('high', close)
        low = row.get('low', close)

        body = abs(close - open_price)
        upper_wick = high - max(close, open_price)
        lower_wick = min(close, open_price) - low

        # Anomaly if either wick > 2× body
        return (upper_wick > 2 * body) or (lower_wick > 2 * body)

    def _get_fakeout_score(self, row: pd.Series) -> float:
        """
        PATCH: Map to tf1h_fakeout_intensity column.
        """
        if 'fakeout_score' in row.index:
            return row.get('fakeout_score', 0.0)

        # Use fakeout intensity as proxy for fakeout score
        return row.get('tf1h_fakeout_intensity', 0.0)

    def _get_wyckoff_phase(self, row: pd.Series) -> str:
        """
        PATCH: Map tf1d_wyckoff_phase to wyckoff_phase.

        Feature store has: tf1d_wyckoff_phase
        Archetype logic expects: wyckoff_phase
        """
        if 'wyckoff_phase' in row.index:
            return row.get('wyckoff_phase', 'transition')

        # Map from daily wyckoff phase
        return row.get('tf1d_wyckoff_phase', 'transition')

    def calculate_fusion_score(self, row: pd.Series) -> float:
        """
        Calculate fusion score from component scores.

        CRITICAL FIX: Feature store already has pre-computed fusion scores!
        Use tf1h_fusion_score directly instead of recomputing from components.

        Args:
            row: DataFrame row with score columns

        Returns:
            Fusion score in [0, 1]
        """
        # CRITICAL FIX: Use pre-computed fusion score from feature store
        if 'tf1h_fusion_score' in row.index:
            return row.get('tf1h_fusion_score', 0.0)

        # Fallback to k2_fusion_score if tf1h not available
        if 'k2_fusion_score' in row.index:
            return row.get('k2_fusion_score', 0.0)

        # Last resort: compute from components (legacy path - should rarely execute)
        wyckoff_score = self._get_wyckoff_score(row)
        liquidity_score = self._get_liquidity_score(row)
        momentum_score = self._get_momentum_score(row)
        smc_score = self._get_smc_score(row)
        fakeout_score = self._get_fakeout_score(row)

        fusion = (
            self.fusion_weights['wyckoff'] * wyckoff_score +
            self.fusion_weights['liquidity'] * liquidity_score +
            self.fusion_weights['momentum'] * momentum_score +
            self.fusion_weights['smc'] * smc_score -
            self.fakeout_penalty * fakeout_score
        )

        return max(0.0, min(fusion, 1.0))

    def check_archetype(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        df: pd.DataFrame,
        index: int
    ) -> Tuple[Optional[str], float, float]:
        """
        **PR#6A: Fixed dispatch - evaluate ALL archetypes, pick best.**

        OLD BEHAVIOR (broken): Return first match → K starves H
        NEW BEHAVIOR (fixed): Evaluate all enabled → pick best by score

        Args:
            row: Current bar
            prev_row: Previous bar (or None if first bar)
            df: Full dataframe for lookback
            index: Current index in df

        Returns:
            (archetype_name_or_None, fusion_score, liquidity_score)
        """
        from engine import feature_flags as features

        if not self.use_archetypes:
            return None, 0.0, 0.0

        # Global precheck: liquidity >= min_threshold
        liquidity_score = self._get_liquidity_score(row)

        # PR#6A Phase 4: Soft filter (penalty) instead of hard reject
        if features.SOFT_LIQUIDITY_FILTER:
            liquidity_penalty = 1.0
            if liquidity_score < self.min_liquidity:
                liquidity_penalty = 0.7  # 30% penalty for low liquidity
        else:
            # Legacy hard filter
            if liquidity_score < self.min_liquidity:
                return None, 0.0, liquidity_score

        fusion_score = self.calculate_fusion_score(row)

        # Apply soft filter penalty
        if features.SOFT_LIQUIDITY_FILTER and liquidity_penalty < 1.0:
            fusion_score *= liquidity_penalty

        # PR#6A Phase 3: Evaluate ALL enabled archetypes
        if features.EVALUATE_ALL_ARCHETYPES:
            return self._check_all_archetypes(row, prev_row, df, index, fusion_score, liquidity_score)
        else:
            # LEGACY: Priority order with early returns (causes the leak)
            return self._check_legacy_priority(row, prev_row, df, index, fusion_score, liquidity_score)

    def _check_all_archetypes(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        df: pd.DataFrame,
        index: int,
        fusion_score: float,
        liquidity_score: float
    ) -> Tuple[Optional[str], float, float]:
        """
        **PR#6A New Dispatch**: Evaluate all enabled archetypes, pick best by fusion score.

        No early returns → no starvation → K and H can both be scored.
        """
        from engine.archetypes.registry import get_archetype_meta
        from engine.observability import get_gate_tracer

        matches = []  # (slug, fusion_score, priority)
        tracer = get_gate_tracer()

        # Archetype checks mapping: letter -> (slug, check_method)
        checks = [
            ('A', 'wyckoff_spring_utad', self._check_A, 'trap_reversal'),
            ('B', 'order_block_retest', self._check_B, 'order_block_retest'),
            ('C', 'bos_choch_reversal', self._check_C, 'fvg_continuation'),
            ('K', 'wick_trap_moneytaur', self._check_K, 'wick_trap'),
            ('H', 'trap_within_trend', self._check_H, 'trap_within_trend'),
            ('L', 'fakeout_real_move', self._check_L, 'volume_exhaustion'),
            ('F', 'expansion_exhaustion', self._check_F, 'expansion_exhaustion'),
            ('D', 'failed_continuation', self._check_D, 'failed_continuation'),
            ('G', 'boms_phase_shift', self._check_G, 'reaccumulation'),
            ('E', 'liquidity_compression', self._check_E, 'liquidity_compression'),
            ('M', 'ratio_coil_break', self._check_M, 'ratio_coil_break'),
        ]

        # Evaluate ALL enabled archetypes (no early returns!)
        for letter, slug, check_fn, return_name in checks:
            if not self.enabled.get(letter, False):
                continue

            try:
                matched = check_fn(row, prev_row, df, index, fusion_score)

                # Gate tracing
                tracer.trace(slug, 'archetype_check', matched)

                if matched:
                    # Get priority from registry
                    try:
                        priority = get_archetype_meta(slug)['priority']
                    except (KeyError, TypeError):
                        priority = 99  # Default low priority

                    matches.append((slug, return_name, fusion_score, priority))

                    # Record match
                    tracer.record_match(slug)
            except Exception as e:
                import logging
                logging.error(f"Error checking archetype {slug}: {e}")
                continue

        # No matches
        if not matches:
            tracer.increment_bars()
            return None, fusion_score, liquidity_score

        # Select best match: highest fusion score, ties broken by priority
        best = max(matches, key=lambda x: (x[2], -x[3]))  # max fusion, min priority
        best_slug, best_return_name, best_fusion, best_priority = best

        tracer.increment_bars()

        return best_return_name, best_fusion, liquidity_score

    def _check_legacy_priority(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        df: pd.DataFrame,
        index: int,
        fusion_score: float,
        liquidity_score: float
    ) -> Tuple[Optional[str], float, float]:
        """
        LEGACY dispatch: First match wins (early returns cause starvation).

        Kept for backward compatibility. Feature flag controls which path is used.
        """
        # Check archetypes in priority order
        # Priority: A, B, C, K, H, L, F, D, G, E, M

        if self.enabled['A']:
            if self._check_A(row, prev_row, df, index, fusion_score):
                return 'trap_reversal', fusion_score, liquidity_score

        if self.enabled['B']:
            if self._check_B(row, prev_row, df, index, fusion_score):
                return 'order_block_retest', fusion_score, liquidity_score

        if self.enabled['C']:
            if self._check_C(row, prev_row, df, index, fusion_score):
                return 'fvg_continuation', fusion_score, liquidity_score

        if self.enabled['K']:
            if self._check_K(row, prev_row, df, index, fusion_score):
                return 'wick_trap', fusion_score, liquidity_score

        if self.enabled['H']:
            if self._check_H(row, prev_row, df, index, fusion_score):
                return 'trap_within_trend', fusion_score, liquidity_score

        if self.enabled['L']:
            if self._check_L(row, prev_row, df, index, fusion_score):
                return 'volume_exhaustion', fusion_score, liquidity_score

        if self.enabled['F']:
            if self._check_F(row, prev_row, df, index, fusion_score):
                return 'expansion_exhaustion', fusion_score, liquidity_score

        if self.enabled['D']:
            if self._check_D(row, prev_row, df, index, fusion_score):
                return 'failed_continuation', fusion_score, liquidity_score

        if self.enabled['G']:
            if self._check_G(row, prev_row, df, index, fusion_score):
                return 'reaccumulation', fusion_score, liquidity_score

        if self.enabled['E']:
            if self._check_E(row, prev_row, df, index, fusion_score):
                return 'liquidity_compression', fusion_score, liquidity_score

        if self.enabled['M']:
            if self._check_M(row, prev_row, df, index, fusion_score):
                return 'ratio_coil_break', fusion_score, liquidity_score

        return None, fusion_score, liquidity_score

    # =========================================================================
    # Archetype Detection Methods
    # =========================================================================

    def _check_A(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        A - Trap Reversal: PTI-based spring/UTAD detection.

        Criteria:
        - pti_trap_type in {spring, utad, bull_trap, bear_trap}
        - tf4h_boms_displacement >= 0.80 * ATR
        - pti_score >= 0.40
        - fusion_score >= 0.33

        PATCHED: Maps to actual tf1h_pti_trap_type and tf1h_pti_score columns
        """
        # PATCH: Use tf1h_pti_trap_type instead of pti_trap_type
        pti_trap = row.get('tf1h_pti_trap_type', '')
        if pti_trap not in ['spring', 'utad', 'bull_trap', 'bear_trap']:
            return False

        atr = row.get('atr_20', 1.0)
        tf4h_disp = row.get('tf4h_boms_displacement', 0.0)
        if tf4h_disp < self.thresh_A.get('disp_atr', 0.80) * atr:
            return False

        # PATCH: Use tf1h_pti_score instead of pti_score
        pti_score = row.get('tf1h_pti_score', 0.0)
        if pti_score < self.thresh_A.get('pti', 0.40):
            return False

        if fusion_score < self.thresh_A.get('fusion', 0.33):
            return False

        return True

    def _check_B(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        B - Order Block Retest: BOMS strength + Wyckoff + near BOS zone.

        Criteria:
        - boms_strength >= 0.30
        - wyckoff_score >= 0.35
        - Near recent BOS/OB zone (≤ 1× ATR)
        - fusion_score >= 0.374

        PATCHED: Maps boms_strength and wyckoff_score to actual columns
        """
        # PATCH: Use tf1d_boms_strength instead of boms_strength
        boms_strength = row.get('tf1d_boms_strength', 0.0)
        if boms_strength < self.thresh_B.get('boms_strength', 0.30):
            return False

        # PATCH: Use helper method for wyckoff_score
        wyckoff_score = self._get_wyckoff_score(row)
        if wyckoff_score < self.thresh_B.get('wyckoff', 0.35):
            return False

        # Check proximity to recent BOS zone
        atr = row.get('atr_20', 1.0)
        close = row.get('close', 0.0)

        # Look back up to 20 bars for recent BOS
        lookback = min(20, index)
        if lookback > 0:
            recent = df.iloc[index-lookback:index]
            # Use synthesized BOS flags from helper method
            bos_bullish = recent.get('tf1h_bos_bullish', pd.Series([False]*lookback))
            bos_bearish = recent.get('tf1h_bos_bearish', pd.Series([False]*lookback))
            # Create synthetic bos_flag column: 1 for bullish, -1 for bearish
            bos_flags = pd.Series([0]*len(recent), index=recent.index)
            bos_flags[bos_bullish & ~bos_bearish] = 1
            bos_flags[bos_bearish & ~bos_bullish] = -1

            # Check if any recent BOS zone is within 1× ATR
            near_bos = False
            for i in range(len(recent)):
                if bos_flags.iloc[i] != 0:
                    bos_price = recent.iloc[i].get('close', 0.0)
                    if abs(close - bos_price) <= atr:
                        near_bos = True
                        break

            if not near_bos:
                return False
        else:
            return False

        if fusion_score < self.thresh_B.get('fusion', 0.374):
            return False

        return True

    def _check_C(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        C - FVG Continuation: Displacement + momentum + recent BOS.

        Criteria:
        - tf4h_boms_displacement >= 1.0 * ATR
        - momentum_score >= 0.45
        - tf4h_fusion_score >= 0.25
        - Recent BOS same direction (≤ 10 bars)
        - fusion_score >= 0.42

        PATCHED: Uses helper method for momentum_score
        """
        atr = row.get('atr_20', 1.0)
        tf4h_disp = row.get('tf4h_boms_displacement', 0.0)
        if tf4h_disp < self.thresh_C.get('disp_atr', 1.00) * atr:
            return False

        # PATCH: Use helper method for momentum_score
        momentum_score = self._get_momentum_score(row)
        if momentum_score < self.thresh_C.get('momentum', 0.45):
            return False

        tf4h_fusion = row.get('tf4h_fusion_score', 0.0)
        if tf4h_fusion < self.thresh_C.get('tf4h_fusion', 0.25):
            return False

        # Check for recent BOS in same direction
        lookback = min(10, index)
        if lookback > 0:
            recent = df.iloc[index-lookback:index]
            bos_flags = recent.get('tf1h_bos_flag', pd.Series([0]*lookback))

            # Determine current direction from displacement
            direction = 1 if tf4h_disp > 0 else -1

            # Check if any recent BOS matches direction
            same_direction_bos = any(
                (bos_flags.iloc[i] > 0 and direction > 0) or
                (bos_flags.iloc[i] < 0 and direction < 0)
                for i in range(len(bos_flags))
            )

            if not same_direction_bos:
                return False
        else:
            return False

        if fusion_score < self.thresh_C.get('fusion', 0.42):
            return False

        return True

    def _check_D(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        D - Failed Continuation: FVG present + RSI < 50 + falling ADX.

        Criteria:
        - tf1h_fvg_present == 1
        - rsi_14 < 50
        - adx_14 falling (compared to prev bar)
        - liquidity_score >= 0.35
        - fusion_score >= 0.42
        """
        fvg_present = row.get('tf1h_fvg_present', 0)
        if fvg_present != 1:
            return False

        rsi = row.get('rsi_14', 50.0)
        if rsi >= self.thresh_D.get('rsi_max', 50.0):
            return False

        # Check if ADX is falling
        if prev_row is not None:
            adx = row.get('adx_14', 0.0)
            prev_adx = prev_row.get('adx_14', 0.0)
            if adx >= prev_adx:
                return False
        else:
            return False

        # PATCH: Use helper method for liquidity_score
        liquidity = self._get_liquidity_score(row)
        if liquidity < 0.35:
            return False

        if fusion_score < self.thresh_D.get('fusion', 0.42):
            return False

        return True

    def _check_E(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        E - Liquidity Compression: Low ATR + narrow range + stable book.

        Criteria:
        - ATR_percentile < 0.25
        - (high-low)/ATR < 0.5 (narrow range)
        - liquidity_score in [0.45, 0.60]
        - fusion_score >= 0.35

        PATCHED: Uses helper methods for liquidity_score and atr_percentile
        """
        atr_pctile = self._compute_atr_percentile(row, df, index)  # PATCH: Use helper method
        if atr_pctile >= self.thresh_E.get('atr_pctile', 0.25):
            return False

        atr = row.get('atr_20', 1.0)
        high = row.get('high', 0.0)
        low = row.get('low', 0.0)
        range_ratio = (high - low) / atr if atr > 0 else 999
        if range_ratio >= 0.5:
            return False

        # PATCH: Use helper method for liquidity_score
        liquidity = self._get_liquidity_score(row)
        if not (0.45 <= liquidity <= 0.60):
            return False

        if fusion_score < self.thresh_E.get('fusion', 0.35):
            return False

        return True

    def _check_F(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        F - Expansion Exhaustion: Extreme RSI + high ATR + volume spike.

        Criteria:
        - RSI > 78 or RSI < 22
        - ATR_percentile > 0.90
        - volume_zscore > 1.0
        - liquidity_score drops or currently < 0.40
        - fusion_score >= 0.38
        """
        rsi = row.get('rsi_14', 50.0)
        rsi_ext = self.thresh_F.get('rsi_ext', 78.0)
        if not (rsi > rsi_ext or rsi < (100 - rsi_ext)):
            return False

        atr_pctile = self._compute_atr_percentile(row, df, index)  # PATCH: Use helper method
        if atr_pctile <= self.thresh_F.get('atr_pctile', 0.90):
            return False

        vol_z = row.get('volume_zscore', 0.0)
        if vol_z <= self.thresh_F.get('vol_z', 1.0):
            return False

        # PATCH: Use helper method for liquidity_score
        liquidity = self._get_liquidity_score(row)
        liquidity_drop = False
        if prev_row is not None:
            prev_liq = self._get_liquidity_score(prev_row)
            if liquidity < prev_liq:
                liquidity_drop = True

        if not (liquidity_drop or liquidity < 0.40):
            return False

        if fusion_score < self.thresh_F.get('fusion', 0.38):
            return False

        return True

    def _check_G(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        G - Re-Accumulate Base: BOMS strength + rising RSI from sub-40.

        Criteria:
        - boms_strength > 0.40
        - liquidity_score > 0.40
        - RSI rising from sub-40 to >45
        - ATR stabilizing (current ATR percentile 0.25-0.75)
        - fusion_score >= 0.40

        PATCHED: Uses helper method for liquidity_score and maps boms_strength
        """
        # PATCH: Use tf1d_boms_strength instead of boms_strength
        boms_strength = row.get('tf1d_boms_strength', 0.0)
        if boms_strength <= self.thresh_G.get('boms_strength', 0.40):
            return False

        # PATCH: Use helper method for liquidity_score
        liquidity = self._get_liquidity_score(row)
        if liquidity <= self.thresh_G.get('liq', 0.40):
            return False

        rsi = row.get('rsi_14', 50.0)
        if rsi <= 45:
            return False

        # Check RSI was below 40 recently (within 5 bars)
        lookback = min(5, index)
        if lookback > 0:
            recent = df.iloc[index-lookback:index]
            rsi_vals = recent.get('rsi_14', pd.Series([50]*lookback))
            had_sub40 = any(rsi_vals.iloc[i] < 40 for i in range(len(rsi_vals)))
            if not had_sub40:
                return False
        else:
            return False

        # ATR stabilizing check
        atr_pctile = self._compute_atr_percentile(row, df, index)  # PATCH: Use helper method
        if not (0.25 <= atr_pctile <= 0.75):
            return False

        if fusion_score < self.thresh_G.get('fusion', 0.40):
            return False

        return True

    def _check_H(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        H - Trap Within Trend: HTF trend + liquidity drop + wick against trend.

        **PR#6A WIRED**: Uses get_param() with canonical slug 'trap_within_trend'.

        Configurable parameters:
        - quality_threshold: HTF fusion minimum (default: 0.55)
        - liquidity_threshold: Max liquidity score (default: 0.30)
        - adx_threshold: Minimum ADX (default: 25.0)
        - fusion_threshold: Minimum fusion score (default: 0.35)
        - wick_multiplier: Wick size vs body (default: 2.0)
        """
        from engine.archetypes.param_accessor import get_param

        # PR#6A: Read from canonical location with migration-safe fallback
        quality_th = get_param(self, 'trap_within_trend', 'quality_threshold', 0.55)
        liquidity_th = get_param(self, 'trap_within_trend', 'liquidity_threshold', 0.30)
        adx_th = get_param(self, 'trap_within_trend', 'adx_threshold', 25.0)
        fusion_th = get_param(self, 'trap_within_trend', 'fusion_threshold', 0.35)
        wick_mult = get_param(self, 'trap_within_trend', 'wick_multiplier', 2.0)

        # Check HTF trend (NOW CONFIGURABLE)
        tf4h_fusion = row.get('tf4h_fusion_score', 0.0)
        if tf4h_fusion <= quality_th:  # ← WAS HARDCODED 0.5
            return False

        # Check liquidity (NOW READS FROM CONFIG)
        liquidity = self._get_liquidity_score(row)
        if liquidity >= liquidity_th:  # ← WAS self.thresh_H.get(...)
            return False

        # Check ADX (NOW READS FROM CONFIG)
        adx = row.get('adx_14', 0.0)
        if adx <= adx_th:  # ← WAS self.thresh_H.get(...)
            return False

        # Check for wick against trend
        close = row.get('close', 0.0)
        open_price = row.get('open', close)
        high = row.get('high', close)
        low = row.get('low', close)

        body = abs(close - open_price)
        upper_wick = high - max(close, open_price)
        lower_wick = min(close, open_price) - low

        # Check wick significance (NOW CONFIGURABLE MULTIPLIER)
        wick_against_trend = (lower_wick > wick_mult * body) or (upper_wick > wick_mult * body)
        if not wick_against_trend:
            return False

        # Check BOS flag alignment
        bos_flag = self._get_bos_flag(row)
        if bos_flag == 0:
            return False

        # Check fusion score (NOW READS FROM CONFIG)
        if fusion_score < fusion_th:  # ← WAS self.thresh_H.get(...)
            return False

        return True

    def _check_K(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        K - Wick Trap (Moneytaur): Wick anomaly + ADX > 25 + BOS context.

        **PR#6A WIRED**: Uses get_param() with canonical slug 'wick_trap_moneytaur'.

        Configurable parameters:
        - adx_threshold: Minimum ADX (default: 25.0)
        - liquidity_threshold: Minimum liquidity (default: 0.30)
        - fusion_threshold: Minimum fusion score (default: 0.36)
        """
        from engine.archetypes.param_accessor import get_param

        # PR#6A: Read from canonical location with migration-safe fallback
        adx_th = get_param(self, 'wick_trap_moneytaur', 'adx_threshold', 25.0)
        liquidity_th = get_param(self, 'wick_trap_moneytaur', 'liquidity_threshold', 0.30)
        fusion_th = get_param(self, 'wick_trap_moneytaur', 'fusion_threshold', 0.36)

        wick_anomaly = self._get_wick_anomaly(row)  # PATCH: Use helper method
        if not wick_anomaly:
            return False

        adx = row.get('adx_14', 0.0)
        if adx <= adx_th:  # PR#6A: Now reads from config!
            return False

        # PATCH: Use helper method for liquidity_score
        liquidity = self._get_liquidity_score(row)
        if liquidity < liquidity_th:  # PR#6A: Now reads from config!
            return False

        bos_flag = self._get_bos_flag(row)  # PATCH: Use helper method
        if bos_flag == 0:
            return False

        if fusion_score < fusion_th:  # PR#6A: Now reads from config!
            return False

        return True

    def _check_L(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        L - Volume Exhaustion (Zeroika): Volume spike + extreme RSI + falling momentum.

        Criteria:
        - volume_zscore > 1.0
        - RSI > 70 or RSI < 30
        - momentum_score falling (compared to prev bar)
        - liquidity_score >= 0.40
        - fusion_score >= 0.38

        PATCHED: Uses helper methods for momentum_score and liquidity_score
        """
        vol_z = row.get('volume_zscore', 0.0)
        if vol_z <= self.thresh_L.get('vol_z', 1.0):
            return False

        rsi = row.get('rsi_14', 50.0)
        rsi_edge = self.thresh_L.get('rsi_edge', 70.0)
        if not (rsi > rsi_edge or rsi < (100 - rsi_edge)):
            return False

        # Check if momentum is falling
        # PATCH: Use helper method for momentum_score
        if prev_row is not None:
            momentum = self._get_momentum_score(row)
            prev_momentum = self._get_momentum_score(prev_row)
            if momentum >= prev_momentum:
                return False
        else:
            return False

        # PATCH: Use helper method for liquidity_score
        liquidity = self._get_liquidity_score(row)
        if liquidity < 0.40:
            return False

        if fusion_score < self.thresh_L.get('fusion', 0.38):
            return False

        return True

    def _check_M(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        M - Ratio Coil Break (Wyckoff Insider): Low ATR + near POC + BOMS strength.

        Criteria:
        - ATR_percentile < 0.30
        - abs(frvp_poc_distance) < 0.50
        - tf4h_boms_strength > 0.40
        - fusion_score >= 0.35

        PATCHED: Maps to actual tf1h_frvp_distance_to_poc and tf1d_boms_strength columns
        """
        atr_pctile = self._compute_atr_percentile(row, df, index)  # PATCH: Use helper method
        if atr_pctile >= self.thresh_M.get('atr_pctile', 0.30):
            return False

        # PATCH: Use tf1h_frvp_distance_to_poc instead of frvp_poc_distance
        poc_dist = abs(row.get('tf1h_frvp_distance_to_poc', 999.0))
        if poc_dist >= self.thresh_M.get('poc_dist', 0.50):
            return False

        # PATCH: Use tf1d_boms_strength instead of tf4h_boms_strength
        tf4h_boms = row.get('tf1d_boms_strength', 0.0)
        if tf4h_boms <= self.thresh_M.get('boms_strength', 0.40):
            return False

        if fusion_score < self.thresh_M.get('fusion', 0.35):
            return False

        return True

    def _check_S1(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        S1 - Breakdown: Support break with volume confirmation (cascade).

        Criteria (short-biased):
        - Liquidity score < 0.22 (breakdown below support)
        - Volume spike > 1.2x average (confirmation)
        - BOS bearish (tf1h_bos_bearish == True or bos_flag < 0)
        - Fusion score >= 0.38 (regime-tuned in risk_off)
        - Optional: Chain of 2+ BOS for cascade (Moneytaur)

        Regime tuning (applied in dispatcher):
        - Risk_off: Fusion floor 0.38, trail_atr 0.85
        - Crisis: Size 0.3x
        """
        liquidity = self._get_liquidity_score(row)
        if liquidity >= self.thresh_S1.get('liq_max', 0.22):
            return False

        vol_z = row.get('volume_zscore', 0.0)
        if vol_z <= self.thresh_S1.get('vol_z', 1.2):
            return False

        # Check for bearish BOS
        bos_bearish = row.get('tf1h_bos_bearish', False)
        bos_flag = row.get('tf1h_bos_flag', 0)
        if not (bos_bearish or bos_flag < 0):
            return False

        if fusion_score < self.thresh_S1.get('fusion', 0.38):
            return False

        # Optional cascade check: Look for 2+ recent bearish BOS
        if self.thresh_S1.get('require_cascade', False):
            lookback = min(10, index)
            if lookback >= 2:
                recent = df.iloc[index-lookback:index]
                bos_bearish_recent = recent.get('tf1h_bos_bearish', pd.Series([False]*lookback))
                if bos_bearish_recent.sum() < 2:
                    return False
            else:
                return False

        return True

    def _check_S2(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        S2 - Rejection: Resistance test with divergence (fade highs).

        Criteria (short-biased):
        - Liquidity score 0.25-0.35 (near resistance)
        - RSI > 70 (overbought)
        - Volume fade < 0.5 (no follow-through)
        - Fusion score >= 0.36
        - Optional: RSI divergence (price higher, RSI lower)

        Regime tuning:
        - Risk_off: RSI threshold 72, max_bars 48
        - Neutral: Veto if VIX low
        """
        liquidity = self._get_liquidity_score(row)
        if not (0.25 <= liquidity <= self.thresh_S2.get('liq_max', 0.35)):
            return False

        rsi = row.get('rsi_14', 50.0)
        if rsi < self.thresh_S2.get('rsi_min', 70.0):
            return False

        vol_z = row.get('volume_zscore', 0.0)
        if vol_z >= self.thresh_S2.get('vol_max', 0.5):
            return False

        if fusion_score < self.thresh_S2.get('fusion', 0.36):
            return False

        # Optional divergence check
        if self.thresh_S2.get('require_divergence', False) and prev_row is not None:
            lookback = min(5, index)
            if lookback >= 2:
                recent = df.iloc[index-lookback:index]
                recent_rsi = recent.get('rsi_14', pd.Series([50.0]*lookback))
                recent_close = recent.get('close', pd.Series([0.0]*lookback))
                # Bearish divergence: price rising, RSI falling
                price_rising = recent_close.iloc[-1] > recent_close.iloc[0]
                rsi_falling = recent_rsi.iloc[-1] < recent_rsi.iloc[0]
                if not (price_rising and rsi_falling):
                    return False

        return True

    def _check_S3(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        S3 - Whipsaw: False break + reversal (upthrust rejection).

        Criteria (short-biased):
        - Wick anomaly > 2x body (false break above resistance)
        - Volume low < 0.5 (no conviction)
        - MTF trend down (tf4h_bos_bearish or trend indicator)
        - Fusion score >= 0.35

        Regime tuning:
        - Risk_off: Wick multiplier 2.5, trail_atr 0.9
        - Crisis: Veto (too risky)
        """
        # Wick anomaly: upper wick much larger than body
        close = row.get('close', 0.0)
        open_price = row.get('open', close)
        high = row.get('high', close)
        low = row.get('low', close)

        body = abs(close - open_price)
        upper_wick = high - max(close, open_price)
        lower_wick = min(close, open_price) - low

        if body == 0:
            return False

        wick_ratio = upper_wick / body
        if wick_ratio < self.thresh_S3.get('wick_ratio', 2.0):
            return False

        vol_z = row.get('volume_zscore', 0.0)
        if vol_z >= self.thresh_S3.get('vol_max', 0.5):
            return False

        # Check MTF downtrend
        tf4h_bos_bearish = row.get('tf4h_bos_bearish', False)
        if not tf4h_bos_bearish:
            # Fallback: Check if tf4h fusion is negative
            tf4h_fusion = row.get('tf4h_fusion_score', 0.5)
            if tf4h_fusion >= 0.5:  # Neutral or bullish
                return False

        if fusion_score < self.thresh_S3.get('fusion', 0.35):
            return False

        return True

    def _check_S4(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        S4 - Distribution: High volume + no follow (exhaustion climax).

        Criteria (short-biased):
        - Volume climax > 1.5x (exhaustion spike)
        - Momentum fading (current < prev shift)
        - Liquidity < 0.3 (distribution phase)
        - Fusion score >= 0.37

        Regime tuning:
        - Risk_off: Volume threshold 1.6, max_bars 36
        - Neutral: Require VIX high
        """
        vol_z = row.get('volume_zscore', 0.0)
        if vol_z <= self.thresh_S4.get('vol_climax', 1.5):
            return False

        # Check momentum fade
        if prev_row is not None:
            momentum = self._get_momentum_score(row)
            prev_momentum = self._get_momentum_score(prev_row)
            if momentum >= prev_momentum:  # Not fading
                return False
        else:
            return False

        liquidity = self._get_liquidity_score(row)
        if liquidity >= self.thresh_S4.get('liq_max', 0.3):
            return False

        if fusion_score < self.thresh_S4.get('fusion', 0.37):
            return False

        return True

    def _check_S5(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        S5 - Short Squeeze Setup: Funding positive + OI spike (inverse squeeze).

        NOTE: This is SHORT-biased, so we look for NEGATIVE funding (longs pay shorts)
        indicating bearish pressure.

        Criteria (short-biased):
        - Funding rate < -0.05% (longs paying shorts = bearish fuel)
        - OI change > +10% (position building)
        - Volume > 1.0x (activity)
        - Fusion score >= 0.35

        Regime tuning:
        - Crisis: Size 0.4x, trail_atr 0.8
        - Risk_off: Require DXY up
        """
        # Check for funding rate (if available)
        funding = row.get('funding_rate', 0.0)
        if funding >= self.thresh_S5.get('funding_max', -0.0005):  # Negative funding
            return False

        # Check OI spike
        oi_change = row.get('oi_change_pct', 0.0)
        if oi_change <= self.thresh_S5.get('oi_min', 0.10):  # 10% increase
            return False

        vol_z = row.get('volume_zscore', 0.0)
        if vol_z <= self.thresh_S5.get('vol_min', 1.0):
            return False

        if fusion_score < self.thresh_S5.get('fusion', 0.35):
            return False

        return True

    def _check_S6(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        S6 - Alt Rotation Down: Altcoin underperformance (TOTAL3 < BTC).

        Criteria (short-biased):
        - Alt rotation < 0 (alts underperforming BTC)
        - Liquidity < 0.22 (risk-off rotation)
        - DXY up (dollar strength)
        - Fusion score >= 0.34

        Regime tuning:
        - Risk_off: Alt_rotation threshold -0.05, max_bars 48
        - Neutral: Veto if VIX low
        """
        # Check alt rotation (if available)
        alt_rotation = row.get('alt_rotation', 0.0)
        if alt_rotation >= self.thresh_S6.get('alt_rot_max', 0.0):
            return False

        liquidity = self._get_liquidity_score(row)
        if liquidity >= self.thresh_S6.get('liq_max', 0.22):
            return False

        # Check DXY (if available)
        dxy = row.get('dxy', 100.0)
        dxy_prev = prev_row.get('dxy', 100.0) if prev_row is not None else 100.0
        if dxy <= dxy_prev:  # DXY not rising
            return False

        if fusion_score < self.thresh_S6.get('fusion', 0.34):
            return False

        return True

    def _check_S7(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        S7 - Curve Inversion Breakdown: Yield curve inversion + support break.

        Criteria (short-biased):
        - Yield curve spread < 0 (10Y-2Y inversion)
        - BOS bearish (support break)
        - Volume > 1.0x (confirmation)
        - Fusion score >= 0.36

        Regime tuning:
        - Risk_off: Spread threshold -0.02, trail_atr 0.85
        - Crisis: Size 0.3x
        """
        # Check yield curve spread (if available)
        yield_spread = row.get('yield_curve_spread', 0.5)  # Default positive
        if yield_spread >= self.thresh_S7.get('spread_max', 0.0):
            return False

        # Check for bearish BOS
        bos_bearish = row.get('tf1h_bos_bearish', False)
        bos_flag = row.get('tf1h_bos_flag', 0)
        if not (bos_bearish or bos_flag < 0):
            return False

        vol_z = row.get('volume_zscore', 0.0)
        if vol_z <= self.thresh_S7.get('vol_min', 1.0):
            return False

        if fusion_score < self.thresh_S7.get('fusion', 0.36):
            return False

        return True

    def _check_S8(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        S8 - Volume Fade in Chop: Low volume drift + failure (chop filter).

        Criteria (short-biased):
        - Volume < 0.5x (low conviction)
        - RSI extreme > 70 or < 30 (overbought/oversold)
        - ADX < 25 (no trend)
        - Fusion score >= 0.34

        Regime tuning:
        - Neutral: Veto if regime neutral
        - Risk_off: Require volume fade
        """
        vol_z = row.get('volume_zscore', 0.0)
        if vol_z >= self.thresh_S8.get('vol_max', 0.5):
            return False

        rsi = row.get('rsi_14', 50.0)
        rsi_threshold = self.thresh_S8.get('rsi_extreme', 70.0)
        if not (rsi > rsi_threshold or rsi < (100 - rsi_threshold)):
            return False

        adx = row.get('adx_14', 0.0)
        if adx >= self.thresh_S8.get('adx_max', 25.0):
            return False

        if fusion_score < self.thresh_S8.get('fusion', 0.34):
            return False

        return True
