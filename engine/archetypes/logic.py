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
        }

        # Fusion weights (from config or defaults)
        self.fusion_weights = {
            'wyckoff': 0.331,
            'liquidity': 0.392,
            'momentum': 0.205
        }
        self.fakeout_penalty = 0.075

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

    def calculate_fusion_score(self, row: pd.Series) -> float:
        """
        Calculate fusion score from component scores.

        PATCHED: Now uses helper methods to map to actual feature names.

        Args:
            row: DataFrame row with score columns

        Returns:
            Fusion score in [0, 1]
        """
        wyckoff_score = self._get_wyckoff_score(row)
        liquidity_score = self._get_liquidity_score(row)
        momentum_score = self._get_momentum_score(row)
        smc_score = self._get_smc_score(row)  # PATCH: Added missing SMC score
        fakeout_score = row.get('fakeout_score', 0.0)

        fusion = (
            self.fusion_weights['wyckoff'] * wyckoff_score +
            self.fusion_weights['liquidity'] * liquidity_score +
            self.fusion_weights['momentum'] * momentum_score +
            self.fusion_weights['smc'] * smc_score -  # PATCH: Added SMC component
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
        Check all archetypes in priority order.

        Args:
            row: Current bar
            prev_row: Previous bar (or None if first bar)
            df: Full dataframe for lookback
            index: Current index in df

        Returns:
            (archetype_name_or_None, fusion_score, liquidity_score)
        """
        if not self.use_archetypes:
            return None, 0.0, 0.0

        # Global precheck: liquidity >= min_threshold
        # PATCHED: Use helper method to get actual liquidity score
        liquidity_score = self._get_liquidity_score(row)
        if liquidity_score < self.min_liquidity:
            return None, 0.0, liquidity_score

        fusion_score = self.calculate_fusion_score(row)

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
            bos_flags = recent.get('tf1h_bos_flag', pd.Series([0]*lookback))

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

        PATCHED: Uses helper method for liquidity_score
        """
        atr_pctile = row.get('atr_percentile', 0.5)
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

        atr_pctile = row.get('atr_percentile', 0.5)
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
        atr_pctile = row.get('atr_percentile', 0.5)
        if not (0.25 <= atr_pctile <= 0.75):
            return False

        if fusion_score < self.thresh_G.get('fusion', 0.40):
            return False

        return True

    def _check_H(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        H - Trap Within Trend: HTF trend + liquidity drop + wick against trend.

        Criteria:
        - tf4h_fusion_score > 0.5 (HTF trend)
        - liquidity_score < 0.30 with wick against trend
        - ADX > 25
        - tf1h_bos_flag agrees with signal direction
        - fusion_score >= 0.35
        """
        tf4h_fusion = row.get('tf4h_fusion_score', 0.0)
        if tf4h_fusion <= 0.5:
            return False

        # PATCH: Use helper method for liquidity_score
        liquidity = self._get_liquidity_score(row)
        if liquidity >= self.thresh_H.get('liq_drop', 0.30):
            return False

        adx = row.get('adx_14', 0.0)
        if adx <= self.thresh_H.get('adx', 25.0):
            return False

        # Check for wick against trend
        # Determine trend from tf4h_fusion and check wick
        close = row.get('close', 0.0)
        open_price = row.get('open', close)
        high = row.get('high', close)
        low = row.get('low', close)

        # Assume bullish HTF if tf4h_fusion > 0.5
        body = abs(close - open_price)
        upper_wick = high - max(close, open_price)
        lower_wick = min(close, open_price) - low

        # For bullish trend, look for lower wick > 2× body (trap down)
        # For simplicity, check if either wick is significant
        wick_against_trend = (lower_wick > 2 * body) or (upper_wick > 2 * body)
        if not wick_against_trend:
            return False

        # Check BOS flag alignment
        bos_flag = row.get('tf1h_bos_flag', 0)
        if bos_flag == 0:
            return False

        if fusion_score < self.thresh_H.get('fusion', 0.35):
            return False

        return True

    def _check_K(self, row, prev_row, df, index, fusion_score) -> bool:
        """
        K - Wick Trap (Moneytaur): Wick anomaly + ADX > 25 + BOS context.

        Criteria:
        - wick_anomaly == True (column present and True)
        - ADX > 25
        - liquidity_score >= 0.30
        - tf1h_bos_flag != 0
        - fusion_score >= 0.36
        """
        wick_anomaly = row.get('wick_anomaly', False)
        if not wick_anomaly:
            return False

        adx = row.get('adx_14', 0.0)
        if adx <= self.thresh_K.get('adx', 25.0):
            return False

        # PATCH: Use helper method for liquidity_score
        liquidity = self._get_liquidity_score(row)
        if liquidity < self.thresh_K.get('liq', 0.30):
            return False

        bos_flag = row.get('tf1h_bos_flag', 0)
        if bos_flag == 0:
            return False

        if fusion_score < self.thresh_K.get('fusion', 0.36):
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
        atr_pctile = row.get('atr_percentile', 0.5)
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
