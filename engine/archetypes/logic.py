#!/usr/bin/env python3
"""
Rule-Based Archetype Detection Logic

Implements 16+1 market archetypes using rule-based heuristics with YAML-driven
hard gates, per-archetype fusion weights, and whale conflict penalties.

Production Archetypes (v17 Whale Footprint Architecture):
- spring: PTI-based spring/UTAD counter-trend reversal
- order_block_retest: BOMS + Wyckoff + BOS proximity continuation
- fvg_continuation: Displacement + momentum + recent BOS breakout
- failed_continuation: FVG present + weak RSI + falling ADX reversal
- liquidity_compression: Low ATR + narrow range compression before expansion
- exhaustion_reversal: Extreme RSI + high ATR + volume spike reversal
- liquidity_sweep: BOMS + rising liquidity from oversold grab reversal
- trap_within_trend: HTF trend + liquidity drop + wick against trend
- wick_trap: Wick anomaly + ADX + BOS context rejection
- retest_cluster: Multi-level confluence fakeout + structural move
- confluence_breakout: Low ATR + near POC + BOMS coil breakout
- liquidity_vacuum: Crisis capitulation reversal at panic lows
- whipsaw: Range-bound mean reversion in choppy markets
- funding_divergence: Negative funding + resilience short squeeze
- long_squeeze: Positive funding extreme + overheating cascade (SHORT)
- volume_fade_chop: Low-volume fade scalping in neutral regimes
- oi_divergence: Open interest / price divergence (shadow mode)

Each archetype has YAML config (configs/archetypes/*.yaml) defining hard_gates,
fusion_weights, regime_preferences, and exit parameters.
"""

import pandas as pd
from typing import Tuple, Optional


class ArchetypeLogic:
    """
    Rule-based archetype detection engine.

    Returns archetype name + scores when pattern matches, otherwise (None, 0, 0).
    Gate checks are driven by per-archetype YAML configs (configs/archetypes/*.yaml).
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
        Compute liquidity_score from available features.

        Fixed: Removed broken boms_strength dependency (mean=0.013, dominated the
        score making liquidity gates nearly impossible). Now uses volume, ATR
        percentile, FVG presence, and OI change.
        """
        # Try direct column first (in case it exists)
        if 'liquidity_score' in row.index:
            val = row.get('liquidity_score', 0.0)
            if isinstance(val, (int, float)) and val == val:
                return float(val)

        # Compute from non-broken components
        vol_z = max(0.0, min(row.get('volume_zscore', 0.0) / 2.5, 1.0))
        atr_pct = row.get('atr_percentile', 0.5)
        if not isinstance(atr_pct, (int, float)) or atr_pct != atr_pct:
            atr_pct = 0.5
        fvg = 1.0 if row.get('tf1h_fvg_present', 0) else 0.0
        oi_raw = row.get('oi_change_4h', 0.0)
        oi_score = min(abs(oi_raw if isinstance(oi_raw, (int, float)) and oi_raw == oi_raw else 0.0) * 10.0, 1.0)

        liquidity = 0.35 * vol_z + 0.25 * float(atr_pct) + 0.20 * fvg + 0.20 * oi_score
        return max(0.0, min(liquidity, 1.0))

    def _get_wyckoff_score(self, row: pd.Series) -> float:
        """Graded Wyckoff score from SM-validated directional scores.

        Uses max of 1H/4H/1D bullish scores (all long archetypes in logic.py).
        Falls back to tf1d_wyckoff_score if graded scores unavailable.
        """
        # Primary: graded directional scores (long-only in this code path)
        scores = []
        for col in ['wyckoff_bullish_score', 'tf4h_wyckoff_bullish_score',
                     'tf1d_wyckoff_bullish_score']:
            val = row.get(col, 0.0)
            if isinstance(val, (int, float)) and val == val and val > 0:
                scores.append(float(val))

        # Non-directional phase scores
        for col in ['tf4h_wyckoff_phase_score', 'wyckoff_event_confidence']:
            val = row.get(col, 0.0)
            if isinstance(val, (int, float)) and val == val and val > 0:
                scores.append(float(val))

        if scores:
            return max(scores)

        # Fallback: non-directional 1D score
        val = row.get('tf1d_wyckoff_score', 0.0)
        return float(val) if isinstance(val, (int, float)) and val == val else 0.0

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

    def _get_wick_anomaly(self, row: pd.Series, wick_threshold: float = 0.35) -> bool:
        """
        Detect wick anomaly from OHLC data.

        Aligned with YAML derived:wick_anomaly definition:
        wick > threshold of total candle range (default 35%).
        """
        close = row.get('close', 0.0)
        open_price = row.get('open', close)
        high = row.get('high', close)
        low = row.get('low', close)

        if not all(isinstance(v, (int, float)) for v in [close, open_price, high, low]):
            return False

        candle_range = float(high) - float(low)
        if candle_range <= 0:
            return False

        lower_wick = (min(float(close), float(open_price)) - float(low)) / candle_range
        upper_wick = (float(high) - max(float(close), float(open_price))) / candle_range

        return lower_wick > wick_threshold or upper_wick > wick_threshold

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

    def _check_A(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        A - Spring: PTI-based spring/UTAD trap detection.

        Identity gate: PTI detector fired (trap type detected).
        Quality filtering (displacement, pti_score, fusion) handled by YAML gates.
        """
        pti_trap = row.get('tf1h_pti_trap_type', '')
        if isinstance(pti_trap, str):
            return pti_trap in ('spring', 'utad', 'bull_trap', 'bear_trap')
        return False

    def _check_B(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        B - Order Block Retest: Price retesting a recent BOS level.

        Identity gate: Price within bos_atr_B ATR of a BOS level in last 20 bars.
        Quality filtering (boms_strength, wyckoff, fusion) handled by YAML gates.
        """
        gp = gate_params or {}
        bos_atr_mult = gp.get('bos_atr_B', 1.5)

        atr = row.get('atr_20', 1.0)
        if not isinstance(atr, (int, float)) or atr != atr or atr <= 0:
            return False
        close = row.get('close', 0.0)

        lookback = min(20, index)
        if lookback <= 0:
            return False

        recent = df.iloc[index-lookback:index]
        for i in range(len(recent)):
            bar = recent.iloc[i]
            bos_bull = bar.get('tf1h_bos_bullish', False)
            bos_bear = bar.get('tf1h_bos_bearish', False)
            if bos_bull or bos_bear:
                bos_price = bar.get('close', 0.0)
                if abs(close - bos_price) <= bos_atr_mult * atr:
                    return True
        return False

    def _check_C(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        C - FVG Continuation: Fair Value Gap with directional context.

        Identity gate: FVG present on 1H or 4H AND recent BOS (within 10 bars).
        CONTINUATION requires both: (1) an FVG to fill, and (2) an established
        trend direction to continue (evidenced by recent BOS).
        Quality filtering (displacement, momentum, fusion) handled by YAML gates.
        """
        fvg_1h = row.get('tf1h_fvg_present', 0)
        fvg_4h = row.get('tf4h_fvg_present', 0)
        # Handle NaN
        if isinstance(fvg_1h, float) and fvg_1h != fvg_1h:
            fvg_1h = 0
        if isinstance(fvg_4h, float) and fvg_4h != fvg_4h:
            fvg_4h = 0
        if not (bool(fvg_1h) or bool(fvg_4h)):
            return False

        # Check for recent BOS (continuation needs an established trend)
        lookback = min(10, index)
        if lookback <= 0:
            return False

        recent = df.iloc[index-lookback:index]
        if 'tf1h_bos_bullish' in recent.columns and 'tf1h_bos_bearish' in recent.columns:
            has_bos = recent['tf1h_bos_bullish'].any() or recent['tf1h_bos_bearish'].any()
            return bool(has_bos)

        return False

    def _check_D(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        D - Failed Continuation: FVG present + momentum dying.

        Identity gate: FVG present AND ADX falling vs previous bar.
        The pattern is a continuation setup that's failing — defined by
        FVG existing but momentum (ADX) weakening.
        Quality filtering (RSI, liquidity, fusion) handled by YAML gates.
        """
        fvg_present = row.get('tf1h_fvg_present', 0)
        if isinstance(fvg_present, float) and fvg_present != fvg_present:
            return False
        if not bool(fvg_present):
            return False

        if prev_row is None:
            return False

        adx = row.get('adx', row.get('adx_14', 0.0))
        prev_adx = prev_row.get('adx', prev_row.get('adx_14', 0.0))
        if not isinstance(adx, (int, float)) or adx != adx:
            return False
        if not isinstance(prev_adx, (int, float)) or prev_adx != prev_adx:
            return False
        return float(adx) < float(prev_adx)

    def _check_E(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        E - Volume Exhaustion at Compression: Volume climax at RSI extreme after compression.

        Identity gate: (Volume climax OR high vol_z) AND RSI extreme AND low ATR.
        Old check (ATR < 25th pctile) had no edge — any quiet bar qualified.
        New check requires climax volume at RSI extremes during compression =
        institutional exhaustion event, which is a genuine reversal signal.
        """
        # Still require compression context
        atr_pctile = self._compute_atr_percentile(row, df, index)
        if atr_pctile >= 0.35:
            return False

        # Volume exhaustion evidence
        climax_flag = row.get('climax_volume_flag', 0)
        vol_climax_3b = row.get('volume_climax_last_3b', 0)
        vol_z = row.get('volume_zscore', 0.0)
        # NaN guards
        if isinstance(climax_flag, float) and climax_flag != climax_flag:
            climax_flag = 0
        if isinstance(vol_climax_3b, float) and vol_climax_3b != vol_climax_3b:
            vol_climax_3b = 0
        if isinstance(vol_z, float) and vol_z != vol_z:
            vol_z = 0.0

        has_volume_event = bool(climax_flag) or bool(vol_climax_3b) or float(vol_z) > 2.0
        if not has_volume_event:
            return False

        # RSI extreme confirms exhaustion
        rsi = row.get('rsi_14', 50.0)
        if not isinstance(rsi, (int, float)) or rsi != rsi:
            return False
        if not (float(rsi) > 65 or float(rsi) < 35):
            return False

        return True

    def _check_F(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        F - Exhaustion Reversal: RSI at extreme level.

        Identity gate: RSI > rsi_upper_F or RSI < rsi_lower_F (extreme exhaustion).
        Quality filtering (ATR percentile, volume, liquidity, fusion) handled by YAML gates.
        """
        gp = gate_params or {}
        rsi_upper = gp.get('rsi_upper_F', 78.0)
        rsi_lower = gp.get('rsi_lower_F', 22.0)

        rsi = row.get('rsi_14', 50.0)
        if not isinstance(rsi, (int, float)) or rsi != rsi:
            return False
        return float(rsi) > rsi_upper or float(rsi) < rsi_lower

    def _check_G(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        G - Liquidity Sweep: Lower wick rejection (downward sweep for longs).

        Identity gate: Lower wick > wick_pct_G of candle range.
        Quality filtering (liquidity, boms, volume) handled by YAML gates.
        """
        gp = gate_params or {}
        wick_threshold = gp.get('wick_pct_G', 0.35)

        close = row.get('close', 0.0)
        open_price = row.get('open', close)
        high = row.get('high', close)
        low = row.get('low', close)

        if not all(isinstance(v, (int, float)) for v in [close, open_price, high, low]):
            return False

        candle_range = float(high) - float(low)
        if candle_range <= 0:
            return False

        lower_wick = (min(float(close), float(open_price)) - float(low)) / candle_range
        return lower_wick > wick_threshold

    def _check_H(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        H - Trap Within Trend: Wick anomaly with prevailing trend context.

        Identity gate: Wick anomaly AND trend context exists AND ADX >= 10.
        Differentiates from K (wick_trap) by requiring trend evidence.
        ADX < 10 = no trend at all (notebook: ADX check critical for trap-in-trend).
        Quality filtering (ADX, liquidity, BOS, fusion) handled by YAML gates.
        """
        gp = gate_params or {}
        # H uses K's wick threshold (shared wick anomaly check)
        wick_threshold = gp.get('wick_pct_K', 0.35)
        if not self._get_wick_anomaly(row, wick_threshold=wick_threshold):
            return False

        # ADX minimum — "trap within TREND" requires a trend to exist
        adx = row.get('adx', row.get('adx_14', 0.0))
        if isinstance(adx, (int, float)) and adx == adx:
            if float(adx) < 10.0:
                return False

        # Check for trend context — what makes this a "trap WITHIN trend"
        # Either EMA data shows alignment, or HTF fusion shows directional bias
        price_above_ema = row.get('price_above_ema_50', None)
        tf4h_fusion = row.get('tf4h_fusion_score', 0.5)
        if isinstance(tf4h_fusion, float) and tf4h_fusion != tf4h_fusion:
            tf4h_fusion = 0.5

        has_trend = False
        # EMA alignment exists (any direction = trend context)
        if price_above_ema is not None:
            if isinstance(price_above_ema, (bool, int, float)):
                if not (isinstance(price_above_ema, float) and price_above_ema != price_above_ema):
                    has_trend = True  # EMA data exists = trend context present
        # HTF fusion shows directional bias (not neutral)
        if not has_trend and (float(tf4h_fusion) > 0.55 or float(tf4h_fusion) < 0.35):
            has_trend = True

        return has_trend

    def _check_K(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        K - Wick Trap: Bar with significant wick rejection.

        Identity gate: Wick > wick_pct_K of candle range (wick anomaly detected).
        Quality checks handled by YAML gates.
        """
        gp = gate_params or {}
        wick_threshold = gp.get('wick_pct_K', 0.35)
        return self._get_wick_anomaly(row, wick_threshold=wick_threshold)

    def _check_L(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        L - Retest Cluster (Volume Exhaustion): Volume spike at RSI extreme.

        Identity gate: Volume z-score > vol_z_L AND RSI extreme.
        Quality filtering (momentum falling, liquidity, fusion) handled by YAML gates.
        """
        gp = gate_params or {}
        vol_z_min = gp.get('vol_z_L', 1.0)
        rsi_upper = gp.get('rsi_upper_L', 70.0)
        rsi_lower = gp.get('rsi_lower_L', 30.0)

        vol_z = row.get('volume_zscore', 0.0)
        if not isinstance(vol_z, (int, float)) or vol_z != vol_z:
            return False
        if float(vol_z) <= vol_z_min:
            return False

        rsi = row.get('rsi_14', 50.0)
        if not isinstance(rsi, (int, float)) or rsi != rsi:
            return False
        return float(rsi) > rsi_upper or float(rsi) < rsi_lower

    def _check_M(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        M - Confluence Breakout: Compressed volatility near FRVP POC.

        Identity gate: ATR < 30th percentile AND price within 5% of POC.
        The pattern is a coil near the value area center — compressed
        volatility near a key volume node sets up directional breakout.
        Quality filtering (boms_strength, fusion) handled by YAML gates.

        POC distance threshold tightened from 50% (meaningless — passes everything)
        to 5% (price within 5% of Point of Control, mean POC distance = 2.7%).
        """
        atr_pctile = self._compute_atr_percentile(row, df, index)
        if atr_pctile >= 0.30:
            return False

        poc_dist = row.get('tf1h_frvp_distance_to_poc', 999.0)
        if not isinstance(poc_dist, (int, float)) or poc_dist != poc_dist:
            return False
        return abs(float(poc_dist)) < 0.05

    def _check_S1(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        S1 - Liquidity Vacuum / Breakdown: Bearish structure break.

        Identity gate: Bearish BOS detected on 1H timeframe.
        The pattern is a structural support break — the defining feature is
        a bearish Break of Structure event.
        Quality filtering (liquidity, volume, fusion) handled by YAML gates.
        """
        bos_bearish = row.get('tf1h_bos_bearish', False)
        if isinstance(bos_bearish, float) and bos_bearish != bos_bearish:
            return False
        return bool(bos_bearish)

    def _check_S2(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
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

    def _check_S3(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        S3 - Whipsaw: False break above resistance (upper wick rejection).

        Identity gate: Upper wick > 2x body (false breakout above).
        The pattern is a failed upside breakout — an upper wick that's
        significantly larger than the body indicates rejection at highs.
        Quality filtering (volume, MTF trend, fusion) handled by YAML gates.
        """
        close = row.get('close', 0.0)
        open_price = row.get('open', close)
        high = row.get('high', close)

        if not all(isinstance(v, (int, float)) for v in [close, open_price, high]):
            return False

        body = abs(float(close) - float(open_price))
        if body <= 0:
            return False

        upper_wick = float(high) - max(float(close), float(open_price))
        return upper_wick > 2.0 * body

    def _check_S4(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        S4 - Funding Divergence: Negative funding rate (shorts overcrowded).

        Identity gate: Funding rate significantly negative.
        Quality filtering (resilience, liquidity, fusion) handled by YAML gates.
        """
        gp = gate_params or {}
        funding_z_thresh = gp.get('funding_z_S4', -1.0)

        # Check binance_funding_rate (in feature store for 2022+)
        funding_rate = row.get('binance_funding_rate', None)
        if funding_rate is not None and isinstance(funding_rate, (int, float)) and funding_rate == funding_rate:
            if float(funding_rate) < -0.0001:  # Negative funding = shorts pay longs
                return True

        # Fallback: funding_Z score
        funding_z = row.get('funding_Z', None)
        if funding_z is not None and isinstance(funding_z, (int, float)) and funding_z == funding_z:
            if float(funding_z) < funding_z_thresh:
                return True

        return False

    def _check_S5(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        S5 - Long Squeeze: Extreme positive funding (overcrowded longs).

        Identity gate: Funding rate significantly positive.
        Quality filtering (OI, RSI, volume, fusion) handled by YAML gates.
        """
        gp = gate_params or {}
        funding_z_thresh = gp.get('funding_z_S5', 1.0)

        # Check binance_funding_rate (in feature store for 2022+)
        funding_rate = row.get('binance_funding_rate', None)
        if funding_rate is not None and isinstance(funding_rate, (int, float)) and funding_rate == funding_rate:
            if float(funding_rate) > 0.0001:  # Positive funding = longs pay shorts
                return True

        # Fallback: funding_Z score
        funding_z = row.get('funding_Z', None)
        if funding_z is not None and isinstance(funding_z, (int, float)) and funding_z == funding_z:
            if float(funding_z) > funding_z_thresh:
                return True

        return False

    def _check_S6(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
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

    def _check_S7(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
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

    def _check_S8(self, row, prev_row, df, index, fusion_score, gate_params=None) -> bool:
        """
        S8 - Volume Fade Chop: Low volume in trendless market.

        Identity gate: Volume z-score < 0.5 AND ADX < 25 (chop conditions).
        The pattern is defined by absence of trend (low ADX) with low conviction
        (low volume) — a choppy, range-bound environment.
        Quality filtering (RSI, fusion) handled by YAML gates.
        """
        vol_z = row.get('volume_zscore', 0.0)
        if not isinstance(vol_z, (int, float)) or vol_z != vol_z:
            return False
        if float(vol_z) >= 0.5:
            return False

        adx = row.get('adx', row.get('adx_14', 0.0))
        if not isinstance(adx, (int, float)) or adx != adx:
            return False
        return float(adx) < 25.0
