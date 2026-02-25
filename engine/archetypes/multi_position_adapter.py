"""
Multi-Position Adapter for ArchetypeLogic

Extends ArchetypeLogic to return ALL valid archetypes per bar instead of just
the highest scoring one. This unlocks 2-3x more trading opportunities.

ARCHITECTURE:
- Evaluates all archetypes just like _detect_all_archetypes
- Returns a LIST of all archetypes that pass their gates
- Allows multiple simultaneous positions to increase profit

USAGE:
    from engine.archetypes.multi_position_adapter import MultiPositionArchetypeLogic

    # Enable multi-position mode
    logic = MultiPositionArchetypeLogic(config)

    # Returns list of (archetype_name, fusion_score, liquidity_score, direction)
    matches = logic.detect_multi(context)

    # Execute each match as separate position
    for archetype_name, fusion, liq, direction in matches:
        execute_trade(archetype_name, fusion, direction)
"""

import logging
from typing import List, Tuple
from engine.archetypes.logic_v2_adapter import ArchetypeLogic
from engine.runtime.context import RuntimeContext

logger = logging.getLogger(__name__)


class MultiPositionArchetypeLogic(ArchetypeLogic):
    """
    Extension of ArchetypeLogic that returns ALL matching archetypes.

    Instead of dispatch competition selecting ONLY the highest scorer,
    this returns ALL archetypes that pass their individual gates.
    """

    def __init__(self, config: dict, min_relative_score: float = 0.7):
        """
        Initialize multi-position logic.

        Args:
            config: Same config as ArchetypeLogic
            min_relative_score: Minimum score relative to best (0.7 = 70% of best)
                               Set to 0.0 to return ALL passing archetypes
                               Set to 1.0 to return only the best (single-position mode)
        """
        super().__init__(config)
        self.min_relative_score = min_relative_score

    def detect_multi(
        self,
        context: RuntimeContext,
        max_positions: int = 3
    ) -> List[Tuple[str, float, float, str]]:
        """
        Detect ALL valid archetypes for current bar.

        This is the core multi-position method that replaces the single-winner
        dispatch competition with a multi-winner approach.

        Args:
            context: RuntimeContext with bar data and regime state
            max_positions: Maximum number of simultaneous positions (default: 3)

        Returns:
            List of (archetype_name, fusion_score, liquidity_score, direction)
            Empty list if no matches

        Example:
            >>> matches = logic.detect_multi(context, max_positions=3)
            >>> # Returns: [
            >>>   ('wick_trap', 0.997, 0.68, 'LONG'),
            >>>   ('trap_reversal', 0.495, 0.68, 'LONG'),
            >>>   ('trap_within_trend', 0.310, 0.68, 'LONG')
            >>> ]
        """
        from engine import feature_flags as features

        if not self.use_archetypes:
            return []

        # Determine which feature flags to use (bull vs bear)
        bull_archetypes_enabled = any(
            self.enabled.get(s, False)
            for s in ["A", "B", "C", "D", "E", "F", "G", "H", "K", "L", "M"]
        )
        bear_archetypes_enabled = any(
            self.enabled.get(s, False) for s in ["S1", "S2", "S3", "S4", "S5", "S6", "S7", "S8"]
        )

        if bear_archetypes_enabled and not bull_archetypes_enabled:
            use_evaluate_all = features.BEAR_EVALUATE_ALL
            use_soft_liquidity = features.BEAR_SOFT_LIQUIDITY
            use_soft_regime = features.BEAR_SOFT_REGIME
            use_soft_session = features.BEAR_SOFT_SESSION
        else:
            use_evaluate_all = features.BULL_EVALUATE_ALL
            use_soft_liquidity = features.BULL_SOFT_LIQUIDITY
            use_soft_regime = features.BULL_SOFT_REGIME
            use_soft_session = features.BULL_SOFT_SESSION

        # Get global scores
        liquidity_score = self._liquidity_score(context.row)
        fusion_score = self._fusion(context.row)

        # Apply domain engine boosts (Wyckoff, SMC, Temporal)
        fusion_score, wyckoff_meta = self._apply_wyckoff_event_boosts(context.row, fusion_score)
        if wyckoff_meta.get("avoided", False):
            return []

        if features.ENABLE_WYCKOFF:
            fusion_score, _ = self._apply_wyckoff_phase_boost(context.row, fusion_score)

        if features.ENABLE_SMC:
            fusion_score, _ = self._apply_smc_confluence_boost(context.row, fusion_score)

        # Apply soft filters (liquidity, regime, session penalties)
        if use_soft_liquidity and liquidity_score < self.min_liquidity:
            fusion_score *= 0.7
        elif not use_soft_liquidity and liquidity_score < self.min_liquidity:
            return []  # Hard veto

        if use_soft_regime:
            regime = context.regime_label if context else "neutral"
            if regime in ["crisis", "risk_off"]:
                fusion_score *= 0.8

        if use_soft_session:
            hour = context.row.name.hour if hasattr(context.row.name, "hour") else 12
            if hour >= 22 or hour < 8:
                fusion_score *= 0.85

        # Evaluate all archetypes (same logic as _detect_all_archetypes)
        candidates = self._evaluate_all_archetypes(context, fusion_score, liquidity_score)

        if not candidates:
            return []

        # Sort by score (highest first)
        candidates.sort(key=lambda x: (x[1], -x[3]), reverse=True)

        # MULTI-POSITION LOGIC: Return top N archetypes
        best_score = candidates[0][1]
        min_score_threshold = best_score * self.min_relative_score

        # Filter candidates by relative score threshold
        valid_candidates = [
            c for c in candidates
            if c[1] >= min_score_threshold
        ][:max_positions]  # Limit to max_positions

        logger.info(
            f"[MULTI-POSITION] {len(candidates)} candidates, "
            f"{len(valid_candidates)} above {self.min_relative_score*100:.0f}% threshold "
            f"(best={best_score:.3f}, min={min_score_threshold:.3f})"
        )

        # Return as list of (name, fusion, liquidity, direction)
        results = [
            (name, score, liquidity_score, direction)
            for name, score, meta, priority, direction in valid_candidates
        ]

        if len(results) > 1:
            logger.info(
                f"[MULTI-POSITION DISPATCH] {len(results)} positions: "
                f"{[(n, f'{s:.3f}', d) for n, s, _, d in results]}"
            )

        return results

    def _evaluate_all_archetypes(
        self,
        context: RuntimeContext,
        global_fusion_score: float,
        liquidity_score: float
    ) -> List[Tuple[str, float, dict, int, str]]:
        """
        Evaluate all enabled archetypes (same as _detect_all_archetypes).

        Returns:
            List of (name, score, meta, priority, direction) tuples
        """
        # Archetype mapping (same as in logic_v2_adapter.py)
        archetype_map = {
            "A": ("trap_reversal", self._check_A, 1),
            "B": ("order_block_retest", self._check_B, 2),
            "C": ("fvg_continuation", self._check_C, 3),
            "K": ("wick_trap", self._check_K, 4),
            "H": ("trap_within_trend", self._check_H, 5),
            "L": ("volume_exhaustion", self._check_L, 6),
            "F": ("expansion_exhaustion", self._check_F, 7),
            "D": ("failed_continuation", self._check_D, 8),
            "G": ("re_accumulate", self._check_G, 9),
            "E": ("liquidity_compression", self._check_E, 10),
            "M": ("ratio_coil_break", self._check_M, 11),
            "S1": ("liquidity_vacuum", self._check_S1, 12),
            "S2": ("failed_rally", self._check_S2, 13),
            "S3": ("whipsaw", self._check_S3, 14),
            "S4": ("funding_divergence", self._check_S4, 15),
            "S5": ("long_squeeze", self._check_S5, 16),
            "S6": ("alt_rotation_down", self._check_S6, 17),
            "S7": ("curve_inversion", self._check_S7, 18),
            "S8": ("volume_fade_chop", self._check_S8, 19),
        }

        candidates = []

        # Evaluate each enabled archetype
        for letter, (name, check_func, priority) in archetype_map.items():
            if not self.enabled[letter]:
                continue

            # Check regime routing (same as original)
            current_regime = context.regime_label if context else "neutral"
            # Import here to avoid circular dependency
            from engine.archetypes.logic_v2_adapter import ARCHETYPE_REGIMES, DEFAULT_ALLOWED_REGIMES
            allowed_regimes = ARCHETYPE_REGIMES.get(name, DEFAULT_ALLOWED_REGIMES)

            if "all" not in allowed_regimes and current_regime not in allowed_regimes:
                continue

            # Call archetype check method
            result = check_func(context)

            # Handle return types (4-tuple, 3-tuple, or bool)
            if isinstance(result, tuple):
                if len(result) == 4:
                    matched, score, meta, direction = result
                    if matched:
                        candidates.append((name, score, meta, priority, direction))
                elif len(result) == 3:
                    matched, score, meta = result
                    if matched:
                        direction = self._infer_direction(name)
                        candidates.append((name, score, meta, priority, direction))
            else:
                if result:
                    direction = self._infer_direction(name)
                    candidates.append((name, global_fusion_score, {}, priority, direction))

        return candidates
