"""
Threshold Policy - Centralized Regime-Aware Threshold Management

Manages all archetype thresholds with regime blending, overrides, and guardrails.
No hardcoded thresholds in archetype logic - everything flows through this policy.
"""

import logging
from typing import Dict, Any, Optional
from copy import deepcopy

logger = logging.getLogger(__name__)


# Archetype name mapping (internal names → config keys)
ARCHETYPE_MAP = {
    'order_block_retest': 'B',
    'wick_trap': 'C',
    'spring': 'A',
    'failed_continuation': 'D',
    'volume_exhaustion': 'E',
    'exhaustion_reversal': 'F',
    'liquidity_sweep': 'G',
    'momentum_continuation': 'H',
    'trap_within_trend': 'K',
    'retest_cluster': 'L',
    'confluence_breakout': 'M'
}


class ThresholdPolicy:
    """
    Centralized policy for computing regime-aware archetype thresholds.

    Workflow:
    1. Start from base (static) thresholds
    2. Blend regime profiles by probability weights (convex combination)
    3. Apply regime-level floors/ceilings from gates_regime_profiles
    4. Apply per-archetype overrides (optional delta adjustments)
    5. Clamp to global guardrails (min/max sanity bounds)
    """

    def __init__(
        self,
        base_cfg: Dict[str, Any],
        regime_profiles: Optional[Dict[str, Dict[str, float]]] = None,
        archetype_overrides: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
        global_clamps: Optional[Dict[str, tuple]] = None,
        locked_regime: Optional[str] = None
    ):
        """
        Initialize threshold policy.

        Args:
            base_cfg: Base config with static archetype thresholds
            regime_profiles: gates_regime_profiles from config
            archetype_overrides: Per-archetype delta adjustments per regime
            global_clamps: Global min/max bounds for params
            locked_regime: If set, bypass blending and force this regime's profile.
                          Used for parity testing (e.g., locked_regime='static' to match legacy).
        """
        self.base = base_cfg
        self.regime_profiles = regime_profiles or {}
        self.overrides = archetype_overrides or {}
        self.clamps = global_clamps or {
            'fusion': (0.20, 0.65),
            'liquidity': (0.08, 0.35),
            'min_liquidity': (0.08, 0.30)
        }
        self.locked_regime = locked_regime

        # Extract base archetype thresholds
        self.base_arch_thresholds = self.base.get('archetypes', {}).get('thresholds', {})

        if self.locked_regime:
            logger.info(f"ThresholdPolicy initialized in LOCKED mode (regime={locked_regime})")
        else:
            logger.info(f"ThresholdPolicy initialized with {len(self.base_arch_thresholds)} archetype configs")

    def resolve(
        self,
        regime_probs: Dict[str, float],
        regime_label: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Compute final thresholds for all archetypes given current regime state.

        Args:
            regime_probs: Probability distribution over regimes
            regime_label: Argmax regime after hysteresis

        Returns:
            Dict mapping archetype_name -> {param: threshold}
            Example: {'order_block_retest': {'fusion': 0.33, 'liquidity': 0.14}, ...}
        """
        # PR-A PARITY FIX: Return empty dict for 'static' mode to match legacy behavior
        # Legacy path uses thresholds={}, causing get_threshold() to return hardcoded defaults
        if self.locked_regime == 'static':
            logger.info("[PR-A PARITY] ThresholdPolicy.resolve() returning EMPTY dict for locked_regime='static'")
            return {}

        # LOCKED MODE: Force specific regime (parity testing)
        if self.locked_regime and self.locked_regime in self.regime_profiles:
            return self._resolve_locked(self.locked_regime)

        # NORMAL MODE: Full 5-step pipeline
        # 1) Start from base thresholds
        final = self._build_base_map()

        # 2) Blend regime gates (weighted average across regimes)
        blended_gates = self._blend_regime_gates(regime_probs)

        # 3) Apply regime floors to all archetypes
        self._apply_regime_floors(final, blended_gates)

        # 4) Apply per-archetype regime overrides (optional deltas)
        self._apply_archetype_overrides(final, regime_label)

        # 5) Clamp to global guardrails
        self._clamp(final)

        return final

    def _build_base_map(self) -> Dict[str, Dict[str, float]]:
        """Build base threshold map from config."""
        base_map = {}

        for arch_name, cfg_key in ARCHETYPE_MAP.items():
            if cfg_key in self.base_arch_thresholds:
                # Copy thresholds for this archetype
                base_map[arch_name] = deepcopy(self.base_arch_thresholds[cfg_key])
            else:
                # Fallback empty dict
                base_map[arch_name] = {}

        return base_map

    def _blend_regime_gates(self, regime_probs: Dict[str, float]) -> Dict[str, float]:
        """
        Blend regime-level gates using probability weights.

        Args:
            regime_probs: {'risk_on': 0.62, 'neutral': 0.28, ...}

        Returns:
            Blended gates: {'final_fusion_floor': 0.32, 'min_liquidity': 0.13}
        """
        if not self.regime_profiles:
            return {}

        # Start with zeros
        blended = {}
        gate_keys = set()
        for profile in self.regime_profiles.values():
            gate_keys.update(profile.keys())

        for key in gate_keys:
            blended[key] = 0.0

        # Weighted sum
        for regime, prob in regime_probs.items():
            if regime not in self.regime_profiles:
                continue
            for key, value in self.regime_profiles[regime].items():
                blended[key] += prob * value

        logger.debug(f"Blended regime gates: {blended}")
        return blended

    def _apply_regime_floors(
        self,
        final: Dict[str, Dict[str, float]],
        blended_gates: Dict[str, float]
    ):
        """
        Apply regime-level floors to all archetypes.

        For each archetype, enforce:
        - fusion >= final_fusion_floor (from blended gates)
        - liquidity >= min_liquidity (from blended gates)
        """
        final_fusion_floor = blended_gates.get('final_fusion_floor')
        min_liquidity = blended_gates.get('min_liquidity')

        for arch_name, thresholds in final.items():
            # Apply fusion floor
            if final_fusion_floor is not None and 'fusion' in thresholds:
                thresholds['fusion'] = max(thresholds['fusion'], final_fusion_floor)

            # Apply liquidity floor
            if min_liquidity is not None:
                # Check for both 'liquidity' and 'liq' keys
                if 'liquidity' in thresholds:
                    thresholds['liquidity'] = max(thresholds['liquidity'], min_liquidity)
                if 'liq' in thresholds:
                    thresholds['liq'] = max(thresholds['liq'], min_liquidity)

    def _apply_archetype_overrides(
        self,
        final: Dict[str, Dict[str, float]],
        regime_label: str
    ):
        """
        Apply per-archetype regime-specific overrides (delta adjustments).

        Example override config:
        {
          'order_block_retest': {
            'risk_on': {'fusion': -0.02, 'liquidity': -0.01},
            'risk_off': {'fusion': +0.04, 'liquidity': +0.03}
          }
        }
        """
        if not self.overrides:
            return

        for arch_name, regime_deltas in self.overrides.items():
            if arch_name not in final:
                continue

            if regime_label in regime_deltas:
                deltas = regime_deltas[regime_label]
                for param, delta in deltas.items():
                    if param in final[arch_name]:
                        final[arch_name][param] += delta

    def _clamp(self, final: Dict[str, Dict[str, float]]):
        """
        Clamp all thresholds to global guardrails.

        Ensures thresholds never drift into absurd ranges.
        """
        for arch_name, thresholds in final.items():
            for param, value in thresholds.items():
                if param in self.clamps:
                    min_val, max_val = self.clamps[param]
                    thresholds[param] = max(min_val, min(max_val, value))

    def _resolve_locked(self, locked_regime: str) -> Dict[str, Dict[str, float]]:
        """
        Resolve thresholds in locked mode - force specific regime profile.

        Used for parity testing to ensure RuntimeContext path matches legacy
        when locked to a specific regime.

        Args:
            locked_regime: Regime to lock to (e.g., 'risk_on', 'static')

        Returns:
            Thresholds with forced regime profile applied, OR empty dict for 'static'
        """
        # PR-A PARITY FIX: 'static' mode returns empty dict to match legacy behavior
        # Legacy path uses thresholds={}, which causes get_threshold() to return
        # hardcoded defaults. This ensures perfect parity.
        if locked_regime == 'static':
            logger.info("Locked mode 'static': Returning empty thresholds (use hardcoded defaults)")
            return {}

        # Start with base thresholds
        final = self._build_base_map()

        # Apply regime floors from locked regime only (100% weight)
        if locked_regime in self.regime_profiles:
            locked_gates = self.regime_profiles[locked_regime]
            self._apply_regime_floors(final, locked_gates)

        # Apply overrides for locked regime
        self._apply_archetype_overrides(final, locked_regime)

        # Clamp to guardrails
        self._clamp(final)

        logger.debug(f"Locked mode: Resolved thresholds for regime={locked_regime}")
        return final
