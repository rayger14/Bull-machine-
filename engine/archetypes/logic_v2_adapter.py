#!/usr/bin/env python3
"""
PR#6A: Rule-Based Archetype Expansion Logic (Adapter Layer Version)
PR#6B: Refactored for RuntimeContext and ThresholdPolicy integration

ADAPTER LAYER: Maps generic feature names to actual TF-prefixed columns
and provides fallback calculations for missing scores.

Implements 11 distinct market archetypes (A-H + K, L, M) using rule-based
heuristics. These create clean labeled data for future PyTorch training.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Optional, Dict
from engine.runtime.context import RuntimeContext

logger = logging.getLogger(__name__)

# Import StateAwareGates for dynamic threshold adjustment (Bull Machine v2)
try:
    from engine.archetypes.state_aware_gates import StateAwareGates, apply_state_aware_gate
    STATE_GATES_AVAILABLE = True
except ImportError:
    STATE_GATES_AVAILABLE = False
    logger.warning("[ArchetypeLogic] StateAwareGates module not available - using static gates")


# ============================================================================
# Helper Functions (Module Level)
# ============================================================================

def _get_first(row, keys, default=0.0):
    """Get first non-null value from list of column names."""
    for k in keys:
        if k in row.index and row[k] is not None and not pd.isna(row[k]):
            return row[k]
    return default


def _norm01(x, lo, hi):
    """Normalize value to [0, 1] range."""
    if hi == lo:
        return 0.0
    v = (x - lo) / (hi - lo)
    return max(0.0, min(1.0, v))


# ============================================================================
# Archetype Logic with Adapter Layer
# ============================================================================

class ArchetypeLogic:
    """
    Rule-based archetype detection for PR#6A with adapter layer.

    The adapter layer intelligently maps generic feature names (liquidity_score,
    wyckoff_score) to actual TF-prefixed columns (tf1d_wyckoff_score, etc.)
    and provides fallback calculations when fields are missing.
    """

    CLASS_VERSION = "archetypes/logic_v2_adapter@r1"

    def __init__(self, config: dict):
        """Initialize with archetype config and alias mappings."""
        import logging
        self.config = config
        self.use_archetypes = config.get('use_archetypes', False)
        logger = logging.getLogger(__name__)
        logger.info(f"[ArchetypeLogic] Using {self.CLASS_VERSION}")
        logger.info(f"[ArchetypeLogic] Config keys received: {list(config.keys())}")
        logger.info(f"[ArchetypeLogic] Has routing: {'routing' in config}")

        # Extract thresholds (kept for backward compatibility with old API)
        # NOTE: With PR#6B RuntimeContext, thresholds come from ThresholdPolicy
        thresholds = config.get('thresholds', {})
        self.min_liquidity = thresholds.get('min_liquidity', 0.30)

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
            # Bear archetypes (short-biased)
            'S1': config.get('enable_S1', False),  # Breakdown
            'S2': config.get('enable_S2', True),   # Failed Rally Rejection (NEW - APPROVED)
            'S3': config.get('enable_S3', False),  # Whipsaw
            'S4': config.get('enable_S4', False),  # Distribution
            'S5': config.get('enable_S5', True),   # Long Squeeze Cascade (NEW - APPROVED with fix)
            'S6': config.get('enable_S6', False),  # Alt Rotation Down (REJECTED)
            'S7': config.get('enable_S7', False),  # Curve Inversion (REJECTED)
            'S8': config.get('enable_S8', False),  # Volume Fade Chop
        }

        # Fusion weights (from config or defaults)
        fusion_cfg = config.get('fusion', {})
        if 'weights' in fusion_cfg:
            self.fusion_weights = fusion_cfg['weights']
        else:
            self.fusion_weights = {
                'wyckoff': 0.331,
                'liquidity': 0.392,
                'momentum': 0.205
            }
        self.fakeout_penalty = 0.075

        # ===================================================================
        # Bull Machine v2: State-Aware Gates
        # ===================================================================
        # Initialize StateAwareGates module if enabled and available
        self.state_gate_module = None
        if STATE_GATES_AVAILABLE and config.get('state_aware_gates', {}).get('enable', False):
            try:
                self.state_gate_module = StateAwareGates(config)
                logger.info("[ArchetypeLogic] State-aware gates ENABLED")
            except Exception as e:
                logger.error(f"[ArchetypeLogic] Failed to initialize StateAwareGates: {e}")
                self.state_gate_module = None
        else:
            logger.info("[ArchetypeLogic] State-aware gates DISABLED (using static thresholds)")

        # ===================================================================
        # ADAPTER LAYER: Alias mappings for TF-prefixed columns
        # ===================================================================
        self.alias = {
            # Composite / domain scores
            # Prefer runtime-injected values (fusion_score, liquidity_score), then TF-prefixed fallbacks
            # Runtime values override broken feature store columns
            "fusion_score": ["fusion_score", "tf4h_fusion_score", "k2_fusion_score"],
            "wyckoff_score": ["wyckoff_score", "tf1d_wyckoff_score"],
            "pti_score": ["pti_score", "tf1h_pti_score", "tf1d_pti_score"],
            "pti_trap_type": ["pti_trap_type", "tf1h_pti_trap_type"],
            "boms_strength": ["boms_strength", "tf1d_boms_strength", "tf4h_boms_strength"],
            "boms_disp": ["tf4h_boms_displacement", "boms_displacement"],

            # Structure helpers
            "bos_bullish": ["tf1h_bos_bullish", "bos_bullish"],
            "bos_bearish": ["tf1h_bos_bearish", "bos_bearish"],
            "fvg_present_1h": ["tf1h_fvg_present", "fvg_present"],
            "fvg_present_4h": ["tf4h_fvg_present"],

            # Momentum / volatility
            "rsi": ["rsi_14", "tf1h_rsi_14"],
            "adx": ["adx_14", "tf1h_adx_14"],
            "atr": ["atr_20", "atr_14", "tf1h_atr_20", "tf1h_atr_14"],
            "atr_percentile": ["atr_percentile", "tf1h_atr_percentile"],
            "vol_z": ["volume_zscore", "tf1h_volume_zscore"],

            # Extras (optional)
            "poc_dist": ["frvp_poc_distance", "tf1h_frvp_distance_to_poc"],
        }

    # =======================================================================
    # Safe Getter Methods
    # =======================================================================

    def g(self, row: pd.Series, key: str, default=0.0):
        """
        Safe getter: resolves aliases and returns first available value.

        Args:
            row: DataFrame row
            key: Logical feature name (e.g., 'wyckoff_score')
            default: Default value if not found

        Returns:
            Feature value or default
        """
        return _get_first(row, self.alias.get(key, [key]), default)

    # =======================================================================
    # Derived Score Calculations
    # =======================================================================

    def _momentum_score(self, row: pd.Series) -> float:
        """
        Derive momentum score if not present.

        Combines RSI distance from 50 and volume z-score, normalized to [0, 1].
        """
        # Check if already present
        mom = row.get("momentum_score")
        if mom is not None and not pd.isna(mom):
            return max(0.0, min(1.0, mom))

        # Derive from components
        rsi = self.g(row, "rsi", 50.0)
        adx = self.g(row, "adx", 20.0)
        vol_z = self.g(row, "vol_z", 0.0)

        # RSI component: 0 at 50, 1 near 25 or 75
        rsi_comp = _norm01(abs(rsi - 50.0), 0.0, 25.0)

        # ADX component: 10→40 maps to 0→1
        adx_comp = _norm01(adx, 10.0, 40.0)

        # Volume component: normalize z-score
        vol_comp = max(0.0, min(1.0, vol_z / 2.0))

        return 0.4 * rsi_comp + 0.3 * adx_comp + 0.3 * vol_comp

    def _liquidity_score(self, row: pd.Series) -> float:
        """
        Get or derive liquidity score.

        Prefers runtime liquidity from PR#4; falls back to blend of
        BOMS strength, FVG presence, and normalized displacement.
        """
        # Prefer existing liquidity_score (from PR#4 runtime calc)
        if "liquidity_score" in row.index and row["liquidity_score"] is not None:
            val = row["liquidity_score"]
            if not pd.isna(val):
                return max(0.0, min(1.0, val))

        # Fallback: derive from available components
        bstr = self.g(row, "boms_strength", 0.0)
        fvg_1h = 1.0 if self.g(row, "fvg_present_1h", 0) else 0.0
        fvg_4h = 1.0 if self.g(row, "fvg_present_4h", 0) else 0.0
        fvg = max(fvg_1h, fvg_4h)

        atr = max(self.g(row, "atr", 0.0), 1e-9)
        disp = self.g(row, "boms_disp", 0.0)
        disp_n = max(0.0, min(1.0, disp / (2.0 * atr)))

        liq_derived = (0.5 * bstr + 0.25 * fvg + 0.25 * disp_n)

        # Log first derivation to diagnose issue
        if not hasattr(self, '_logged_liquidity_derivation'):
            logger.info(f"[ARCHETYPE DEBUG] Derived liquidity: {liq_derived:.3f} (boms={bstr:.3f}, fvg={fvg:.1f}, disp_n={disp_n:.3f})")
            self._logged_liquidity_derivation = True

        return liq_derived

    def _fusion(self, row: pd.Series) -> float:
        """
        Get or recompute fusion score.

        Prefers engine's fusion if present; otherwise recomputes with weights.
        """
        # Prefer existing fusion_score
        fuse = self.g(row, "fusion_score", None)

        # Debug first call
        if not hasattr(self, '_logged_fusion'):
            logger.info(f"[FUSION DEBUG] Got fusion from features: {fuse}")
            self._logged_fusion = True

        if fuse is not None:
            return max(0.0, min(1.0, fuse))

        # Recompute from components
        w = self.fusion_weights
        wy = self.g(row, "wyckoff_score", 0.0)
        liq = self._liquidity_score(row)
        mom = self._momentum_score(row)
        fake = row.get("fakeout_score", 0.0) or 0.0

        f = w.get("wyckoff", 0.331) * wy + \
            w.get("liquidity", 0.392) * liq + \
            w.get("momentum", 0.205) * mom - \
            self.fakeout_penalty * fake

        if not hasattr(self, '_logged_fusion_recompute'):
            logger.info(f"[FUSION DEBUG] Recomputed: {f:.3f} (wy={wy:.3f}, liq={liq:.3f}, mom={mom:.3f}, fake={fake:.3f})")
            self._logged_fusion_recompute = True

        return max(0.0, min(1.0, f))

    # =======================================================================
    # Main Archetype Detection Logic
    # =======================================================================

    def detect(self, context: RuntimeContext) -> Tuple[Optional[str], float, float]:
        """
        Detect which archetype (if any) matches the current context.

        PR#6B: Now accepts RuntimeContext with regime-aware thresholds from ThresholdPolicy.

        **ARCHITECTURE FIX #1**: Now respects EVALUATE_ALL_ARCHETYPES feature flag to prevent
        archetype starvation caused by early returns.

        **ARCHITECTURE FIX #2**: Soft filters apply penalties instead of hard vetoes, allowing
        marginal signals to compete (critical for choppy 2022 conditions).

        **ARCHITECTURE FIX #3 (Bull/Bear Split)**: Different feature flags for bull vs bear
        archetypes to prevent cross-contamination. Bull archetypes use legacy priority dispatch
        with hard liquidity filter. Bear archetypes use evaluate-all with soft liquidity filter.

        Args:
            context: RuntimeContext with row, regime state, and resolved thresholds

        Returns:
            (archetype_name_or_None, fusion_score, liquidity_score)
        """
        from engine import feature_flags as features

        if not self.use_archetypes:
            return None, 0.0, 0.0

        # Determine if bear archetypes ONLY (no bull archetypes enabled)
        # This prevents gold standard configs with mixed archetypes from using bear flags
        bull_archetypes_enabled = any(
            self.enabled.get(s, False) for s in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'K', 'L', 'M']
        )
        bear_archetypes_enabled = any(
            self.enabled.get(s, False) for s in ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8']
        )

        # Select appropriate feature flags based on enabled archetypes
        # CRITICAL: Use bull flags by default to preserve gold standard
        # Only use bear flags if ONLY bear archetypes are enabled (pure bear config)
        if bear_archetypes_enabled and not bull_archetypes_enabled:
            use_evaluate_all = features.BEAR_EVALUATE_ALL
            use_soft_liquidity = features.BEAR_SOFT_LIQUIDITY
            use_soft_regime = features.BEAR_SOFT_REGIME
            use_soft_session = features.BEAR_SOFT_SESSION
            flag_source = "BEAR"
        else:
            # Default to bull flags (preserves gold standard for mixed/bull-only configs)
            use_evaluate_all = features.BULL_EVALUATE_ALL
            use_soft_liquidity = features.BULL_SOFT_LIQUIDITY
            use_soft_regime = features.BULL_SOFT_REGIME
            use_soft_session = features.BULL_SOFT_SESSION
            flag_source = "BULL"

        # Global precheck: liquidity >= min_threshold
        liquidity_score = self._liquidity_score(context.row)
        fusion_score = self._fusion(context.row)

        # DEBUG: Log first liquidity check with flag source
        if not hasattr(self, '_logged_first_liquidity_check'):
            logger.info(f"[LIQUIDITY DEBUG] First check - liquidity_score={liquidity_score:.3f}, min_liquidity={self.min_liquidity:.3f}, use_soft_liquidity={use_soft_liquidity} (source={flag_source}, bull_enabled={bull_archetypes_enabled}, bear_enabled={bear_archetypes_enabled})")
            self._logged_first_liquidity_check = True

        # SOFT FILTER #1: Liquidity penalty (30%) instead of hard veto
        if use_soft_liquidity:
            liquidity_penalty = 1.0
            if liquidity_score < self.min_liquidity:
                liquidity_penalty = 0.7  # 30% penalty for low liquidity
                logger.debug(f"Soft liquidity filter ({flag_source}): {liquidity_score:.3f} < {self.min_liquidity:.3f}, applying 0.7x penalty")
            fusion_score *= liquidity_penalty
        else:
            # Legacy hard filter (kills signal completely)
            if liquidity_score < self.min_liquidity:
                if not hasattr(self, '_logged_liquidity_veto'):
                    logger.info(f"[LIQUIDITY VETO] ({flag_source}) liquidity_score={liquidity_score:.3f} < min_liquidity={self.min_liquidity:.3f} - VETOING")
                    self._logged_liquidity_veto = True
                return None, fusion_score, liquidity_score

        # SOFT FILTER #2: Regime penalty during crisis/risk_off (20%)
        if use_soft_regime:
            regime = context.regime_label if context else 'neutral'
            regime_penalty = 1.0
            if regime in ['crisis', 'risk_off']:
                regime_penalty = 0.8  # 20% penalty during macro stress
                logger.debug(f"Soft regime filter ({flag_source}): regime={regime}, applying 0.8x penalty")
            fusion_score *= regime_penalty

        # SOFT FILTER #3: Session penalty for low-volume periods (15%)
        if use_soft_session:
            hour = context.row.name.hour if hasattr(context.row.name, 'hour') else 12
            session_penalty = 1.0
            # Asian session (22:00-08:00 UTC) gets slight penalty
            if hour >= 22 or hour < 8:
                session_penalty = 0.85  # 15% penalty for Asian session
                logger.debug(f"Soft session filter ({flag_source}): hour={hour}, applying 0.85x penalty")
            fusion_score *= session_penalty

        # Route to evaluate-all or legacy dispatcher based on feature flag
        if use_evaluate_all:
            if not hasattr(self, '_logged_dispatcher_path'):
                logger.info(f"[DISPATCHER PATH] Using EVALUATE_ALL ({flag_source}_EVALUATE_ALL={use_evaluate_all})")
                self._logged_dispatcher_path = True
            return self._detect_all_archetypes(context, fusion_score, liquidity_score)
        else:
            if not hasattr(self, '_logged_dispatcher_path'):
                logger.info(f"[DISPATCHER PATH] Using LEGACY_PRIORITY ({flag_source}_EVALUATE_ALL={use_evaluate_all})")
                self._logged_dispatcher_path = True
            return self._detect_legacy_priority(context, fusion_score, liquidity_score)

    def _detect_all_archetypes(self, context: RuntimeContext, global_fusion_score: float, liquidity_score: float) -> Tuple[Optional[str], float, float]:
        """
        Evaluate ALL enabled archetypes and pick best by archetype-specific score.

        **DISPATCH FIX**: Uses archetype-specific scoring instead of global fusion_score.
        No early returns → prevents archetype starvation.
        """
        # Map letter codes to canonical names and check methods
        archetype_map = {
            # Bull-biased archetypes
            'A': ('trap_reversal', self._check_A, 1),
            'B': ('order_block_retest', self._check_B, 2),
            'C': ('fvg_continuation', self._check_C, 3),
            'K': ('wick_trap', self._check_K, 4),
            'H': ('trap_within_trend', self._check_H, 5),
            'L': ('volume_exhaustion', self._check_L, 6),
            'F': ('expansion_exhaustion', self._check_F, 7),
            'D': ('failed_continuation', self._check_D, 8),
            'G': ('re_accumulate', self._check_G, 9),
            'E': ('liquidity_compression', self._check_E, 10),
            'M': ('ratio_coil_break', self._check_M, 11),
            # Bear-biased archetypes (short-biased)
            'S1': ('breakdown', self._check_S1, 12),
            'S2': ('failed_rally', self._check_S2, 13),  # Failed Rally Rejection
            'S3': ('whipsaw', self._check_S3, 14),
            'S4': ('distribution', self._check_S4, 15),
            'S5': ('long_squeeze', self._check_S5, 16),  # Long Squeeze Cascade
            'S6': ('alt_rotation_down', self._check_S6, 17),
            'S7': ('curve_inversion', self._check_S7, 18),
            'S8': ('volume_fade_chop', self._check_S8, 19),
        }

        candidates = []

        # DEBUG: Log S2/S5 enable status
        if not hasattr(self, '_logged_bear_enable'):
            logger.info(f"[DISPATCHER DEBUG] S2 enabled: {self.enabled.get('S2', False)}, S5 enabled: {self.enabled.get('S5', False)}")
            self._logged_bear_enable = True

        # Evaluate all enabled archetypes without early returns
        for letter, (name, check_func, priority) in archetype_map.items():
            if not self.enabled[letter]:
                continue

            result = check_func(context)

            # Handle both new (tuple) and legacy (bool) return types
            if isinstance(result, tuple):
                matched, score, meta = result
                if matched:
                    candidates.append((name, score, meta, priority))
                    logger.debug(f"[DISPATCH] {name} matched with score={score:.3f}, meta={meta}")
            else:
                # Legacy bool return (not yet upgraded)
                if result:
                    candidates.append((name, global_fusion_score, {}, priority))
                    logger.debug(f"[DISPATCH] {name} matched (legacy bool), using global_fusion={global_fusion_score:.3f}")

        # No matches
        if not candidates:
            return None, global_fusion_score, liquidity_score

        # STEP 2: Apply regime-specific routing weights
        # Use RuntimeContext regime (respects forced regime override)
        regime = context.regime_label
        if len(candidates) > 1:
            logger.info(f"[ROUTING DEBUG] context.regime_label={regime}, candidates={[(n, f'{s:.3f}') for n,s,_,_ in candidates]}")
        # FIX: routing is at self.config['routing'], NOT self.config['archetypes']['routing']
        # (archetype_config already IS the archetypes section)
        routing_config = self.config.get('routing', {})
        regime_routing = routing_config.get(regime, {})
        regime_weights = regime_routing.get('weights', {})

        if not regime_weights:
            logger.info(f"[ROUTING CHECK] regime={regime}, routing_config_keys={list(routing_config.keys())}, regime_found={regime in routing_config}")

        if regime_weights:
            logger.info(f"[REGIME ROUTING] regime={regime}, applying weights: {regime_weights}")
            adjusted_candidates = []
            for name, score, meta, priority in candidates:
                regime_mult = regime_weights.get(name, 1.0)
                adjusted_score = score * regime_mult
                adjusted_candidates.append((name, adjusted_score, meta, priority, score))  # Keep original score for logging
                if regime_mult != 1.0:
                    logger.info(f"[REGIME ROUTING] {name}: {score:.3f} × {regime_mult:.2f} = {adjusted_score:.3f}")
            candidates = [(n, s, m, p) for n, s, m, p, _ in adjusted_candidates]

        # Pick best match by score (highest score wins, priority breaks ties)
        candidates.sort(key=lambda x: (x[1], -x[3]), reverse=True)
        best = candidates[0]

        # Structured logging for diagnosis
        if len(candidates) > 1:
            logger.info(f"[DISPATCH] {len(candidates)} candidates: {[(c[0], f'{c[1]:.3f}') for c in candidates]} → chose {best[0]}")

        return best[0], best[1], liquidity_score

    def _detect_legacy_priority(self, context: RuntimeContext, fusion_score: float, liquidity_score: float) -> Tuple[Optional[str], float, float]:
        """
        Legacy dispatcher with early returns (causes archetype starvation).

        Kept for A/B testing only.

        **BUGFIX**: Handle both bool and tuple returns from archetype checks.
        Some archetypes (B, H, L) return (matched, score, meta) tuples,
        while others return booleans. Must extract matched flag from tuples.
        """
        # Helper to handle both return types
        def _is_match(result):
            if isinstance(result, tuple):
                return result[0]  # Extract matched flag from (matched, score, meta)
            return result  # Boolean return

        # Check archetypes in priority order: A, B, C, K, H, L, F, D, G, E, M
        if self.enabled['A'] and _is_match(self._check_A(context)):
            return 'trap_reversal', fusion_score, liquidity_score

        if self.enabled['B'] and _is_match(self._check_B(context)):
            return 'order_block_retest', fusion_score, liquidity_score

        if self.enabled['C'] and _is_match(self._check_C(context)):
            return 'fvg_continuation', fusion_score, liquidity_score

        if self.enabled['K'] and _is_match(self._check_K(context)):
            return 'wick_trap', fusion_score, liquidity_score

        if self.enabled['H'] and _is_match(self._check_H(context)):
            return 'trap_within_trend', fusion_score, liquidity_score

        if self.enabled['L'] and _is_match(self._check_L(context)):
            return 'volume_exhaustion', fusion_score, liquidity_score

        if self.enabled['F'] and _is_match(self._check_F(context)):
            return 'expansion_exhaustion', fusion_score, liquidity_score

        if self.enabled['D'] and _is_match(self._check_D(context)):
            return 'failed_continuation', fusion_score, liquidity_score

        if self.enabled['G'] and _is_match(self._check_G(context)):
            return 're_accumulate', fusion_score, liquidity_score

        if self.enabled['E'] and _is_match(self._check_E(context)):
            return 'liquidity_compression', fusion_score, liquidity_score

        if self.enabled['M'] and _is_match(self._check_M(context)):
            return 'ratio_coil_break', fusion_score, liquidity_score

        # No match
        return None, fusion_score, liquidity_score

    # =======================================================================
    # Backward Compatibility Wrapper (DEPRECATED)
    # =======================================================================

    def check_archetype(
        self,
        row: pd.Series,
        prev_row: Optional[pd.Series],
        df: pd.DataFrame,
        index: int
    ) -> Tuple[Optional[str], float, float]:
        """
        DEPRECATED: Backward compatibility wrapper for old API.

        Use detect(RuntimeContext) instead. This wrapper creates a minimal
        RuntimeContext without regime awareness.

        **PR#6A FIX**: Reads archetype parameters from self.config to populate
        thresholds dict. This allows optimizer-written params to actually work!
        """
        # PR#6A: Build thresholds from config instead of using empty dict
        # This fixes the zero-variance bug where all trials used hardcoded defaults
        thresholds = self._build_thresholds_from_config()

        # Create minimal context without regime data
        ctx = RuntimeContext(
            ts=row.name if hasattr(row, 'name') else index,
            row=row,
            regime_probs={'neutral': 1.0},
            regime_label='neutral',
            adapted_params={},
            thresholds=thresholds  # PR#6A: Now populated from config!
        )
        return self.detect(ctx)

    def _build_thresholds_from_config(self) -> dict:
        """
        **PR#6A + PHASE1 FIX**: Build thresholds dict from self.config.

        CRITICAL FIX: self.config IS ALREADY the 'archetypes' subdictionary
        (passed from backtest as runtime_config.get('archetypes', {})),
        so we look for archetype params directly in self.config, NOT in
        self.config['archetypes'] (which doesn't exist).

        **PHASE1 FIX**: Also checks legacy 'thresholds' subdirectory with letter
        codes (e.g., config['thresholds']['H']) for backward compatibility.
        Uses same logic as ThresholdPolicy._build_base_map().

        This is the missing link that allows optimizer-written parameters
        to actually affect backtest behavior!

        Returns:
            Dict mapping archetype_name -> {param: value}
            Example: {'trap_within_trend': {'fusion_threshold': 0.35, ...}}
        """
        from engine.archetypes.threshold_policy import ARCHETYPE_NAMES, LEGACY_ARCHETYPE_MAP

        thresholds = {}
        thresholds_subdir = self.config.get('thresholds', {})

        # PR#6A FIX: self.config IS the archetypes section, not the full config
        # Priority order (same as ThresholdPolicy):
        # 1. Top-level archetype config (where optimizer writes)
        # 2. Descriptive name in thresholds subdirectory (legacy new-style)
        # 3. Letter code in thresholds subdirectory (legacy old-style)
        for arch_name in ARCHETYPE_NAMES:
            # Try top-level archetype config FIRST (optimizer writes here)
            if arch_name in self.config and isinstance(self.config[arch_name], dict):
                # Skip metadata keys that aren't actual archetype configs
                if arch_name not in ('use_archetypes', 'thresholds', 'exits'):
                    thresholds[arch_name] = self.config[arch_name]
                    logger.debug(f"[PHASE1] Loaded {arch_name} from top-level config")
                    continue

            # Try descriptive name in thresholds subdirectory (LEGACY)
            if arch_name in thresholds_subdir:
                thresholds[arch_name] = thresholds_subdir[arch_name]
                logger.debug(f"[PHASE1] Loaded {arch_name} from thresholds/{arch_name}")
            # Fallback to letter code in thresholds subdirectory (LEGACY)
            elif arch_name in LEGACY_ARCHETYPE_MAP:
                letter_code = LEGACY_ARCHETYPE_MAP[arch_name]
                if letter_code in thresholds_subdir:
                    thresholds[arch_name] = thresholds_subdir[letter_code]
                    logger.debug(f"[PHASE1] Loaded {arch_name} from thresholds/{letter_code}")
                else:
                    thresholds[arch_name] = {}
            else:
                # No config found, use empty dict
                thresholds[arch_name] = {}

        logger.info(f"[PHASE1] Built thresholds from config: {len([v for v in thresholds.values() if v])} archetypes with params")
        return thresholds

    # =======================================================================
    # Individual Archetype Checks (Using Safe Getters)
    # =======================================================================

    def _check_A(self, context: RuntimeContext) -> tuple:
        """
        Archetype A: Trap Reversal (PTI spring/UTAD + displacement).

        **LAYER 5 FIX**: Read ALL thresholds from context to enable optimization.
        **REFACTOR #2**: Standardized to return (matched, score, meta) tuple.

        Returns:
            (matched: bool, score: float, meta: dict)
        """
        # PR#6A: Read ALL thresholds from context (not hardcoded!)
        fusion_th = context.get_threshold('spring', 'fusion_threshold', 0.33)
        pti_score_th = context.get_threshold('spring', 'pti_score_threshold', 0.40)
        disp_multiplier = context.get_threshold('spring', 'disp_atr_multiplier', 0.80)

        # Get features from context
        pti_trap = self.g(context.row, "pti_trap_type", '')
        if not pti_trap or pti_trap not in ['spring', 'utad']:
            return False, 0.0, {"reason": "no_pti_trap", "pti_trap": pti_trap}

        pti_score = self.g(context.row, "pti_score", 0.0)
        disp = self.g(context.row, "boms_disp", 0.0)
        atr = max(self.g(context.row, "atr", 0.0), 1e-9)
        fusion = context.row.get('fusion_score', 0.0)

        # Gate checks
        if pti_score < pti_score_th:
            return False, 0.0, {"reason": "pti_score_low", "value": pti_score, "threshold": pti_score_th}

        if disp < disp_multiplier * atr:
            return False, 0.0, {"reason": "disp_insufficient", "value": disp, "threshold": disp_multiplier * atr}

        # Archetype-specific scoring
        components = {
            "fusion": self._fusion(context.row),
            "pti_score": pti_score,
            "displacement": min(disp / (disp_multiplier * atr), 1.0)  # Normalize
        }

        weights = context.get_threshold('spring', 'weights', {
            "fusion": 0.5,
            "pti_score": 0.3,
            "displacement": 0.2
        })

        base_score = sum(components.get(k, 0.0) * weights.get(k, 0.0) for k in components)
        archetype_weight = context.get_threshold('spring', 'archetype_weight', 1.0)
        score = max(0.0, min(1.0, base_score * archetype_weight))

        if score < fusion_th:
            return False, score, {"reason": "score_below_threshold", "score": score, "threshold": fusion_th}

        meta = {
            "components": components,
            "weights": weights,
            "base_score": base_score,
            "archetype_weight": archetype_weight,
            "pti_trap_type": pti_trap
        }

        return True, score, meta

    def _check_B(self, context: RuntimeContext) -> tuple:
        """
        Archetype B: Order Block Retest (BOS + BOMS + Wyckoff).

        **LAYER 5 FIX**: Read ALL thresholds from context to enable optimization.
        **DISPATCH FIX**: Returns (matched, score, meta) for true evaluate-all behavior.

        Returns:
            (matched: bool, score: float, meta: dict)
        """
        # Gates (pass/fail thresholds, with state-aware adjustment)
        boms_str_th = context.get_threshold('order_block_retest', 'boms_strength_min', 0.30)
        wyckoff_th = context.get_threshold('order_block_retest', 'wyckoff_min', 0.35)
        base_fusion_th = context.get_threshold('order_block_retest', 'fusion_threshold', 0.374)

        # Apply state-aware gate adjustment (Bull Machine v2)
        fusion_th = apply_state_aware_gate(
            'order_block_retest',
            base_fusion_th,
            context,
            self.state_gate_module,
            log_components=False
        ) if STATE_GATES_AVAILABLE else base_fusion_th

        # Get features from context
        bos_bullish = self.g(context.row, "bos_bullish", 0)
        boms_str = self.g(context.row, "boms_strength", 0.0)
        wyckoff = self.g(context.row, "wyckoff_score", 0.0)

        # Gate checks
        if not bos_bullish:
            return False, 0.0, {"reason": "no_bos"}
        if boms_str < boms_str_th:
            return False, 0.0, {"reason": "boms_weak", "value": boms_str, "threshold": boms_str_th}
        if wyckoff < wyckoff_th:
            return False, 0.0, {"reason": "wyckoff_weak", "value": wyckoff, "threshold": wyckoff_th}

        # Archetype-specific scoring (not global fusion!)
        components = {
            "fusion": self._fusion(context.row),
            "liquidity": self._liquidity_score(context.row),
            "momentum": self._momentum_score(context.row),
            "wyckoff": wyckoff,
            "boms": boms_str
        }

        # Archetype-specific weights (configurable)
        weights = context.get_threshold('order_block_retest', 'weights', {
            "fusion": 0.35,
            "liquidity": 0.20,
            "momentum": 0.15,
            "wyckoff": 0.20,
            "boms": 0.10
        })

        # Weighted score
        base_score = sum(components.get(k, 0.0) * weights.get(k, 0.0) for k in components)

        # Soft penalties (from Fix #2)
        penalties = {"liquidity": 1.0, "regime": 1.0, "session": 1.0}
        # (These are applied globally in detect(), but we track them here for logging)

        # Archetype bias knob (allows fine-tuning without changing weights)
        archetype_weight = context.get_threshold('order_block_retest', 'archetype_weight', 1.0)

        # Final score
        score = max(0.0, min(1.0, base_score * archetype_weight))

        # Gate check on fusion threshold
        if score < fusion_th:
            return False, score, {"reason": "score_below_fusion_th", "score": score, "threshold": fusion_th}

        meta = {
            "components": components,
            "weights": weights,
            "base_score": base_score,
            "archetype_weight": archetype_weight,
            "penalties": penalties
        }

        return True, score, meta

    def _check_C(self, context: RuntimeContext) -> bool:
        """
        Archetype C: FVG Continuation (displacement + momentum).

        **LAYER 5 FIX**: Read ALL thresholds from context to enable optimization.
        """
        # PR#6A: Read ALL thresholds from context (not hardcoded!)
        fusion_th = context.get_threshold('wick_trap', 'fusion_threshold', 0.42)
        disp_multiplier = context.get_threshold('wick_trap', 'disp_atr_multiplier', 1.00)
        momentum_th = context.get_threshold('wick_trap', 'momentum_min', 0.45)
        tf4h_fusion_th = context.get_threshold('wick_trap', 'tf4h_fusion_min', 0.25)

        # Get features from context
        fvg_4h = self.g(context.row, "fvg_present_4h", 0)
        if not fvg_4h:
            return False

        disp = self.g(context.row, "boms_disp", 0.0)
        atr = max(self.g(context.row, "atr", 0.0), 1e-9)
        momentum = self._momentum_score(context.row)
        tf4h_fusion = self.g(context.row, "fusion_score", 0.0)
        fusion = context.row.get('fusion_score', 0.0)

        # Now use DYNAMIC thresholds instead of hardcoded!
        return (disp >= disp_multiplier * atr and
                momentum >= momentum_th and
                tf4h_fusion >= tf4h_fusion_th and
                fusion >= fusion_th)

    def _check_D(self, context: RuntimeContext) -> bool:
        """
        Archetype D: Failed Continuation (FVG + weak RSI).

        **LAYER 5 FIX**: Read ALL thresholds from context to enable optimization.
        """
        # PR#6A: Read ALL thresholds from context (not hardcoded!)
        fusion_th = context.get_threshold('failed_continuation', 'fusion_threshold', 0.42)
        rsi_max = context.get_threshold('failed_continuation', 'rsi_max', 50.0)

        # Get features from context
        fvg_1h = self.g(context.row, "fvg_present_1h", 0)
        if not fvg_1h:
            return False

        rsi = self.g(context.row, "rsi", 50.0)
        fusion = context.row.get('fusion_score', 0.0)

        # Now use DYNAMIC thresholds instead of hardcoded!
        return (rsi <= rsi_max and
                fusion >= fusion_th)

    def _check_E(self, context: RuntimeContext) -> bool:
        """
        Archetype E: Liquidity Compression (low ATR + volume cluster).

        **LAYER 5 FIX**: Read ALL thresholds from context to enable optimization.
        """
        # PR#6A: Read ALL thresholds from context (with state-aware adjustment)
        base_fusion_th = context.get_threshold('volume_exhaustion', 'fusion_threshold', 0.35)
        atr_pct_max = context.get_threshold('volume_exhaustion', 'atr_percentile_max', 0.25)
        vol_z_min = context.get_threshold('volume_exhaustion', 'vol_z_min', 0.5)
        vol_z_max = context.get_threshold('volume_exhaustion', 'vol_z_max', 1.5)
        vol_cluster_min = context.get_threshold('volume_exhaustion', 'vol_cluster_min', 0.70)

        # Apply state-aware gate adjustment (Bull Machine v2)
        fusion_th = apply_state_aware_gate(
            'volume_exhaustion',
            base_fusion_th,
            context,
            self.state_gate_module,
            log_components=False
        ) if STATE_GATES_AVAILABLE else base_fusion_th

        # Get features from context
        atr_pct = self.g(context.row, "atr_percentile", 0.5)
        vol_z = self.g(context.row, "vol_z", 0.0)
        fusion = context.row.get('fusion_score', 0.0)

        # Volume clustering: check if vol_z is moderate (dynamic thresholds)
        vol_cluster = 1.0 if vol_z_min <= vol_z <= vol_z_max else 0.0

        # Now use DYNAMIC thresholds instead of hardcoded!
        return (atr_pct <= atr_pct_max and
                vol_cluster >= vol_cluster_min and
                fusion >= fusion_th)

    def _check_F(self, context: RuntimeContext) -> bool:
        """
        Archetype F: Expansion Exhaustion (extreme RSI + high ATR + vol spike).

        **LAYER 5 FIX**: Read ALL thresholds from context to enable optimization.
        """
        # PR#6A: Read ALL thresholds from context (not hardcoded!)
        fusion_th = context.get_threshold('exhaustion_reversal', 'fusion_threshold', 0.38)
        rsi_min = context.get_threshold('exhaustion_reversal', 'rsi_min', 78.0)
        atr_pct_min = context.get_threshold('exhaustion_reversal', 'atr_percentile_min', 0.90)
        vol_z_min = context.get_threshold('exhaustion_reversal', 'vol_z_min', 1.0)

        # Get features from context
        rsi = self.g(context.row, "rsi", 50.0)
        atr_pct = self.g(context.row, "atr_percentile", 0.5)
        vol_z = self.g(context.row, "vol_z", 0.0)
        fusion = context.row.get('fusion_score', 0.0)

        # Now use DYNAMIC thresholds instead of hardcoded!
        return (rsi >= rsi_min and
                atr_pct >= atr_pct_min and
                vol_z >= vol_z_min and
                fusion >= fusion_th)

    def _check_G(self, context: RuntimeContext) -> bool:
        """
        Archetype G: Re-Accumulate Base (BOMS strength + high liquidity).

        **LAYER 5 FIX**: Read ALL thresholds from context to enable optimization.
        """
        # PR#6A: Read ALL thresholds from context (not hardcoded!)
        fusion_th = context.get_threshold('liquidity_sweep', 'fusion_threshold', 0.40)
        boms_str_min = context.get_threshold('liquidity_sweep', 'boms_strength_min', 0.40)
        liq_min = context.get_threshold('liquidity_sweep', 'liquidity_min', 0.40)

        # Get features from context
        boms_str = self.g(context.row, "boms_strength", 0.0)
        liq = self._liquidity_score(context.row)
        fusion = context.row.get('fusion_score', 0.0)

        # Now use DYNAMIC thresholds instead of hardcoded!
        return (boms_str >= boms_str_min and
                liq >= liq_min and
                fusion >= fusion_th)

    def _check_H(self, context: RuntimeContext) -> tuple:
        """
        Archetype H: Trap Within Trend (ADX trend + liquidity drop).

        **LAYER 5 FIX**: Read ALL thresholds from context to enable optimization.
        **DISPATCH FIX**: Returns (matched, score, meta) for true evaluate-all behavior.

        Returns:
            (matched: bool, score: float, meta: dict)
        """
        # Gates (with state-aware adjustment)
        base_fusion_th = context.get_threshold('trap_within_trend', 'fusion_threshold', 0.35)
        adx_th = context.get_threshold('trap_within_trend', 'adx_threshold', 25.0)
        liq_th = context.get_threshold('trap_within_trend', 'liquidity_threshold', 0.30)

        # Apply state-aware gate adjustment (Bull Machine v2)
        fusion_th = apply_state_aware_gate(
            'trap_within_trend',
            base_fusion_th,
            context,
            self.state_gate_module,
            log_components=False
        ) if STATE_GATES_AVAILABLE else base_fusion_th

        # Get features
        adx = self.g(context.row, "adx", 0.0)
        liq = self._liquidity_score(context.row)

        # Gate checks
        if adx < adx_th:
            return False, 0.0, {"reason": "adx_weak", "value": adx, "threshold": adx_th}
        if liq >= liq_th:  # Trap within trend needs LOW liquidity
            return False, 0.0, {"reason": "liquidity_too_high", "value": liq, "threshold": liq_th}

        # Archetype-specific scoring
        components = {
            "fusion": self._fusion(context.row),
            "momentum": self._momentum_score(context.row),
            "adx": min(adx / 50.0, 1.0),  # Normalize ADX to 0-1
            "liquidity_inverse": max(0.0, 1.0 - liq),  # Inverse liquidity (lower is better for traps)
        }

        # Trap within trend weights (emphasizes momentum + low liquidity)
        weights = context.get_threshold('trap_within_trend', 'weights', {
            "fusion": 0.40,
            "momentum": 0.30,
            "adx": 0.20,
            "liquidity_inverse": 0.10
        })

        base_score = sum(components.get(k, 0.0) * weights.get(k, 0.0) for k in components)
        archetype_weight = context.get_threshold('trap_within_trend', 'archetype_weight', 0.95)  # Slightly disfavor to reduce dominance

        score = max(0.0, min(1.0, base_score * archetype_weight))

        if score < fusion_th:
            return False, score, {"reason": "score_below_threshold", "score": score, "threshold": fusion_th}

        meta = {
            "components": components,
            "weights": weights,
            "base_score": base_score,
            "archetype_weight": archetype_weight
        }

        return True, score, meta

    def _check_K(self, context: RuntimeContext) -> bool:
        """
        Archetype K: Wick Trap / Moneytaur (ADX + liquidity + wicks).

        **LAYER 5 FIX**: Read ALL thresholds from context, not just fusion!
        This was causing zero-variance because ADX and liquidity were hardcoded.
        """
        # PR#6A: Read ALL thresholds from context (not hardcoded!)
        fusion_th = context.get_threshold('wick_trap_moneytaur', 'fusion_threshold', 0.36)
        adx_th = context.get_threshold('wick_trap_moneytaur', 'adx_threshold', 25.0)
        liq_th = context.get_threshold('wick_trap_moneytaur', 'liquidity_threshold', 0.30)

        # Get features from context
        adx = self.g(context.row, "adx", 0.0)
        liq = self._liquidity_score(context.row)
        fusion = context.row.get('fusion_score', 0.0)

        # Now use DYNAMIC thresholds instead of hardcoded!
        return (adx >= adx_th and
                liq >= liq_th and
                fusion >= fusion_th)

    def _check_L(self, context: RuntimeContext) -> tuple:
        """
        Archetype L: Volume Exhaustion / Zeroika (vol spike + extreme RSI).

        **LAYER 5 FIX**: Read ALL thresholds from context to enable optimization.
        **DISPATCH FIX**: Returns (matched, score, meta) for true evaluate-all behavior.

        Returns:
            (matched: bool, score: float, meta: dict)
        """
        # Gates
        fusion_th = context.get_threshold('volume_exhaustion', 'fusion_threshold', 0.38)
        vol_z_min = context.get_threshold('volume_exhaustion', 'vol_z_min', 1.0)
        rsi_min = context.get_threshold('volume_exhaustion', 'rsi_min', 70.0)

        # Get features
        vol_z = self.g(context.row, "vol_z", 0.0)
        rsi = self.g(context.row, "rsi", 50.0)

        # Gate checks
        if vol_z < vol_z_min:
            return False, 0.0, {"reason": "vol_z_low", "value": vol_z, "threshold": vol_z_min}
        if rsi < rsi_min:
            return False, 0.0, {"reason": "rsi_low", "value": rsi, "threshold": rsi_min}

        # Archetype-specific scoring
        components = {
            "fusion": self._fusion(context.row),
            "liquidity": self._liquidity_score(context.row),
            "vol_z": min(vol_z / 3.0, 1.0),  # Normalize vol_z to 0-1 (3+ sigma = max)
            "rsi": min((rsi - 50.0) / 50.0, 1.0),  # Normalize RSI above 50 to 0-1
            "momentum": self._momentum_score(context.row)
        }

        # Volume exhaustion weights (emphasizes vol spike + RSI extremes)
        weights = context.get_threshold('volume_exhaustion', 'weights', {
            "fusion": 0.30,
            "liquidity": 0.20,
            "vol_z": 0.25,
            "rsi": 0.15,
            "momentum": 0.10
        })

        base_score = sum(components.get(k, 0.0) * weights.get(k, 0.0) for k in components)
        archetype_weight = context.get_threshold('volume_exhaustion', 'archetype_weight', 1.05)  # Slightly favor (high quality archetype)

        score = max(0.0, min(1.0, base_score * archetype_weight))

        if score < fusion_th:
            return False, score, {"reason": "score_below_threshold", "score": score, "threshold": fusion_th}

        meta = {
            "components": components,
            "weights": weights,
            "base_score": base_score,
            "archetype_weight": archetype_weight
        }

        return True, score, meta

    def _check_M(self, context: RuntimeContext) -> bool:
        """
        Archetype M: Ratio Coil Break / Wyckoff Insider (low ATR + near POC + BOMS).

        **LAYER 5 FIX**: Read ALL thresholds from context to enable optimization.
        """
        # PR#6A: Read ALL thresholds from context (not hardcoded!)
        fusion_th = context.get_threshold('confluence_breakout', 'fusion_threshold', 0.35)
        atr_pct_max = context.get_threshold('confluence_breakout', 'atr_percentile_max', 0.30)
        poc_dist_max = context.get_threshold('confluence_breakout', 'poc_dist_max', 0.50)
        boms_str_min = context.get_threshold('confluence_breakout', 'boms_strength_min', 0.40)

        # Get features from context
        atr_pct = self.g(context.row, "atr_percentile", 0.5)
        poc_dist = self.g(context.row, "poc_dist", 1.0)
        boms_str = self.g(context.row, "boms_strength", 0.0)
        fusion = context.row.get('fusion_score', 0.0)

        # Now use DYNAMIC thresholds instead of hardcoded!
        return (atr_pct <= atr_pct_max and
                poc_dist <= poc_dist_max and
                boms_str >= boms_str_min and
                fusion >= fusion_th)

    # =======================================================================
    # Bear Archetype Check Methods (Short-Biased)
    # =======================================================================

    def _check_S1(self, context: RuntimeContext) -> bool:
        """
        S1 - Breakdown: Support break with volume confirmation.

        Criteria (short-biased):
        - Low liquidity (breakdown below support)
        - Volume spike confirmation
        - Fusion score threshold
        """
        # Read thresholds from config
        fusion_th = context.get_threshold('breakdown', 'fusion', 0.38)
        liq_max = context.get_threshold('breakdown', 'liq_max', 0.22)
        vol_z_min = context.get_threshold('breakdown', 'vol_z', 1.2)

        # Get features
        liq = self._liquidity_score(context.row)
        vol_z = self.g(context.row, "vol_z", 0.0)
        fusion = context.row.get('fusion_score', 0.0)

        # Check conditions
        return (liq < liq_max and
                vol_z > vol_z_min and
                fusion >= fusion_th)

    def _check_S2(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """
        Archetype S2: Failed Rally Rejection

        Trader: Zeroika (dead cat bounce specialist)
        Edge: Order block retest + volume fade + RSI divergence = bull trap

        Validated Performance (2022):
        - Win Rate: 58.5%
        - Estimated PF: 1.4
        - Forward 24H: -0.68%

        Detection Logic:
        1. Price retests order block (resistance)
        2. RSI divergence (price higher, RSI lower) or overbought
        3. Volume fading (volume_z declining)
        4. Long wick rejection (wick_ratio > 2.0)
        5. 4H trend down (MTF confirmation)

        Returns:
            (matched: bool, score: float, meta: dict)
        """
        # Get thresholds
        fusion_th = context.get_threshold('failed_rally', 'fusion_threshold', 0.36)
        wick_ratio_min = context.get_threshold('failed_rally', 'wick_ratio_min', 2.0)
        rsi_div_required = context.get_threshold('failed_rally', 'require_rsi_divergence', False)

        # Extract features
        ob_high = self.g(context.row, 'tf1h_ob_high', None)
        close = self.g(context.row, 'close', 0)
        high = self.g(context.row, 'high', 0)
        low = self.g(context.row, 'low', 0)
        open_price = self.g(context.row, 'open', close)
        rsi = self.g(context.row, 'rsi_14', 50)
        volume_z = self.g(context.row, 'volume_zscore', 0)
        tf4h_trend_raw = self.g(context.row, 'tf4h_external_trend', 0)  # -1=down
        # Convert to numeric if string (handle data type inconsistency)
        try:
            tf4h_trend = float(tf4h_trend_raw) if tf4h_trend_raw is not None else 0
        except (ValueError, TypeError):
            tf4h_trend = 0

        # Gate 1: Order block retest (within 2% of resistance)
        if ob_high is None or close < ob_high * 0.98:
            return False, 0.0, {"reason": "no_ob_retest", "ob_high": ob_high, "close": close}

        # Gate 2: Wick ratio (rejection)
        wick_top = high - max(close, open_price)
        body = abs(close - open_price)
        wick_ratio = wick_top / body if body > 0 else 0

        if wick_ratio < wick_ratio_min:
            return False, 0.0, {"reason": "weak_rejection", "wick_ratio": wick_ratio}

        # Gate 3: RSI signal (overbought as proxy for divergence)
        # TODO: Implement proper RSI divergence detection when feature available
        rsi_signal = 1.0 if rsi > 65 else 0.5

        # Gate 4: Volume fade
        # Lower volume_z indicates fading buying pressure
        vol_fade = volume_z < 0.4
        vol_fade_score = 1.0 if vol_fade else 0.3

        # Gate 5: 4H trend down (MTF confirmation)
        tf4h_confirm = tf4h_trend < 0  # -1 = downtrend
        tf4h_score = 1.0 if tf4h_confirm else 0.2

        # Compute weighted score
        components = {
            "ob_retest": 1.0,
            "wick_rejection": min(wick_ratio / 3.0, 1.0),  # Normalize
            "rsi_signal": rsi_signal,
            "volume_fade": vol_fade_score,
            "tf4h_confirm": tf4h_score
        }

        weights = context.get_threshold('failed_rally', 'weights', {
            "ob_retest": 0.25,
            "wick_rejection": 0.25,
            "rsi_signal": 0.20,
            "volume_fade": 0.15,
            "tf4h_confirm": 0.15
        })

        score = sum(components[k] * weights.get(k, 0.2) for k in components)

        # Final gate
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_threshold",
                "components": components,
                "score": score,
                "threshold": fusion_th
            }

        return True, score, {
            "components": components,
            "wick_ratio": wick_ratio,
            "rsi": rsi,
            "volume_z": volume_z,
            "tf4h_trend": tf4h_trend,
            "ob_high": ob_high
        }

    def _check_S3(self, context: RuntimeContext) -> bool:
        """
        S3 - Whipsaw: False break + reversal (upthrust rejection).

        Criteria (short-biased):
        - RSI extreme (overbought)
        - Low volume (weak momentum)
        - Fusion score threshold

        Note: Simplified from wick_ratio check (field may not exist)
        """
        # Read thresholds from config (use friendly name, ThresholdPolicy maps letter codes)
        fusion_th = context.get_threshold('whipsaw', 'fusion', 0.35)
        rsi_min = context.get_threshold('whipsaw', 'rsi_extreme', 70.0)
        vol_max = context.get_threshold('whipsaw', 'vol_max', 0.5)

        # Get features
        rsi = self.g(context.row, "rsi", 50.0)
        vol_z = self.g(context.row, "vol_z", 0.0)
        fusion = context.row.get('fusion_score', 0.0)

        # Check conditions
        return (rsi >= rsi_min and
                vol_z <= vol_max and
                fusion >= fusion_th)

    def _check_S4(self, context: RuntimeContext) -> bool:
        """
        S4 - Distribution: High volume + no follow (exhaustion climax).

        Criteria (short-biased):
        - Volume climax (very high volume)
        - Low liquidity (distribution/selling)
        - Fusion score threshold
        """
        # Read thresholds from config
        fusion_th = context.get_threshold('distribution', 'fusion', 0.37)
        vol_climax = context.get_threshold('distribution', 'vol_climax', 1.5)
        liq_max = context.get_threshold('distribution', 'liq_max', 0.3)

        # Get features
        vol_z = self.g(context.row, "vol_z", 0.0)
        liq = self._liquidity_score(context.row)
        fusion = context.row.get('fusion_score', 0.0)

        # Check conditions
        return (vol_z >= vol_climax and
                liq < liq_max and
                fusion >= fusion_th)

    def _check_S5(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """
        Archetype S5: Long Squeeze Cascade

        Trader: Moneytaur (funding rate specialist)
        Edge: Overcrowded longs + exhaustion = cascade down

        CRITICAL FIX: Original user logic was BACKWARDS
        - User claimed: funding > +0.08 = short squeeze (wrong!)
        - Reality: Positive funding = longs pay shorts = LONG SQUEEZE DOWN

        Detection Logic (with graceful OI degradation):
        1. REQUIRED: High positive funding (funding_Z > threshold) -> longs overcrowded
        2. REQUIRED: RSI overbought (rsi > threshold) -> price exhaustion
        3. REQUIRED: Low liquidity (liquidity < threshold) -> thin books amplify cascade
        4. OPTIONAL: OI spike (oi_change > threshold) -> BONUS if available (late longs entering)

        Graceful Degradation:
        - 2024 data (OI available): Full 4-component scoring with OI bonus
        - 2022-2023 data (0% OI coverage): 3-component scoring, OI weight redistributed
        - Pattern fires in both cases, but with adjusted confidence scoring

        Mechanism:
        - Longs paying high funding -> unsustainable
        - New longs piling in (OI spike) -> fuel for cascade [if OI data available]
        - High RSI -> no buyers left
        - Thin liquidity -> cascades faster (no bids to catch fall)

        Returns:
            (matched: bool, score: float, meta: dict)
        """
        # Get thresholds (relaxed from original proposal)
        fusion_th = context.get_threshold('long_squeeze', 'fusion_threshold', 0.35)
        funding_z_min = context.get_threshold('long_squeeze', 'funding_z_min', 1.2)  # Relaxed from 1.5
        rsi_min = context.get_threshold('long_squeeze', 'rsi_min', 70)  # Relaxed from 75
        liq_max = context.get_threshold('long_squeeze', 'liquidity_max', 0.25)  # Relaxed from 0.22

        # Extract features
        funding_z = self.g(context.row, 'funding_Z', 0)
        rsi = self.g(context.row, 'rsi_14', 50)
        liquidity = self._liquidity_score(context.row)

        # DEBUG: Log first call to verify S5 is being evaluated
        if not hasattr(self, '_s5_first_call_logged'):
            logger.info(f"[S5 DEBUG] First evaluation - funding_z={funding_z:.3f}, rsi={rsi:.1f}, liquidity={liquidity:.3f}")
            self._s5_first_call_logged = True

        # Gate 1: High positive funding (longs overcrowded) - REQUIRED
        if funding_z < funding_z_min:
            return False, 0.0, {
                "reason": "funding_not_extreme",
                "funding_z": funding_z,
                "threshold": funding_z_min
            }

        # Gate 2: RSI overbought (exhaustion) - REQUIRED
        if rsi < rsi_min:
            return False, 0.0, {
                "reason": "rsi_not_overbought",
                "rsi": rsi,
                "threshold": rsi_min
            }

        # Gate 3: Low liquidity (amplification factor) - REQUIRED
        if liquidity > liq_max:
            return False, 0.0, {
                "reason": "liquidity_not_thin",
                "liquidity": liquidity,
                "threshold": liq_max
            }

        # Optional: OI spike (BONUS scoring if available)
        oi_change = self.g(context.row, 'oi_change_24h', None)
        has_oi_data = oi_change is not None and not pd.isna(oi_change)

        # Compute score components
        components = {
            "funding_extreme": min((funding_z - 1.0) / 2.0, 1.0),  # Normalize z-score
            "rsi_exhaustion": min((rsi - 50) / 50, 1.0),
            "liquidity_thin": 1.0 - (liquidity / 0.5),  # Lower liquidity = higher score
            "oi_spike": 0.0  # Default: no OI data
        }

        # Cap liquidity_thin component at 1.0
        components["liquidity_thin"] = min(components["liquidity_thin"], 1.0)

        # Add OI spike component if data is available
        if has_oi_data and oi_change > 0.08:  # 8% increase threshold
            components["oi_spike"] = min(oi_change / 0.30, 1.0)  # Normalize (30% = max)

        # Adaptive weights based on OI availability
        if has_oi_data:
            # Full 4-component scoring (2024 data)
            weights = context.get_threshold('long_squeeze', 'weights', {
                "funding_extreme": 0.40,
                "rsi_exhaustion": 0.30,
                "oi_spike": 0.15,
                "liquidity_thin": 0.15
            })
        else:
            # Graceful degradation: redistribute OI weight (2022-2023 data)
            weights = {
                "funding_extreme": 0.50,  # +0.10 from OI
                "rsi_exhaustion": 0.35,   # +0.05 from OI
                "liquidity_thin": 0.15,
                "oi_spike": 0.0           # No weight when unavailable
            }

        score = sum(components[k] * weights.get(k, 0.0) for k in components)

        # Final gate
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_threshold",
                "components": components,
                "score": score,
                "threshold": fusion_th,
                "has_oi_data": has_oi_data
            }

        return True, score, {
            "components": components,
            "weights": weights,
            "has_oi_data": has_oi_data,
            "funding_z": funding_z,
            "oi_change": oi_change if has_oi_data else "N/A",
            "rsi": rsi,
            "liquidity": liquidity,
            "mechanism": "longs_overcrowded_cascade_risk"
        }

    def _check_S6(self, context: RuntimeContext) -> bool:
        """
        S6 - Alt Rotation Down: Altcoin underperformance (TOTAL3 < BTC).

        DISABLED: Requires altcoin dominance data not in feature store.
        """
        return False

    def _check_S7(self, context: RuntimeContext) -> bool:
        """
        S7 - Curve Inversion Breakdown: Yield curve inversion + support break.

        DISABLED: Requires yield curve data not in feature store.
        """
        return False

    def _check_S8(self, context: RuntimeContext) -> bool:
        """
        S8 - Volume Fade in Chop: Low volume drift + failure (chop filter).

        Criteria (short-biased):
        - Low volume (weak momentum)
        - RSI extreme (overbought)
        - Low ADX (choppy market)
        - Fusion score threshold
        """
        # Read thresholds from config
        fusion_th = context.get_threshold('volume_fade_chop', 'fusion', 0.34)
        vol_max = context.get_threshold('volume_fade_chop', 'vol_max', 0.5)
        rsi_extreme = context.get_threshold('volume_fade_chop', 'rsi_extreme', 70.0)
        adx_max = context.get_threshold('volume_fade_chop', 'adx_max', 25.0)

        # Get features
        vol_z = self.g(context.row, "vol_z", 0.0)
        rsi = self.g(context.row, "rsi", 50.0)
        adx = self.g(context.row, "adx", 0.0)
        fusion = context.row.get('fusion_score', 0.0)

        # Check conditions
        return (vol_z <= vol_max and
                rsi >= rsi_extreme and
                adx <= adx_max and
                fusion >= fusion_th)
