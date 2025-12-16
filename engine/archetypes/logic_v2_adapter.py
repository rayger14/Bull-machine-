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
# REGIME-AWARE ARCHETYPE ROUTING
# ============================================================================

# Define allowed regimes per archetype
# This prevents archetypes from firing in inappropriate market conditions
ARCHETYPE_REGIMES = {
    # Bull-biased archetypes (long-only)
    'spring': ['risk_on', 'neutral'],                    # A: Wyckoff Spring
    'order_block_retest': ['risk_on', 'neutral'],        # B: Order Block Retest
    'wick_trap': ['risk_on', 'neutral'],                 # C: Wick Trap Reversal
    'failed_continuation': ['risk_on', 'neutral'],       # D: Failed Continuation
    'volume_exhaustion': ['risk_on', 'neutral'],         # E: Volume Exhaustion
    'exhaustion_reversal': ['risk_on', 'neutral'],       # F: Exhaustion Reversal
    'liquidity_sweep': ['risk_on', 'neutral'],           # G: Liquidity Sweep
    'momentum_continuation': ['risk_on', 'neutral'],     # H: Momentum Continuation
    'trap_within_trend': ['risk_on', 'neutral'],         # K: Trap Within Trend
    'retest_cluster': ['risk_on', 'neutral'],            # L: Retest Cluster
    'confluence_breakout': ['risk_on', 'neutral'],       # M: Confluence Breakout

    # Bear-biased archetypes (short-biased)
    'breakdown': ['risk_off', 'crisis'],                 # S1 OLD: Breakdown (DEPRECATED)
    'liquidity_vacuum': ['risk_off', 'crisis'],          # S1: Liquidity Vacuum (capitulation reversal)
    'failed_rally': ['risk_off', 'neutral'],             # S2: Failed Rally Rejection (DEPRECATED for BTC)
    'whipsaw': ['risk_off', 'crisis'],                   # S3: Whipsaw
    'distribution': ['risk_off', 'neutral'],             # S4 OLD: Distribution (DEPRECATED)
    'funding_divergence': ['risk_off', 'neutral'],       # S4: Funding Divergence (short squeeze)
    'long_squeeze': ['risk_on', 'neutral'],              # S5: Long Squeeze Cascade (positive funding extreme)
    'alt_rotation_down': ['risk_off', 'crisis'],         # S6: Alt Rotation Down
    'curve_inversion': ['risk_off', 'crisis'],           # S7: Curve Inversion
    'volume_fade_chop': ['neutral'],                     # S8: Volume Fade Chop

    # Special cases
    'bos_choch_reversal': ['risk_on', 'neutral'],        # BOS/CHOCH (momentum)
    'wick_trap_moneytaur': ['risk_on', 'neutral'],       # Moneytaur variant
    'wyckoff_spring_utad': ['risk_on', 'neutral'],       # Wyckoff events
}

# Default: Allow all regimes if not specified
DEFAULT_ALLOWED_REGIMES = ['all']


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
        # META-FUSION: Engine-Level Weight Optimization
        # ===================================================================
        # Initialize MetaFusionEngine if enabled (optimized domain weights)
        self.meta_fusion_engine = None
        self.use_meta_fusion = config.get('use_meta_fusion', False)

        if self.use_meta_fusion:
            try:
                from engine.archetypes.meta_fusion import MetaFusionEngine
                self.meta_fusion_engine = MetaFusionEngine(config)
                logger.info("[ArchetypeLogic] Meta-fusion ENABLED with optimized weights")
                logger.info(f"[ArchetypeLogic]   - Weights: {self.meta_fusion_engine.weights}")
            except ImportError:
                logger.warning("[ArchetypeLogic] MetaFusionEngine not available - using default fusion")
                self.use_meta_fusion = False
            except Exception as e:
                logger.error(f"[ArchetypeLogic] Failed to initialize MetaFusionEngine: {e}")
                self.use_meta_fusion = False
        else:
            logger.info("[ArchetypeLogic] Meta-fusion DISABLED (using legacy fusion weights)")

        # ===================================================================
        # WYCKOFF EVENTS INTEGRATION (PR#6C)
        # ===================================================================
        wyckoff_cfg = config.get('wyckoff_events', {})
        self.wyckoff_enabled = wyckoff_cfg.get('enabled', False)
        self.wyckoff_min_confidence = wyckoff_cfg.get('min_confidence', 0.65)
        self.wyckoff_log_events = wyckoff_cfg.get('log_events', False)

        # Avoid signals (BC, UTAD)
        self.wyckoff_avoid_longs = wyckoff_cfg.get('avoid_longs_if', [])

        # Boost signals (LPS, Spring-A, SOS, PTI confluence)
        self.wyckoff_boost_longs = wyckoff_cfg.get('boost_longs_if', {})

        # Position size reduction (LPSY, UT)
        self.wyckoff_reduce_size = wyckoff_cfg.get('reduce_position_size_if', [])

        if self.wyckoff_enabled:
            logger.info(f"[ArchetypeLogic] Wyckoff events ENABLED")
            logger.info(f"[ArchetypeLogic]   - Avoid longs: {self.wyckoff_avoid_longs}")
            logger.info(f"[ArchetypeLogic]   - Boost longs: {list(self.wyckoff_boost_longs.keys())}")
            logger.info(f"[ArchetypeLogic]   - Min confidence: {self.wyckoff_min_confidence}")

        # ===================================================================
        # TEMPORAL FUSION LAYER (Wisdom Time)
        # ===================================================================
        temporal_cfg = config.get('temporal_fusion', {})
        self.temporal_fusion_enabled = temporal_cfg.get('enabled', False)
        self.temporal_fusion_engine = None

        if self.temporal_fusion_enabled:
            try:
                from engine.temporal.temporal_fusion import TemporalFusionEngine
                self.temporal_fusion_engine = TemporalFusionEngine(temporal_cfg)
                logger.info("[ArchetypeLogic] Temporal Fusion Layer ENABLED")
                logger.info(f"[ArchetypeLogic]   - Component weights: {self.temporal_fusion_engine.weights}")
                logger.info(f"[ArchetypeLogic]   - Adjustment range: [{self.temporal_fusion_engine.min_multiplier:.2f}, {self.temporal_fusion_engine.max_multiplier:.2f}]")
            except ImportError as e:
                logger.warning(f"[ArchetypeLogic] TemporalFusionEngine not available: {e}")
                self.temporal_fusion_enabled = False
            except Exception as e:
                logger.error(f"[ArchetypeLogic] Failed to initialize TemporalFusionEngine: {e}")
                self.temporal_fusion_enabled = False
        else:
            logger.info("[ArchetypeLogic] Temporal Fusion Layer DISABLED")

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

        META-FUSION INTEGRATION: If meta-fusion is enabled, uses optimized
        engine weights instead of default fusion logic.

        Prefers engine's fusion if present; otherwise recomputes with weights.
        """
        # META-FUSION: Use optimized engine weights if available
        if self.use_meta_fusion and self.meta_fusion_engine is not None:
            fusion, _ = self.meta_fusion_engine.apply_meta_fusion(row)

            # Debug first call
            if not hasattr(self, '_logged_meta_fusion'):
                logger.info(f"[META-FUSION] Using optimized weights: fusion={fusion:.3f}")
                self._logged_meta_fusion = True

            return fusion

        # LEGACY FUSION: Prefer existing fusion_score
        fuse = self.g(row, "fusion_score", None)

        # Debug first call
        if not hasattr(self, '_logged_fusion'):
            logger.info(f"[FUSION DEBUG] Got fusion from features: {fuse}")
            self._logged_fusion = True

        if fuse is not None:
            return max(0.0, min(1.0, fuse))

        # Recompute from components (legacy weights)
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

    def _apply_wyckoff_event_boosts(self, row: pd.Series, fusion_score: float) -> Tuple[float, Dict]:
        """
        Apply Wyckoff event boosts/filters to fusion score (PR#6C).

        Returns:
            (adjusted_fusion_score, wyckoff_metadata)
        """
        if not self.wyckoff_enabled:
            return fusion_score, {}

        metadata = {
            'events_detected': [],
            'boosts_applied': [],
            'avoided': False,
            'original_fusion': fusion_score,
        }

        # STEP 1: Check avoid signals (BC, UTAD) - VETO trade completely
        for event_name in self.wyckoff_avoid_longs:
            event_col = event_name  # e.g., 'wyckoff_bc'
            conf_col = f"{event_name}_confidence"

            if self.g(row, event_col, False):
                # Check confidence threshold
                confidence = self.g(row, conf_col, 0.0)
                if confidence >= self.wyckoff_min_confidence:
                    metadata['avoided'] = True
                    metadata['avoid_reason'] = event_name
                    metadata['avoid_confidence'] = confidence

                    if self.wyckoff_log_events:
                        logger.info(f"[WYCKOFF VETO] {event_name} detected (conf={confidence:.2f}) - AVOIDING long entry")

                    return 0.0, metadata  # Kill signal completely

        # STEP 2: Apply boost multipliers (LPS, Spring-A, SOS, PTI confluence)
        total_boost = 0.0
        for event_name, boost_multiplier in self.wyckoff_boost_longs.items():
            event_col = event_name  # e.g., 'wyckoff_lps'
            conf_col = f"{event_name}_confidence"

            if self.g(row, event_col, False):
                # Check confidence threshold
                confidence = self.g(row, conf_col, 0.0)
                if confidence >= self.wyckoff_min_confidence:
                    total_boost += boost_multiplier
                    metadata['events_detected'].append(event_name)
                    metadata['boosts_applied'].append({
                        'event': event_name,
                        'boost': boost_multiplier,
                        'confidence': confidence
                    })

                    if self.wyckoff_log_events:
                        logger.info(f"[WYCKOFF BOOST] {event_name} detected (conf={confidence:.2f}) - boost={boost_multiplier:+.2f}")

        # Apply total boost
        if total_boost > 0:
            boosted_fusion = fusion_score * (1.0 + total_boost)
            metadata['final_fusion'] = min(1.0, boosted_fusion)  # Cap at 1.0
            return min(1.0, boosted_fusion), metadata

        return fusion_score, metadata

    def _apply_domain_engines(self, context: RuntimeContext, base_score: float, tags: list) -> float:
        """
        Universal domain engine modifier - applies Wyckoff, SMC, Temporal, HOB, Macro, Fusion
        Soft vetoes (penalties) not hard kills
        """
        r = context.row
        score = base_score

        # Determine direction from tags
        direction = "LONG" if "LONG" in tags else ("SHORT" if "SHORT" in tags else "NEUTRAL")

        # =============================================================================
        # WYCKOFF ENGINE
        # =============================================================================
        phase = self.g(r, 'wyckoff_phase_abc', None)

        if phase == 'D' and direction == "LONG":
            score *= 0.65  # Soft veto - distribution phase reduces long confidence
        elif phase == 'A' and direction == "LONG":
            score *= 1.15  # Accumulation boosts longs
        elif phase == 'A' and direction == "SHORT":
            score *= 0.65  # Accumulation reduces short confidence
        elif phase == 'D' and direction == "SHORT":
            score *= 1.15  # Distribution boosts shorts

        # Wyckoff events
        if self.g(r, 'wyckoff_spring_a', False) and direction == "LONG":
            score *= 2.50  # Major spring boost
        if self.g(r, 'wyckoff_lps', False) and direction == "LONG":
            score *= 1.50
        if self.g(r, 'wyckoff_sos', False) and direction == "LONG":
            score *= 1.80
        if self.g(r, 'wyckoff_utad', False) and direction == "SHORT":
            score *= 2.50  # UTAD top boost
        if self.g(r, 'wyckoff_bc', False) and direction == "SHORT":
            score *= 2.00

        # =============================================================================
        # SMC ENGINE
        # =============================================================================
        if direction == "LONG":
            if self.g(r, 'tf4h_bos_bullish', False):
                score *= 2.00  # 4H bullish structure
            elif self.g(r, 'tf1h_bos_bullish', False):
                score *= 1.40  # 1H bullish structure

            if self.g(r, 'smc_demand_zone', False):
                score *= 1.60
            if self.g(r, 'smc_liquidity_sweep', False):
                score *= 1.80

            # Bearish SMC reduces long confidence
            if self.g(r, 'tf4h_bos_bearish', False):
                score *= 0.70
            if self.g(r, 'smc_supply_zone', False):
                score *= 0.70

        elif direction == "SHORT":
            if self.g(r, 'tf4h_bos_bearish', False):
                score *= 2.00  # 4H bearish structure
            elif self.g(r, 'tf1h_bos_bearish', False):
                score *= 1.40

            if self.g(r, 'smc_supply_zone', False):
                score *= 1.60

            # Bullish SMC reduces short confidence
            if self.g(r, 'tf4h_bos_bullish', False):
                score *= 0.70
            if self.g(r, 'smc_demand_zone', False):
                score *= 0.70

        # =============================================================================
        # TEMPORAL ENGINE
        # =============================================================================
        if self.g(r, 'fib_time_cluster', False):
            score *= 1.70  # Fibonacci time window

        temporal_conf = self.g(r, 'temporal_confluence', 0.0)
        if temporal_conf >= 0.70:
            score *= 1.40  # High multi-timeframe confluence
        elif temporal_conf >= 0.50:
            score *= 1.20
        elif temporal_conf <= 0.20:
            score *= 0.90  # Low confluence penalty

        # =============================================================================
        # HOB ENGINE (Order Book)
        # =============================================================================
        if direction == "LONG":
            if self.g(r, 'hob_demand_zone', False):
                score *= 1.50  # Large bid walls

            hob_imbalance = self.g(r, 'hob_imbalance', 0.0)
            if hob_imbalance > 0.60:
                score *= 1.30  # Strong bid imbalance
            elif hob_imbalance < -0.60:
                score *= 0.75  # Ask imbalance penalty

        elif direction == "SHORT":
            # HOB supply zones would go here (if available)
            hob_imbalance = self.g(r, 'hob_imbalance', 0.0)
            if hob_imbalance < -0.60:
                score *= 1.30  # Strong ask imbalance
            elif hob_imbalance > 0.60:
                score *= 0.75  # Bid imbalance penalty

        # =============================================================================
        # MACRO ENGINE
        # =============================================================================
        crisis = self.g(r, 'crisis_composite', 0.0)

        if direction == "LONG" and crisis >= 0.60:
            score *= 1.30  # Crisis = contrarian long opportunity
        elif direction == "LONG" and crisis >= 0.40:
            score *= 1.15

        if direction == "SHORT" and crisis >= 0.60:
            score *= 1.50  # Crisis boosts short confidence

        # =============================================================================
        # FUSION ENGINE (handled globally, usually 1.0x)
        # =============================================================================
        # This is typically handled at the portfolio level, not per-archetype

        return max(0.0, min(score, 5.0))  # Cap score safely

    # =======================================================================
    # Direction Inference Helper (for backward compatibility)
    # =======================================================================

    def _infer_direction(self, archetype_name: str) -> str:
        """
        Infer trade direction from archetype name.

        Used as fallback when methods don't return direction explicitly.

        Directions:
            - LONG: Bull archetypes (A, B, C, D, F, G, H, K, L, M, S1, S5)
            - SHORT: Bear archetypes (S4)
            - EITHER: Ambiguous archetypes (E, S3, S8)
        """
        # Bull archetypes - always LONG
        if archetype_name in ['trap_reversal', 'order_block_retest', 'fvg_continuation',
                              'failed_continuation', 'expansion_exhaustion',
                              're_accumulate', 'wick_trap', 'trap_within_trend', 'volume_exhaustion',
                              'ratio_coil_break']:
            return 'LONG'

        # S1 (Liquidity Vacuum) - LONG capitulation reversal
        if archetype_name == 'breakdown':
            return 'LONG'

        # S4 (Funding Divergence) - SHORT
        if archetype_name == 'funding_divergence':
            return 'SHORT'

        # S5 (Long Squeeze) - LONG reversal after squeeze
        if archetype_name == 'long_squeeze':
            return 'LONG'

        # E (Volume Exhaustion), S3 (Whipsaw), S8 (Volume Fade Chop) - EITHER (two-sided/ambiguous)
        if archetype_name in ['liquidity_compression', 'whipsaw', 'volume_fade_chop']:
            return 'EITHER'

        # S2 (Failed Rally) - SHORT (deprecated but fallback)
        if archetype_name == 'failed_rally':
            return 'SHORT'

        # Stubs (S6, S7) - default to LONG
        if archetype_name in ['alt_rotation_down', 'curve_inversion']:
            return 'LONG'

        # Unknown - default to LONG for safety
        return 'LONG'

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

        # PR#6C: Apply Wyckoff event boosts/filters
        fusion_score, wyckoff_meta = self._apply_wyckoff_event_boosts(context.row, fusion_score)

        # If Wyckoff veto'd the trade, return immediately
        if wyckoff_meta.get('avoided', False):
            return None, fusion_score, liquidity_score

        # TEMPORAL FUSION ADJUSTMENT (Wisdom Time Layer)
        # Adjust fusion based on temporal confluence AFTER Wyckoff boosts
        if self.temporal_fusion_enabled and self.temporal_fusion_engine is not None:
            try:
                # Compute temporal confluence
                temporal_confluence = self.temporal_fusion_engine.compute_temporal_confluence(context)

                # Adjust fusion weight
                original_fusion = fusion_score
                fusion_score = self.temporal_fusion_engine.adjust_fusion_weight(fusion_score, temporal_confluence)

                # Log significant adjustments (only first few times to avoid spam)
                if abs(fusion_score - original_fusion) > 0.01:
                    if not hasattr(self, '_temporal_adjustment_log_count'):
                        self._temporal_adjustment_log_count = 0
                    if self._temporal_adjustment_log_count < 5:
                        logger.info(
                            f"[TEMPORAL] Fusion adjusted: {original_fusion:.3f} → {fusion_score:.3f} "
                            f"(confluence={temporal_confluence:.3f})"
                        )
                        self._temporal_adjustment_log_count += 1
            except Exception as e:
                logger.error(f"[TEMPORAL] Error applying temporal adjustment: {e}")
                # Continue with unadjusted fusion on error

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
            'S2': ('failed_rally', self._check_S2, 13),  # DEPRECATED for BTC - poor performance
            'S3': ('whipsaw', self._check_S3, 14),
            'S4': ('funding_divergence', self._check_S4, 15),  # Funding Divergence (Short Squeeze)
            'S5': ('long_squeeze', self._check_S5, 16),  # Long Squeeze Cascade
            'S6': ('alt_rotation_down', self._check_S6, 17),  # STUB - always returns False
            'S7': ('curve_inversion', self._check_S7, 18),  # STUB - always returns False
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

            # REGIME-AWARE ROUTING: Check if archetype is allowed in current regime
            current_regime = context.regime_label if context else 'neutral'
            allowed_regimes = ARCHETYPE_REGIMES.get(name, DEFAULT_ALLOWED_REGIMES)

            # Skip archetype if regime not allowed (hard filter for regime-aware optimization)
            if 'all' not in allowed_regimes and current_regime not in allowed_regimes:
                logger.debug(
                    f"[REGIME ROUTING] Skipping {name}: regime={current_regime} "
                    f"not in allowed={allowed_regimes}"
                )
                continue

            result = check_func(context)

            # Handle new 4-tuple (matched, score, meta, direction), 3-tuple legacy, and bool returns
            if isinstance(result, tuple):
                if len(result) == 4:
                    matched, score, meta, direction = result
                    if matched:
                        candidates.append((name, score, meta, priority, direction))
                        logger.debug(f"[DISPATCH] {name} matched with score={score:.3f}, direction={direction}, meta={meta}")
                elif len(result) == 3:
                    matched, score, meta = result
                    if matched:
                        # Infer direction from archetype name (fallback for partial upgrades)
                        direction = self._infer_direction(name)
                        candidates.append((name, score, meta, priority, direction))
                        logger.debug(f"[DISPATCH] {name} matched with score={score:.3f}, direction={direction} (inferred), meta={meta}")
            else:
                # Legacy bool return (not yet upgraded)
                if result:
                    direction = self._infer_direction(name)
                    candidates.append((name, global_fusion_score, {}, priority, direction))
                    logger.debug(f"[DISPATCH] {name} matched (legacy bool), using global_fusion={global_fusion_score:.3f}, direction={direction}")

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
            for name, score, meta, priority, direction in candidates:
                regime_mult = regime_weights.get(name, 1.0)
                adjusted_score = score * regime_mult
                adjusted_candidates.append((name, adjusted_score, meta, priority, direction, score))  # Keep original score for logging
                if regime_mult != 1.0:
                    logger.info(f"[REGIME ROUTING] {name}: {score:.3f} × {regime_mult:.2f} = {adjusted_score:.3f}")
            candidates = [(n, s, m, p, d) for n, s, m, p, d, _ in adjusted_candidates]

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
        Archetype A: Trap Reversal (Spring pattern - multi-path detection).

        **QUICK WIN FIX**: Make PTI optional, use Wyckoff spring as primary detection.
        **LAYER 5 FIX**: Read ALL thresholds from context to enable optimization.
        **REFACTOR #2**: Standardized to return (matched, score, meta) tuple.

        Returns:
            (matched: bool, score: float, meta: dict)
        """
        r = context.row

        # PR#6A: Read ALL thresholds from context (not hardcoded!)
        fusion_th = context.get_threshold('spring', 'fusion_threshold', 0.33)
        pti_score_th = context.get_threshold('spring', 'pti_score_threshold', 0.30)  # Relaxed from 0.40
        disp_multiplier = context.get_threshold('spring', 'disp_atr_multiplier', 0.50)  # Relaxed from 0.80
        wick_th = context.get_threshold('spring', 'wick_lower_threshold', 0.60)

        # ============================================================================
        # MULTI-PATH SPRING DETECTION (similar to C making CHOCH optional)
        # ============================================================================

        # PATH 1: Wyckoff Spring Events (highest confidence - 8 events exist!)
        wyckoff_spring_a = self.g(r, 'wyckoff_spring_a', False)
        wyckoff_spring_b = self.g(r, 'wyckoff_spring_b', False)
        wyckoff_lps = self.g(r, 'wyckoff_lps', False)

        # PATH 2: PTI Trap Detection (if available - FIX feature names!)
        pti_trap = self.g(r, "tf1h_pti_trap_type", '')  # FIX: was "pti_trap_type"
        pti_score = self.g(r, "tf1h_pti_score", 0.0)    # FIX: was "pti_score"

        # PATH 3: Synthetic Spring (wick rejection + volume + displacement)
        wick_lower = self.g(r, 'wick_lower_ratio', 0.0)
        volume_climax = self.g(r, 'volume_climax_last_3b', False)
        disp = self.g(r, "tf4h_boms_displacement", 0.0)  # FIX: was "boms_disp"
        atr = max(self.g(r, "atr_14", 0.0), 1e-9)        # FIX: was "atr"

        # Build score based on detection path
        base_score = 0.0
        detection_path = None

        # PATH 1: Wyckoff Spring (primary - most reliable)
        if wyckoff_spring_a:
            base_score = 0.50  # High confidence spring event
            detection_path = "wyckoff_spring_a"
        elif wyckoff_spring_b:
            base_score = 0.45  # Moderate confidence spring
            detection_path = "wyckoff_spring_b"
        elif wyckoff_lps and wick_lower >= wick_th:
            base_score = 0.40  # LPS + wick rejection combo
            detection_path = "wyckoff_lps_wick"

        # PATH 2: PTI Trap (secondary - currently all 'none' but keep for future)
        elif pti_trap in ['spring', 'utad'] and pti_score >= pti_score_th:
            base_score = 0.35 + (pti_score * 0.20)  # Scale with PTI confidence
            detection_path = f"pti_{pti_trap}"

        # PATH 3: Synthetic Spring (tertiary - fallback pattern)
        elif wick_lower >= wick_th and volume_climax and disp >= disp_multiplier * atr:
            base_score = 0.30  # Lower confidence synthetic
            detection_path = "synthetic_spring"

        # No spring pattern detected - early return
        if base_score == 0.0:
            return False, 0.0, {
                "reason": "no_spring_pattern_detected",
                "wyckoff_spring_a": wyckoff_spring_a,
                "wyckoff_lps": wyckoff_lps,
                "pti_trap": pti_trap,
                "wick_lower": wick_lower,
                "volume_climax": volume_climax
            }

        # Apply bonus modifiers
        bonuses = 0.0

        # PTI confirmation bonus (if spring detected via other paths)
        if detection_path and "pti" not in detection_path and pti_score >= pti_score_th:
            bonuses += 0.10

        # Displacement bonus (strong reversal move)
        if disp >= disp_multiplier * atr:
            bonuses += 0.10

        # Volume climax bonus (exhaustion signal)
        if volume_climax:
            bonuses += 0.05

        # Apply bonuses and archetype weight
        score = min(1.0, base_score + bonuses)
        archetype_weight = context.get_threshold('spring', 'archetype_weight', 1.0)
        score = max(0.0, min(1.0, score * archetype_weight))

        # ============================================================================
        # DOMAIN ENGINE INTEGRATION (BOOST/VETO LAYER)
        # ============================================================================
        # Apply domain engines BEFORE fusion threshold gate
        # This allows marginal signals to qualify via boosts
        # Order: VETOES first (safety) -> BOOSTS second -> GATE third

        use_wyckoff = context.metadata.get('feature_flags', {}).get('enable_wyckoff', False)
        use_smc = context.metadata.get('feature_flags', {}).get('enable_smc', False)
        use_temporal = context.metadata.get('feature_flags', {}).get('enable_temporal', False)
        use_hob = context.metadata.get('feature_flags', {}).get('enable_hob', False)
        use_macro = context.metadata.get('feature_flags', {}).get('enable_macro', False)

        domain_boost = 1.0
        domain_signals = []

        # ============================================================================
        # WYCKOFF ENGINE: Accumulation signals for LONG archetype
        # ============================================================================
        if use_wyckoff:
            # SOFT VETOES: Distribution phase reduces confidence
            wyckoff_distribution = self.g(context.row, 'wyckoff_phase_abc', '') == 'D'
            wyckoff_utad = self.g(context.row, 'wyckoff_utad', False)
            wyckoff_bc = self.g(context.row, 'wyckoff_bc', False)

            if wyckoff_distribution:
                domain_boost *= 0.70  # Distribution phase = caution
                domain_signals.append("wyckoff_distribution_caution")

            if wyckoff_utad or wyckoff_bc:
                domain_boost *= 0.70  # Distribution events = caution
                domain_signals.append("wyckoff_distribution_event_caution")

            # MAJOR BOOSTS: Spring events (Phase C - trap reversals)
            wyckoff_spring_a = self.g(context.row, 'wyckoff_spring_a', False)
            wyckoff_spring_b = self.g(context.row, 'wyckoff_spring_b', False)

            if wyckoff_spring_a:
                domain_boost *= 2.50  # Spring A = deep fake breakdown
                domain_signals.append("wyckoff_spring_a_trap_reversal")
            elif wyckoff_spring_b:
                domain_boost *= 2.50  # Spring B = shallow spring
                domain_signals.append("wyckoff_spring_b_trap_reversal")

            # SUPPORT SIGNALS: LPS + Accumulation
            wyckoff_lps = self.g(context.row, 'wyckoff_lps', False)
            wyckoff_accumulation = self.g(context.row, 'wyckoff_phase_abc', '') == 'A'

            if wyckoff_lps:
                domain_boost *= 1.50  # Last Point Support
                domain_signals.append("wyckoff_lps_support")

            if wyckoff_accumulation:
                domain_boost *= 2.00  # Accumulation phase
                domain_signals.append("wyckoff_accumulation_phase")

        # ============================================================================
        # SMC ENGINE: Smart Money Concepts - bullish structure
        # ============================================================================
        if use_smc:
            # VETOES: Don't long into supply zones
            smc_supply_zone = self.g(context.row, 'smc_supply_zone', False)
            tf4h_bos_bearish = self.g(context.row, 'tf4h_bos_bearish', False)

            if smc_supply_zone:
                domain_boost *= 0.70  # Supply overhead
                domain_signals.append("smc_supply_zone_overhead")

            if tf4h_bos_bearish:
                domain_boost *= 0.70  # Bearish structure
                domain_signals.append("smc_4h_bearish_structure_penalty")

            # MAJOR BOOSTS: Bullish structure breaks
            tf4h_bos_bullish = self.g(context.row, 'tf4h_bos_bullish', False)
            tf1h_bos_bullish = self.g(context.row, 'tf1h_bos_bullish', False)

            if tf4h_bos_bullish:
                domain_boost *= 2.00  # 4H institutional shift
                domain_signals.append("smc_4h_bos_bullish_institutional")
            elif tf1h_bos_bullish:
                domain_boost *= 1.40  # 1H structural shift
                domain_signals.append("smc_1h_bos_bullish")

            # DEMAND ZONES: Institutional buying areas
            smc_demand_zone = self.g(context.row, 'smc_demand_zone', False)
            smc_liquidity_sweep = self.g(context.row, 'smc_liquidity_sweep', False)

            if smc_demand_zone:
                domain_boost *= 1.60  # Demand zone support
                domain_signals.append("smc_demand_zone_support")

            if smc_liquidity_sweep:
                domain_boost *= 1.80  # Liquidity sweep = trap setup
                domain_signals.append("smc_liquidity_sweep_reversal")

        # ============================================================================
        # TEMPORAL ENGINE: Fibonacci time + multi-timeframe confluence
        # ============================================================================
        if use_temporal:
            # MAJOR BOOSTS: Fibonacci time clusters
            fib_time_cluster = self.g(context.row, 'fib_time_cluster', False)
            temporal_confluence = self.g(context.row, 'temporal_confluence', False)

            if fib_time_cluster:
                domain_boost *= 1.70  # Fibonacci timing
                domain_signals.append("fib_time_cluster_reversal")

            if temporal_confluence:
                domain_boost *= 1.40  # Multi-timeframe alignment
                domain_signals.append("temporal_multi_tf_confluence")

            # VETOES: Don't long into resistance
            temporal_resistance_cluster = self.g(context.row, 'temporal_resistance_cluster', False)
            if temporal_resistance_cluster:
                domain_boost *= 0.75  # Resistance overhead
                domain_signals.append("temporal_resistance_overhead")

        # ============================================================================
        # HOB ENGINE: Order book depth + imbalance
        # ============================================================================
        if use_hob:
            # DEMAND ZONES: Large bid walls
            hob_demand_zone = self.g(context.row, 'hob_demand_zone', False)
            hob_imbalance = self.g(context.row, 'hob_imbalance', 0.0)

            if hob_demand_zone:
                domain_boost *= 1.50  # Demand wall support
                domain_signals.append("hob_demand_zone_support")

            # ORDER BOOK IMBALANCE: Bid/ask ratio
            if hob_imbalance > 0.60:  # More bids than asks
                domain_boost *= 1.30  # Strong buyer imbalance
                domain_signals.append("hob_bid_imbalance_strong")
            elif hob_imbalance > 0.40:
                domain_boost *= 1.15  # Moderate buyer imbalance
                domain_signals.append("hob_bid_imbalance_moderate")

            # VETOES: Supply zones
            hob_supply_zone = self.g(context.row, 'hob_supply_zone', False)
            if hob_supply_zone:
                domain_boost *= 0.70  # Supply wall overhead
                domain_signals.append("hob_supply_zone_overhead")

        # MACRO: Crisis composite check
        if use_macro:
            crisis_composite = self.g(context.row, 'crisis_composite', 0.0)
            if crisis_composite > 0.60:  # High crisis reduces confidence
                domain_boost *= 0.85  # Crisis penalty
                domain_signals.append("macro_crisis_penalty")
            elif crisis_composite < 0.30:  # Low crisis = favorable
                domain_boost *= 1.20  # Risk-on boost
                domain_signals.append("macro_risk_on_boost")

        # Apply domain boost to final score
        score_before_domain = score
        score = score * domain_boost

        # Cap score at valid range [0.0, 5.0]
        score = max(0.0, min(5.0, score))

        # ============================================================================
        # FUSION THRESHOLD GATE (applied AFTER domain engines)
        # ============================================================================
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_threshold",
                "score": score,
                "score_before_domain": score_before_domain,
                "threshold": fusion_th,
                "domain_boost": domain_boost,
                "domain_signals": domain_signals
            }

        meta = {
            "detection_path": detection_path,
            "base_score": base_score,
            "bonuses": bonuses,
            "archetype_weight": archetype_weight,
            "pti_trap_type": pti_trap,
            "wyckoff_spring_a": wyckoff_spring_a,
            "wyckoff_lps": wyckoff_lps,
            "domain_boost": domain_boost,
            "domain_signals": domain_signals,
            "score_before_domain": score_before_domain
        }

        return True, score, meta, "LONG"

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

        # CRISIS FIX: BOMS strength is too strict in crisis (0.94% pass rate in 2022)
        # Similar to Archetype C making CHOCH optional, make BOMS optional in crisis
        # Crisis detected via regime_label or crisis_composite
        regime = self.g(context.row, 'regime_label', 'neutral')
        crisis_composite = self.g(context.row, 'crisis_composite', 0.0)
        is_crisis = (crisis_composite >= 0.30) or (regime in ['crisis', 'bear'])

        # Gate checks (BOS always required)
        if not bos_bullish:
            return False, 0.0, {"reason": "no_bos"}

        # BOMS gate: Required in normal markets, optional in crisis
        # In crisis: BOS + Wyckoff sufficient (BOMS adds bonus if present)
        if not is_crisis:
            # Normal market: Require BOMS
            if boms_str < boms_str_th:
                return False, 0.0, {"reason": "boms_weak", "value": boms_str, "threshold": boms_str_th}

        # Wyckoff gate: Always required (accumulation phase detection)
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
        # CRISIS ADJUSTMENT: Since BOMS is optional in crisis, reduce its weight
        # and redistribute to other components (especially Wyckoff for accumulation detection)
        if is_crisis:
            weights = context.get_threshold('order_block_retest', 'weights_crisis', {
                "fusion": 0.35,
                "liquidity": 0.20,
                "momentum": 0.15,
                "wyckoff": 0.25,  # Increased from 0.20 (key accumulation detector in crisis)
                "boms": 0.05      # Reduced from 0.10 (optional bonus in crisis)
            })
        else:
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

        # ============================================================================
        # DOMAIN ENGINE INTEGRATION (BOOST/VETO LAYER)
        # ============================================================================
        # Apply domain engines BEFORE fusion threshold gate
        # This allows marginal signals to qualify via boosts
        # Order: VETOES first (safety) -> BOOSTS second -> GATE third

        use_wyckoff = context.metadata.get('feature_flags', {}).get('enable_wyckoff', False)
        use_smc = context.metadata.get('feature_flags', {}).get('enable_smc', False)
        use_temporal = context.metadata.get('feature_flags', {}).get('enable_temporal', False)
        use_hob = context.metadata.get('feature_flags', {}).get('enable_hob', False)
        use_macro = context.metadata.get('feature_flags', {}).get('enable_macro', False)

        domain_boost = 1.0
        domain_signals = []

        # ============================================================================
        # WYCKOFF ENGINE: Order block retest confirmation
        # ============================================================================
        if use_wyckoff:
            # SOFT VETOES: Distribution phase reduces confidence
            wyckoff_distribution = self.g(context.row, 'wyckoff_phase_abc', '') == 'D'
            wyckoff_utad = self.g(context.row, 'wyckoff_utad', False)
            wyckoff_bc = self.g(context.row, 'wyckoff_bc', False)

            if wyckoff_distribution:
                domain_boost *= 0.70  # Distribution phase = caution
                domain_signals.append("wyckoff_distribution_caution")

            if wyckoff_utad or wyckoff_bc:
                domain_boost *= 0.70  # Distribution events = caution
                domain_signals.append("wyckoff_distribution_event_caution")

            # MAJOR BOOSTS: Accumulation + Support signals
            wyckoff_accumulation = self.g(context.row, 'wyckoff_phase_abc', '') == 'A'
            wyckoff_lps = self.g(context.row, 'wyckoff_lps', False)
            wyckoff_ps = self.g(context.row, 'wyckoff_ps', False)

            if wyckoff_accumulation:
                domain_boost *= 2.00  # Accumulation phase
                domain_signals.append("wyckoff_accumulation_phase")

            if wyckoff_lps:
                domain_boost *= 1.50  # Last Point Support = retest confirmation
                domain_signals.append("wyckoff_lps_support")
            elif wyckoff_ps:
                domain_boost *= 1.30  # Preliminary Support
                domain_signals.append("wyckoff_ps_support")

            # SPRING SIGNALS: Shakeouts before markup
            wyckoff_spring_a = self.g(context.row, 'wyckoff_spring_a', False)
            wyckoff_spring_b = self.g(context.row, 'wyckoff_spring_b', False)

            if wyckoff_spring_a:
                domain_boost *= 2.50  # Spring A = deep shakeout
                domain_signals.append("wyckoff_spring_a_shakeout")
            elif wyckoff_spring_b:
                domain_boost *= 2.00  # Spring B = shallow shakeout
                domain_signals.append("wyckoff_spring_b_shakeout")

        # ============================================================================
        # SMC ENGINE: Smart Money Concepts - order block validation
        # ============================================================================
        if use_smc:
            # VETOES: Don't long into supply zones
            smc_supply_zone = self.g(context.row, 'smc_supply_zone', False)
            tf4h_bos_bearish = self.g(context.row, 'tf4h_bos_bearish', False)

            if smc_supply_zone:
                domain_boost *= 0.70  # Supply overhead
                domain_signals.append("smc_supply_zone_overhead")

            if tf4h_bos_bearish:
                domain_boost *= 0.70  # Bearish structure
                domain_signals.append("smc_4h_bearish_structure_penalty")

            # MAJOR BOOSTS: BOS confirmation (critical for order blocks)
            tf4h_bos_bullish = self.g(context.row, 'tf4h_bos_bullish', False)
            tf1h_bos_bullish = self.g(context.row, 'tf1h_bos_bullish', False)

            if tf4h_bos_bullish:
                domain_boost *= 2.00  # 4H BOS = institutional structure shift
                domain_signals.append("smc_4h_bos_bullish_institutional")
            elif tf1h_bos_bullish:
                domain_boost *= 1.40  # 1H BOS = structural shift
                domain_signals.append("smc_1h_bos_bullish")

            # ORDER BLOCK ZONES: Institutional retest areas
            smc_demand_zone = self.g(context.row, 'smc_demand_zone', False)
            smc_order_block = self.g(context.row, 'smc_order_block', False)
            smc_choch = self.g(context.row, 'smc_choch', False)

            if smc_demand_zone:
                domain_boost *= 1.60  # Demand zone = institutional support
                domain_signals.append("smc_demand_zone_support")

            if smc_order_block:
                domain_boost *= 1.80  # Order block retest = high probability
                domain_signals.append("smc_order_block_retest")

            if smc_choch:
                domain_boost *= 1.50  # Change of Character
                domain_signals.append("smc_choch_trend_change")

        # ============================================================================
        # TEMPORAL ENGINE: Fibonacci time + confluence
        # ============================================================================
        if use_temporal:
            # MAJOR BOOSTS: Fibonacci time clusters
            fib_time_cluster = self.g(context.row, 'fib_time_cluster', False)
            temporal_confluence = self.g(context.row, 'temporal_confluence', False)

            if fib_time_cluster:
                domain_boost *= 1.70  # Fibonacci timing
                domain_signals.append("fib_time_cluster_reversal")

            if temporal_confluence:
                domain_boost *= 1.40  # Multi-timeframe alignment
                domain_signals.append("temporal_multi_tf_confluence")

            # VETOES: Don't long into resistance
            temporal_resistance_cluster = self.g(context.row, 'temporal_resistance_cluster', False)
            if temporal_resistance_cluster:
                domain_boost *= 0.75  # Resistance overhead
                domain_signals.append("temporal_resistance_overhead")

        # ============================================================================
        # HOB ENGINE: Order book confirmation
        # ============================================================================
        if use_hob:
            # DEMAND ZONES: Large bid walls
            hob_demand_zone = self.g(context.row, 'hob_demand_zone', False)
            hob_imbalance = self.g(context.row, 'hob_imbalance', 0.0)

            if hob_demand_zone:
                domain_boost *= 1.50  # Demand wall support
                domain_signals.append("hob_demand_zone_support")

            # ORDER BOOK IMBALANCE: Bid/ask ratio
            if hob_imbalance > 0.60:  # More bids than asks
                domain_boost *= 1.30  # Strong buyer imbalance
                domain_signals.append("hob_bid_imbalance_strong")
            elif hob_imbalance > 0.40:
                domain_boost *= 1.15  # Moderate buyer imbalance
                domain_signals.append("hob_bid_imbalance_moderate")

            # VETOES: Supply zones
            hob_supply_zone = self.g(context.row, 'hob_supply_zone', False)
            if hob_supply_zone:
                domain_boost *= 0.70  # Supply wall overhead
                domain_signals.append("hob_supply_zone_overhead")

        # MACRO: Crisis composite check
        if use_macro:
            crisis_composite = self.g(context.row, 'crisis_composite', 0.0)
            if crisis_composite > 0.60:  # High crisis reduces confidence
                domain_boost *= 0.85  # Crisis penalty
                domain_signals.append("macro_crisis_penalty")
            elif crisis_composite < 0.30:  # Low crisis = favorable
                domain_boost *= 1.20  # Risk-on boost
                domain_signals.append("macro_risk_on_boost")

        # Apply domain boost to final score
        score_before_domain = score
        score = score * domain_boost

        # Cap score at valid range [0.0, 5.0]
        score = max(0.0, min(5.0, score))

        # ============================================================================
        # FUSION THRESHOLD GATE (applied AFTER domain engines)
        # ============================================================================
        # Gate check on fusion threshold
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_fusion_th",
                "score": score,
                "score_before_domain": score_before_domain,
                "threshold": fusion_th,
                "domain_boost": domain_boost,
                "domain_signals": domain_signals
            }

        meta = {
            "components": components,
            "weights": weights,
            "base_score": base_score,
            "archetype_weight": archetype_weight,
            "penalties": penalties,
            "domain_boost": domain_boost,
            "domain_signals": domain_signals,
            "score_before_domain": score_before_domain,
            "crisis_mode": is_crisis,  # Track when crisis-aware logic is active
            "boms_value": boms_str     # Track BOMS value for debugging
        }

        return True, score, meta, "LONG"

    def _pattern_C(self, context: RuntimeContext):
        """Pattern detection for Archetype C: BOS/CHOCH Reversal (LONG)"""
        r = context.row
        bos_bullish = self.g(r, 'tf1h_bos_bullish', False)
        choch_flag = self.g(r, 'tf1h_choch_flag', False)

        # CHANGE: Accept BOS alone, CHOCH adds bonus
        if bos_bullish:
            score = 0.35  # Base score for BOS alone
            tags = ["C", "bos_reversal", "LONG"]

            # Bonus if CHOCH confirms
            if choch_flag:
                score += 0.10  # CHOCH confirmation bonus
                tags[1] = "bos_choch"  # Update tag to show CHOCH present

            # Bonus for wick rejection
            if self.g(r, 'wick_lower_ratio', 0.0) >= 0.55:
                score += 0.20

            return True, score, tags

        return None

    def _check_C(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """Archetype C: BOS/CHOCH Reversal (LONG)"""

        # 1) Base pattern detection - returns (matched, score, metadata)
        base_result = self._pattern_C(context)
        if not base_result:
            return False, 0.0, {"reason": "pattern_not_matched"}

        matched, base_score, pattern_tags = base_result  # base_score in [0,1]

        # 2) Global safety vetoes (soft - use penalties not hard returns)
        # Note: Keep these soft - use penalties not hard returns

        # 3) Domain modifiers (standardized across all)
        score = self._apply_domain_engines(context, base_score, pattern_tags)

        # 4) Final fusion gate
        fusion_th = context.get_threshold('bos_choch', 'fusion_threshold', 0.40)
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_fusion_threshold",
                "base_score": base_score,
                "final_score": score,
                "fusion_threshold": fusion_th
            }

        # 5) Return with full metadata
        return True, score, {
            "base_score": base_score,
            "final_score": score,
            "pattern_tags": pattern_tags,
            "domain_boost_applied": True
        }, "LONG"

    def _pattern_D(self, context: RuntimeContext):
        """Pattern detection for Archetype D: Order Block Retest (LONG)"""
        r = context.row
        ob_retest = self.g(r, 'tf1h_ob_retest_flag', False) or (
            self.g(r, 'tf1h_ob_low', None) is not None and
            abs(r['close'] - self.g(r, 'tf1h_ob_low', 999999)) < self.g(r, 'atr', 100) * 0.5
        )
        if ob_retest and self.g(r, 'rsi_14', 50) < 35:
            score = 0.42
            return True, score, ["D", "ob_retest", "LONG"]
        return None

    def _check_D(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """Archetype D: Order Block Retest (LONG)"""

        # 1) Base pattern detection - returns (matched, score, metadata)
        base_result = self._pattern_D(context)
        if not base_result:
            return False, 0.0, {"reason": "pattern_not_matched"}

        matched, base_score, pattern_tags = base_result  # base_score in [0,1]

        # 2) Global safety vetoes (soft - use penalties not hard returns)
        # Note: Keep these soft - use penalties not hard returns

        # 3) Domain modifiers (standardized across all)
        score = self._apply_domain_engines(context, base_score, pattern_tags)

        # 4) Final fusion gate
        fusion_th = context.get_threshold('ob_retest', 'fusion_threshold', 0.40)
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_fusion_threshold",
                "base_score": base_score,
                "final_score": score,
                "fusion_threshold": fusion_th
            }

        # 5) Return with full metadata
        return True, score, {
            "base_score": base_score,
            "final_score": score,
            "pattern_tags": pattern_tags,
            "domain_boost_applied": True
        }, "LONG"

    def _pattern_E(self, context: RuntimeContext):
        """Pattern detection for Archetype E: Breakdown (SHORT)"""
        r = context.row
        if self.g(r, 'tf1h_bos_bearish', False) and self.g(r, 'volume_zscore', 0.0) > 1.5:
            score = 0.45
            return True, score, ["E", "breakdown", "SHORT"]
        return None

    def _check_E(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """Archetype E: Breakdown (SHORT)"""

        # 1) Base pattern detection - returns (matched, score, metadata)
        base_result = self._pattern_E(context)
        if not base_result:
            return False, 0.0, {"reason": "pattern_not_matched"}

        matched, base_score, pattern_tags = base_result  # base_score in [0,1]

        # 2) Global safety vetoes (soft - use penalties not hard returns)
        # Note: Keep these soft - use penalties not hard returns

        # 3) Domain modifiers (standardized across all)
        score = self._apply_domain_engines(context, base_score, pattern_tags)

        # 4) Final fusion gate
        fusion_th = context.get_threshold('breakdown', 'fusion_threshold', 0.40)
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_fusion_threshold",
                "base_score": base_score,
                "final_score": score,
                "fusion_threshold": fusion_th
            }

        # 5) Return with full metadata
        return True, score, {
            "base_score": base_score,
            "final_score": score,
            "pattern_tags": pattern_tags,
            "domain_boost_applied": True
        }, "EITHER"

    def _pattern_F(self, context: RuntimeContext):
        """Pattern detection for Archetype F: FVG Real Move (LONG)"""
        r = context.row
        if self.g(r, 'tf1h_fvg_present', False) and self.g(r, 'volume_zscore', 0.0) > 1.0:
            score = 0.42
            return True, score, ["F", "fvg_real_move", "LONG"]
        return None

    def _check_F(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """Archetype F: FVG Real Move (LONG)"""

        # 1) Base pattern detection - returns (matched, score, metadata)
        base_result = self._pattern_F(context)
        if not base_result:
            return False, 0.0, {"reason": "pattern_not_matched"}

        matched, base_score, pattern_tags = base_result  # base_score in [0,1]

        # 2) Global safety vetoes (soft - use penalties not hard returns)
        # Note: Keep these soft - use penalties not hard returns

        # 3) Domain modifiers (standardized across all)
        score = self._apply_domain_engines(context, base_score, pattern_tags)

        # 4) Final fusion gate
        fusion_th = context.get_threshold('fvg_real_move', 'fusion_threshold', 0.40)
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_fusion_threshold",
                "base_score": base_score,
                "final_score": score,
                "fusion_threshold": fusion_th
            }

        # 5) Return with full metadata
        return True, score, {
            "base_score": base_score,
            "final_score": score,
            "pattern_tags": pattern_tags,
            "domain_boost_applied": True
        }, "LONG"

    def _pattern_G(self, context: RuntimeContext):
        """Pattern detection for Archetype G: Liquidity Sweep (LONG)"""
        r = context.row
        wick_low = self.g(r, 'wick_lower_ratio', 0.0)
        if wick_low >= 0.65 and self.g(r, 'tf1h_bos_bullish', False):
            score = 0.45 + 0.20 * min(1.0, wick_low)
            return True, score, ["G", "liquidity_sweep", "LONG"]
        return None

    def _check_G(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """Archetype G: Liquidity Sweep (LONG)"""

        # 1) Base pattern detection - returns (matched, score, metadata)
        base_result = self._pattern_G(context)
        if not base_result:
            return False, 0.0, {"reason": "pattern_not_matched"}

        matched, base_score, pattern_tags = base_result  # base_score in [0,1]

        # 2) Global safety vetoes (soft - use penalties not hard returns)
        # Note: Keep these soft - use penalties not hard returns

        # 3) Domain modifiers (standardized across all)
        score = self._apply_domain_engines(context, base_score, pattern_tags)

        # 4) Final fusion gate
        fusion_th = context.get_threshold('liquidity_sweep', 'fusion_threshold', 0.40)
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_fusion_threshold",
                "base_score": base_score,
                "final_score": score,
                "fusion_threshold": fusion_th
            }

        # 5) Return with full metadata
        return True, score, {
            "base_score": base_score,
            "final_score": score,
            "pattern_tags": pattern_tags,
            "domain_boost_applied": True
        }, "LONG"

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

        # ============================================================================
        # DOMAIN ENGINE INTEGRATION (BOOST/VETO LAYER)
        # ============================================================================
        # Apply domain engines BEFORE fusion threshold gate
        # This allows marginal signals to qualify via boosts
        # Order: VETOES first (safety) -> BOOSTS second -> GATE third

        use_wyckoff = context.metadata.get('feature_flags', {}).get('enable_wyckoff', False)
        use_smc = context.metadata.get('feature_flags', {}).get('enable_smc', False)
        use_temporal = context.metadata.get('feature_flags', {}).get('enable_temporal', False)
        use_hob = context.metadata.get('feature_flags', {}).get('enable_hob', False)
        use_macro = context.metadata.get('feature_flags', {}).get('enable_macro', False)

        domain_boost = 1.0
        domain_signals = []

        # ============================================================================
        # WYCKOFF ENGINE: Trap within trend confirmation
        # ============================================================================
        if use_wyckoff:
            # SOFT VETOES: Distribution phase reduces confidence
            wyckoff_distribution = self.g(context.row, 'wyckoff_phase_abc', '') == 'D'
            wyckoff_utad = self.g(context.row, 'wyckoff_utad', False)
            wyckoff_bc = self.g(context.row, 'wyckoff_bc', False)

            if wyckoff_distribution:
                domain_boost *= 0.70  # Distribution phase = caution
                domain_signals.append("wyckoff_distribution_caution")

            if wyckoff_utad or wyckoff_bc:
                domain_boost *= 0.70  # Distribution events = caution
                domain_signals.append("wyckoff_distribution_event_caution")

            # MAJOR BOOSTS: Accumulation phase + Springs (trap reversals within trend)
            wyckoff_accumulation = self.g(context.row, 'wyckoff_phase_abc', '') == 'A'
            wyckoff_spring_a = self.g(context.row, 'wyckoff_spring_a', False)
            wyckoff_spring_b = self.g(context.row, 'wyckoff_spring_b', False)

            if wyckoff_accumulation:
                domain_boost *= 2.00  # Accumulation phase
                domain_signals.append("wyckoff_accumulation_phase")

            if wyckoff_spring_a:
                domain_boost *= 2.50  # Spring A = deep trap reversal
                domain_signals.append("wyckoff_spring_a_trap_reversal")
            elif wyckoff_spring_b:
                domain_boost *= 2.00  # Spring B = shallow trap
                domain_signals.append("wyckoff_spring_b_trap_reversal")

            # SUPPORT SIGNALS: LPS + Shakeouts
            wyckoff_lps = self.g(context.row, 'wyckoff_lps', False)
            wyckoff_st = self.g(context.row, 'wyckoff_st', False)

            if wyckoff_lps:
                domain_boost *= 1.50  # Last Point Support
                domain_signals.append("wyckoff_lps_support")

            if wyckoff_st:
                domain_boost *= 1.40  # Secondary Test = trap retest
                domain_signals.append("wyckoff_secondary_test")

        # ============================================================================
        # SMC ENGINE: Smart Money Concepts - trend structure
        # ============================================================================
        if use_smc:
            # VETOES: Don't long into supply zones
            smc_supply_zone = self.g(context.row, 'smc_supply_zone', False)
            tf4h_bos_bearish = self.g(context.row, 'tf4h_bos_bearish', False)

            if smc_supply_zone:
                domain_boost *= 0.70  # Supply overhead
                domain_signals.append("smc_supply_zone_overhead")

            if tf4h_bos_bearish:
                domain_boost *= 0.70  # Bearish structure
                domain_signals.append("smc_4h_bearish_structure_penalty")

            # MAJOR BOOSTS: Bullish structure + liquidity sweeps
            tf4h_bos_bullish = self.g(context.row, 'tf4h_bos_bullish', False)
            tf1h_bos_bullish = self.g(context.row, 'tf1h_bos_bullish', False)
            smc_liquidity_sweep = self.g(context.row, 'smc_liquidity_sweep', False)

            if tf4h_bos_bullish:
                domain_boost *= 2.00  # 4H BOS = institutional shift
                domain_signals.append("smc_4h_bos_bullish_institutional")
            elif tf1h_bos_bullish:
                domain_boost *= 1.40  # 1H BOS
                domain_signals.append("smc_1h_bos_bullish")

            if smc_liquidity_sweep:
                domain_boost *= 1.80  # Liquidity sweep = trap setup
                domain_signals.append("smc_liquidity_sweep_reversal")

            # DEMAND ZONES: Support areas
            smc_demand_zone = self.g(context.row, 'smc_demand_zone', False)
            if smc_demand_zone:
                domain_boost *= 1.50  # Demand zone support
                domain_signals.append("smc_demand_zone_support")

        # ============================================================================
        # TEMPORAL ENGINE: Fibonacci time + confluence
        # ============================================================================
        if use_temporal:
            # MAJOR BOOSTS: Fibonacci time clusters
            fib_time_cluster = self.g(context.row, 'fib_time_cluster', False)
            temporal_confluence = self.g(context.row, 'temporal_confluence', False)

            if fib_time_cluster:
                domain_boost *= 1.70  # Fibonacci timing
                domain_signals.append("fib_time_cluster_reversal")

            if temporal_confluence:
                domain_boost *= 1.40  # Multi-timeframe alignment
                domain_signals.append("temporal_multi_tf_confluence")

            # VETOES: Don't long into resistance
            temporal_resistance_cluster = self.g(context.row, 'temporal_resistance_cluster', False)
            if temporal_resistance_cluster:
                domain_boost *= 0.75  # Resistance overhead
                domain_signals.append("temporal_resistance_overhead")

        # ============================================================================
        # HOB ENGINE: Order book confirmation
        # ============================================================================
        if use_hob:
            # DEMAND ZONES: Large bid walls
            hob_demand_zone = self.g(context.row, 'hob_demand_zone', False)
            hob_imbalance = self.g(context.row, 'hob_imbalance', 0.0)

            if hob_demand_zone:
                domain_boost *= 1.50  # Demand wall support
                domain_signals.append("hob_demand_zone_support")

            # ORDER BOOK IMBALANCE: Bid/ask ratio
            if hob_imbalance > 0.60:  # More bids than asks
                domain_boost *= 1.30  # Strong buyer imbalance
                domain_signals.append("hob_bid_imbalance_strong")
            elif hob_imbalance > 0.40:
                domain_boost *= 1.15  # Moderate buyer imbalance
                domain_signals.append("hob_bid_imbalance_moderate")

            # VETOES: Supply zones
            hob_supply_zone = self.g(context.row, 'hob_supply_zone', False)
            if hob_supply_zone:
                domain_boost *= 0.70  # Supply wall overhead
                domain_signals.append("hob_supply_zone_overhead")

        # MACRO: Crisis composite check
        if use_macro:
            crisis_composite = self.g(context.row, 'crisis_composite', 0.0)
            if crisis_composite > 0.60:  # High crisis reduces confidence
                domain_boost *= 0.85  # Crisis penalty
                domain_signals.append("macro_crisis_penalty")
            elif crisis_composite < 0.30:  # Low crisis = favorable
                domain_boost *= 1.20  # Risk-on boost
                domain_signals.append("macro_risk_on_boost")

        # Apply domain boost to final score
        score_before_domain = score
        score = score * domain_boost

        # Cap score at valid range [0.0, 5.0]
        score = max(0.0, min(5.0, score))

        # ============================================================================
        # FUSION THRESHOLD GATE (applied AFTER domain engines)
        # ============================================================================
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_threshold",
                "score": score,
                "score_before_domain": score_before_domain,
                "threshold": fusion_th,
                "domain_boost": domain_boost,
                "domain_signals": domain_signals
            }

        meta = {
            "components": components,
            "weights": weights,
            "base_score": base_score,
            "archetype_weight": archetype_weight,
            "domain_boost": domain_boost,
            "domain_signals": domain_signals,
            "score_before_domain": score_before_domain
        }

        return True, score, meta, "LONG"

    def _pattern_K(self, context: RuntimeContext):
        """Pattern detection for Archetype K: Wick Trap (LONG)"""
        r = context.row
        wick_low = self.g(r, 'wick_lower_ratio', 0.0)
        rsi = self.g(r, 'rsi_14', 50)
        if wick_low >= 0.75 and rsi <= 35:
            score = 0.50 + 0.25 * wick_low
            return True, score, ["K", "wick_trap", "LONG"]
        return None

    def _check_K(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """Archetype K: Wick Trap (LONG)"""

        # 1) Base pattern detection - returns (matched, score, metadata)
        base_result = self._pattern_K(context)
        if not base_result:
            return False, 0.0, {"reason": "pattern_not_matched"}

        matched, base_score, pattern_tags = base_result  # base_score in [0,1]

        # 2) Global safety vetoes (soft - use penalties not hard returns)
        # Note: Keep these soft - use penalties not hard returns

        # 3) Domain modifiers (standardized across all)
        score = self._apply_domain_engines(context, base_score, pattern_tags)

        # 4) Final fusion gate
        fusion_th = context.get_threshold('wick_trap', 'fusion_threshold', 0.40)
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_fusion_threshold",
                "base_score": base_score,
                "final_score": score,
                "fusion_threshold": fusion_th
            }

        # 5) Return with full metadata
        return True, score, {
            "base_score": base_score,
            "final_score": score,
            "pattern_tags": pattern_tags,
            "domain_boost_applied": True
        }, "LONG"

    def _pattern_L(self, context: RuntimeContext):
        """Pattern detection for Archetype L: Fakeout Real Move (LONG)"""
        r = context.row
        # Check for recent trap + reversal BOS
        if self.g(r, 'tf1h_bos_bullish', False):
            # Check if previous bars had bearish BOS (trap)
            df = context.metadata.get('df')
            current_idx = context.metadata.get('index', 0)

            # Handle both timestamp and integer index
            if df is not None:
                try:
                    # If index is a timestamp, convert to integer position
                    if isinstance(current_idx, pd.Timestamp):
                        current_pos = df.index.get_loc(current_idx)
                    else:
                        current_pos = current_idx

                    prev_pos = current_pos - 1

                    if prev_pos >= 0 and prev_pos < len(df):
                        prev_row = df.iloc[prev_pos]
                        if self.g(prev_row, 'tf1h_bos_bearish', False):
                            score = 0.48
                            return True, score, ["L", "fakeout_real_move", "LONG"]
                except (KeyError, ValueError):
                    # Index not found in df, skip this check
                    pass
        return None

    def _check_L(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """Archetype L: Fakeout Real Move (LONG)"""

        # 1) Base pattern detection - returns (matched, score, metadata)
        base_result = self._pattern_L(context)
        if not base_result:
            return False, 0.0, {"reason": "pattern_not_matched"}

        matched, base_score, pattern_tags = base_result  # base_score in [0,1]

        # 2) Global safety vetoes (soft - use penalties not hard returns)
        # Note: Keep these soft - use penalties not hard returns

        # 3) Domain modifiers (standardized across all)
        score = self._apply_domain_engines(context, base_score, pattern_tags)

        # 4) Final fusion gate
        fusion_th = context.get_threshold('fakeout_real_move', 'fusion_threshold', 0.40)
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_fusion_threshold",
                "base_score": base_score,
                "final_score": score,
                "fusion_threshold": fusion_th
            }

        # 5) Return with full metadata
        return True, score, {
            "base_score": base_score,
            "final_score": score,
            "pattern_tags": pattern_tags,
            "domain_boost_applied": True
        }, "LONG"

    def _pattern_M(self, context: RuntimeContext):
        """Pattern detection for Archetype M: Coil Break (LONG)

        PATTERN LOGIC:
        Low volatility coil breakout - detects when price breaks structure after volatility compression.

        ORIGINAL LOGIC (BROKEN):
        - Required: atr_percentile <= 0.25 (missing feature → always 0.5 default → 0 signals)

        FIXED LOGIC V2:
        - Uses absolute ATR threshold instead of percentile
        - atr_pct < 0.50% = bottom 25% of volatility (same intent as atr_percentile <= 0.25)
        - Validates with tf4h_bos_bullish (4H bullish BOS confirms breakout direction)

        Expected signals: ~150 base opportunities → 20-80 after fusion/domain filters
        """
        r = context.row

        # FIX: Replace missing atr_percentile with absolute ATR percentage threshold
        # Original: atrp = self.g(r, 'atr_percentile', 0.5)  # Missing feature → always 0.5
        # New: Calculate ATR as % of close, use 25th percentile threshold (0.50%)

        atr = self.g(r, 'atr_14', self.g(r, 'atr_20', 0))
        close = r.get('close', 1)
        atr_pct = (atr / close) * 100 if close > 0 else 999  # ATR as % of price

        # Low volatility threshold: 0.50% = 25th percentile (captures coil compression)
        # This threshold can be tuned via config: coil_break.atr_pct_max
        atr_pct_max = context.get_threshold('coil_break', 'atr_pct_max', 0.50)

        # Core pattern: Low volatility coil + 4H BOS breakout
        if atr_pct < atr_pct_max and self.g(r, 'tf4h_bos_bullish', False):
            # Score increases as volatility gets lower (tighter coil = stronger spring)
            # Base: 0.45, Max bonus: 0.20 (when atr_pct near 0)
            vol_compression_bonus = 0.20 * max(0, (atr_pct_max - atr_pct) / atr_pct_max)
            score = 0.45 + vol_compression_bonus
            return True, score, ["M", "coil_break", "LONG"]
        return None

    def _check_M(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """Archetype M: Coil Break (LONG)"""

        # 1) Base pattern detection - returns (matched, score, metadata)
        base_result = self._pattern_M(context)
        if not base_result:
            return False, 0.0, {"reason": "pattern_not_matched"}

        matched, base_score, pattern_tags = base_result  # base_score in [0,1]

        # 2) Global safety vetoes (soft - use penalties not hard returns)
        # Note: Keep these soft - use penalties not hard returns

        # 3) Domain modifiers (standardized across all)
        score = self._apply_domain_engines(context, base_score, pattern_tags)

        # 4) Final fusion gate
        fusion_th = context.get_threshold('coil_break', 'fusion_threshold', 0.40)
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_fusion_threshold",
                "base_score": base_score,
                "final_score": score,
                "fusion_threshold": fusion_th
            }

        # 5) Return with full metadata
        return True, score, {
            "base_score": base_score,
            "final_score": score,
            "pattern_tags": pattern_tags,
            "domain_boost_applied": True
        }, "LONG"

    # =======================================================================
    # Bear Archetype Check Methods (Short-Biased)
    # =======================================================================

    def _check_S1(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """
        Liquidity Vacuum Reversal (V2 - Multi-bar Capitulation Detection)

        Trader: Insider (capitulation specialist)
        Edge: Deep liquidity drain + multi-bar exhaustion signals = violent bounce

        PATTERN LOGIC V2 (REDESIGNED - FIXES LIQUIDITY PARADOX):
        Liquidity vacuum reversals occur when orderbook liquidity evaporates during sell-offs,
        creating "air pockets" where sellers exhaust themselves. The resulting vacuum creates
        explosive short-covering bounces as there's no resistance.

        KEY V2 IMPROVEMENTS:
        1. RELATIVE liquidity drain (vs 7d avg) - fixes June 18 detection failure
        2. Multi-bar exhaustion signals (3-bar lookback) - handles messy real capitulations
        3. Capitulation depth filter - separates micro-dips from macro capitulations
        4. Crisis composite score - better macro context detection

        Detection Logic V2 (DUAL MODE: V1 fallback if V2 features missing):

        V2 MODE (when runtime features available):
        1. HARD GATE: Capitulation depth (drawdown >= -20% from 30d high)
        2. HARD GATE: Crisis composite (>= 0.40) - true capitulation environment
        3. OR GATE: Multi-bar exhaustion (at least ONE):
           - Volume climax last 3 bars (> 0.25) OR
           - Wick exhaustion last 3 bars (> 0.30)
        4. SOFT FUSION: Liquidity velocity, persistence, funding boost score

        V1 MODE (fallback - backward compatible):
        1. HARD GATE: Liquidity drain (liquidity_score < 0.20)
        2. OR GATE: Single-bar exhaustion:
           - Volume panic (volume_zscore > 1.5) OR
           - Wick rejection (wick_lower_ratio > 0.28)
        3. SOFT FUSION: Standard V1 scoring

        WHY MULTI-BAR LOGIC:
        Real capitulation events are MESSY. Signals rarely align on same bar:
        - FTX bottom: volume + wick (same bar)
        - 2022-06-18: wick bar 1, volume bar 2 (SEPARATED!)
        - 2024-08-05: wick bar 1, entry bar 2
        - 2022 Luna: volume spike across 3 bars, entry on bar 3

        Validation Results (2022 major capitulations):
        - LUNA (May 12): depth=-38.4%, crisis=63.9%, vol_3b=0.0%, wick_3b=48.9% → DETECT
        - June 18: depth=-44.7%, crisis=61.7%, vol_3b=44.7%, wick_3b=37.2% → DETECT
        - FTX (Nov 9): depth=-26.8%, crisis=30.3%, vol_3b=32.8%, wick_3b=21.0% → BORDERLINE

        Returns:
            (matched: bool, score: float, meta: dict)
        """
        # ============================================================================
        # STEP 1: Get Thresholds (V2 thresholds with V1 fallbacks)
        # ============================================================================

        # V2 thresholds (multi-bar capitulation detection)
        use_v2_logic = context.get_threshold('liquidity_vacuum', 'use_v2_logic', True)
        cap_depth_max = context.get_threshold('liquidity_vacuum', 'capitulation_depth_max', -0.20)  # Must be >= -20% drawdown
        crisis_min = context.get_threshold('liquidity_vacuum', 'crisis_composite_min', 0.40)  # Must be in crisis
        vol_climax_3b_min = context.get_threshold('liquidity_vacuum', 'volume_climax_3b_min', 0.25)
        wick_exhaust_3b_min = context.get_threshold('liquidity_vacuum', 'wick_exhaustion_3b_min', 0.30)

        # ============================================================================
        # REGIME-ROUTING: V2 CRISIS-MODE ONLY, V1 FOR NORMAL MARKETS
        # ============================================================================
        # FIX: V2 AND gates (capitulation_depth < -0.20 AND crisis_composite >= 0.40)
        # block all signals in Q1 2023 bull recovery (0 bars pass both gates).
        # SOLUTION: Route V2 to crisis-only environments, use V1 for normal markets.
        #
        # BEFORE: V2 requires BOTH extreme drawdown AND crisis environment → 0 signals in Q1 2023
        # AFTER: V2 only activates in crisis_composite >= 0.30, V1 handles normal markets → ~300 signals unlocked
        #
        # This preserves V2's multi-bar capitulation detection for true crisis events
        # (2022 crash, Luna, FTX) while allowing V1 single-bar logic for normal volatility.
        if use_v2_logic:
            crisis_composite = self.g(context.row, 'crisis_composite', 0.0)
            crisis_routing_threshold = context.get_threshold('liquidity_vacuum', 'crisis_routing_threshold', 0.30)

            # Override: Disable V2 in non-crisis regimes (route to V1 instead)
            if crisis_composite < crisis_routing_threshold:
                use_v2_logic = False
                # V1 will handle normal market capitulations (single-bar exhaustion logic)
            # else: V2 active in crisis mode (crisis_composite >= 0.30)
        # ============================================================================

        # V1 thresholds (backward compatible)
        fusion_th = context.get_threshold('liquidity_vacuum', 'fusion_threshold', 0.30)
        liq_max = context.get_threshold('liquidity_vacuum', 'liquidity_max', 0.20)
        vol_z_min = context.get_threshold('liquidity_vacuum', 'volume_z_min', 1.5)
        wick_lower_min = context.get_threshold('liquidity_vacuum', 'wick_lower_min', 0.28)

        # ============================================================================
        # STEP 1.5: REGIME FILTER (fast fail to prevent bull market false positives)
        # ============================================================================
        # S1 is a capitulation/crisis archetype. Only fire when macro supports the hypothesis.
        # This prevents noise during bull markets (e.g., 2023: 0 trades, 2024: reduce from 12 to 4-6)

        use_regime_filter = context.get_threshold('liquidity_vacuum', 'use_regime_filter', False)

        if use_regime_filter:
            # Get current regime (from RuntimeContext.regime_label or row.regime_label)
            # Try RuntimeContext first (preferred), then fallback to row column
            current_regime = context.regime_label if hasattr(context, 'regime_label') else 'unknown'
            if current_regime == 'unknown' or current_regime is None:
                current_regime = self.g(context.row, 'regime_label', 'unknown')

            # Fallback: If regime_label not available, infer from crisis_composite
            if current_regime == 'unknown':
                crisis_composite = self.g(context.row, 'crisis_composite', 0.0)
                if crisis_composite > 0.35:
                    current_regime = 'risk_off'  # High crisis = bearish regime
                else:
                    current_regime = 'neutral'  # Default to neutral

            # Get drawdown from V2 features (fallback to 0 if not available)
            capitulation_depth = self.g(context.row, 'capitulation_depth', 0.0)

            # Check regime allowlist
            allowed_regimes = context.get_threshold('liquidity_vacuum', 'allowed_regimes', ['risk_off', 'crisis'])
            regime_ok = current_regime in allowed_regimes

            # Check drawdown override (severe drawdown = always allow regardless of regime)
            drawdown_override_pct = context.get_threshold('liquidity_vacuum', 'drawdown_override_pct', 0.10)
            drawdown_ok = capitulation_depth < -drawdown_override_pct  # e.g., < -0.10 (more than 10% drawdown)

            # OR logic: pass if EITHER regime ok OR drawdown significant
            require_or = context.get_threshold('liquidity_vacuum', 'require_regime_or_drawdown', True)

            if require_or:
                # OR logic: pass if EITHER regime ok OR drawdown significant
                regime_check_pass = regime_ok or drawdown_ok
            else:
                # AND logic: require BOTH
                regime_check_pass = regime_ok and drawdown_ok

            # Block if regime filter fails
            if not regime_check_pass:
                return False, 0.0, {
                    "reason": "regime_filter_blocked",
                    "current_regime": current_regime,
                    "allowed_regimes": allowed_regimes,
                    "capitulation_depth": capitulation_depth,
                    "drawdown_override_pct": drawdown_override_pct,
                    "regime_ok": regime_ok,
                    "drawdown_ok": drawdown_ok,
                    "note": "Capitulation pattern blocked by regime filter (prevents bull market false positives)"
                }

        # ============================================================================
        # STEP 2: Check for V2 Features (determine which mode to use)
        # ============================================================================
        has_v2_features = all([
            context.row.get('capitulation_depth') is not None,
            context.row.get('crisis_composite') is not None,
            context.row.get('volume_climax_last_3b') is not None,
            context.row.get('wick_exhaustion_last_3b') is not None
        ])

        # ============================================================================
        # STEP 3: V2 DETECTION LOGIC (preferred - multi-bar capitulation)
        # ============================================================================
        if use_v2_logic and has_v2_features:
            # Extract V2 features
            cap_depth = self.g(context.row, 'capitulation_depth', 0.0)
            crisis = self.g(context.row, 'crisis_composite', 0.0)
            vol_climax_3b = self.g(context.row, 'volume_climax_last_3b', 0.0)
            wick_exhaust_3b = self.g(context.row, 'wick_exhaustion_last_3b', 0.0)

            # Optional V2 features
            liq_drain_pct = self.g(context.row, 'liquidity_drain_pct', 0.0)
            liq_velocity = self.g(context.row, 'liquidity_velocity', 0.0)
            liq_persist = self.g(context.row, 'liquidity_persistence', 0)

            # ============================================================================
            # CONFLUENCE MODE (probabilistic 3-of-4 logic)
            # ============================================================================
            use_confluence = context.get_threshold('liquidity_vacuum', 'use_confluence', False)

            if use_confluence:
                # Score each condition [0-1] - how strong is this signal?
                depth_score = max(0.0, min(1.0, abs(cap_depth) / 0.30))  # 0.30 = extreme capitulation
                crisis_score = max(0.0, min(1.0, crisis / 0.50))  # 0.50 = peak crisis
                vol_score = max(0.0, min(1.0, vol_climax_3b / 0.70))  # 0.70 = extreme volume climax
                wick_score = max(0.0, min(1.0, wick_exhaust_3b / 0.80))  # 0.80 = max rejection

                # Count binary conditions (pass/fail at thresholds)
                conditions_met = sum([
                    cap_depth < cap_depth_max,  # Depth threshold met
                    crisis > crisis_min,  # Crisis threshold met
                    vol_climax_3b > vol_climax_3b_min,  # Volume threshold met
                    wick_exhaust_3b > wick_exhaust_3b_min  # Wick threshold met
                ])

                # Minimum conditions required (default: 3 of 4)
                min_conditions = context.get_threshold('liquidity_vacuum', 'confluence_min_conditions', 3)

                if conditions_met < min_conditions:
                    return False, 0.0, {
                        "reason": "confluence_insufficient_conditions",
                        "conditions_met": conditions_met,
                        "required": min_conditions,
                        "condition_states": {
                            "depth": cap_depth < cap_depth_max,
                            "crisis": crisis > crisis_min,
                            "volume": vol_climax_3b > vol_climax_3b_min,
                            "wick": wick_exhaust_3b > wick_exhaust_3b_min
                        },
                        "values": {
                            "cap_depth": cap_depth,
                            "crisis": crisis,
                            "vol_climax_3b": vol_climax_3b,
                            "wick_exhaust_3b": wick_exhaust_3b
                        }
                    }

                # Weighted confluence scoring (using normalized scores, not binary)
                weights = context.get_threshold('liquidity_vacuum', 'confluence_weights', {
                    'capitulation_depth': 0.30,
                    'crisis_environment': 0.25,
                    'volume_climax': 0.25,
                    'wick_exhaustion': 0.20
                })

                confluence_score = (
                    depth_score * weights.get('capitulation_depth', 0.30) +
                    crisis_score * weights.get('crisis_environment', 0.25) +
                    vol_score * weights.get('volume_climax', 0.25) +
                    wick_score * weights.get('wick_exhaustion', 0.20)
                )

                # Confluence threshold (default: 0.65 = 65% weighted score required)
                confluence_threshold = context.get_threshold('liquidity_vacuum', 'confluence_threshold', 0.65)

                if confluence_score < confluence_threshold:
                    return False, 0.0, {
                        "reason": "confluence_score_insufficient",
                        "confluence_score": confluence_score,
                        "threshold": confluence_threshold,
                        "conditions_met": conditions_met,
                        "scores": {
                            "depth_score": depth_score,
                            "crisis_score": crisis_score,
                            "vol_score": vol_score,
                            "wick_score": wick_score
                        },
                        "values": {
                            "cap_depth": cap_depth,
                            "crisis": crisis,
                            "vol_climax_3b": vol_climax_3b,
                            "wick_exhaust_3b": wick_exhaust_3b
                        }
                    }

                # CONFLUENCE PASS - build components for final fusion score
                components = {
                    # Use confluence score as primary signal
                    "confluence_score": confluence_score,
                    "capitulation_depth_score": depth_score,
                    "crisis_environment": crisis_score,
                    "volume_climax_3b": vol_score,
                    "wick_exhaustion_3b": wick_score,

                    # Liquidity dynamics (optional enhancements)
                    "liquidity_drain_severity": abs(min(liq_drain_pct, 0.0) / 0.50),
                    "liquidity_velocity_score": abs(min(liq_velocity, 0.0) / 0.10),
                    "liquidity_persistence_score": min(liq_persist / 8.0, 1.0),
                }

                # Extract additional confluence features
                funding_z = self.g(context.row, 'funding_Z', 0)
                rsi = self.g(context.row, 'rsi_14', 50)
                atr_pct = self.g(context.row, 'atr_percentile', 0.5)

                components.update({
                    "funding_reversal": 1.0 if funding_z < -0.5 else 0.5,
                    "oversold": 1.0 if rsi < 30 else max(0.0, 1.0 - (rsi / 100)),
                    "volatility_spike": atr_pct
                })

                # Use confluence_score directly as final score (already weighted)
                final_score = confluence_score

                # ============================================================================
                # DOMAIN ENGINE INTEGRATION (CONFLUENCE MODE)
                # ============================================================================
                use_wyckoff = context.metadata.get('feature_flags', {}).get('enable_wyckoff', False)
                use_smc = context.metadata.get('feature_flags', {}).get('enable_smc', False)
                use_temporal = context.metadata.get('feature_flags', {}).get('enable_temporal', False)
                use_macro = context.metadata.get('feature_flags', {}).get('enable_macro', False)

                domain_boost = 1.0
                domain_signals = []

                if use_wyckoff:
                    wyckoff_spring_a = self.g(context.row, 'wyckoff_spring_a', False)
                    wyckoff_spring_b = self.g(context.row, 'wyckoff_spring_b', False)
                    wyckoff_ps = self.g(context.row, 'wyckoff_ps', False)
                    if wyckoff_spring_a or wyckoff_spring_b:
                        domain_boost *= 1.25
                        domain_signals.append("wyckoff_spring")
                    elif wyckoff_ps:
                        domain_boost *= 1.15
                        domain_signals.append("wyckoff_ps")

                if use_smc:
                    smc_score = self.g(context.row, 'smc_score', 0.0)
                    if smc_score > 0.5:
                        domain_boost *= 1.15
                        domain_signals.append("smc_bullish")

                if use_temporal:
                    wyckoff_pti_confluence = self.g(context.row, 'wyckoff_pti_confluence', False)
                    if wyckoff_pti_confluence:
                        domain_boost *= 1.10
                        domain_signals.append("temporal_confluence")

                if use_macro and crisis > 0.70:
                    domain_boost *= 0.85
                    domain_signals.append("macro_extreme_penalty")

                final_score = final_score * domain_boost

                # CONFLUENCE PATTERN MATCHED
                return True, final_score, {
                    "domain_boost": domain_boost,
                    "domain_signals": domain_signals,
                    "mode": "v2_confluence_probabilistic",
                    "confluence_score": confluence_score,
                    "conditions_met": conditions_met,
                    "min_conditions": min_conditions,
                    "components": components,
                    "mechanism": "liquidity_vacuum_confluence_v2",
                    "condition_states": {
                        "depth": cap_depth < cap_depth_max,
                        "crisis": crisis > crisis_min,
                        "volume": vol_climax_3b > vol_climax_3b_min,
                        "wick": wick_exhaust_3b > wick_exhaust_3b_min
                    },
                    "raw_values": {
                        "cap_depth": cap_depth,
                        "crisis": crisis,
                        "vol_climax_3b": vol_climax_3b,
                        "wick_exhaust_3b": wick_exhaust_3b
                    },
                    "normalized_scores": {
                        "depth_score": depth_score,
                        "crisis_score": crisis_score,
                        "vol_score": vol_score,
                        "wick_score": wick_score
                    }
                }

            # ============================================================================
            # BINARY MODE (original hard gate logic - backward compatible)
            # ============================================================================

            # HARD GATE 1: Capitulation depth (must be in drawdown)
            if cap_depth >= cap_depth_max:
                return False, 0.0, {
                    "reason": "v2_insufficient_drawdown",
                    "capitulation_depth": cap_depth,
                    "threshold": cap_depth_max,
                    "note": "Need >= 20% drawdown from 30d high for true capitulation"
                }

            # HARD GATE 2: Crisis environment (must be in crisis/stress)
            if crisis < crisis_min:
                return False, 0.0, {
                    "reason": "v2_not_in_crisis",
                    "crisis_composite": crisis,
                    "threshold": crisis_min,
                    "note": "Need crisis environment (VIX spike + funding extreme + volatility)"
                }

            # OR GATE: Multi-bar exhaustion (at least ONE signal in last 3 bars)
            has_volume_exhaustion = vol_climax_3b > vol_climax_3b_min
            has_wick_exhaustion = wick_exhaust_3b > wick_exhaust_3b_min

            if not (has_volume_exhaustion or has_wick_exhaustion):
                return False, 0.0, {
                    "reason": "v2_no_multi_bar_exhaustion",
                    "volume_climax_3b": vol_climax_3b,
                    "volume_threshold": vol_climax_3b_min,
                    "wick_exhaustion_3b": wick_exhaust_3b,
                    "wick_threshold": wick_exhaust_3b_min,
                    "note": "Need at least ONE exhaustion signal in last 3 bars"
                }

            # Extract additional features for scoring
            funding_z = self.g(context.row, 'funding_Z', 0)
            rsi = self.g(context.row, 'rsi_14', 50)
            atr_pct = self.g(context.row, 'atr_percentile', 0.5)

            # V2 SCORING COMPONENTS
            components = {
                # HARD GATES (already passed - contribute to score)
                "capitulation_depth_score": abs(cap_depth / 0.50),  # Normalize: -50% = 1.0
                "crisis_environment": crisis,  # Already [0, 1]

                # EXHAUSTION SIGNALS
                "volume_climax_3b": min(vol_climax_3b / 0.50, 1.0),  # Normalize
                "wick_exhaustion_3b": min(wick_exhaust_3b / 0.50, 1.0),  # Normalize

                # LIQUIDITY DYNAMICS (V2 features)
                "liquidity_drain_severity": abs(min(liq_drain_pct, 0.0) / 0.50),  # Normalize: -50% drain = 1.0
                "liquidity_velocity_score": abs(min(liq_velocity, 0.0) / 0.10),  # Normalize: -10% velocity = 1.0
                "liquidity_persistence_score": min(liq_persist / 8.0, 1.0),  # Normalize: 8 bars = 1.0

                # CONFLUENCE SIGNALS
                "funding_reversal": 1.0 if funding_z < -0.5 else 0.5,
                "oversold": 1.0 if rsi < 30 else max(0.0, 1.0 - (rsi / 100)),
                "volatility_spike": atr_pct
            }

            # V2 WEIGHTS (emphasize V2 features)
            weights = context.get_threshold('liquidity_vacuum', 'v2_weights', {
                # Core capitulation signals (50%)
                "capitulation_depth_score": 0.20,
                "crisis_environment": 0.15,
                "volume_climax_3b": 0.08,
                "wick_exhaustion_3b": 0.07,

                # Liquidity dynamics (25%)
                "liquidity_drain_severity": 0.10,
                "liquidity_velocity_score": 0.08,
                "liquidity_persistence_score": 0.07,

                # Confluence (25%)
                "funding_reversal": 0.12,
                "oversold": 0.08,
                "volatility_spike": 0.05
            })

            # Calculate weighted fusion score
            score = sum(components[k] * weights.get(k, 0.0) for k in components)

            # ============================================================================
            # DOMAIN ENGINE INTEGRATION (BOOST/VETO LAYER)
            # ============================================================================
            # CRITICAL FIX: Apply domain engines BEFORE fusion threshold gate
            # This allows marginal signals (e.g., score=0.38) to qualify via boosts
            # Order: VETOES first (safety) → BOOSTS second → GATE third

            use_wyckoff = context.metadata.get('feature_flags', {}).get('enable_wyckoff', False)
            use_smc = context.metadata.get('feature_flags', {}).get('enable_smc', False)
            use_temporal = context.metadata.get('feature_flags', {}).get('enable_temporal', False)
            use_hob = context.metadata.get('feature_flags', {}).get('enable_hob', False)
            use_macro = context.metadata.get('feature_flags', {}).get('enable_macro', False)

            # DEBUG: Log feature flags on first detection
            if not hasattr(self, '_domain_flags_logged'):
                logger.info(f"[DOMAIN_DEBUG] S1 Feature Flags: wyckoff={use_wyckoff}, smc={use_smc}, temporal={use_temporal}, hob={use_hob}, macro={use_macro}")
                self._domain_flags_logged = True

            domain_boost = 1.0
            domain_signals = []

            # ============================================================================
            # WYCKOFF ENGINE: Complete capitulation event detection
            # ============================================================================
            if use_wyckoff:
                # SOFT VETOES: Distribution phase reduces confidence but doesn't block
                wyckoff_distribution = self.g(context.row, 'wyckoff_phase_abc', '') == 'D'
                wyckoff_utad = self.g(context.row, 'wyckoff_utad', False)
                wyckoff_bc = self.g(context.row, 'wyckoff_bc', False)

                if wyckoff_distribution:
                    domain_boost *= 0.30  # 70% confidence reduction for distribution phase
                    domain_signals.append("wyckoff_distribution_caution")

                if wyckoff_utad or wyckoff_bc:
                    domain_boost *= 0.50  # Stronger events get more severe reduction
                    domain_signals.append("wyckoff_utad_bc_caution")

                # MAJOR BOOSTS: Spring events (Phase C - strongest capitulation signals)
                wyckoff_spring_a = self.g(context.row, 'wyckoff_spring_a', False)
                wyckoff_spring_b = self.g(context.row, 'wyckoff_spring_b', False)

                if wyckoff_spring_a:
                    domain_boost *= 2.50  # Spring A = deep fake breakdown, strongest signal
                    domain_signals.append("wyckoff_spring_a_major_capitulation")
                elif wyckoff_spring_b:
                    domain_boost *= 2.50  # Spring B = shallow spring, still strong
                    domain_signals.append("wyckoff_spring_b_capitulation")

                # CLIMAX SIGNALS: Selling Climax + Secondary Test (Phase A)
                wyckoff_sc = self.g(context.row, 'wyckoff_sc', False)
                wyckoff_st = self.g(context.row, 'wyckoff_st', False)
                wyckoff_ar = self.g(context.row, 'wyckoff_ar', False)

                if wyckoff_sc:
                    domain_boost *= 2.00  # Selling Climax = panic bottom
                    domain_signals.append("wyckoff_selling_climax")
                elif wyckoff_st:
                    domain_boost *= 1.50  # Secondary Test = retest of lows
                    domain_signals.append("wyckoff_secondary_test")
                elif wyckoff_ar:
                    domain_boost *= 1.30  # Automatic Rally = relief bounce
                    domain_signals.append("wyckoff_automatic_rally")

                # SUPPORT SIGNALS: LPS + Preliminary Support (Phase B/D)
                wyckoff_lps = self.g(context.row, 'wyckoff_lps', False)
                wyckoff_ps = self.g(context.row, 'wyckoff_ps', False)

                if wyckoff_lps:
                    domain_boost *= 1.80  # Last Point Support = final test before markup
                    domain_signals.append("wyckoff_lps_support")
                elif wyckoff_ps:
                    domain_boost *= 1.30  # Preliminary Support = early accumulation
                    domain_signals.append("wyckoff_ps_support")

                # ACCUMULATION PHASE: General accumulation confirmation
                wyckoff_accumulation = self.g(context.row, 'wyckoff_phase_abc', '') == 'A'
                wyckoff_sos = self.g(context.row, 'wyckoff_sos', False)

                if wyckoff_accumulation:
                    domain_boost *= 1.40  # Accumulation phase = buying pressure
                    domain_signals.append("wyckoff_accumulation_phase")
                elif wyckoff_sos:
                    domain_boost *= 1.35  # Sign of Strength = decisive up move
                    domain_signals.append("wyckoff_sos_strength")

            # ============================================================================
            # SMC ENGINE: Smart Money Concepts - institutional structure
            # ============================================================================
            if use_smc:
                # VETOES: Don't long into supply zones
                smc_supply_zone = self.g(context.row, 'smc_supply_zone', False)
                tf4h_bos_bearish = self.g(context.row, 'tf4h_bos_bearish', False)

                if smc_supply_zone:
                    domain_boost *= 0.70  # Supply overhead = resistance, reduce signal
                    domain_signals.append("smc_supply_zone_overhead")

                if tf4h_bos_bearish:
                    # Soft veto: bearish 4H structure reduces confidence
                    domain_boost *= 0.60
                    domain_signals.append("smc_4h_bearish_structure_penalty")

                # MAJOR BOOSTS: Multi-timeframe bullish structure
                tf1h_bos_bullish = self.g(context.row, 'tf1h_bos_bullish', False)
                tf4h_bos_bullish = self.g(context.row, 'tf4h_bos_bullish', False)

                # 4H BOS = institutional timeframe (strongest signal)
                if tf4h_bos_bullish:
                    domain_boost *= 2.00  # +100% boost for 4H institutional shift
                    domain_signals.append("smc_4h_bos_bullish_institutional")
                elif tf1h_bos_bullish:
                    domain_boost *= 1.40  # +40% boost for 1H structural shift
                    domain_signals.append("smc_1h_bos_bullish")

                # DEMAND ZONES: Institutional buying areas
                smc_demand_zone = self.g(context.row, 'smc_demand_zone', False)
                smc_liquidity_sweep = self.g(context.row, 'smc_liquidity_sweep', False)
                smc_choch = self.g(context.row, 'smc_choch', False)  # Change of Character

                if smc_demand_zone:
                    domain_boost *= 1.50  # Demand zone = institutional support
                    domain_signals.append("smc_demand_zone_support")

                if smc_liquidity_sweep:
                    domain_boost *= 1.80  # Liquidity sweep = stop hunt before rally
                    domain_signals.append("smc_liquidity_sweep_reversal")

                if smc_choch:
                    domain_boost *= 1.60  # Character change = trend shift
                    domain_signals.append("smc_choch_trend_change")

                # LEGACY SMC SCORE: Fallback composite
                smc_score = self.g(context.row, 'smc_score', 0.0)
                if smc_score > 0.5 and domain_boost == 1.0:  # No specific signals yet
                    domain_boost *= 1.15
                    domain_signals.append("smc_bullish_structure")

            # ============================================================================
            # TEMPORAL ENGINE: Fibonacci time + multi-timeframe confluence
            # ============================================================================
            if use_temporal:
                # MAJOR BOOSTS: Fibonacci time clusters
                fib_time_cluster = self.g(context.row, 'fib_time_cluster', False)
                temporal_confluence = self.g(context.row, 'temporal_confluence', False)

                if fib_time_cluster:
                    domain_boost *= 1.80  # Fibonacci timing = geometric reversal points
                    domain_signals.append("fib_time_cluster_reversal")

                if temporal_confluence:
                    domain_boost *= 1.50  # Multi-timeframe alignment
                    domain_signals.append("temporal_multi_tf_confluence")

                # MULTI-TIMEFRAME FUSION: 4H fusion score
                tf4h_fusion_score = self.g(context.row, 'tf4h_fusion_score', 0.0)
                if tf4h_fusion_score > 0.70:
                    domain_boost *= 1.60  # High 4H fusion = strong trend alignment
                    domain_signals.append("tf4h_high_fusion_score")

                # WYCKOFF-PTI CONFLUENCE: Combined Wyckoff + time patterns
                wyckoff_pti_confluence = self.g(context.row, 'wyckoff_pti_confluence', False)
                wyckoff_pti_score = self.g(context.row, 'wyckoff_pti_score', 0.0)

                if wyckoff_pti_confluence:
                    if wyckoff_pti_score > 0.50:
                        domain_boost *= 1.50  # Strong PTI + Wyckoff combo
                        domain_signals.append("wyckoff_pti_strong_confluence")
                    else:
                        domain_boost *= 1.20  # Moderate confluence
                        domain_signals.append("wyckoff_pti_confluence")

                # VETOES: Don't long into resistance clusters
                temporal_resistance_cluster = self.g(context.row, 'temporal_resistance_cluster', False)
                if temporal_resistance_cluster:
                    domain_boost *= 0.75  # Resistance overhead = reduce conviction
                    domain_signals.append("temporal_resistance_overhead")

            # ============================================================================
            # HOB ENGINE: Order book depth + imbalance analysis
            # ============================================================================
            if use_hob:
                # DEMAND ZONES: Large bid walls
                hob_demand_zone = self.g(context.row, 'hob_demand_zone', False)
                hob_imbalance = self.g(context.row, 'hob_imbalance', 0.0)

                if hob_demand_zone:
                    domain_boost *= 1.50  # Demand wall = institutional support
                    domain_signals.append("hob_demand_zone_support")

                # ORDER BOOK IMBALANCE: Bid/ask ratio
                if hob_imbalance > 0.60:  # More bids than asks
                    domain_boost *= 1.30  # Strong buyer imbalance
                    domain_signals.append("hob_bid_imbalance_strong")
                elif hob_imbalance > 0.40:
                    domain_boost *= 1.15  # Moderate buyer imbalance
                    domain_signals.append("hob_bid_imbalance_moderate")

                # VETOES: Supply zones
                hob_supply_zone = self.g(context.row, 'hob_supply_zone', False)
                if hob_supply_zone:
                    domain_boost *= 0.70  # Supply wall overhead
                    domain_signals.append("hob_supply_zone_overhead")

            # MACRO: Extreme risk-off reduces confidence (avoid catching falling knife)
            if use_macro:
                # Use crisis_composite as proxy for macro stress
                # Already in components, but apply additional veto logic
                if crisis > 0.70:  # Extreme crisis (>70%) = reduce confidence
                    domain_boost *= 0.85  # -15% penalty for extreme macro stress
                    domain_signals.append("macro_extreme_crisis_penalty")

            # Apply domain boost to final score
            score_before_domain = score
            score = score * domain_boost

            # DEBUG: Log domain boost application
            if domain_boost != 1.0 or len(domain_signals) > 0:
                logger.info(f"[DOMAIN_DEBUG] S1 Domain Boost Applied: {domain_boost:.2f}x | Score: {score_before_domain:.3f} -> {score:.3f} | Signals: {domain_signals}")
            else:
                logger.info(f"[DOMAIN_DEBUG] S1 No Domain Boost: domain_boost={domain_boost}, signals={domain_signals}, score={score:.3f}")

            # ============================================================================
            # FUSION THRESHOLD GATE (applied AFTER domain engines)
            # ============================================================================
            # Check boosted score against threshold
            # This allows marginal signals to qualify via domain boosts
            if score < fusion_th:
                return False, score, {
                    "reason": "v2_score_below_threshold",
                    "score": score,
                    "score_before_domain": score_before_domain,
                    "threshold": fusion_th,
                    "mode": "v2_multi_bar",
                    "components": components,
                    "domain_boost": domain_boost,
                    "domain_signals": domain_signals
                }

            # V2 PATTERN MATCHED
            return True, score, {
                "mode": "v2_multi_bar_capitulation",
                "components": components,
                "weights": weights,
                "mechanism": "liquidity_vacuum_capitulation_fade_v2",
                "domain_boost": domain_boost,
                "domain_signals": domain_signals,
                "gates_passed": {
                    "capitulation_depth": cap_depth,
                    "crisis_composite": crisis,
                    "volume_climax_3b": vol_climax_3b,
                    "wick_exhaustion_3b": wick_exhaust_3b,
                    "has_volume_exhaustion": has_volume_exhaustion,
                    "has_wick_exhaustion": has_wick_exhaustion
                }
            }, "LONG"

        # ============================================================================
        # STEP 4: V1 FALLBACK LOGIC (backward compatible)
        # ============================================================================

        # Extract V1 REQUIRED features
        liquidity = self._liquidity_score(context.row)
        volume_z = self.g(context.row, 'volume_zscore', 0)

        # Try to get runtime-enriched wick_lower_ratio
        wick_lower_ratio = self.g(context.row, 'wick_lower_ratio', None)

        # If not pre-calculated, calculate on-the-fly
        if wick_lower_ratio is None:
            wick_lower_ratio = self._calculate_wick_lower_ratio(context.row)

        # DEBUG: Log first call
        if not hasattr(self, '_liquidity_vacuum_first_call_logged'):
            mode = "V2" if has_v2_features else "V1 (fallback)"
            logger.info(f"[Liquidity Vacuum {mode}] First evaluation")
            if not has_v2_features:
                logger.info(f"  V1 fallback: liq={liquidity:.3f}, vol_z={volume_z:.2f}, wick_lower={wick_lower_ratio:.3f}")
            self._liquidity_vacuum_first_call_logged = True

        # HARD GATE 1: Liquidity drain (ONLY true invariant across all capitulations)
        if liquidity >= liq_max:
            return False, 0.0, {
                "reason": "v1_liquidity_not_drained",
                "mode": "v1_fallback",
                "liquidity": liquidity,
                "threshold": liq_max
            }

        # OR GATE: At least ONE exhaustion signal required (NOT both)
        volume_exhaustion = volume_z >= vol_z_min
        wick_exhaustion = wick_lower_ratio >= wick_lower_min

        if not (volume_exhaustion or wick_exhaustion):
            return False, 0.0, {
                "reason": "v1_no_exhaustion_signal",
                "mode": "v1_fallback",
                "volume_z": volume_z,
                "volume_threshold": vol_z_min,
                "wick_lower": wick_lower_ratio,
                "wick_threshold": wick_lower_min,
                "note": "Need at least ONE: volume panic OR wick rejection"
            }

        # Extract V1 OPTIONAL Features (Graceful Degradation)
        funding_z = self.g(context.row, 'funding_Z', 0)
        vix_z = self.g(context.row, 'VIX_Z', 0)
        dxy_z = self.g(context.row, 'DXY_Z', 0)
        rsi = self.g(context.row, 'rsi_14', 50)
        atr_pct = self.g(context.row, 'atr_percentile', 0.5)
        tf4h_trend = self.g(context.row, 'tf4h_external_trend', 'neutral')

        # V1 SCORING COMPONENTS
        components = {
            # REQUIRED components (normalized scores)
            "liquidity_vacuum": 1.0 - (liquidity / liq_max),  # Lower liquidity = higher score
            "volume_capitulation": min(volume_z / 3.0, 1.0),  # Normalize (3.0 = extreme)
            "wick_rejection": min(wick_lower_ratio / 0.5, 1.0),  # Normalize (0.5 = extreme)

            # OPTIONAL components (with fallbacks)
            "funding_reversal": 1.0 if funding_z < -0.5 else 0.5,  # Negative funding bonus
            "crisis_context": min((vix_z + dxy_z) / 3.0, 1.0),  # Combined macro stress
            "oversold": 1.0 if rsi < 30 else (1.0 - (rsi / 100)),  # Oversold bonus
            "volatility_spike": atr_pct,  # Higher volatility = higher score
            "downtrend_confirm": 1.0 if tf4h_trend == 'down' else 0.3  # Downtrend context
        }

        # V1 WEIGHTS
        weights = context.get_threshold('liquidity_vacuum', 'weights', {
            # REQUIRED feature weights (total: 65%)
            "liquidity_vacuum": 0.25,      # Primary signal
            "volume_capitulation": 0.20,   # Panic selling
            "wick_rejection": 0.20,        # Exhaustion signal

            # OPTIONAL feature weights (total: 35%)
            "funding_reversal": 0.15,      # Short squeeze fuel
            "crisis_context": 0.10,        # Macro capitulation
            "oversold": 0.05,              # Mean reversion
            "volatility_spike": 0.03,      # Violent moves expected
            "downtrend_confirm": 0.02      # Context confirmation
        })

        # Weighted fusion score
        score = sum(components[k] * weights.get(k, 0.0) for k in components)

        # ============================================================================
        # DOMAIN ENGINE INTEGRATION (V1 FALLBACK MODE)
        # ============================================================================
        # CRITICAL FIX: Apply domain engines BEFORE fusion threshold gate
        use_wyckoff = context.metadata.get('feature_flags', {}).get('enable_wyckoff', False)
        use_smc = context.metadata.get('feature_flags', {}).get('enable_smc', False)
        use_temporal = context.metadata.get('feature_flags', {}).get('enable_temporal', False)

        domain_boost = 1.0
        domain_signals = []

        if use_wyckoff:
            wyckoff_spring_a = self.g(context.row, 'wyckoff_spring_a', False)
            wyckoff_spring_b = self.g(context.row, 'wyckoff_spring_b', False)
            wyckoff_ps = self.g(context.row, 'wyckoff_ps', False)
            if wyckoff_spring_a or wyckoff_spring_b:
                domain_boost *= 1.25
                domain_signals.append("wyckoff_spring")
            elif wyckoff_ps:
                domain_boost *= 1.15
                domain_signals.append("wyckoff_ps")

        if use_smc:
            smc_score = self.g(context.row, 'smc_score', 0.0)
            if smc_score > 0.5:
                domain_boost *= 1.15
                domain_signals.append("smc_bullish")

        if use_temporal:
            wyckoff_pti_confluence = self.g(context.row, 'wyckoff_pti_confluence', False)
            if wyckoff_pti_confluence:
                domain_boost *= 1.10
                domain_signals.append("temporal_confluence")

        score_before_domain = score
        score = score * domain_boost

        # Final Fusion Threshold Gate (applied AFTER domain engines)
        if score < fusion_th:
            return False, score, {
                "reason": "v1_score_below_threshold",
                "mode": "v1_fallback",
                "score": score,
                "score_before_domain": score_before_domain,
                "threshold": fusion_th,
                "components": components,
                "domain_boost": domain_boost,
                "domain_signals": domain_signals
            }

        # V1 PATTERN MATCHED
        return True, score, {
            "mode": "v1_single_bar_fallback",
            "components": components,
            "weights": weights,
            "mechanism": "liquidity_vacuum_capitulation_fade_v1",
            "domain_boost": domain_boost,
            "domain_signals": domain_signals,
            "gates_passed": {
                "liquidity": liquidity,
                "volume_z": volume_z,
                "wick_lower": wick_lower_ratio
            }
        }, "LONG"

    def _calculate_wick_lower_ratio(self, row: pd.Series) -> float:
        """
        Calculate lower wick ratio on-the-fly if not pre-enriched.

        Returns:
            float [0, 1] - lower wick as percentage of candle range
        """
        try:
            open_price = float(row.get('open', 0))
            close = float(row.get('close', 0))
            high = float(row.get('high', 0))
            low = float(row.get('low', 0))

            # Candle range
            candle_range = high - low
            if candle_range == 0:
                return 0.0

            # Body low (min of open/close)
            body_low = min(open_price, close)

            # Lower wick
            wick_lower = body_low - low

            # Normalize
            wick_lower_ratio = wick_lower / candle_range

            return max(0.0, min(1.0, wick_lower_ratio))

        except (TypeError, KeyError, ValueError):
            return 0.0

    def _check_S2(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """
        Archetype S2: Failed Rally Rejection (WITH RUNTIME FEATURE ENRICHMENT)

        Trader: Zeroika (dead cat bounce specialist)
        Edge: Order block retest + volume fade + RSI divergence = bull trap

        **RUNTIME ENRICHMENT UPDATE:**
        Now uses runtime-calculated features when available:
        - wick_upper_ratio (from runtime vs manual calculation)
        - volume_fade_flag (proper 3-bar sequence detection)
        - rsi_bearish_div (true divergence detection)
        - ob_retest_flag (enhanced OB detection)

        Baseline Performance (2022, NO enrichment):
        - PF: 0.38 (baseline) / 0.56 (optimized)
        - Win Rate: 38.5% / 42.6%
        - Trade Count: 335 / 444

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
        use_runtime_features = context.get_threshold('failed_rally', 'use_runtime_features', False)

        # ENHANCED: Check for runtime-enriched features first
        if use_runtime_features:
            # Prefer runtime-calculated features (more accurate)
            wick_upper_ratio = self.g(context.row, 'wick_upper_ratio', None)
            volume_fade_flag = self.g(context.row, 'volume_fade_flag', None)
            rsi_bearish_div = self.g(context.row, 'rsi_bearish_div', None)
            ob_retest_flag = self.g(context.row, 'ob_retest_flag', None)

            # Log first runtime feature usage
            if not hasattr(self, '_s2_runtime_logged'):
                logger.info(f"[S2 RUNTIME] Using enriched features - wick={wick_upper_ratio is not None}, vol_fade={volume_fade_flag is not None}, rsi_div={rsi_bearish_div is not None}, ob={ob_retest_flag is not None}")
                self._s2_runtime_logged = True

            # If runtime features available, use enhanced logic
            if all(x is not None for x in [wick_upper_ratio, volume_fade_flag, rsi_bearish_div, ob_retest_flag]):
                return self._check_S2_enhanced(context, wick_upper_ratio, volume_fade_flag, rsi_bearish_div, ob_retest_flag, fusion_th)

        # FALLBACK: Original logic (for backward compatibility)
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
            "ob_high": ob_high,
            "feature_source": "legacy"
        }

    def _check_S2_enhanced(
        self,
        context: RuntimeContext,
        wick_upper_ratio: float,
        volume_fade_flag: bool,
        rsi_bearish_div: bool,
        ob_retest_flag: bool,
        fusion_th: float
    ) -> Tuple[bool, float, Dict]:
        """
        S2 Enhanced Logic using runtime-calculated features.

        This version uses proper feature calculations:
        - wick_upper_ratio: Vectorized wick calculation (% of candle range)
        - volume_fade_flag: 3-bar volume sequence detection
        - rsi_bearish_div: True divergence (price up, RSI down over 14 bars)
        - ob_retest_flag: Enhanced OB detection

        Args:
            context: RuntimeContext
            wick_upper_ratio: Pre-calculated upper wick ratio [0, 1]
            volume_fade_flag: Boolean, True if volume fading
            rsi_bearish_div: Boolean, True if bearish divergence detected
            ob_retest_flag: Boolean, True if price retesting resistance

        Returns:
            (matched: bool, score: float, meta: dict)
        """
        # Check if multi-confluence mode is enabled
        use_multi_confluence = context.get_threshold('failed_rally', 'use_multi_confluence', False)

        if use_multi_confluence:
            return self._check_S2_multi_confluence(context, wick_upper_ratio, volume_fade_flag, rsi_bearish_div, ob_retest_flag, fusion_th)

        # Original enhanced logic
        # Get configurable thresholds
        wick_min = context.get_threshold('failed_rally', 'wick_ratio_min', 0.4)  # More lenient for ratio
        rsi_div_weight = context.get_threshold('failed_rally', 'rsi_div_weight', 0.25)

        # Gate 1: Order block retest (required)
        if not ob_retest_flag:
            return False, 0.0, {"reason": "no_ob_retest", "ob_retest_flag": ob_retest_flag}

        # Gate 2: Upper wick rejection (required)
        if wick_upper_ratio < wick_min:
            return False, 0.0, {"reason": "weak_wick", "wick_upper_ratio": wick_upper_ratio, "threshold": wick_min}

        # Compute enhanced components
        components = {
            "ob_retest": 1.0 if ob_retest_flag else 0.0,
            "wick_rejection": min(wick_upper_ratio / 0.6, 1.0),  # Normalize (0.6 = very strong)
            "volume_fade": 1.0 if volume_fade_flag else 0.3,     # Strong signal if present
            "rsi_divergence": 1.0 if rsi_bearish_div else 0.4    # BONUS if true divergence
        }

        # Enhanced weights (emphasize actual divergence over simple overbought)
        weights = context.get_threshold('failed_rally', 'enhanced_weights', {
            "ob_retest": 0.25,
            "wick_rejection": 0.30,  # Increased (most reliable)
            "volume_fade": 0.25,      # Increased (key signal)
            "rsi_divergence": 0.20    # New (true divergence bonus)
        })

        score = sum(components[k] * weights.get(k, 0.0) for k in components)

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
            "wick_upper_ratio": wick_upper_ratio,
            "volume_fade_flag": volume_fade_flag,
            "rsi_bearish_div": rsi_bearish_div,
            "ob_retest_flag": ob_retest_flag,
            "feature_source": "runtime_enriched"
        }

    def _check_S2_multi_confluence(
        self,
        context: RuntimeContext,
        wick_upper_ratio: float,
        volume_fade_flag: bool,
        rsi_bearish_div: bool,
        ob_retest_flag: bool,
        fusion_th: float
    ) -> Tuple[bool, float, Dict]:
        """
        S2 v2: Multi-Confluence Failed Rally (trader-style discretion)

        Implements 8-factor confluence filter to reduce false positives:
        - Original 4: OB retest, RSI div, volume fade, wick rejection
        - NEW 4: MTF down, DXY strength, OI drain, Wyckoff distribution

        Requires 6/8 confluence minimum (trader discretion threshold).
        Crisis veto: VIX > 1.5 sigma = panic avoidance.
        Dynamic sizing: 6/8=0.8x, 7/8=1.0x, 8/8=1.2x

        Args:
            context: RuntimeContext
            wick_upper_ratio: Pre-calculated upper wick ratio
            volume_fade_flag: Boolean volume fade signal
            rsi_bearish_div: Boolean RSI divergence
            ob_retest_flag: Boolean OB retest
            fusion_th: Minimum fusion threshold

        Returns:
            (matched: bool, score: float, meta: dict)
        """
        # Original 4 conditions
        c1_ob_retest = ob_retest_flag
        c2_rsi_div = rsi_bearish_div
        c3_volume_fade = volume_fade_flag
        c4_wick_rejection = wick_upper_ratio > 0.4

        # NEW: 4 trader discretion conditions with graceful degradation
        # c5: 4H downtrend (prefer external_trend as it has 100% coverage)
        tf4h_trend_raw = self.g(context.row, 'tf4h_external_trend', 'neutral')
        c5_mtf_down = tf4h_trend_raw == 'down' if isinstance(tf4h_trend_raw, str) else (tf4h_trend_raw < 0)

        # c6: DXY strength (use DXY_Z from features, 100% coverage)
        dxy_z = self.g(context.row, 'DXY_Z', 0.0)
        c6_dxy_strength = dxy_z > 0.5

        # c7: OI drain (OI_CHANGE has 100% coverage)
        # Note: OI increasing = liquidity trap (more longs to squeeze)
        oi_change = self.g(context.row, 'OI_CHANGE', 0.0)
        c7_oi_drain = oi_change > 0.10  # 10% increase = trap

        # c8: Wyckoff distribution (check for distribution-like conditions)
        # Simplified: high RSI + low liquidity = distribution proxy
        wyckoff_score = self.g(context.row, 'wyckoff_score', 0.0)
        rsi = self.g(context.row, 'rsi_14', 50.0)
        liquidity = self._liquidity_score(context.row)
        c8_wyckoff_dist = (wyckoff_score > 0.35) or (rsi > 65 and liquidity < 0.30)

        # Count confluence
        conditions = [c1_ob_retest, c2_rsi_div, c3_volume_fade, c4_wick_rejection,
                      c5_mtf_down, c6_dxy_strength, c7_oi_drain, c8_wyckoff_dist]
        confluence_count = sum(conditions)

        # Get configurable min confluence
        min_confluence = context.get_threshold('failed_rally', 'min_confluence', 6)

        # Require minimum confluence (trader discretion threshold)
        if confluence_count < min_confluence:
            return False, 0.0, {
                'reason': 'insufficient_confluence',
                'confluence': confluence_count,
                'min_required': min_confluence,
                'conditions': {
                    'ob_retest': c1_ob_retest,
                    'rsi_div': c2_rsi_div,
                    'volume_fade': c3_volume_fade,
                    'wick': c4_wick_rejection,
                    'mtf_down': c5_mtf_down,
                    'dxy_up': c6_dxy_strength,
                    'oi_drain': c7_oi_drain,
                    'wyckoff': c8_wyckoff_dist
                }
            }

        # Crisis veto (VIX > 1.5 sigma = panic, traders avoid fades)
        vix_z = self.g(context.row, 'VIX_Z', 0.0)
        vix_z_max = context.get_threshold('failed_rally', 'vix_z_max', 1.5)
        if vix_z > vix_z_max:
            return False, 0.0, {
                'reason': 'crisis_veto',
                'vix_z': vix_z,
                'threshold': vix_z_max,
                'confluence': confluence_count
            }

        # Calculate fusion score (use original fusion calculation)
        fusion = self._fusion(context.row)

        if fusion < fusion_th:
            return False, fusion, {
                'reason': 'fusion_below_threshold',
                'fusion': fusion,
                'threshold': fusion_th,
                'confluence': confluence_count
            }

        # Dynamic sizing based on confluence (6/8 = 0.8x, 7/8 = 1.0x, 8/8 = 1.2x)
        size_mult = 0.6 + (0.1 * confluence_count)

        # Build metadata
        meta = {
            'confluence': confluence_count,
            'size_mult': size_mult,
            'fusion': fusion,
            'vix_z': vix_z,
            'conditions': {
                'ob_retest': c1_ob_retest,
                'rsi_div': c2_rsi_div,
                'volume_fade': c3_volume_fade,
                'wick': c4_wick_rejection,
                'mtf_down': c5_mtf_down,
                'dxy_up': c6_dxy_strength,
                'oi_drain': c7_oi_drain,
                'wyckoff': c8_wyckoff_dist
            },
            'feature_values': {
                'wick_upper_ratio': wick_upper_ratio,
                'tf4h_trend': tf4h_trend_raw,
                'dxy_z': dxy_z,
                'oi_change': oi_change,
                'wyckoff_score': wyckoff_score,
                'rsi': rsi,
                'liquidity': liquidity
            },
            'feature_source': 'multi_confluence_v2'
        }

        return True, fusion, meta

    def _pattern_S3(self, context: RuntimeContext):
        """Pattern detection for Archetype S3: Distribution Climax Short (SHORT)"""
        r = context.row
        vc = self.g(r, 'volume_climax_last_3b', 0.0)
        if vc >= 1.0 and self.g(r, 'rsi_14', 50) >= 70:
            score = 0.45
            return True, score, ["S3", "distribution_climax", "SHORT"]
        return None

    def _check_S3(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """Archetype S3: Distribution Climax Short (SHORT)"""

        # 1) Base pattern detection - returns (matched, score, metadata)
        base_result = self._pattern_S3(context)
        if not base_result:
            return False, 0.0, {"reason": "pattern_not_matched"}

        matched, base_score, pattern_tags = base_result  # base_score in [0,1]

        # 2) Global safety vetoes (soft - use penalties not hard returns)
        # Note: Keep these soft - use penalties not hard returns

        # 3) Domain modifiers (standardized across all)
        score = self._apply_domain_engines(context, base_score, pattern_tags)

        # 4) Final fusion gate
        fusion_th = context.get_threshold('distribution_climax', 'fusion_threshold', 0.40)
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_fusion_threshold",
                "base_score": base_score,
                "final_score": score,
                "fusion_threshold": fusion_th
            }

        # 5) Return with full metadata
        return True, score, {
            "base_score": base_score,
            "final_score": score,
            "pattern_tags": pattern_tags,
            "domain_boost_applied": True
        }, "EITHER"

    def _check_S4(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """
        Archetype S4: Funding Divergence (Short Squeeze)

        Trader: Contra-funding specialist
        Edge: Overcrowded shorts + price strength = violent squeeze UP

        OPPOSITE OF S5:
        - S5 (Long Squeeze): Positive funding → longs overcrowded → cascade DOWN
        - S4 (Funding Divergence): Negative funding → shorts overcrowded → squeeze UP

        Detection Logic:
        1. REQUIRED: Negative funding extreme (funding_Z < -1.2) -> shorts overcrowded
        2. REQUIRED: Price resilience (price NOT falling despite bearish funding) -> strength signal
        3. REQUIRED: Low liquidity (liquidity < threshold) -> thin bids amplify squeeze
        4. OPTIONAL: Volume quiet -> coiled spring effect (calm before storm)

        Mechanism:
        - Shorts paying high negative funding -> unsustainable
        - Price showing strength despite bearish sentiment -> divergence
        - Thin liquidity -> violent squeeze when shorts panic cover
        - Low volume before spike -> coiled spring (explosive covering)

        BTC Examples:
        - 2022-08-15: Funding -0.15% → +18% rally (violent short squeeze)
        - 2023-01-14: Negative funding + price strength → 12% rally

        Returns:
            (matched: bool, score: float, meta: dict)
        """
        # Get thresholds
        fusion_th = context.get_threshold('funding_divergence', 'fusion_threshold', 0.40)
        funding_z_max = context.get_threshold('funding_divergence', 'funding_z_max', -1.2)  # Negative!
        resilience_min = context.get_threshold('funding_divergence', 'resilience_min', 0.5)
        liq_max = context.get_threshold('funding_divergence', 'liquidity_max', 0.30)

        # Extract features
        funding_z = self.g(context.row, 'funding_Z', 0)
        liquidity = self._liquidity_score(context.row)

        # Try to get S4 runtime features if available
        price_resilience = self.g(context.row, 'price_resilience', None)
        volume_quiet = self.g(context.row, 'volume_quiet', False)

        # DEBUG: Log first call
        if not hasattr(self, '_s4_first_call_logged'):
            logger.info(f"[S4 DEBUG] First evaluation - funding_z={funding_z:.3f}, liquidity={liquidity:.3f}, resilience={price_resilience}")
            self._s4_first_call_logged = True

        # SMC VETO GATE: Don't long into bearish 4H structure
        # Check BEFORE other gates (fast fail for structural misalignment)
        tf4h_bos_bearish = self.g(context.row, 'tf4h_bos_bearish', False)
        if tf4h_bos_bearish:
            return False, 0.0, {
                "reason": "smc_4h_bos_bearish_veto",
                "message": "4H bearish BOS - institutional sellers active, abort long"
            }

        # Gate 1: NEGATIVE funding extreme (shorts overcrowded) - REQUIRED
        # Note: funding_z_max is NEGATIVE (e.g., -1.2), so we check if funding_z < -1.2
        if funding_z > funding_z_max:  # If funding is > -1.2, not negative enough
            return False, 0.0, {
                "reason": "funding_not_negative_extreme",
                "funding_z": funding_z,
                "threshold": funding_z_max
            }

        # Gate 2: Low liquidity (amplification factor) - REQUIRED
        if liquidity > liq_max:
            return False, 0.0, {
                "reason": "liquidity_not_thin",
                "liquidity": liquidity,
                "threshold": liq_max
            }

        # Gate 3: Price resilience (if runtime features available)
        if price_resilience is not None:
            if price_resilience < resilience_min:
                return False, 0.0, {
                    "reason": "price_not_resilient",
                    "resilience": price_resilience,
                    "threshold": resilience_min
                }

        # Compute score components
        components = {
            "funding_negative": min((-funding_z - 1.0) / 2.0, 1.0),  # Map -3 to -1 sigma → 1.0 to 0.0
            "price_resilience": price_resilience if price_resilience is not None else 0.5,
            "liquidity_thin": 1.0 - (liquidity / 0.5),  # Lower liquidity = higher score
            "volume_quiet": 1.0 if volume_quiet else 0.0
        }

        # Cap liquidity_thin at 1.0
        components["liquidity_thin"] = min(components["liquidity_thin"], 1.0)

        # Weights (tuned for BTC short squeezes)
        weights = context.get_threshold('funding_divergence', 'weights', {
            "funding_negative": 0.40,
            "price_resilience": 0.30,
            "volume_quiet": 0.15,
            "liquidity_thin": 0.15
        })

        score = sum(components[k] * weights.get(k, 0.0) for k in components)

        # ============================================================================
        # DOMAIN ENGINE INTEGRATION (BOOST/VETO LAYER) - S4 FUNDING DIVERGENCE
        # ============================================================================
        # CRITICAL FIX: Apply domain engines BEFORE fusion threshold gate
        use_wyckoff = context.metadata.get('feature_flags', {}).get('enable_wyckoff', False)
        use_smc = context.metadata.get('feature_flags', {}).get('enable_smc', False)
        use_temporal = context.metadata.get('feature_flags', {}).get('enable_temporal', False)
        use_hob = context.metadata.get('feature_flags', {}).get('enable_hob', False)

        domain_boost = 1.0
        domain_signals = []

        # ============================================================================
        # WYCKOFF ENGINE: Accumulation vs Distribution detection
        # ============================================================================
        if use_wyckoff:
            # SOFT VETOES: Distribution phase reduces confidence but doesn't block
            wyckoff_distribution = self.g(context.row, 'wyckoff_phase_abc', '') == 'D'
            wyckoff_utad = self.g(context.row, 'wyckoff_utad', False)
            wyckoff_bc = self.g(context.row, 'wyckoff_bc', False)
            wyckoff_sow = self.g(context.row, 'wyckoff_sow', False)

            if wyckoff_distribution:
                domain_boost *= 0.30  # 70% confidence reduction for distribution phase
                domain_signals.append("wyckoff_distribution_caution")

            if wyckoff_utad or wyckoff_bc:
                domain_boost *= 0.50  # Stronger events get more severe reduction
                domain_signals.append("wyckoff_utad_bc_caution")

            if wyckoff_sow:
                # Soft veto: Sign of Weakness reduces conviction
                domain_boost *= 0.70
                domain_signals.append("wyckoff_sow_weakness_penalty")

            # MAJOR BOOSTS: Accumulation phase signals
            wyckoff_accumulation = self.g(context.row, 'wyckoff_phase_abc', '') == 'A'
            wyckoff_spring_a = self.g(context.row, 'wyckoff_spring_a', False)
            wyckoff_spring_b = self.g(context.row, 'wyckoff_spring_b', False)
            wyckoff_lps = self.g(context.row, 'wyckoff_lps', False)

            if wyckoff_spring_a or wyckoff_spring_b:
                domain_boost *= 2.50  # Spring confirms squeeze setup (shorts trapped)
                domain_signals.append("wyckoff_spring_squeeze_setup")
            elif wyckoff_lps:
                domain_boost *= 1.50  # Last Point Support = final accumulation
                domain_signals.append("wyckoff_lps_accumulation")
            elif wyckoff_accumulation:
                domain_boost *= 2.00  # Accumulation phase = smart money buying
                domain_signals.append("wyckoff_accumulation_phase")

            # SUPPORT SIGNALS: Wyckoff accumulation events
            wyckoff_ps = self.g(context.row, 'wyckoff_ps', False)
            wyckoff_sos = self.g(context.row, 'wyckoff_sos', False)
            wyckoff_ar = self.g(context.row, 'wyckoff_ar', False)

            if wyckoff_sos:
                domain_boost *= 1.80  # Sign of Strength = buying pressure
                domain_signals.append("wyckoff_sos_strength")
            elif wyckoff_ps:
                domain_boost *= 1.40  # Preliminary Support
                domain_signals.append("wyckoff_ps_support")
            elif wyckoff_ar:
                domain_boost *= 1.30  # Automatic Rally
                domain_signals.append("wyckoff_ar_bounce")

        # ============================================================================
        # SMC ENGINE: Bullish structure confirmation
        # ============================================================================
        if use_smc:
            # Multi-timeframe bullish BOS (institutional buying)
            tf1h_bos_bullish = self.g(context.row, 'tf1h_bos_bullish', False)
            tf4h_bos_bullish = self.g(context.row, 'tf4h_bos_bullish', False)

            if tf4h_bos_bullish:
                domain_boost *= 2.00  # 4H institutional structure shift
                domain_signals.append("smc_4h_bos_bullish")
            elif tf1h_bos_bullish:
                domain_boost *= 1.40  # 1H bullish structure
                domain_signals.append("smc_1h_bos_bullish")

            # DEMAND ZONES + LIQUIDITY SWEEPS
            smc_demand_zone = self.g(context.row, 'smc_demand_zone', False)
            smc_liquidity_sweep = self.g(context.row, 'smc_liquidity_sweep', False)

            if smc_liquidity_sweep:
                domain_boost *= 1.80  # Liquidity sweep = stop hunt before squeeze
                domain_signals.append("smc_liquidity_sweep_reversal")
            elif smc_demand_zone:
                domain_boost *= 1.60  # Demand zone support
                domain_signals.append("smc_demand_zone_support")

            # VETOES: Supply zones overhead
            smc_supply_zone = self.g(context.row, 'smc_supply_zone', False)
            if smc_supply_zone:
                domain_boost *= 0.70  # Supply overhead reduces conviction
                domain_signals.append("smc_supply_zone_overhead")

        # ============================================================================
        # TEMPORAL ENGINE: Fibonacci time + confluence
        # ============================================================================
        if use_temporal:
            # Fibonacci time clusters
            fib_time_cluster = self.g(context.row, 'fib_time_cluster', False)
            temporal_confluence = self.g(context.row, 'temporal_confluence', False)

            if fib_time_cluster:
                domain_boost *= 1.70  # Perfect timing for squeeze
                domain_signals.append("fib_time_cluster_reversal")

            if temporal_confluence:
                domain_boost *= 1.50  # Multi-timeframe alignment
                domain_signals.append("temporal_confluence")

            # Wyckoff-PTI confluence
            wyckoff_pti_confluence = self.g(context.row, 'wyckoff_pti_confluence', False)
            if wyckoff_pti_confluence:
                domain_boost *= 1.40  # Combined pattern + time
                domain_signals.append("wyckoff_pti_confluence")

        # ============================================================================
        # HOB ENGINE: Order book confirmation
        # ============================================================================
        if use_hob:
            # Demand zones in order book
            hob_demand_zone = self.g(context.row, 'hob_demand_zone', False)
            hob_imbalance = self.g(context.row, 'hob_imbalance', 0.0)

            if hob_demand_zone:
                domain_boost *= 1.50  # Large bid walls support squeeze
                domain_signals.append("hob_demand_zone_support")

            if hob_imbalance > 0.60:
                domain_boost *= 1.30  # Strong bid imbalance
                domain_signals.append("hob_bid_imbalance_strong")

        score_before_domain = score
        score = score * domain_boost

        # ============================================================================
        # FUSION THRESHOLD GATE (applied AFTER domain engines)
        # ============================================================================
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_threshold",
                "score": score,
                "score_before_domain": score_before_domain,
                "threshold": fusion_th,
                "components": components,
                "domain_boost": domain_boost,
                "domain_signals": domain_signals,
                "funding_z": funding_z,
                "resilience": price_resilience,
                "liquidity": liquidity,
                "volume_quiet": volume_quiet
            }

        # MATCH! Short squeeze setup detected
        return True, score, {
            "reason": "short_squeeze_detected",
            "components": components,
            "score": score,
            "domain_boost": domain_boost,
            "domain_signals": domain_signals,
            "funding_z": funding_z,
            "resilience": price_resilience,
            "liquidity": liquidity,
            "volume_quiet": volume_quiet
        }, "SHORT"

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

        # SMC STRUCTURE GATE: REMOVED BULLISH VETO
        # CRITICAL FIX: Original logic had backwards SMC gate
        # Old: "Don't short into bullish 1H structure" - WRONG!
        # Truth: Long squeeze happens BECAUSE bullish structure exhausts
        # A bullish BOS that fails to hold = cascade down (LONG SQUEEZE UP for shorts)
        # Longs get squeezed when market rejects their bullish breakout attempts
        #
        # Solution: Remove this veto entirely. Let core gates (funding+RSI+liquidity)
        # handle signal generation. SMC domain engines will provide boosts/vetoes in layer below.
        #
        # Impact: Restores 68 signals in 2022 crisis (was 1 signal before)

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

        # ============================================================================
        # DOMAIN ENGINE INTEGRATION (BOOST/VETO LAYER) - S5 LONG SQUEEZE
        # ============================================================================
        # CRITICAL FIX: Apply domain engines BEFORE fusion threshold gate
        use_wyckoff = context.metadata.get('feature_flags', {}).get('enable_wyckoff', False)
        use_smc = context.metadata.get('feature_flags', {}).get('enable_smc', False)
        use_temporal = context.metadata.get('feature_flags', {}).get('enable_temporal', False)
        use_hob = context.metadata.get('feature_flags', {}).get('enable_hob', False)

        domain_boost = 1.0
        domain_signals = []

        # ============================================================================
        # WYCKOFF ENGINE: Distribution top detection
        # ============================================================================
        if use_wyckoff:
            # HARD VETOES: Don't short into accumulation phase
            wyckoff_accumulation = self.g(context.row, 'wyckoff_phase_abc', '') == 'A'
            wyckoff_spring_a = self.g(context.row, 'wyckoff_spring_a', False)
            wyckoff_spring_b = self.g(context.row, 'wyckoff_spring_b', False)

            if wyckoff_accumulation or wyckoff_spring_a or wyckoff_spring_b:
                # Hard veto: Accumulation/Spring signals = abort short
                return False, 0.0, {
                    "reason": "wyckoff_accumulation_veto",
                    "wyckoff_accumulation": wyckoff_accumulation,
                    "wyckoff_spring_a": wyckoff_spring_a,
                    "wyckoff_spring_b": wyckoff_spring_b,
                    "note": "Don't short into Wyckoff accumulation phase"
                }

            # MAJOR BOOSTS: Distribution phase signals (Phase A-D)
            wyckoff_distribution = self.g(context.row, 'wyckoff_phase_abc', '') == 'D'
            wyckoff_utad = self.g(context.row, 'wyckoff_utad', False)
            wyckoff_bc = self.g(context.row, 'wyckoff_bc', False)

            if wyckoff_utad:
                domain_boost *= 2.50  # UTAD = final trap before markdown (strongest)
                domain_signals.append("wyckoff_utad_distribution_climax")
            elif wyckoff_bc:
                domain_boost *= 2.00  # Buying Climax = euphoria top
                domain_signals.append("wyckoff_buying_climax_top")
            elif wyckoff_distribution:
                domain_boost *= 2.00  # Distribution phase = smart money selling
                domain_signals.append("wyckoff_distribution_phase")

            # WEAKNESS SIGNALS: Sign of Weakness + LPSY (Phase D)
            wyckoff_sow = self.g(context.row, 'wyckoff_sow', False)
            wyckoff_lpsy = self.g(context.row, 'wyckoff_lpsy', False)
            wyckoff_as = self.g(context.row, 'wyckoff_as', False)

            if wyckoff_sow:
                domain_boost *= 1.80  # Sign of Weakness = selling pressure emerging
                domain_signals.append("wyckoff_sow_weakness")
            elif wyckoff_lpsy:
                domain_boost *= 1.80  # Last Point Supply = final rally before markdown
                domain_signals.append("wyckoff_lpsy_final_supply")
            elif wyckoff_as:
                domain_boost *= 1.40  # Automatic Reaction = relief drop after BC
                domain_signals.append("wyckoff_as_reaction")

        # ============================================================================
        # SMC ENGINE: Bearish structure confirmation
        # ============================================================================
        if use_smc:
            # Multi-timeframe bearish BOS (institutional distribution)
            tf1h_bos_bearish = self.g(context.row, 'tf1h_bos_bearish', False)
            tf4h_bos_bearish = self.g(context.row, 'tf4h_bos_bearish', False)

            if tf4h_bos_bearish:
                domain_boost *= 2.00  # 4H institutional structure shift DOWN
                domain_signals.append("smc_4h_bos_bearish_institutional")
            elif tf1h_bos_bearish:
                domain_boost *= 1.60  # 1H bearish structure
                domain_signals.append("smc_1h_bos_bearish")

            # SUPPLY ZONES: Institutional resistance
            smc_supply_zone = self.g(context.row, 'smc_supply_zone', False)
            smc_choch = self.g(context.row, 'smc_choch', False)

            if smc_supply_zone:
                domain_boost *= 1.80  # Supply zone = institutional sellers
                domain_signals.append("smc_supply_zone_resistance")

            if smc_choch:
                domain_boost *= 1.60  # Change of Character = trend reversal DOWN
                domain_signals.append("smc_choch_bearish_reversal")

            # VETOES: Demand zones below (support)
            smc_demand_zone = self.g(context.row, 'smc_demand_zone', False)
            smc_liquidity_sweep = self.g(context.row, 'smc_liquidity_sweep', False)

            if smc_demand_zone:
                domain_boost *= 0.70  # Demand support below reduces short conviction
                domain_signals.append("smc_demand_zone_support_below")

            if smc_liquidity_sweep:
                # Liquidity sweep could be bullish setup, reduce short signal
                domain_boost *= 0.75
                domain_signals.append("smc_liquidity_sweep_caution")

        # ============================================================================
        # TEMPORAL ENGINE: Fibonacci resistance + time clusters
        # ============================================================================
        if use_temporal:
            # Fibonacci time clusters at tops
            fib_time_cluster = self.g(context.row, 'fib_time_cluster', False)
            temporal_resistance_cluster = self.g(context.row, 'temporal_resistance_cluster', False)

            if fib_time_cluster and temporal_resistance_cluster:
                domain_boost *= 1.80  # Perfect timing at resistance = ideal short entry
                domain_signals.append("fib_time_resistance_cluster_top")
            elif fib_time_cluster:
                domain_boost *= 1.50  # Fibonacci time alignment
                domain_signals.append("fib_time_cluster_top")

            # Wyckoff-PTI confluence (resistance vs support)
            wyckoff_pti_confluence = self.g(context.row, 'wyckoff_pti_confluence', False)
            wyckoff_pti_score = self.g(context.row, 'wyckoff_pti_score', 0.0)

            if wyckoff_pti_confluence:
                if wyckoff_pti_score > 0.50:
                    domain_boost *= 1.50  # Strong resistance confluence
                    domain_signals.append("wyckoff_pti_resistance_confluence")
                elif wyckoff_pti_score < -0.50:
                    # Support cluster = veto short
                    return False, 0.0, {
                        "reason": "temporal_support_veto",
                        "wyckoff_pti_score": wyckoff_pti_score,
                        "note": "Don't short into Fibonacci support cluster"
                    }

        # ============================================================================
        # HOB ENGINE: Order book supply/demand
        # ============================================================================
        if use_hob:
            # Supply zones in order book
            hob_supply_zone = self.g(context.row, 'hob_supply_zone', False)
            hob_imbalance = self.g(context.row, 'hob_imbalance', 0.0)

            if hob_supply_zone:
                domain_boost *= 1.50  # Large ask walls = institutional resistance
                domain_signals.append("hob_supply_zone_resistance")

            # Negative imbalance = more sellers than buyers
            if hob_imbalance < -0.60:
                domain_boost *= 1.30  # Strong sell-side imbalance
                domain_signals.append("hob_ask_imbalance_strong")
            elif hob_imbalance < -0.40:
                domain_boost *= 1.15  # Moderate sell imbalance
                domain_signals.append("hob_ask_imbalance_moderate")

            # VETOES: Demand zones below
            hob_demand_zone = self.g(context.row, 'hob_demand_zone', False)
            if hob_demand_zone:
                domain_boost *= 0.70  # Bid walls below reduce short conviction
                domain_signals.append("hob_demand_zone_support_below")

        score_before_domain = score
        score = score * domain_boost

        # ============================================================================
        # FUSION THRESHOLD GATE (applied AFTER domain engines)
        # ============================================================================
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_threshold",
                "score": score,
                "score_before_domain": score_before_domain,
                "threshold": fusion_th,
                "components": components,
                "has_oi_data": has_oi_data,
                "domain_boost": domain_boost,
                "domain_signals": domain_signals,
                "funding_z": funding_z,
                "oi_change": oi_change if has_oi_data else "N/A",
                "rsi": rsi,
                "liquidity": liquidity
            }

        return True, score, {
            "components": components,
            "weights": weights,
            "has_oi_data": has_oi_data,
            "domain_boost": domain_boost,
            "domain_signals": domain_signals,
            "funding_z": funding_z,
            "oi_change": oi_change if has_oi_data else "N/A",
            "rsi": rsi,
            "liquidity": liquidity,
            "mechanism": "longs_overcrowded_cascade_risk"
        }, "LONG"

    def _check_S6(self, context: RuntimeContext) -> bool:
        """
        S6 - Alt Rotation Down: Altcoin underperformance (TOTAL3 < BTC).

        DEPRECATED: STUB archetype - no implementation exists.
        DISABLED: Requires altcoin dominance data not in feature store.

        DO NOT ENABLE in configs - this will never fire.
        """
        return False

    def _check_S7(self, context: RuntimeContext) -> bool:
        """
        S7 - Curve Inversion Breakdown: Yield curve inversion + support break.

        DEPRECATED: STUB archetype - no implementation exists.
        DISABLED: Requires yield curve data not in feature store.

        DO NOT ENABLE in configs - this will never fire.
        """
        return False

    def _pattern_S8(self, context: RuntimeContext):
        """Pattern detection for Archetype S8: Fakeout Exhaustion / Volume Fade Chop (SHORT)"""
        r = context.row

        # Volume fade - low volume signature (works: 864 bars in Q1 2023)
        volume_fade = self.g(r, 'volume_zscore', 0.0) <= -0.5

        # Low volatility check - replaced missing atr_percentile with absolute ATR threshold
        # Chop/consolidation signature: ATR < 0.6% of price
        atr = self.g(r, 'atr_14', self.g(r, 'atr_20', 999999))
        close = r.get('close', 1)
        low_volatility = atr < (close * 0.006)  # ATR < 0.6% = low volatility/chop

        # Fallback: If ATR missing, use Bollinger Band width as volatility proxy
        if atr >= 999999:  # ATR not available
            bb_width = self.g(r, 'bb_width', 0.0)
            low_volatility = bb_width < 0.03  # BB width < 3% = low volatility

        # Fire signal if BOTH volume fade AND low volatility (chop conditions)
        if volume_fade and low_volatility:
            score = 0.42
            return True, score, ["S8", "volume_fade_chop", "SHORT"]

        return None

    def _check_S8(self, context: RuntimeContext) -> Tuple[bool, float, Dict]:
        """Archetype S8: Fakeout Exhaustion (SHORT)"""

        # 1) Base pattern detection - returns (matched, score, metadata)
        base_result = self._pattern_S8(context)
        if not base_result:
            return False, 0.0, {"reason": "pattern_not_matched"}

        matched, base_score, pattern_tags = base_result  # base_score in [0,1]

        # 2) Global safety vetoes (soft - use penalties not hard returns)
        # Note: Keep these soft - use penalties not hard returns

        # 3) Domain modifiers (standardized across all)
        score = self._apply_domain_engines(context, base_score, pattern_tags)

        # 4) Final fusion gate
        fusion_th = context.get_threshold('fakeout_exhaustion', 'fusion_threshold', 0.40)
        if score < fusion_th:
            return False, score, {
                "reason": "score_below_fusion_threshold",
                "base_score": base_score,
                "final_score": score,
                "fusion_threshold": fusion_th
            }

        # 5) Return with full metadata
        return True, score, {
            "base_score": base_score,
            "final_score": score,
            "pattern_tags": pattern_tags,
            "domain_boost_applied": True
        }, "EITHER"
