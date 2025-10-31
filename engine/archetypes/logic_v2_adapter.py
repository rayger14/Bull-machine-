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
from typing import Tuple, Optional
from engine.runtime.context import RuntimeContext

logger = logging.getLogger(__name__)


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
        logging.getLogger(__name__).info(f"[ArchetypeLogic] Using {self.CLASS_VERSION}")

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

    def detect(self, ctx: RuntimeContext) -> Tuple[Optional[str], float, float]:
        """
        Detect which archetype (if any) matches the current context.

        PR#6B: Now accepts RuntimeContext with regime-aware thresholds from ThresholdPolicy.

        Args:
            ctx: RuntimeContext with row, regime state, and resolved thresholds

        Returns:
            (archetype_name_or_None, fusion_score, liquidity_score)
        """
        if not self.use_archetypes:
            return None, 0.0, 0.0

        # Global precheck: liquidity >= min_threshold
        liquidity_score = self._liquidity_score(ctx.row)
        fusion_score = self._fusion(ctx.row)

        # Check global liquidity floor
        if liquidity_score < self.min_liquidity:
            return None, fusion_score, liquidity_score

        # Check archetypes in priority order: A, B, C, K, H, L, F, D, G, E, M
        if self.enabled['A'] and self._check_A(ctx):
            return 'trap_reversal', fusion_score, liquidity_score

        if self.enabled['B'] and self._check_B(ctx):
            return 'order_block_retest', fusion_score, liquidity_score

        if self.enabled['C'] and self._check_C(ctx):
            return 'fvg_continuation', fusion_score, liquidity_score

        if self.enabled['K'] and self._check_K(ctx):
            return 'wick_trap', fusion_score, liquidity_score

        if self.enabled['H'] and self._check_H(ctx):
            return 'trap_within_trend', fusion_score, liquidity_score

        if self.enabled['L'] and self._check_L(ctx):
            return 'volume_exhaustion', fusion_score, liquidity_score

        if self.enabled['F'] and self._check_F(ctx):
            return 'expansion_exhaustion', fusion_score, liquidity_score

        if self.enabled['D'] and self._check_D(ctx):
            return 'failed_continuation', fusion_score, liquidity_score

        if self.enabled['G'] and self._check_G(ctx):
            return 're_accumulate', fusion_score, liquidity_score

        if self.enabled['E'] and self._check_E(ctx):
            return 'liquidity_compression', fusion_score, liquidity_score

        if self.enabled['M'] and self._check_M(ctx):
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
        """
        # Create minimal context without regime data
        ctx = RuntimeContext(
            ts=row.name if hasattr(row, 'name') else index,
            row=row,
            regime_probs={'neutral': 1.0},
            regime_label='neutral',
            adapted_params={},
            thresholds={}  # Empty thresholds - methods will use defaults
        )
        return self.detect(ctx)

    # =======================================================================
    # Individual Archetype Checks (Using Safe Getters)
    # =======================================================================

    def _check_A(self, ctx: RuntimeContext) -> bool:
        """Archetype A: Trap Reversal (PTI spring/UTAD + displacement)."""
        # Get regime-aware fusion threshold from policy
        fusion_th = ctx.get_threshold('spring', 'fusion', 0.33)

        # Get features from context
        pti_trap = self.g(ctx.row, "pti_trap_type", '')
        if not pti_trap or pti_trap not in ['spring', 'utad']:
            return False

        pti_score = self.g(ctx.row, "pti_score", 0.0)
        disp = self.g(ctx.row, "boms_disp", 0.0)
        atr = max(self.g(ctx.row, "atr", 0.0), 1e-9)
        fusion = ctx.row.get('fusion_score', 0.0)

        # Structural thresholds (static)
        return (pti_score >= 0.40 and
                disp >= 0.80 * atr and
                fusion >= fusion_th)

    def _check_B(self, ctx: RuntimeContext) -> bool:
        """Archetype B: Order Block Retest (BOS + BOMS + Wyckoff)."""
        # Get regime-aware fusion threshold from policy
        fusion_th = ctx.get_threshold('order_block_retest', 'fusion', 0.374)

        # Get features from context
        bos_bullish = self.g(ctx.row, "bos_bullish", 0)
        boms_str = self.g(ctx.row, "boms_strength", 0.0)
        wyckoff = self.g(ctx.row, "wyckoff_score", 0.0)
        fusion = ctx.row.get('fusion_score', 0.0)

        # Structural thresholds (static)
        return (bos_bullish and
                boms_str >= 0.30 and
                wyckoff >= 0.35 and
                fusion >= fusion_th)

    def _check_C(self, ctx: RuntimeContext) -> bool:
        """Archetype C: FVG Continuation (displacement + momentum)."""
        # Get regime-aware fusion threshold from policy
        fusion_th = ctx.get_threshold('wick_trap', 'fusion', 0.42)

        # Get features from context
        fvg_4h = self.g(ctx.row, "fvg_present_4h", 0)
        if not fvg_4h:
            return False

        disp = self.g(ctx.row, "boms_disp", 0.0)
        atr = max(self.g(ctx.row, "atr", 0.0), 1e-9)
        momentum = self._momentum_score(ctx.row)
        tf4h_fusion = self.g(ctx.row, "fusion_score", 0.0)
        fusion = ctx.row.get('fusion_score', 0.0)

        # Structural thresholds (static)
        return (disp >= 1.00 * atr and
                momentum >= 0.45 and
                tf4h_fusion >= 0.25 and
                fusion >= fusion_th)

    def _check_D(self, ctx: RuntimeContext) -> bool:
        """Archetype D: Failed Continuation (FVG + weak RSI)."""
        # Get regime-aware fusion threshold from policy
        fusion_th = ctx.get_threshold('failed_continuation', 'fusion', 0.42)

        # Get features from context
        fvg_1h = self.g(ctx.row, "fvg_present_1h", 0)
        if not fvg_1h:
            return False

        rsi = self.g(ctx.row, "rsi", 50.0)
        fusion = ctx.row.get('fusion_score', 0.0)

        # Structural thresholds (static)
        return (rsi <= 50 and
                fusion >= fusion_th)

    def _check_E(self, ctx: RuntimeContext) -> bool:
        """Archetype E: Liquidity Compression (low ATR + volume cluster)."""
        # Get regime-aware fusion threshold from policy
        fusion_th = ctx.get_threshold('volume_exhaustion', 'fusion', 0.35)

        # Get features from context
        atr_pct = self.g(ctx.row, "atr_percentile", 0.5)
        vol_z = self.g(ctx.row, "vol_z", 0.0)
        fusion = ctx.row.get('fusion_score', 0.0)

        # Volume clustering: check if vol_z is moderate
        vol_cluster = 1.0 if 0.5 <= vol_z <= 1.5 else 0.0

        # Structural thresholds (static)
        return (atr_pct <= 0.25 and
                vol_cluster >= 0.70 and
                fusion >= fusion_th)

    def _check_F(self, ctx: RuntimeContext) -> bool:
        """Archetype F: Expansion Exhaustion (extreme RSI + high ATR + vol spike)."""
        # Get regime-aware fusion threshold from policy
        fusion_th = ctx.get_threshold('exhaustion_reversal', 'fusion', 0.38)

        # Get features from context
        rsi = self.g(ctx.row, "rsi", 50.0)
        atr_pct = self.g(ctx.row, "atr_percentile", 0.5)
        vol_z = self.g(ctx.row, "vol_z", 0.0)
        fusion = ctx.row.get('fusion_score', 0.0)

        # Structural thresholds (static)
        return (rsi >= 78 and
                atr_pct >= 0.90 and
                vol_z >= 1.0 and
                fusion >= fusion_th)

    def _check_G(self, ctx: RuntimeContext) -> bool:
        """Archetype G: Re-Accumulate Base (BOMS strength + high liquidity)."""
        # Get regime-aware fusion threshold from policy
        fusion_th = ctx.get_threshold('liquidity_sweep', 'fusion', 0.40)

        # Get features from context
        boms_str = self.g(ctx.row, "boms_strength", 0.0)
        liq = self._liquidity_score(ctx.row)
        fusion = ctx.row.get('fusion_score', 0.0)

        # Structural thresholds (static)
        return (boms_str >= 0.40 and
                liq >= 0.40 and
                fusion >= fusion_th)

    def _check_H(self, ctx: RuntimeContext) -> bool:
        """Archetype H: Trap Within Trend (ADX trend + liquidity drop)."""
        # Get regime-aware fusion threshold from policy
        fusion_th = ctx.get_threshold('momentum_continuation', 'fusion', 0.35)

        # Get features from context
        adx = self.g(ctx.row, "adx", 0.0)
        liq = self._liquidity_score(ctx.row)
        fusion = ctx.row.get('fusion_score', 0.0)

        # Liquidity drop check (simplified - ideally compare to prev)
        liq_drop = liq < 0.30  # Proxy for drop

        # Structural thresholds (static)
        return (adx >= 25 and
                liq_drop and
                fusion >= fusion_th)

    def _check_K(self, ctx: RuntimeContext) -> bool:
        """Archetype K: Wick Trap / Moneytaur (ADX + liquidity + wicks)."""
        # Get regime-aware fusion threshold from policy
        fusion_th = ctx.get_threshold('trap_within_trend', 'fusion', 0.36)

        # Get features from context
        adx = self.g(ctx.row, "adx", 0.0)
        liq = self._liquidity_score(ctx.row)
        fusion = ctx.row.get('fusion_score', 0.0)

        # Wick detection (simplified - could check high/low ranges)
        # For now, just check ADX + liquidity

        # Structural thresholds (static)
        return (adx >= 25 and
                liq >= 0.30 and
                fusion >= fusion_th)

    def _check_L(self, ctx: RuntimeContext) -> bool:
        """Archetype L: Volume Exhaustion / Zeroika (vol spike + extreme RSI)."""
        # Get regime-aware fusion threshold from policy
        fusion_th = ctx.get_threshold('retest_cluster', 'fusion', 0.38)

        # Get features from context
        vol_z = self.g(ctx.row, "vol_z", 0.0)
        rsi = self.g(ctx.row, "rsi", 50.0)
        fusion = ctx.row.get('fusion_score', 0.0)

        # Structural thresholds (static)
        return (vol_z >= 1.0 and
                rsi >= 70 and
                fusion >= fusion_th)

    def _check_M(self, ctx: RuntimeContext) -> bool:
        """Archetype M: Ratio Coil Break / Wyckoff Insider (low ATR + near POC + BOMS)."""
        # Get regime-aware fusion threshold from policy
        fusion_th = ctx.get_threshold('confluence_breakout', 'fusion', 0.35)

        # Get features from context
        atr_pct = self.g(ctx.row, "atr_percentile", 0.5)
        poc_dist = self.g(ctx.row, "poc_dist", 1.0)
        boms_str = self.g(ctx.row, "boms_strength", 0.0)
        fusion = ctx.row.get('fusion_score', 0.0)

        # Structural thresholds (static)
        return (atr_pct <= 0.30 and
                poc_dist <= 0.50 and
                boms_str >= 0.40 and
                fusion >= fusion_th)
