#!/usr/bin/env python3
"""
Runtime Liquidity Scorer

Computes a bounded liquidity score in [0, 1] per bar using:
1. Strength/Intent (S): BOMS strength + displacement proxy
2. Structure Context (C): FVG quality + BOS freshness
3. Liquidity Conditions (L): Volume z-score + spread proxy
4. Positioning & Timing (P): EQ discount/premium + ATR regime + time-of-day

Final score composition:
    liquidity = 0.35*S + 0.30*C + 0.20*L + 0.15*P + 0.08*HTF_boost

Target distribution (after calibration):
    median ≈ 0.45–0.55
    p75 ≈ 0.68–0.75
    p90 ≈ 0.80–0.90

Architecture:
- Pure function (no side effects)
- No lookahead (all HTF values from closed bars)
- No feature store rebuild required
- Configurable caps and weights via cfg dict

Author: PR#4 - Runtime Intelligence
"""

from __future__ import annotations
import math
from typing import Dict, Any, Optional


def _clip01(x: float | None) -> float:
    """
    Clip value to [0, 1] range with safety for None/NaN.

    Args:
        x: Input value (may be None or NaN)

    Returns:
        Value clipped to [0, 1], or 0.0 if invalid
    """
    try:
        if x is None or math.isnan(x):
            return 0.0
        return max(0.0, min(1.0, float(x)))
    except (ValueError, TypeError):
        return 0.0


def _sigmoid01(z: float, k: float = 1.0) -> float:
    """
    Map real line to (0, 1) using sigmoid function.

    Args:
        z: Input value (real number)
        k: Steepness parameter (default 1.0)

    Returns:
        Sigmoid output in (0, 1)

    Example:
        >>> _sigmoid01(0.0)  # Returns ~0.5 (neutral)
        0.5
        >>> _sigmoid01(2.0)  # Returns ~0.88 (high)
        0.8807970779778823
    """
    try:
        return 1.0 / (1.0 + math.exp(-k * z))
    except OverflowError:
        # Very large negative z → 0, very large positive z → 1
        return 0.0 if z < 0 else 1.0


def compute_liquidity_score(
    ctx: Dict[str, Any],
    side: str,
    cfg: Optional[Dict[str, Any]] = None
) -> float:
    """
    Compute runtime liquidity score in [0, 1].

    Args:
        ctx: Per-bar context dict with:
            - OHLCV: high, low, close
            - Features: tf1d_boms_strength, tf4h_boms_displacement,
                       fvg_quality (or fvg_present), tf4h_fusion_score
            - Runtime: volume_zscore, atr, fresh_bos_flag,
                      range_eq (or rolling_high/rolling_low),
                      tod_boost (time-of-day, optional)
        side: 'long' or 'short' (for discount/premium positioning)
        cfg: Optional config overrides:
            - disp_cap: displacement normalization cap (default 1.5)
            - spr_cap: spread ratio cap (default 0.01)
            - weights: dict with wS, wC, wL, wP (defaults sum to 1.0)

    Returns:
        Liquidity score in [0, 1]

    Example:
        >>> ctx = {
        ...     'close': 60000.0, 'high': 60500.0, 'low': 59800.0,
        ...     'tf1d_boms_strength': 0.6, 'tf4h_boms_displacement': 850.0,
        ...     'fvg_quality': 0.8, 'volume_zscore': 1.2, 'atr': 600.0,
        ...     'tf4h_fusion_score': 0.5
        ... }
        >>> score = compute_liquidity_score(ctx, 'long')
        >>> 0.0 <= score <= 1.0
        True

    Notes:
        - All missing/invalid values default to 0.0 or neutral fallbacks
        - No exceptions raised (defensive coding for backtesting)
        - Pillars are weighted to avoid redundancy with BOMS strength
    """
    cfg = cfg or {}

    # --- 1. Strength/Intent Pillar (S) ---
    # Primary: tf1d_boms_strength (already normalized 0-1)
    S = _clip01(ctx.get("tf1d_boms_strength", 0.0))

    # Secondary: displacement proxy (down-weighted to avoid double-count)
    disp_raw = float(ctx.get("tf4h_boms_displacement", 0.0) or 0.0)
    disp_cap = cfg.get("disp_cap", 1.5)  # Cap at ~1.5x typical displacement
    disp_norm = _clip01(disp_raw / max(1e-9, disp_cap))

    # Combine: 75% strength, 25% displacement
    S_star = 0.75 * S + 0.25 * disp_norm

    # --- 2. Structure Context Pillar (C) ---
    # FVG quality (0-1) or fallback to binary fvg_present
    fvg_q = ctx.get("fvg_quality")
    if fvg_q is None:
        fvg_q = 1.0 if ctx.get("fvg_present", False) else 0.0
    fvg_q = _clip01(fvg_q)

    # BOS freshness bonus (recent CHoCH/BOS within lookback)
    bos_fresh = 1.0 if ctx.get("fresh_bos_flag", False) else 0.0

    # Combine: FVG quality + small BOS freshness boost
    C = _clip01(fvg_q + 0.10 * bos_fresh)

    # --- 3. Liquidity Conditions Pillar (L) ---
    # Volume z-score mapped to (0, 1) via sigmoid
    # z=0 → 0.5 (neutral), z>0 → higher, z<0 → lower
    z = float(ctx.get("volume_zscore", 0.0) or 0.0)
    vol_score = _sigmoid01(z - 0.0, k=1.0)

    # Spread proxy: tighter spreads = better liquidity
    # Use candle spread if no bid/ask available
    hi = float(ctx.get("high", 0.0) or 0.0)
    lo = float(ctx.get("low", 0.0) or 0.0)
    cl = float(ctx.get("close", 0.0) or 0.0)

    spr = abs(hi - lo) / max(1e-9, abs(cl))  # Spread ratio
    spr_cap = cfg.get("spr_cap", 0.01)  # ~1% typical tight spread

    # Invert and squash: lower spread → higher score
    spread_score = _clip01(max(0.0, 1.0 - spr / max(1e-6, spr_cap)))

    # Combine: 70% volume, 30% spread
    L_star = _clip01(0.70 * vol_score + 0.30 * spread_score)

    # --- 4. Positioning & Timing Pillar (P) ---
    # EQ mid from rolling range (precomputed or fallback)
    eq = ctx.get("range_eq")
    if eq is None:
        rolling_hi = float(ctx.get("rolling_high", hi) or hi)
        rolling_lo = float(ctx.get("rolling_low", lo) or lo)
        eq = (rolling_hi + rolling_lo) / 2.0

    # Discount/premium positioning (directional)
    in_discount = 1.0 if (
        (side == "long" and cl <= eq) or
        (side == "short" and cl >= eq)
    ) else 0.0

    # ATR regime adjustment: prefer mid-regime (not too hot/cold)
    atr = float(ctx.get("atr", 0.0) or 0.0)
    tr = abs(hi - lo)
    atr_regime = _clip01(tr / max(1e-9, (2.0 * atr)))  # ~0.5 is mid-regime

    # Penalize extremes: peak around 0.5, drop at 0.0 and 1.0
    atr_adj = max(0.0, 1.0 - abs(atr_regime - 0.5) * 1.6)

    # Time-of-day boost (optional, defaults to neutral 0.5)
    # Crypto: boost during US/EU overlap, stocks: boost during RTH peak
    tod_boost = _clip01(ctx.get("tod_boost", 0.5))

    # Combine: 50% positioning, 30% ATR regime, 20% time-of-day
    P = _clip01(0.50 * in_discount + 0.30 * atr_adj + 0.20 * tod_boost)

    # --- 5. Compose Base Score ---
    # Weights sum to 1.0 (configurable via cfg)
    wS = cfg.get("wS", 0.35)  # Strength/intent
    wC = cfg.get("wC", 0.30)  # Structure context
    wL = cfg.get("wL", 0.20)  # Liquidity conditions
    wP = cfg.get("wP", 0.15)  # Positioning & timing

    base = wS * S_star + wC * C + wL * L_star + wP * P

    # --- 6. HTF Fusion Nudge (Bounded) ---
    # Small boost from 4H fusion score (capped to avoid over-weighting)
    fusion4h = _clip01(ctx.get("tf4h_fusion_score", 0.0))
    htf_boost = 0.08 * fusion4h  # Max +0.08 boost

    # Final score (clipped to [0, 1])
    liquidity = _clip01(base + htf_boost)

    return liquidity


def compute_liquidity_telemetry(
    scores: list[float],
    window_size: int = 500
) -> Dict[str, float]:
    """
    Compute telemetry statistics for liquidity scores.

    Args:
        scores: List of liquidity scores from recent bars
        window_size: Number of recent scores to analyze

    Returns:
        Dict with:
            - p25, p50, p75, p90: Percentiles
            - nonzero_pct: Percentage of non-zero scores
            - mean: Average score

    Example:
        >>> scores = [0.4, 0.5, 0.6, 0.7, 0.3]
        >>> stats = compute_liquidity_telemetry(scores)
        >>> stats['p50']  # Median
        0.5
    """
    if not scores:
        return {
            'p25': 0.0, 'p50': 0.0, 'p75': 0.0, 'p90': 0.0,
            'nonzero_pct': 0.0, 'mean': 0.0
        }

    # Take last N scores
    recent = scores[-window_size:] if len(scores) > window_size else scores

    # Sort for percentile calculation
    sorted_scores = sorted(recent)
    n = len(sorted_scores)

    def percentile(p: float) -> float:
        """Get p-th percentile (p in [0, 100])"""
        k = (n - 1) * p / 100.0
        f = math.floor(k)
        c = math.ceil(k)
        if f == c:
            return sorted_scores[int(k)]
        d0 = sorted_scores[int(f)] * (c - k)
        d1 = sorted_scores[int(c)] * (k - f)
        return d0 + d1

    # Calculate statistics
    nonzero_count = sum(1 for s in recent if s > 0.0)
    nonzero_pct = (nonzero_count / n) * 100.0 if n > 0 else 0.0

    return {
        'p25': percentile(25),
        'p50': percentile(50),
        'p75': percentile(75),
        'p90': percentile(90),
        'nonzero_pct': nonzero_pct,
        'mean': sum(recent) / n if n > 0 else 0.0
    }


# Example usage / testing
if __name__ == '__main__':
    # Test case 1: Strong BOMS setup with good context
    ctx_strong = {
        'close': 60000.0,
        'high': 60500.0,
        'low': 59800.0,
        'tf1d_boms_strength': 0.8,
        'tf4h_boms_displacement': 1200.0,
        'fvg_quality': 0.9,
        'fresh_bos_flag': True,
        'volume_zscore': 1.5,
        'atr': 800.0,
        'tf4h_fusion_score': 0.7,
        'range_eq': 59900.0,
        'tod_boost': 0.7
    }

    score_strong = compute_liquidity_score(ctx_strong, 'long')
    print(f"Strong setup score: {score_strong:.3f}")
    print("  Expected: > 0.70 (high liquidity)")

    # Test case 2: Weak setup (low BOMS, no FVG, low volume)
    ctx_weak = {
        'close': 60000.0,
        'high': 60100.0,
        'low': 59900.0,
        'tf1d_boms_strength': 0.1,
        'tf4h_boms_displacement': 100.0,
        'fvg_present': False,
        'volume_zscore': -0.5,
        'atr': 800.0,
        'tf4h_fusion_score': 0.2,
        'range_eq': 59950.0
    }

    score_weak = compute_liquidity_score(ctx_weak, 'long')
    print(f"\nWeak setup score: {score_weak:.3f}")
    print("  Expected: < 0.40 (low liquidity)")

    # Test case 3: Missing fields (defensive fallbacks)
    ctx_minimal = {
        'close': 60000.0,
        'high': 60200.0,
        'low': 59800.0
    }

    score_minimal = compute_liquidity_score(ctx_minimal, 'long')
    print(f"\nMinimal context score: {score_minimal:.3f}")
    print("  Expected: 0.20–0.40 (neutral fallbacks)")

    # Test telemetry
    test_scores = [0.45, 0.52, 0.38, 0.61, 0.72, 0.44, 0.58, 0.66, 0.41, 0.55]
    stats = compute_liquidity_telemetry(test_scores)
    print(f"\nTelemetry (n={len(test_scores)}):")
    print(f"  p25: {stats['p25']:.3f}")
    print(f"  p50: {stats['p50']:.3f}")
    print(f"  p75: {stats['p75']:.3f}")
    print(f"  p90: {stats['p90']:.3f}")
    print(f"  Non-zero: {stats['nonzero_pct']:.1f}%")
