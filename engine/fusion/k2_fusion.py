"""
Knowledge v2.0 Meta-Fusion Engine

Combines multi-timeframe fusion scores into a single coherent signal that represents
market agreement across temporal scales and cross-asset correlation.

Architecture:
    tf1h_fusion_score: Local momentum / trap potential (fast, noisy)
    tf4h_fusion_score: Structural context / trend clarity (medium, reliable)
    tf1d_fusion_score: Macro trend bias (slow, strong filter)
    macro_correlation_score: Cross-asset regime (dampens/boosts conviction)

Returns:
    k2_fusion_score: Dynamic confluence score [0-1] with disagreement penalty
"""

import numpy as np
import logging
from typing import Dict, Optional

logger = logging.getLogger(__name__)


def compute_k2_fusion(row, weights: Optional[Dict[str, float]] = None) -> float:
    """
    Compute multi-timeframe meta-fusion score (Knowledge v2.0).

    Combines local (1H), context (4H), macro (1D), and cross-asset correlation signals
    into a single coherent score that reflects multi-timeframe agreement.

    Parameters:
        row: pandas Series with tf1h_fusion_score, tf4h_fusion_score,
             tf1d_fusion_score, macro_correlation_score
        weights: dict of weights for each timeframe
                 Default: {"1h": 0.35, "4h": 0.35, "1d": 0.2, "macro": 0.1}

    Returns:
        float (0-1): Dynamic fusion score with disagreement penalty

    Algorithm:
        1. Extract timeframe fusion scores from row
        2. Compute weighted mean (baseline fusion)
        3. Calculate disagreement penalty (std of values)
        4. Apply penalty to reduce confidence when timeframes conflict
        5. Clip to [0, 1] range

    Example:
        >>> row = pd.Series({
        ...     'tf1h_fusion_score': 0.7,
        ...     'tf4h_fusion_score': 0.65,
        ...     'tf1d_fusion_score': 0.6,
        ...     'macro_correlation_score': 0.5
        ... })
        >>> k2 = compute_k2_fusion(row)
        >>> print(f"K2 Fusion: {k2:.3f}")
        K2 Fusion: 0.612
    """
    # Default weights: prioritize 1H and 4H (operational timeframes),
    # with 1D as trend filter and macro as regime dampener
    if weights is None:
        weights = {
            "1h": 0.35,    # Local momentum, entry timing
            "4h": 0.35,    # Structural context, primary signal
            "1d": 0.20,    # Macro trend bias, directional filter
            "macro": 0.10  # Cross-asset correlation, regime damper
        }

    try:
        # Extract fusion scores from row
        vals = np.array([
            row.get("tf1h_fusion_score", np.nan),
            row.get("tf4h_fusion_score", np.nan),
            row.get("tf1d_fusion_score", np.nan),
            row.get("macro_correlation_score", np.nan)
        ])

        w = np.array(list(weights.values()))

        # Handle NaNs safely - if all values are missing, return neutral
        mask = ~np.isnan(vals)
        if not mask.any():
            logger.warning("All fusion components are NaN, returning neutral 0.5")
            return 0.5  # fallback neutral

        # Filter out NaN values and corresponding weights
        valid_vals = vals[mask]
        valid_weights = w[mask]

        # Normalize weights to sum to 1.0
        normalized_weights = valid_weights / valid_weights.sum()

        # Compute weighted mean (baseline fusion score)
        base = np.dot(valid_vals, normalized_weights)

        # Disagreement penalty: reduce confidence when timeframes conflict
        # High std → low agreement → reduce final score
        disagreement = np.std(valid_vals)

        # Penalty formula: penalty ∈ [0.7, 1.0]
        # - If disagreement = 0 (perfect agreement) → penalty = 1.0 (no reduction)
        # - If disagreement = 0.2 (moderate conflict) → penalty = 0.7 (30% reduction)
        penalty = max(0.7, 1.0 - disagreement * 1.5)

        # Apply penalty to base score
        fused = base * penalty

        # Clip to [0, 1] range
        return float(np.clip(fused, 0.0, 1.0))

    except Exception as e:
        logger.error(f"Error computing k2_fusion_score: {e}")
        return 0.5  # safe fallback


def compute_k2_fusion_with_regime_adaptation(
    row,
    base_weights: Optional[Dict[str, float]] = None
) -> float:
    """
    Compute K2 fusion with dynamic weight adjustment based on macro regime.

    In high volatility regimes, increase weight on 1D and macro components
    to favor stability over short-term noise.

    Parameters:
        row: pandas Series with fusion scores AND macro_vix_level
        base_weights: starting weights before regime adjustment

    Returns:
        float (0-1): Regime-adapted fusion score
    """
    # Detect macro regime
    vix_level = row.get("macro_vix_level", "medium")

    # Adjust weights based on VIX regime
    if vix_level == "high":
        # High volatility → trust slower timeframes more
        weights = {"1h": 0.20, "4h": 0.30, "1d": 0.40, "macro": 0.10}
        logger.debug("High VIX regime: shifting weight to 1D")
    elif vix_level == "low":
        # Low volatility → can be more aggressive with faster signals
        weights = {"1h": 0.40, "4h": 0.35, "1d": 0.15, "macro": 0.10}
        logger.debug("Low VIX regime: shifting weight to 1H")
    else:
        # Medium volatility → use base weights
        weights = base_weights if base_weights else None

    return compute_k2_fusion(row, weights=weights)


def validate_k2_fusion_inputs(df) -> Dict[str, bool]:
    """
    Validate that all required inputs for k2_fusion exist and have variance.

    Used in feature store builder to ensure data quality before computing k2.

    Parameters:
        df: DataFrame with fusion score columns

    Returns:
        dict with validation results for each component
    """
    required_cols = [
        "tf1h_fusion_score",
        "tf4h_fusion_score",
        "tf1d_fusion_score",
        "macro_correlation_score"
    ]

    results = {}

    for col in required_cols:
        if col not in df.columns:
            results[col] = False
            logger.warning(f"Missing required column: {col}")
        else:
            variance = df[col].var()
            has_variance = variance > 1e-6
            results[col] = has_variance

            if not has_variance:
                logger.warning(f"Column {col} is flatlined (variance={variance:.8f})")
            else:
                logger.info(f"Column {col} validated (variance={variance:.6f})")

    return results
