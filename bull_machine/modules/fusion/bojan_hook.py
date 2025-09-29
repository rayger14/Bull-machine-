"""
Bojan Fusion Hook - Bull Machine v1.6.2
Conservative Bojan microstructure integration with anti-double-counting
"""

from typing import Tuple, Dict
import pandas as pd

from ..bojan.bojan import compute_bojan_score


def apply_bojan(layer_scores: Dict[str, float],
                df: pd.DataFrame,
                tf: str,
                config: dict,
                last_hooks: dict) -> Tuple[Dict[str, float], Dict]:
    """
    Apply Bojan microstructure analysis to layer scores with conservative capping

    Args:
        layer_scores: Current layer scores dict
        df: OHLCV data
        tf: Timeframe string
        config: Configuration dict
        last_hooks: Previous hook results for anti-double-counting

    Returns:
        Tuple of (updated_layer_scores, telemetry)
    """
    if not config.get("features", {}).get("bojan", False):
        return layer_scores, {}

    # Compute Bojan signals
    bojan_config = config.get("bojan", {})
    out = compute_bojan_score(df, bojan_config)
    bscore = out["bojan_score"]

    # Anti double-count with PO3/Liquidity if they already boosted this bar
    po3_boost = last_hooks.get("po3_boost", 0.0)
    liq_boost = last_hooks.get("liquidity_boost", 0.0)

    # Conservative cap
    cap = 0.10
    if po3_boost > 0.0:
        cap = min(cap, 0.07)
    if liq_boost > 0.0:
        cap = min(cap, 0.08)

    add = min(bscore, cap)

    # Blend: mostly into Structure, then Wyckoff/Volume
    layer_scores = dict(layer_scores)  # copy
    layer_scores["structure"] = layer_scores.get("structure", 0.0) + add * 0.50
    layer_scores["wyckoff"]   = layer_scores.get("wyckoff", 0.0)   + add * 0.25
    layer_scores["volume"]    = layer_scores.get("volume", 0.0)    + add * 0.25

    telemetry = {
        "bojan": out,
        "bojan_applied": add,
        "bojan_cap_used": cap,
        "po3_overlap": po3_boost > 0.0,
        "liq_overlap": liq_boost > 0.0
    }

    return layer_scores, telemetry