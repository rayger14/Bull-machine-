import logging
from typing import Optional, List, Tuple
from ..core.types import WyckoffResult, LiquidityResult, Signal, Series
from ..core.utils import detect_sweep_displacement

def combine(w: WyckoffResult, l: LiquidityResult, cfg: dict, state: dict) -> Tuple[Optional[Signal], Optional[str]]:
    """Combine Wyckoff + Liquidity with range suppression and confidence floor.

    Returns a tuple (Signal|None, reason|None). Reason is a short machine-friendly string
    explaining why no signal was produced when Signal is None.
    """
    try:
        signals_cfg = cfg.get('signals', {})
        thr = signals_cfg.get('confidence_threshold', 0.72)
        weights = signals_cfg.get('weights', {'wyckoff': 0.60, 'liquidity': 0.40, 'smt':0.0, 'macro':0.0, 'temporal':0.0})
        # assemble available confidence sources
        confidences = {
            'wyckoff': (w.phase_confidence + w.trend_confidence)/2,
            'liquidity': l.score,
            'smt': 0.0,
            'macro': 0.0,
            'temporal': 0.0,
        }
        # compute weighted combined confidence using only keys present in both dicts
        num = 0.0
        denom = 0.0
        for k, wk in weights.items():
            if k in confidences and wk and wk > 0:
                num += confidences.get(k, 0.0) * wk
                denom += wk
        combined = (num / denom) if denom > 0 else 0.0
        if combined < thr:
            return None, 'confidence_below_threshold'
        if _is_range_suppressed(w, l, cfg, state):
            return None, 'range_suppressed'
        side = _determine_signal_side(w, l)
        if side == 'neutral':
            return None, 'side_neutral'
        reasons = _build_signal_reasons(w, l, combined)
        ttl_bars = cfg.get('risk',{}).get('ttl_bars', 18)
        return Signal(ts=0, side=side, confidence=combined, reasons=reasons, ttl_bars=ttl_bars), None
    except Exception as e:
        logging.error(f"Fusion error: {e}")
        return None, 'fusion_error'

def _is_range_suppressed(w: WyckoffResult, l: LiquidityResult, cfg: dict, state: dict) -> bool:
    """Check if signal should be suppressed due to ranging conditions"""
    range_cfg = cfg.get('range', {})
    time_in_range_min = range_cfg.get('time_in_range_bars_min', 20)
    net_progress_threshold = range_cfg.get('net_progress_threshold', 0.25)
    liq_cfg = cfg.get('liquidity', {})
    candidate_min = liq_cfg.get('candidate_min', 0.65)

    # Only consider suppression when Wyckoff marks the chart as ranging
    if w.range and w.regime in ['ranging']:
        within = w.range.get('within_range', False)
        shallow = w.range.get('penetration', 1.0) < net_progress_threshold
        # Allow ONLY if we have sweep→displacement NOW and liquidity is strong
        # (Use live detection to avoid stale state)
        had_sweep_disp = bool(state.get('had_recent_sweep_displacement', False))
        strong_liq = (l.score >= candidate_min)
        if within and shallow and not (had_sweep_disp and strong_liq):
            return True

    return False

def _determine_signal_side(w: WyckoffResult, l: LiquidityResult) -> str:
    if w.bias == 'long' and l.pressure in ['bullish','neutral']: return 'long'
    if w.bias == 'short' and l.pressure in ['bearish','neutral']: return 'short'
    return 'neutral'

def _build_signal_reasons(w: WyckoffResult, l: LiquidityResult, conf: float) -> List[str]:
    reasons = [
        f"Wyckoff {w.phase} phase, {w.bias} bias",
        f"Liquidity pressure: {l.pressure}",
        f"Combined confidence: {conf:.2f}"
    ]
    if l.fvgs: reasons.append(f"{len(l.fvgs)} FVGs detected")
    if l.order_blocks: reasons.append(f"{len(l.order_blocks)} Order Blocks detected")
    return reasons
