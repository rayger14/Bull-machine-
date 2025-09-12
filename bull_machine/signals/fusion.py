import logging
from typing import Optional, List
from ..core.types import WyckoffResult, LiquidityResult, Signal

def combine(w: WyckoffResult, l: LiquidityResult, cfg: dict, state: dict) -> Optional[Signal]:
    """Combine Wyckoff + Liquidity with range suppression and confidence floor."""
    try:
        signals_cfg = cfg.get('signals', {})
        thr = signals_cfg.get('confidence_threshold', 0.70)
        weights = signals_cfg.get('weights', {'wyckoff': 0.60, 'liquidity': 0.40})
        wyckoff_conf = (w.phase_confidence + w.trend_confidence)/2
        liq_conf = l.score
        combined = wyckoff_conf * weights['wyckoff'] + liq_conf * weights['liquidity']
        if combined < thr: return None
        if _is_range_suppressed(w, cfg): return None
        side = _determine_signal_side(w, l)
        if side == 'neutral': return None
        reasons = _build_signal_reasons(w, l, combined)
        ttl_bars = cfg.get('risk',{}).get('ttl_bars', 18)
        return Signal(ts=0, side=side, confidence=combined, reasons=reasons, ttl_bars=ttl_bars)
    except Exception as e:
        logging.error(f"Fusion error: {e}")
        return None

def _is_range_suppressed(w: WyckoffResult, cfg: dict) -> bool:
    range_cfg = cfg.get('range', {})
    net_thresh = range_cfg.get('net_progress_threshold', 0.25)
    if w.range and w.regime in ['ranging','accumulation']:
        if w.range.get('within_range') and w.range.get('penetration',1.0) < net_thresh:
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
