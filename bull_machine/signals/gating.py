import logging

from ..core.types import Series, Signal
from ..core.utils import calculate_atr


def _compute_dynamic_ttl_bars(series, w, cfg):
    """Compute dynamic TTL based on ATR%%, regime and range conditions."""
    risk_cfg = cfg.get("risk", {})
    base = risk_cfg.get("ttl_bars", 18)
    dyn = risk_cfg.get("ttl_dynamic", {})
    if not series or not getattr(series, 'bars', None):
        return base
    if not dyn:
        return base
    lo = dyn.get("min", 8)
    hi = dyn.get("max", 30)
    atr_period = dyn.get("atr_period", 14)
    atr_high = dyn.get("atr_high_pct", 0.015)
    atr_low  = dyn.get("atr_low_pct", 0.007)
    d_high   = dyn.get("delta_high_vol", 4)
    d_low    = dyn.get("delta_low_vol", -3)
    trend_bonus = dyn.get("trend_bonus", 3)
    range_penalty = dyn.get("range_penalty", -3)
    pen_thresh = dyn.get("range_penetration_thresh", 0.25)
    atr = calculate_atr(series, atr_period)
    price = series.bars[-1].close if series.bars else 0.0
    atr_pct = (atr / price) if price > 0 else 0.0
    ttl = base
    if atr_pct >= atr_high: ttl += d_high
    elif atr_pct <= atr_low: ttl += d_low
    if w and getattr(w, "regime", None) in ("trending","accumulation"):
        ttl += trend_bonus
    if w and getattr(w, "range", None):
        rng = w.range
        within = rng.get("within_range", False)
        penetration = rng.get("penetration", 1.0)
        if within and penetration < pen_thresh:
            ttl += range_penalty
    return max(lo, min(hi, int(round(ttl))))

def enforce_hysteresis(prev_bias: str, new_side: str, series: Series, cfg: dict) -> bool:
    """Secondary check (kept simple in v1.1)."""
    side_to_bias = {'long': 'long', 'short': 'short'}
    new_bias = side_to_bias.get(new_side, 'neutral')
    if prev_bias == new_bias:
        return True
    return True

def assign_ttl(signal: Signal, series: Series, cfg: dict, wyckoff_result=None) -> Signal:
    """Assign bar timestamp and dynamic TTL if enabled."""
    if len(series.bars) > 0:
        signal.ts = series.bars[-1].ts
    if cfg.get("features", {}).get("dynamic_ttl", False):
        signal.ttl_bars = _compute_dynamic_ttl_bars(series, wyckoff_result, cfg)
        logging.info(f"Dynamic TTL: {signal.ttl_bars} bars")
    else:
        logging.info(f"Fixed TTL: {signal.ttl_bars} bars")
    return signal

def is_expired(signal: Signal, current_ts: int, series: Series, cfg: dict) -> bool:
    if signal.ts == 0:
        return False
    bars_passed = current_ts - signal.ts
    return bars_passed > signal.ttl_bars
