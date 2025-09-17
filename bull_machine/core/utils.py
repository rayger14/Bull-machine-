import numpy as np
from typing import List, Dict, Tuple
from .types import Series, Bar

def _nearly_equal(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(1.0, (abs(a) + abs(b)) / 2.0)

def detect_sweep_displacement(
    series: Series,
    window: int = 20,
    equal_tol: float = 0.001,      # 0.1% “equal high/low”
    impulse_pct: float = 0.01      # 1% displacement impulse bar
) -> bool:
    """
    Detect a simple 'liquidity sweep → impulse displacement' pattern in the last `window` bars.
    Heuristic:
      - Bullish case: price sweeps prior equal lows (within equal_tol) then prints an up impulse bar (> impulse_pct).
      - Bearish case: price sweeps prior equal highs then prints a down impulse bar.
    This is deliberately light-weight for v1.1.
    """
    n = len(series.bars)
    if n < max(10, window):
        return False
    bars = series.bars[-window:]

    # find clusters of equal highs/lows (last half vs first half)
    mid = len(bars) // 2
    first = bars[:mid]
    last  = bars[mid:]

    prior_highs = [b.high for b in first]
    prior_lows  = [b.low  for b in first]
    last_highs  = [b.high for b in last]
    last_lows   = [b.low  for b in last]

    # equal highs/lows from prior segment (proxy for liquidity)
    if not prior_highs or not prior_lows:
        return False

    # take representative equal levels as medians
    import statistics as _stats
    eq_high = _stats.median(prior_highs)
    eq_low  = _stats.median(prior_lows)

    # check if last segment swept above eq_high or below eq_low
    swept_high = any(h > eq_high and _nearly_equal(h, eq_high, equal_tol) for h in last_highs)
    swept_low  = any(low < eq_low  and _nearly_equal(low, eq_low,  equal_tol) for low in last_lows)

    # displacement: check last bar vs previous close
    c0 = last[-2].close if len(last) >= 2 else bars[-2].close
    c1 = last[-1].close
    move = (c1 - c0) / max(1e-12, c0)

    # Bullish: swept lows then +impulse; Bearish: swept highs then -impulse
    if swept_low and move >= impulse_pct:
        return True
    if swept_high and move <= -impulse_pct:
        return True
    return False

def calculate_atr(series: Series, period: int = 14) -> float:
    """Calculate Average True Range (simple moving)."""
    if len(series.bars) < period + 1:
        return 0.0
    true_ranges = []
    for i in range(1, len(series.bars)):
        curr = series.bars[i]
        prev = series.bars[i-1]
        tr = max(
            curr.high - curr.low,
            abs(curr.high - prev.close),
            abs(curr.low - prev.close)
        )
        true_ranges.append(tr)
    if len(true_ranges) < period:
        return (sum(true_ranges) / max(1, len(true_ranges)))
    return sum(true_ranges[-period:]) / period

def find_swing_high_low(series: Series, lookback: int = 20) -> Tuple[float, float]:
    """Find recent swing high and low within lookback bars."""
    bars = series.bars[-lookback:] if len(series.bars) >= lookback else series.bars
    if not bars:
        return 0.0, 0.0
    swing_high = max(bar.high for bar in bars)
    swing_low = min(bar.low for bar in bars)
    return swing_high, swing_low

def detect_structure_breaks(series: Series, lookback: int = 10) -> Dict:
    """Detect simple CHoCH/BOS-like conditions (heuristic)."""
    if len(series.bars) < lookback + 5:
        return {'choch_bull': False, 'choch_bear': False, 'bos_strength': 0.3}
    recent_bars = series.bars[-lookback:]
    current_price = series.bars[-1].close
    recent_highs = [bar.high for bar in recent_bars[:-2]]
    recent_lows = [bar.low for bar in recent_bars[:-2]]
    max_recent_high = max(recent_highs) if recent_highs else current_price
    min_recent_low = min(recent_lows) if recent_lows else current_price
    choch_bull = current_price > max_recent_high
    choch_bear = current_price < min_recent_low
    return {
        'choch_bull': choch_bull,
        'choch_bear': choch_bear,
        'bos_strength': 0.7 if (choch_bull or choch_bear) else 0.3
    }
