import numpy as np
from typing import List, Dict, Tuple
from .types import Series, Bar

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
