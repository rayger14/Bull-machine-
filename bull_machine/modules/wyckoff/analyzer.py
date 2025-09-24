import logging
from typing import Dict, Optional, Tuple

from ...core.types import Series, WyckoffResult


def analyze(series: Series, cfg: dict, state: dict) -> WyckoffResult:
    """v1.1 Wyckoff Analysis (heuristic)."""
    wyckoff_cfg = cfg.get("wyckoff", {})
    lookback = wyckoff_cfg.get("lookback_bars", 50)
    hysteresis_bars = wyckoff_cfg.get("bias_hysteresis_bars", 2)
    if len(series.bars) < 10:
        return _get_neutral_wyckoff()
    try:
        range_data = _build_range_model(series, lookback)
        current_bias, trend_confidence = _calculate_bias_with_hysteresis(
            series, state, hysteresis_bars
        )
        phase, phase_confidence = _detect_wyckoff_phase(series, current_bias, range_data)
        regime = _determine_regime(phase, current_bias, range_data)
        return WyckoffResult(
            regime=regime,
            phase=phase,
            bias=current_bias,
            phase_confidence=phase_confidence,
            trend_confidence=trend_confidence,
            range=range_data,
        )
    except Exception as e:
        logging.error(f"Wyckoff analysis error: {e}")
        return _get_neutral_wyckoff()


def _build_range_model(series: Series, lookback: int) -> Optional[Dict]:
    if len(series.bars) < 10:
        return None
    recent = series.bars[-min(lookback, len(series.bars)) :]
    range_high = max(b.high for b in recent)
    range_low = min(b.low for b in recent)
    range_mid = (range_high + range_low) / 2
    current_price = series.bars[-1].close
    height = range_high - range_low
    if height == 0:
        return None
    within = range_low <= current_price <= range_high
    penetration = abs(current_price - range_mid) / (height / 2)
    return {
        "low": range_low,
        "high": range_high,
        "mid": range_mid,
        "height": height,
        "within_range": within,
        "penetration": penetration,
    }


def _calculate_bias_with_hysteresis(
    series: Series, state: dict, hysteresis_bars: int
) -> Tuple[str, float]:
    if len(series.bars) < hysteresis_bars + 5:
        return "neutral", 0.5
    prev_bias = state.get("prev_bias", "neutral")
    recent = series.bars[-hysteresis_bars - 5 :]
    changes = []
    for i in range(1, len(recent)):
        prev = recent[i - 1].close
        changes.append((recent[i].close - prev) / prev if prev else 0.0)
    avg_change = sum(changes) / len(changes) if changes else 0.0
    if avg_change > 0.001:
        new_bias = "long"
    elif avg_change < -0.001:
        new_bias = "short"
    else:
        new_bias = "neutral"
    if prev_bias != "neutral" and new_bias != prev_bias:
        consistent = 0
        for i in range(len(recent) - hysteresis_bars, len(recent)):
            if i > 0:
                inc = (
                    (recent[i].close - recent[i - 1].close) / recent[i - 1].close
                    if recent[i - 1].close
                    else 0.0
                )
                if (new_bias == "long" and inc > 0) or (new_bias == "short" and inc < 0):
                    consistent += 1
        if consistent < hysteresis_bars // 2:
            new_bias = prev_bias
    trend_strength = abs(avg_change) * 100
    confidence = min(0.5 + trend_strength * 10, 0.9)
    return new_bias, confidence


def _detect_wyckoff_phase(
    series: Series, bias: str, range_data: Optional[Dict]
) -> Tuple[str, float]:
    if len(series.bars) < 20:
        return "neutral", 0.5
    recent = series.bars[-20:]
    changes = [
        abs(recent[i].close - recent[i - 1].close) / recent[i - 1].close
        for i in range(1, len(recent))
        if recent[i - 1].close
    ]
    avg_vol = sum(changes) / len(changes) if changes else 0.0
    if range_data and range_data["within_range"]:
        if avg_vol < 0.01:
            return ("B", 0.7) if bias == "neutral" else ("C", 0.8)
        else:
            return "D", 0.6
    else:
        if bias == "long":
            return "E", 0.7
        elif bias == "short":
            return "E", 0.7
        else:
            return "A", 0.5


def _determine_regime(phase: str, bias: str, range_data: Optional[Dict]) -> str:
    if phase in ["C", "D"] and bias != "neutral":
        return "accumulation" if bias == "long" else "distribution"
    elif phase == "E":
        return "trending"
    elif range_data and range_data["within_range"]:
        return "ranging"
    else:
        return "neutral"


def _get_neutral_wyckoff() -> WyckoffResult:
    return WyckoffResult(
        regime="neutral",
        phase="neutral",
        bias="neutral",
        phase_confidence=0.0,
        trend_confidence=0.0,
        range=None,
    )
