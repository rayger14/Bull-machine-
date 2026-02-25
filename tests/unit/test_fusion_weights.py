import json
from bull_machine.signals.fusion import combine
from bull_machine.core.types import WyckoffResult, LiquidityResult


class DummyW:
    def __init__(self, phase_conf=0.8, trend_conf=0.8, bias="long", phase="markup"):
        self.phase_confidence = phase_conf
        self.trend_confidence = trend_conf
        self.bias = bias
        self.phase = phase
        self.regime = "trend"
        self.range = None


class DummyL:
    def __init__(self, score=0.5, pressure="bullish"):
        self.score = score
        self.pressure = pressure
        self.fvgs = []
        self.order_blocks = []


def test_fusion_respects_weights_high_wyckoff():
    w = DummyW(phase_conf=0.9, trend_conf=0.9)
    l = DummyL(score=0.1, pressure="bullish")
    cfg = {"signals": {"confidence_threshold": 0.5, "weights": {"wyckoff": 1.0, "liquidity": 0.0}}}
    signal, reason = combine(w, l, cfg, {})
    assert signal is not None
    assert signal.confidence > 0.8


def test_fusion_respects_weights_high_liquidity():
    w = DummyW(phase_conf=0.2, trend_conf=0.2)
    l = DummyL(score=0.9, pressure="bullish")
    cfg = {"signals": {"confidence_threshold": 0.5, "weights": {"wyckoff": 0.0, "liquidity": 1.0}}}
    signal, reason = combine(w, l, cfg, {})
    assert signal is not None
    assert signal.confidence > 0.8
