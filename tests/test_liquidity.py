import pytest
from bull_machine.modules.liquidity.advanced import AdvancedLiquidityAnalyzer
from bull_machine.core.types import Series, Bar, WyckoffResult


def _series(prices):
    bars = [
        Bar(ts=i, open=p, high=p * 1.01, low=p * 0.99, close=p, volume=1000)
        for i, p in enumerate(prices)
    ]
    return Series(bars=bars, symbol="TEST", timeframe="1h")


def _wy():
    return WyckoffResult(
        regime="trending",
        phase="D",
        bias="long",
        phase_confidence=0.8,
        trend_confidence=0.8,
        range=None,
    )


def test_sweeps_tick_guard_no_crash():
    cfg = {"liquidity": {"tick_size": 0, "sweep_recent_bars": 5}}
    a = AdvancedLiquidityAnalyzer(cfg)
    s = _series([100, 101, 100.5, 101.2, 101.3, 101.29, 101.31, 101.1, 100.9, 101.0, 101.2, 101.3])
    sweeps = a._detect_liquidity_sweeps(s)
    assert isinstance(sweeps, list)


def test_phob_pct_handling():
    cfg = {"liquidity": {"phob_mitigation_pct": 75}}
    a = AdvancedLiquidityAnalyzer(cfg)
    s = _series([100, 99, 98, 102, 103, 104, 103.5, 103.7, 103.8])
    # provide a minimal order block stub format
    obs = [
        {"index": 1, "direction": "bullish", "bottom": 98, "top": 101, "mid": 99.5, "strength": 0.6}
    ]
    phobs = a._detect_phobs(obs, s)
    assert isinstance(phobs, list)
