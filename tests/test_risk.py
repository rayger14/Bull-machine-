
import pytest
from bull_machine.modules.risk.advanced import AdvancedRiskManager
from bull_machine.core.types import Series, Bar, Signal, WyckoffResult

def _series(prices, vol=0.01):
    bars = [Bar(ts=i, open=p, high=p*(1+vol), low=p*(1-vol), close=p, volume=1000) for i,p in enumerate(prices)]
    return Series(bars=bars, symbol="BTCUSD", timeframe="1h")

def _wy():
    return WyckoffResult(regime='trending', phase='E', bias='long', phase_confidence=0.8, trend_confidence=0.8, range=None)

def test_position_size_scales_with_vol():
    cfg = {
        "risk": {
            "account_risk_percent": 1.0,
            "max_risk_percent": 2.0,
            "max_risk_per_trade": 10000,
            "stop": {"method":"atr","atr_mult":2.0,"volatility_scaling":True,"target_volatility":0.012},
            "tp_ladder": {"tp1":{"r":1.0,"pct":50,"action":"move_stop_to_breakeven"}}
        }
    }
    rm = AdvancedRiskManager(cfg)
    s_low = _series([100 + i*0.1 for i in range(60)], vol=0.005)
    s_high = _series([100 + (i%2)*3 for i in range(60)], vol=0.03)
    sig = Signal(ts=0, side='long', confidence=0.8, reasons=[], ttl_bars=20)
    liq = {"best_candidate": None, "phobs": []}
    wy = _wy()
    rp_low = rm.plan_trade(s_low, sig, liq, wy, 10000)
    rp_high = rm.plan_trade(s_high, sig, liq, wy, 10000)
    assert rp_high.size < rp_low.size
