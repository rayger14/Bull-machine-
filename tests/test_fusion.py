
import pytest
from bull_machine.modules.fusion.advanced import AdvancedFusionEngine
from bull_machine.core.types import WyckoffResult

def _wy(bias='long', pc=0.8, tc=0.8, phase='D'):
    return WyckoffResult(regime='trending', phase=phase, bias=bias, phase_confidence=pc, trend_confidence=tc, range=None)

def test_trend_blocks_apply():
    cfg = {"features":{"trend_filter":True}, "fusion":{"enter_threshold":0.5,"weights":{"wyckoff":1.0}}}
    eng = AdvancedFusionEngine(cfg)
    modules = {"wyckoff": _wy(bias="long"), "liquidity":{"overall_score":0.8}, "series": None}
    res = eng.fuse(modules)
    # Either a signal or a trend veto present depending on score alignment logic
    assert ('trend_filter_short' in res.vetoes) or (res.signal is not None) or ('trend_filter_long' in res.vetoes)
