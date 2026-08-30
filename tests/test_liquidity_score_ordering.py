"""liquidity_score ordering fix (metrology audit 2026-08-28)."""
from bin.live.live_feature_computer import _liquidity_score_from


def test_uses_real_inputs_when_present():
    f={'volume_zscore':2.5,'atr_percentile':0.9,'tf1h_fvg_present':1.0,'oi_change_4h':-0.08}
    # 0.35*1.0 + 0.25*0.9 + 0.20*1.0 + 0.20*0.8 = 0.935
    assert abs(_liquidity_score_from(f)-0.935)<1e-9


def test_pinned_value_only_when_inputs_absent():
    f={'volume_zscore':0.0}
    assert abs(_liquidity_score_from(f)-(0.25*0.5))<1e-9  # the old permanent state


def test_nan_safe():
    f={'volume_zscore':float('nan'),'atr_percentile':float('nan'),'oi_change_4h':float('nan')}
    v=_liquidity_score_from(f)
    assert 0.0<=v<=1.0
