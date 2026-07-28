"""Spring direction-purity (H4b, 2026-07-27): a LONG archetype must not
trigger on BEARISH trap signatures. utad/bull_trap acceptance was a lifetime
defect (train 0.84->1.31, holdout 0.68->0.92 when removed). Pure is default."""
import pandas as pd
from engine.archetypes.logic import ArchetypeLogic


def check(trap, gate_params=None):
    logic = ArchetypeLogic({'use_archetypes': True})
    row = pd.Series({'tf1h_pti_trap_type': trap})
    return logic._check_A(row, None, pd.DataFrame([row]), 0, 1.0, gate_params)


def test_pure_default_accepts_bullish_traps():
    assert check('spring') and check('bear_trap')


def test_pure_default_rejects_bearish_traps():
    assert not check('utad') and not check('bull_trap')


def test_legacy_reproducible():
    assert check('utad', {'spring_pure_A': 0})


def test_none_rejected():
    assert not check('none') and not check('')


def test_twt_real_trend_default_on():
    """A1 (2026-07-27): trap_within_trend requires an ACTUAL uptrend by
    default (price_above_ema_50 >= 1), not mere column existence."""
    logic = ArchetypeLogic({'use_archetypes': True})
    row_dn = pd.Series({'tf1h_pti_trap_type': 'none', 'price_above_ema_50': 0,
                        'tf4h_fusion_score': 0.5, 'adx': 25.0,
                        'wick_lower_ratio': 0.6, 'low': 99.0, 'high': 101.0,
                        'open': 100.6, 'close': 100.5})
    df = pd.DataFrame([row_dn])
    # downtrend + neutral 4H bias -> must NOT pass the trend context
    assert logic._check_H(row_dn, None, df, 0, 1.0, None) is False
    row_up = row_dn.copy(); row_up['price_above_ema_50'] = 1
    r = logic._check_H(row_up, None, pd.DataFrame([row_up]), 0, 1.0, None)
    assert isinstance(r, bool)
