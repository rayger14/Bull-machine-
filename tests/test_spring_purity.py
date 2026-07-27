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
