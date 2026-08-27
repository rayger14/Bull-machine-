"""Sizing-package dial tests — causality is the point."""
import numpy as np
import pandas as pd

from engine.risk.tape_dial import (compute_tape_dial, dial_for_day,
                                   MULT_IMPULSE, MULT_BASE, MULT_DEAD)


def _closes(vals, start="2024-01-01"):
    return pd.Series(vals, index=pd.date_range(start, periods=len(vals), freq="1D"))


def test_impulse_day_is_not_self_labelled():
    """STRICT CAUSALITY: the day an impulse completes must NOT get 1.25x —
    only the following days may (live can't know today's close)."""
    vals = [100.0]*20 + [100, 101, 112, 113, 114, 115]   # +12% burst completing at idx 22
    m = compute_tape_dial(_closes(vals))
    burst_complete = m.index[22]
    assert m.loc[burst_complete] != MULT_IMPULSE, "impulse day must not self-label"
    assert m.iloc[23] == MULT_IMPULSE, "day AFTER the impulse gets the boost"


def test_tight_base_is_base_mult():
    vals = [100 + 0.3*np.sin(i) for i in range(30)]      # flat 12d range << 8%
    m = compute_tape_dial(_closes(vals))
    assert m.iloc[-1] == MULT_BASE


def test_dead_tape_is_desized():
    rng = np.random.default_rng(7)
    vals = list(100 * np.exp(np.cumsum(rng.normal(-0.004, 0.02, 40))))  # drifting, wide, no impulse
    m = compute_tape_dial(_closes(vals))
    assert m.iloc[-1] == MULT_DEAD


def test_truncation_no_repaint():
    rng = np.random.default_rng(1)
    vals = list(100 * np.exp(np.cumsum(rng.normal(0, 0.03, 60))))
    full = compute_tape_dial(_closes(vals))
    part = compute_tape_dial(_closes(vals[:45]))
    pd.testing.assert_series_equal(part, full.iloc[:45], check_names=False)


def test_dial_for_day_unknown_is_neutral():
    assert dial_for_day(pd.Series(dtype=float), pd.Timestamp("2024-01-01")) == 1.0
