"""Regression tests: structural checks must use a frame-local bar index.

Callers pass a rolling lookback window (last row = current bar) plus a GLOBAL
bar counter. History-based checks (_check_B, _check_C) slice
df.iloc[index-lookback:index]; with the global index that slice is empty on
any bar past the window length — which kept order_block_retest and
fvg_continuation structurally dead in backtest AND live even after the BOS
feature repair. check_structure now normalizes to len(lookback_df)-1.
"""
import numpy as np
import pandas as pd
import pytest

from engine.archetypes.structural_check import StructuralChecker


def make_window(n=61, bos_at=-5, fvg_now=True):
    """Rolling window: BOS fired `bos_at` bars from the end, FVG on last bar."""
    df = pd.DataFrame({
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5,
        "volume": 10.0,
        "tf1h_bos_bullish": 0, "tf1h_bos_bearish": 0,
        "tf1h_fvg_present": 0, "tf4h_fvg_present": 0,
    }, index=range(n))
    df.loc[df.index[bos_at], "tf1h_bos_bullish"] = 1
    if fvg_now:
        df.loc[df.index[-1], "tf1h_fvg_present"] = 1
    return df


def run_check(df, global_index):
    sc = StructuralChecker(config={}, mode="backtest")
    row = df.iloc[-1]
    prev = df.iloc[-2]
    return sc.check_structure("fvg_continuation", row, prev, df, global_index)


def test_check_c_passes_with_global_index():
    """The historical failure: global bar index (e.g. 30000) on a 61-row
    window must NOT produce an empty history slice."""
    df = make_window()
    passed, reason = run_check(df, global_index=30_000)
    assert passed, f"rejected with {reason} — global-index slice regression"


def test_check_c_fails_without_recent_bos():
    df = make_window(bos_at=-30)  # BOS too old (>10 bars back)
    passed, _ = run_check(df, global_index=30_000)
    assert not passed


def test_check_c_fails_without_fvg():
    df = make_window(fvg_now=False)
    passed, _ = run_check(df, global_index=30_000)
    assert not passed


def test_row_only_checks_unaffected():
    """wick_trap (_check_K) uses only the current bar — same verdict for any
    global index value."""
    df = make_window()
    sc = StructuralChecker(config={}, mode="backtest")
    row, prev = df.iloc[-1], df.iloc[-2]
    r1, _ = sc.check_structure("wick_trap", row, prev, df, 5)
    sc2 = StructuralChecker(config={}, mode="backtest")
    r2, _ = sc2.check_structure("wick_trap", row, prev, df, 50_000)
    assert r1 == r2


def test_no_lookback_df_still_works():
    sc = StructuralChecker(config={}, mode="backtest")
    df = make_window()
    passed, reason = sc.check_structure("fvg_continuation", df.iloc[-1],
                                        df.iloc[-2], None, 12_345)
    # no history available -> _check_C requires it -> clean reject, no crash
    assert isinstance(passed, bool) and "error" not in reason
