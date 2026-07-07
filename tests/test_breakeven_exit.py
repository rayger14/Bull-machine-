"""Tests for the breakeven-stop rule in ExitLogic.

Validated 2026-06-29 (wick_trap: gross loss -13% w/ flat gross profit).
OFF by default everywhere; enabled only via per-archetype config override
(breakeven_trigger_r). These tests exercise the REAL ExitLogic path.
"""
import pandas as pd
import pytest

from engine.archetypes.exit_logic import ExitLogic, create_default_exit_config
from engine.runtime.context import RuntimeContext


class FakePosition:
    """Minimal Position for ExitLogic (mirrors _PositionAdapter's interface)."""

    def __init__(self, entry_price, stop_loss, direction="long",
                 entry_time=None, quantity=1.0):
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.direction = direction
        self.entry_time = entry_time or pd.Timestamp("2024-01-01 00:00")
        self.runner_trailing_stop = None
        self.quantity = quantity
        self.current_quantity = quantity
        self.original_quantity = quantity
        self.metadata = {}


def make_bar(close, ts="2024-01-01 02:00", atr=100.0, **extra):
    data = {"close": close, "high": close, "low": close, "open": close,
            "atr_14": atr, "chop_score": 0.2, **extra}
    bar = pd.Series(data)
    bar.name = pd.Timestamp(ts)
    return bar


def make_context(regime="neutral"):
    return RuntimeContext(ts=pd.Timestamp("2024-01-01 02:00"), row=None,
                          regime_probs={regime: 1.0}, regime_label=regime,
                          adapted_params={}, thresholds={})


def exit_logic_with_be(trigger=1.0, buffer=0.0):
    cfg = create_default_exit_config()
    cfg["wick_trap"] = {"breakeven_trigger_r": trigger, "breakeven_buffer_r": buffer}
    return ExitLogic(cfg)


def test_breakeven_fires_at_trigger():
    """Long at 100, stop 90 (R=10). Close 110 = +1R -> stop moves to entry."""
    el = exit_logic_with_be(trigger=1.0)
    pos = FakePosition(entry_price=100.0, stop_loss=90.0)
    el.check_exit(make_bar(110.0), pos, "wick_trap", make_context())
    assert pos.stop_loss == pytest.approx(100.0)


def test_breakeven_not_fired_below_trigger():
    el = exit_logic_with_be(trigger=1.0)
    pos = FakePosition(entry_price=100.0, stop_loss=90.0)
    el.check_exit(make_bar(105.0), pos, "wick_trap", make_context())  # +0.5R
    assert pos.stop_loss == pytest.approx(90.0)


def test_breakeven_buffer():
    """buffer 0.1 -> stop at entry + 0.1*R = 101."""
    el = exit_logic_with_be(trigger=1.0, buffer=0.1)
    pos = FakePosition(entry_price=100.0, stop_loss=90.0)
    el.check_exit(make_bar(110.0), pos, "wick_trap", make_context())
    assert pos.stop_loss == pytest.approx(101.0)


def test_breakeven_ratchet_only_never_lowers():
    """If the stop is already above breakeven (trailing raised it), BE must not lower it."""
    el = exit_logic_with_be(trigger=1.0)
    pos = FakePosition(entry_price=100.0, stop_loss=105.0)  # already locked in
    el.check_exit(make_bar(120.0), pos, "wick_trap", make_context())
    assert pos.stop_loss >= 105.0


def test_breakeven_short_mirror():
    """Short at 100, stop 110 (R=10). Close 90 = +1R -> stop moves DOWN to entry."""
    el = exit_logic_with_be(trigger=1.0)
    pos = FakePosition(entry_price=100.0, stop_loss=110.0, direction="short")
    el.check_exit(make_bar(90.0), pos, "wick_trap", make_context())
    assert pos.stop_loss == pytest.approx(100.0)


def test_breakeven_off_by_default():
    """Default config: no archetype has breakeven enabled -> stops untouched below trailing."""
    el = ExitLogic(create_default_exit_config())
    for arch in ("wick_trap", "liquidity_sweep", "confluence_breakout", "spring"):
        rules = el.exit_rules.get(arch) or {}
        assert rules.get("breakeven_trigger_r") is None, f"{arch} must be BE-off by default"
    pos = FakePosition(entry_price=100.0, stop_loss=90.0)
    # +0.4R: below trailing_start (0.5) AND no BE -> stop must remain original
    el.check_exit(make_bar(104.0), pos, "wick_trap", make_context())
    assert pos.stop_loss == pytest.approx(90.0)


def test_breakeven_does_not_break_unrealized_r():
    """After BE fires, next bar's R calc uses the adapter-rebuilt ORIGINAL stop in
    the engines. Within ExitLogic, a re-check on a fresh position object with the
    original stop must still compute R off initial risk (no div-by-zero path)."""
    el = exit_logic_with_be(trigger=1.0)
    pos = FakePosition(entry_price=100.0, stop_loss=90.0)
    el.check_exit(make_bar(110.0), pos, "wick_trap", make_context())
    assert pos.stop_loss == pytest.approx(100.0)
    # engines rebuild the adapter from the ORIGINAL stop next bar; simulate that:
    pos2 = FakePosition(entry_price=100.0, stop_loss=90.0)
    r = el._calculate_unrealized_r(pos2, 120.0, atr=100.0)
    assert r == pytest.approx(2.0)
