"""Tests for the maker-first shadow fill ledger (measurement only).

Fill rule: resting limit fills only on trade-THROUGH on a LATER bar
(long: low < limit; short: high > limit). Miss after HORIZON_BARS records
signed chase cost. Never touches real fills.
"""
import json
import ast
from pathlib import Path

import pytest

from bin.live.maker_shadow import MakerShadowLedger, HORIZON_BARS

REPO = Path(__file__).resolve().parents[1]


@pytest.fixture
def ledger(tmp_path):
    return MakerShadowLedger(tmp_path)


def records(ledger):
    if not ledger.ledger_path.exists():
        return []
    return [json.loads(l) for l in ledger.ledger_path.read_text().splitlines()]


def test_long_fills_on_trade_through(ledger):
    ledger.record_entry("p1", "wick_trap", "long", 100.0, "t0")
    ledger.on_bar(high=101.0, low=99.5, close=100.5, ts="t1")  # low < 100 -> fill
    recs = records(ledger)
    assert len(recs) == 1 and recs[0]["filled"] and recs[0]["bars_waited"] == 1
    assert not ledger.pending


def test_long_touch_does_not_fill(ledger):
    """low == limit is a touch, not a trade-through -> stays pending."""
    ledger.record_entry("p1", "wick_trap", "long", 100.0, "t0")
    ledger.on_bar(high=101.0, low=100.0, close=100.5, ts="t1")
    assert not records(ledger) and len(ledger.pending) == 1


def test_miss_records_chase_cost(ledger):
    ledger.record_entry("p1", "liquidity_compression", "long", 100.0, "t0")
    for i in range(HORIZON_BARS):
        ledger.on_bar(high=103.0, low=100.5, close=102.0, ts=f"t{i+1}")
    recs = records(ledger)
    assert len(recs) == 1 and not recs[0]["filled"]
    assert recs[0]["chase_bps"] == pytest.approx(200.0)  # (102-100)/100 = +200bp
    assert not ledger.pending


def test_short_mirror(ledger):
    ledger.record_entry("p1", "wick_trap", "short", 100.0, "t0")
    ledger.on_bar(high=100.5, low=99.0, close=99.5, ts="t1")  # high > 100 -> fill
    assert records(ledger)[0]["filled"]


def test_short_miss_chase_sign(ledger):
    """Short misses when price falls away; chase positive = pay down to enter."""
    ledger.record_entry("p1", "wick_trap", "short", 100.0, "t0")
    for i in range(HORIZON_BARS):
        ledger.on_bar(high=99.5, low=98.0, close=98.0, ts=f"t{i+1}")
    rec = records(ledger)[0]
    assert not rec["filled"] and rec["chase_bps"] == pytest.approx(200.0)


def test_pending_persists_across_restart(tmp_path):
    l1 = MakerShadowLedger(tmp_path)
    l1.record_entry("p1", "wick_trap", "long", 100.0, "t0")
    l2 = MakerShadowLedger(tmp_path)  # simulated service restart
    assert len(l2.pending) == 1
    l2.on_bar(high=101.0, low=99.0, close=100.0, ts="t1")
    assert records(l2)[0]["filled"]


def test_bad_bar_data_is_ignored(ledger):
    ledger.record_entry("p1", "wick_trap", "long", 100.0, "t0")
    ledger.on_bar(high=None, low=None, close=None, ts="t1")
    ledger.on_bar(high=float("nan"), low=float("nan"), close=float("nan"), ts="t2")
    assert len(ledger.pending) == 1  # bars_waited must not advance on junk


def test_zero_limit_not_recorded(ledger):
    ledger.record_entry("p1", "wick_trap", "long", 0.0, "t0")
    assert not ledger.pending


def test_runner_wiring():
    """Runner records entries at the PRE-slippage signal price and resolves
    pending BEFORE new entries; ledger is lazy-bound to output_dir."""
    src = (REPO / "bin/live/v11_shadow_runner.py").read_text()
    ast.parse(src)
    assert "record_entry(\n                pos_id, archetype, direction, entry_price, timestamp)" in src
    assert "_maker_shadow().on_bar(" in src
    assert src.index("_maker_shadow().on_bar(") < src.index("Step 3: Generate signals")
