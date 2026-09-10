from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research.causal_parent_ledger import build_parent_ledger
from scripts.research.parent_context_policy import (
    annotate_h3_events,
    evaluate_h3_permission,
)


LC_POLICY = "lc_fixed_parent_reclaim_v1"
MINUTE_POLICY = "minute_child_sweep_parent_location_v1"
LC_CASES = [
    (99, 101, True),
    (100, 101, False),
    (99, 100, False),
    (99, 120, False),
    (99, 121, False),
]
MINUTE_CASES = [
    (100, 101, True),
    (110, 119, True),
    (110.01, 111, False),
    (99.99, 101, False),
    (105, 100, False),
    (105, 120, False),
]
SOURCE_PATHS = {
    "htf": "/Users/rayghandchi/Bull Machine/one-strategy/idea_lab/htf_pivots.py",
    "range": "/Users/rayghandchi/Bull Machine/one-strategy/idea_lab/structural_range.py",
}
EXPECTED_HASHES = {
    "htf": "0c37d254cba93f40638ffeb2422d53649ea326adecee0a0690ed417b2f492f0b",
    "range": "22e3e1ab21d8db585d89569c3d1d4a3fe3c74ce17ca72c5080a0ca7cf0caa8ca",
}
ATR_CONTRACT = {
    "source": "fixture-explicit-atr",
    "formula_id": "atr-14-wilder-fixture",
    "version": "1",
    "availability_policy": "hour_close_assumed",
}
LOCAL_SOURCE_AVAILABLE = all(Path(path).is_file() for path in SOURCE_PATHS.values())


def _version(
    version_id="version-1",
    *,
    lineage_id="lineage-1",
    predecessor=None,
    reason="formation",
    low=100.0,
    high=120.0,
    formation_hour="2026-01-01 00:00:00+00:00",
    available_at="2026-01-01 01:00:00+00:00",
):
    return {
        "id": version_id,
        "lineage_id": lineage_id,
        "predecessor_version_id": predecessor,
        "creation_reason": reason,
        "range_low": low,
        "range_high": high,
        "low_pivot_id": "pivot-low-" + version_id,
        "high_pivot_id": "pivot-high-1",
        "formation_hour": formation_hour,
        "available_at": available_at,
    }


def _pivot(pivot_id, side, level, *, anchor_timeframe="4H", pivot_n=3):
    return {
        "id": pivot_id,
        "side": side,
        "level": level,
        "pivot_open": "2025-12-30 00:00:00+00:00",
        "pivot_close": "2025-12-30 04:00:00+00:00",
        "confirming_close": "2025-12-31 00:00:00+00:00",
        "available_at": "2025-12-31 00:00:00+00:00",
        "instrument": "BTC-USD",
        "data_stream_id": "same-stream",
        "anchor_timeframe": anchor_timeframe,
        "pivot_n": pivot_n,
        "evidence_id": "evidence-" + pivot_id,
        "supporting_anchor_ids": ["anchor-1"],
    }


def _transition(
    hour,
    *,
    pre_version=None,
    post_version="version-1",
    pre_lineage=None,
    post_lineage="lineage-1",
    pre_state="forming",
    post_state="active",
    break_direction=None,
):
    source_hour = f"2026-01-01 {hour:02d}:00:00+00:00"
    available_at = f"2026-01-01 {hour + 1:02d}:00:00+00:00"
    return {
        "id": f"transition-{hour}",
        "source_hour": source_hour,
        "available_at": available_at,
        "pre_state": pre_state,
        "pre_range_low": 100.0 if pre_version else None,
        "pre_range_high": 120.0 if pre_version else None,
        "pre_lineage_id": pre_lineage,
        "pre_version_id": pre_version,
        "evaluated_version_id": pre_version,
        "source_break_direction": break_direction,
        "source_sweep_low": 0,
        "source_sweep_high": 0,
        "source_range_state": post_state,
        "source_range_low": 100.0 if post_version else None,
        "source_range_high": 120.0 if post_version else None,
        "source_range_age_bars": hour,
        "source_range_pos": 0.5 if post_version else None,
        "source_range_width_atr": 10.0 if post_version else None,
        "post_state": post_state,
        "post_range_low": 100.0 if post_version else None,
        "post_range_high": 120.0 if post_version else None,
        "post_lineage_id": post_lineage,
        "post_version_id": post_version,
        "latest_low_pivot_id": "pivot-low-version-1",
        "latest_high_pivot_id": "pivot-high-1",
        "adopted_low_pivot_id": "pivot-low-version-1" if post_version else None,
        "adopted_high_pivot_id": "pivot-high-1" if post_version else None,
        "atr_14": 2.0,
        "quality_flags": [],
    }


def ledger(*, anchor_timeframe="4H", pivot_n=3):
    transitions = [_transition(0)]
    transitions.extend(
        _transition(
            hour,
            pre_version="version-1",
            pre_lineage="lineage-1",
            pre_state="active",
        )
        for hour in range(1, 5)
    )
    return {
        "certified": False,
        "manifest": {
            "schema": "causal_parent_ledger.v1",
            "contract_id": f"parent-contract-{anchor_timeframe}-{pivot_n}",
            "instrument": "BTC-USD",
            "data_stream_id": "same-stream",
            "parameters": {"anchor_timeframe": anchor_timeframe, "pivot_n": pivot_n},
        },
        "coverage": {
            "first_open": "2026-01-01 00:00:00+00:00",
            "first_processed_close": "2026-01-01 01:00:00+00:00",
            "last_processed_close": "2026-01-01 05:00:00+00:00",
            "query_exclusive_end": "2026-01-01 06:00:00+00:00",
            "input_hours": 5,
        },
        "pivots": [
            _pivot("pivot-low-version-1", "low", 100.0, anchor_timeframe=anchor_timeframe, pivot_n=pivot_n),
            _pivot("pivot-high-1", "high", 120.0, anchor_timeframe=anchor_timeframe, pivot_n=pivot_n),
        ],
        "versions": [_version()],
        "transitions": transitions,
    }


def lc_event(low=99.0, close=101.0, *, event_id="lc-1"):
    return {
        "id": event_id,
        "contract_id": "native-lc-event.v1",
        "kind": "hourly_lc",
        "instrument": "BTC-USD",
        "data_stream_id": "same-stream",
        "evidence_id": "lc-evidence-1",
        "first_sweep_open": "2026-01-01 03:00:00+00:00",
        "reclaim_bar_open": "2026-01-01 03:00:00+00:00",
        "decision_time": "2026-01-01 04:00:00+00:00",
        "available_at": "2026-01-01 04:00:00+00:00",
        "values": {"low": low, "close": close},
    }


def minute_event(child_level=105.0, reclaim_close=101.0, *, event_id="minute-1"):
    return {
        "id": event_id,
        "contract_id": "native-minute-event.v1",
        "kind": "minute_equal_low_sweep",
        "instrument": "BTC-USD",
        "data_stream_id": "same-stream",
        "evidence_id": "minute-evidence-1",
        "first_sweep_open": "2026-01-01 03:00:00+00:00",
        "reclaim_bar_open": "2026-01-01 03:01:00+00:00",
        "decision_time": "2026-01-01 03:02:00+00:00",
        "available_at": "2026-01-01 03:02:00+00:00",
        "values": {
            "child_level": child_level,
            "sweep_low": 99.0,
            "reclaim_close": reclaim_close,
        },
    }


@pytest.mark.parametrize("low,close,expected", LC_CASES)
def test_lc_geometry_uses_strict_breach_and_reclaim_bounds(low, close, expected):
    """Break caught: relaxing any strict LC comparison at the frozen 100/120 bounds."""
    original = ledger()
    event = lc_event(low, close)

    result = evaluate_h3_permission(original, policy_id=LC_POLICY, child_event=event)

    assert result["status"] == ("pass" if expected else "reject")
    assert result["would_allow"] is expected
    assert result["reasons"] == ([] if expected else ["frozen_geometry_rejected"])
    assert result["evaluated_values"] == {
        "low": low,
        "close": close,
        "parent_low": 100.0,
        "parent_high": 120.0,
    }
    assert original == ledger()


@pytest.mark.parametrize("level,reclaim_close,expected", MINUTE_CASES)
def test_minute_geometry_has_inclusive_lower_half_and_strict_reclaim(
    level, reclaim_close, expected
):
    """Break caught: moving the minute midpoint/low boundaries or admitting bound reclaims."""
    result = evaluate_h3_permission(
        ledger(),
        policy_id=MINUTE_POLICY,
        child_event=minute_event(level, reclaim_close),
    )

    assert result["status"] == ("pass" if expected else "reject")
    assert result["would_allow"] is expected
    assert result["reasons"] == ([] if expected else ["frozen_geometry_rejected"])
    assert result["evaluated_values"] == {
        "child_level": level,
        "sweep_low": 99.0,
        "reclaim_close": reclaim_close,
        "parent_low": 100.0,
        "parent_high": 120.0,
        "parent_midpoint": 110.0,
    }


def _break(row):
    row.update(
        source_break_direction="down",
        source_range_state="broken_down",
        post_state="broken_down",
        post_range_low=100.0,
        post_range_high=120.0,
        post_lineage_id=None,
        post_version_id=None,
        adopted_low_pivot_id=None,
        adopted_high_pivot_id=None,
    )


def _tighten(book, hour):
    version = _version(
        "version-2",
        predecessor="version-1",
        reason="floor_tightening",
        low=105.0,
        formation_hour=f"2026-01-01 {hour:02d}:00:00+00:00",
        available_at=f"2026-01-01 {hour + 1:02d}:00:00+00:00",
    )
    book["versions"].append(version)
    config = book["manifest"]["parameters"]
    book["pivots"].append(
        _pivot(
            "pivot-low-version-2",
            "low",
            105.0,
            anchor_timeframe=config["anchor_timeframe"],
            pivot_n=config["pivot_n"],
        )
    )
    row = book["transitions"][hour]
    row.update(
        pre_version_id="version-1",
        evaluated_version_id="version-1",
        pre_lineage_id="lineage-1",
        post_version_id="version-2",
        post_lineage_id="lineage-1",
        source_range_low=105.0,
        post_range_low=105.0,
        adopted_low_pivot_id="pivot-low-version-2",
    )
    for later in book["transitions"][hour + 1 :]:
        later.update(
            pre_range_low=105.0,
            pre_lineage_id="lineage-1",
            pre_version_id="version-2",
            evaluated_version_id="version-2",
            source_range_low=105.0,
            post_range_low=105.0,
            post_lineage_id="lineage-1",
            post_version_id="version-2",
            adopted_low_pivot_id="pivot-low-version-2",
        )


@pytest.mark.parametrize("hour", [2, 3])
def test_bound_lineage_break_at_sweep_or_decision_rejects(hour):
    """Break caught: consuming frozen geometry after its bound lineage broke."""
    book = ledger()
    _break(book["transitions"][hour])
    if hour == 2:
        book["transitions"][3].update(
            pre_state="broken_down",
            pre_range_low=None,
            pre_range_high=None,
            pre_lineage_id=None,
            pre_version_id=None,
            evaluated_version_id=None,
            source_range_state="forming",
            post_state="forming",
            post_range_low=None,
            post_range_high=None,
            post_lineage_id=None,
            post_version_id=None,
            adopted_low_pivot_id=None,
            adopted_high_pivot_id=None,
        )

    result = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=lc_event())

    assert result["status"] == "reject"
    assert result["would_allow"] is False
    assert result["reasons"] == ["bound_lineage_broken"]
    assert result["bound_lineage_broken"] is True


def test_transition_after_decision_is_not_consumed_or_hashed():
    """Break caught: future transition leakage into a fixed event result or its ID."""
    book = ledger()
    baseline = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=lc_event())
    _break(book["transitions"][4])

    future_extended = evaluate_h3_permission(
        book, policy_id=LC_POLICY, child_event=lc_event()
    )

    assert future_extended == baseline
    assert baseline["interval_transition_ids"] == ["transition-2", "transition-3"]


def test_unvalidated_intrahour_tail_cannot_supply_a_parent():
    """Break caught: bind_parent consuming a malformed row beyond the validated prefix."""
    book = ledger()
    fake = _version(
        "version-fake",
        lineage_id="lineage-fake",
        low=90.0,
        high=130.0,
        formation_hour="2026-01-01 02:00:00+00:00",
        available_at="2026-01-01 03:00:00+00:00",
    )
    book["versions"].append(fake)
    injected = deepcopy(book["transitions"][2])
    injected.update(
        id="transition-injected",
        source_hour="2026-01-01 02:15:00+00:00",
        available_at="2026-01-01 03:15:00+00:00",
        post_lineage_id="lineage-fake",
        post_version_id="version-fake",
        post_range_low=90.0,
        post_range_high=130.0,
    )
    book["transitions"].insert(3, injected)
    event = minute_event(child_level=95.0, reclaim_close=101.0)
    event["values"]["sweep_low"] = 89.0
    event.update(
        first_sweep_open="2026-01-01 03:30:00+00:00",
        reclaim_bar_open="2026-01-01 03:30:00+00:00",
        decision_time="2026-01-01 03:31:00+00:00",
        available_at="2026-01-01 03:31:00+00:00",
    )

    result = evaluate_h3_permission(book, policy_id=MINUTE_POLICY, child_event=event)

    assert result["status"] == "unknown"
    assert result["would_allow"] is False
    assert result["reasons"] == ["malformed_causal_prefix"]
    assert result["binding"] is None


def test_valid_future_append_does_not_change_fixed_event_result():
    """Break caught: treating a valid later ledger extension as consumed evidence."""
    book = ledger()
    baseline = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=lc_event())
    extended = deepcopy(book)
    extended["coverage"].update(
        last_processed_close="2026-01-01 06:00:00+00:00",
        query_exclusive_end="2026-01-01 07:00:00+00:00",
        input_hours=6,
    )
    extended["transitions"].append(
        _transition(
            5,
            pre_version="version-1",
            pre_lineage="lineage-1",
            pre_state="active",
        )
    )

    assert evaluate_h3_permission(
        extended, policy_id=LC_POLICY, child_event=lc_event()
    ) == baseline


def test_duplicate_transition_identity_in_consumed_prefix_is_unknown():
    """Break caught: ambiguous interval evidence reusing one transition identity twice."""
    book = ledger()
    book["transitions"][2]["id"] = book["transitions"][1]["id"]

    result = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=lc_event())

    assert result["status"] == "unknown"
    assert result["reasons"] == ["malformed_causal_prefix"]


def test_causal_prefix_must_start_from_constructor_cold_state():
    """Break caught: accepting a carried active parent before the ledger's first row."""
    book = ledger()
    book["transitions"][0].update(
        pre_state="active",
        pre_range_low=100.0,
        pre_range_high=120.0,
        pre_lineage_id="lineage-1",
        pre_version_id="version-1",
        evaluated_version_id="version-1",
    )

    result = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=lc_event())

    assert result["status"] == "unknown"
    assert result["reasons"] == ["malformed_causal_prefix"]


def test_future_version_cannot_be_consumed_or_bound_before_availability():
    """Break caught: a future version entering through an initially-active first row."""
    book = ledger()
    book["transitions"][0].update(
        pre_state="active",
        pre_range_low=100.0,
        pre_range_high=120.0,
        pre_lineage_id="lineage-1",
        pre_version_id="version-1",
        evaluated_version_id="version-1",
    )
    book["versions"][0].update(
        formation_hour="2026-01-01 10:00:00+00:00",
        available_at="2026-01-01 11:00:00+00:00",
    )

    result = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=lc_event())

    assert result["status"] == "unknown"
    assert result["would_allow"] is False
    assert result["reasons"] == ["malformed_causal_prefix"]
    assert result["binding"] is None


def test_break_between_minute_sweep_and_reclaim_close_rejects():
    """Break caught: inspecting only child boundary transitions and missing an interval break."""
    book = ledger()
    _break(book["transitions"][2])
    event = minute_event()
    event.update(
        first_sweep_open="2026-01-01 02:30:00+00:00",
        reclaim_bar_open="2026-01-01 03:29:00+00:00",
        decision_time="2026-01-01 03:30:00+00:00",
        available_at="2026-01-01 03:30:00+00:00",
    )

    result = evaluate_h3_permission(book, policy_id=MINUTE_POLICY, child_event=event)

    assert result["status"] == "reject"
    assert result["reasons"] == ["bound_lineage_broken"]
    assert result["interval_transition_ids"] == ["transition-2"]


@pytest.mark.parametrize("hour,expected_superseded", [(2, True), (3, True), (4, False)])
def test_tightening_never_rewrites_frozen_geometry(hour, expected_superseded):
    """Break caught: applying a new floor to the already-bound event geometry."""
    book = ledger()
    _tighten(book, hour)

    result = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=lc_event())

    assert result["status"] == "pass"
    assert result["reasons"] == []
    assert result["bound_version_superseded"] is expected_superseded
    assert result["evaluated_values"]["parent_low"] == 100.0


def test_tightening_between_minute_sweep_and_reclaim_close_only_supersedes():
    """Break caught: treating an interval tightening as a veto or mutable geometry."""
    book = ledger()
    _tighten(book, 2)
    event = minute_event()
    event.update(
        first_sweep_open="2026-01-01 02:30:00+00:00",
        reclaim_bar_open="2026-01-01 03:29:00+00:00",
        decision_time="2026-01-01 03:30:00+00:00",
        available_at="2026-01-01 03:30:00+00:00",
    )

    result = evaluate_h3_permission(book, policy_id=MINUTE_POLICY, child_event=event)

    assert result["status"] == "pass"
    assert result["bound_version_superseded"] is True
    assert result["evaluated_values"]["parent_low"] == 100.0


def test_break_after_tightening_is_a_definite_lifecycle_reject():
    """Break caught: treating a post-tightening break as unknown or testing the old floor."""
    book = ledger()
    _tighten(book, 2)
    break_row = book["transitions"][3]
    break_row.update(
        pre_version_id="version-2",
        evaluated_version_id="version-2",
        pre_lineage_id="lineage-1",
        pre_range_low=105.0,
    )
    _break(break_row)

    result = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=lc_event())

    assert result["status"] == "reject"
    assert result["reasons"] == ["bound_lineage_broken"]
    assert result["bound_lineage_broken"] is True
    assert result["bound_version_superseded"] is True


def test_break_then_reformation_does_not_rescue_bound_lineage():
    """Break caught: authorizing from a same-anchor reformation after interval break."""
    book = ledger()
    _break(book["transitions"][2])
    reformed = _version(
        "version-reformed",
        lineage_id="lineage-2",
        formation_hour="2026-01-01 03:00:00+00:00",
        available_at="2026-01-01 04:00:00+00:00",
    )
    book["versions"].append(reformed)
    book["pivots"].append(_pivot("pivot-low-version-reformed", "low", 100.0))
    book["transitions"][3].update(
        pre_state="broken_down",
        pre_range_low=None,
        pre_range_high=None,
        pre_lineage_id=None,
        pre_version_id=None,
        evaluated_version_id=None,
        source_range_state="active",
        post_state="active",
        post_lineage_id="lineage-2",
        post_version_id="version-reformed",
        adopted_low_pivot_id="pivot-low-version-reformed",
    )

    result = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=lc_event())

    assert result["status"] == "reject"
    assert result["reasons"] == ["bound_lineage_broken"]
    assert result["binding"]["parent_lineage_id"] == "lineage-1"


def test_reformation_strictly_before_sweep_is_eligible():
    """Break caught: permanently vetoing a newly formed lineage visible before the sweep."""
    book = ledger()
    _break(book["transitions"][1])
    reformed = _version(
        "version-reformed",
        lineage_id="lineage-2",
        formation_hour="2026-01-01 02:00:00+00:00",
        available_at="2026-01-01 03:00:00+00:00",
    )
    book["versions"].append(reformed)
    book["pivots"].append(_pivot("pivot-low-version-reformed", "low", 100.0))
    book["transitions"][2].update(
        pre_state="broken_down",
        pre_range_low=None,
        pre_range_high=None,
        pre_lineage_id=None,
        pre_version_id=None,
        evaluated_version_id=None,
        source_range_state="active",
        post_state="active",
        post_lineage_id="lineage-2",
        post_version_id="version-reformed",
        adopted_low_pivot_id="pivot-low-version-reformed",
    )
    for row in book["transitions"][3:]:
        row.update(
            pre_lineage_id="lineage-2",
            post_lineage_id="lineage-2",
            pre_version_id="version-reformed",
            evaluated_version_id="version-reformed",
            post_version_id="version-reformed",
            adopted_low_pivot_id="pivot-low-version-reformed",
        )

    event = lc_event()
    event.update(
        first_sweep_open="2026-01-01 04:00:00+00:00",
        reclaim_bar_open="2026-01-01 04:00:00+00:00",
        decision_time="2026-01-01 05:00:00+00:00",
        available_at="2026-01-01 05:00:00+00:00",
    )
    result = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=event)

    assert result["status"] == "pass"
    assert result["binding"]["parent_lineage_id"] == "lineage-2"


def test_parent_formed_at_sweep_time_is_absent_not_unknown():
    """Break caught: equal-time parent formation leaking into strict sweep binding."""
    book = ledger()
    equal_version = _version(
        available_at="2026-01-01 03:00:00+00:00",
        formation_hour="2026-01-01 02:00:00+00:00",
    )
    book["versions"] = [equal_version]
    for row in book["transitions"][:2]:
        row.update(
            pre_state="forming",
            pre_range_low=None,
            pre_range_high=None,
            pre_lineage_id=None,
            pre_version_id=None,
            evaluated_version_id=None,
            source_range_state="forming",
            post_state="forming",
            post_range_low=None,
            post_range_high=None,
            post_lineage_id=None,
            post_version_id=None,
            adopted_low_pivot_id=None,
            adopted_high_pivot_id=None,
        )
    book["transitions"][2].update(
        pre_state="forming",
        pre_range_low=None,
        pre_range_high=None,
        pre_lineage_id=None,
        pre_version_id=None,
        evaluated_version_id=None,
    )

    result = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=lc_event())

    assert result["status"] == "reject"
    assert result["reasons"] == ["absent_parent"]
    assert result["binding"]["status"] == "rejected"


def test_parent_formed_after_sweep_cannot_authorize_event():
    """Break caught: binding a parent created during the child event interval."""
    book = ledger()
    late_version = _version(
        available_at="2026-01-01 04:00:00+00:00",
        formation_hour="2026-01-01 03:00:00+00:00",
    )
    book["versions"] = [late_version]
    for row in book["transitions"][:3]:
        row.update(
            pre_state="forming",
            pre_range_low=None,
            pre_range_high=None,
            pre_lineage_id=None,
            pre_version_id=None,
            evaluated_version_id=None,
            source_range_state="forming",
            post_state="forming",
            post_range_low=None,
            post_range_high=None,
            post_lineage_id=None,
            post_version_id=None,
            adopted_low_pivot_id=None,
            adopted_high_pivot_id=None,
        )
    book["transitions"][3].update(
        pre_state="forming",
        pre_range_low=None,
        pre_range_high=None,
        pre_lineage_id=None,
        pre_version_id=None,
        evaluated_version_id=None,
    )

    result = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=lc_event())

    assert result["status"] == "reject"
    assert result["reasons"] == ["absent_parent"]


@pytest.mark.parametrize(
    "bad_value",
    [None, float("nan"), float("inf"), True, "99"],
)
def test_missing_nonfinite_boolean_or_string_numeric_evidence_is_unknown(bad_value):
    """Break caught: coercing absent or invalid numeric evidence into a decision."""
    event = lc_event()
    event["values"]["low"] = bad_value

    result = evaluate_h3_permission(ledger(), policy_id=LC_POLICY, child_event=event)

    assert result["status"] == "unknown"
    assert result["reasons"] == ["invalid_child_event"]


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (
            lambda book: book.update(transitions=book["transitions"][:3]),
            "insufficient_causal_coverage",
        ),
        (
            lambda book: book["transitions"][3].update(post_version_id="missing-version"),
            "malformed_causal_prefix",
        ),
        (
            lambda book: book["pivots"].pop(0),
            "malformed_causal_prefix",
        ),
        (
            lambda book: book["transitions"][2].update(
                post_lineage_id=None, post_version_id=None, post_state="forming"
            ),
            "malformed_causal_prefix",
        ),
    ],
)
def test_incomplete_or_unexplained_causal_prefix_is_unknown(mutate, reason):
    """Break caught: converting incomplete provenance into a rule rejection or permission."""
    book = ledger()
    mutate(book)

    result = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=lc_event())

    assert result["status"] == "unknown"
    assert result["would_allow"] is False
    assert result["reasons"] == [reason]


@pytest.mark.parametrize(
    "mutate,reason",
    [
        (lambda event: event.update(decision_time="2026-01-01 04:01:00+00:00"), "invalid_child_event"),
        (lambda event: event["values"].update(low=float("nan")), "invalid_child_event"),
        (lambda event: event["values"].update(low=True), "invalid_child_event"),
        (lambda event: event.update(instrument="ETH-USD"), "instrument_mismatch"),
        (lambda event: event.update(data_stream_id="other-stream"), "data_stream_mismatch"),
    ],
)
def test_invalid_event_or_source_identity_is_unknown(mutate, reason):
    """Break caught: treating invalid evidence as a strategy-rule rejection."""
    event = lc_event()
    mutate(event)

    result = evaluate_h3_permission(ledger(), policy_id=LC_POLICY, child_event=event)

    assert result["status"] == "unknown"
    assert result["would_allow"] is False
    assert result["reasons"] == [reason]
    assert result["certified"] is False


@pytest.mark.parametrize(
    "mutate",
    [
        lambda book: book.update(coverage=None),
        lambda book: book["manifest"]["parameters"].update(anchor_timeframe=[]),
    ],
)
def test_malformed_ledger_containers_return_unknown(mutate):
    """Break caught: leaking AttributeError/TypeError instead of evidence-unknown."""
    book = ledger()
    mutate(book)

    result = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=lc_event())

    assert result["status"] == "unknown"
    assert result["would_allow"] is False
    assert result["reasons"] == ["invalid_parent_ledger"]


def test_equivalent_offset_clocks_have_identical_utc_result_and_id():
    """Break caught: timezone spelling changing causal ordering or record identity."""
    utc_result = evaluate_h3_permission(
        ledger(), policy_id=LC_POLICY, child_event=lc_event()
    )
    offset_book = deepcopy(ledger())
    offset_book["versions"][0]["available_at"] = "2025-12-31 17:00:00-08:00"
    event = lc_event()
    event.update(
        first_sweep_open="2025-12-31 19:00:00-08:00",
        reclaim_bar_open="2025-12-31 19:00:00-08:00",
        decision_time="2025-12-31 20:00:00-08:00",
        available_at="2025-12-31 20:00:00-08:00",
    )

    offset_result = evaluate_h3_permission(
        offset_book, policy_id=LC_POLICY, child_event=event
    )

    assert offset_result == utc_result
    assert offset_result["decision_time"] == "2026-01-01 04:00:00+00:00"


def test_unknown_policy_raises_and_outputs_do_not_alias_inputs():
    """Break caught: silently accepting an unregistered rule or returning mutable aliases."""
    with pytest.raises(ValueError, match="unknown.*policy"):
        evaluate_h3_permission(ledger(), policy_id="invented", child_event=lc_event())

    book = ledger()
    event = lc_event()
    result = evaluate_h3_permission(book, policy_id=LC_POLICY, child_event=event)
    result["binding"]["parent_range_low"] = -1
    result["comparison_contract"]["geometry"] = "changed"
    assert book == ledger()
    assert event == lc_event()


def test_batch_requires_four_unique_hypotheses_and_counts_every_event():
    """Break caught: dropping configurations/events or turning counts into winner selection."""
    books = [ledger(anchor_timeframe=tf, pivot_n=n) for tf, n in sorted({
        ("4H", 3), ("4H", 5), ("1D", 3), ("1D", 5)
    })]
    events = [lc_event(event_id="pass"), lc_event(100, 101, event_id="reject")]

    batch = annotate_h3_events(books, policy_id=LC_POLICY, child_events=events)

    assert len(batch["rows"]) == 8
    assert {row["child_event_id"] for row in batch["rows"]} == {"pass", "reject"}
    assert batch["counts"] == {
        "1D:3": {"pass": 1, "reject": 1, "unknown": 0},
        "1D:5": {"pass": 1, "reject": 1, "unknown": 0},
        "4H:3": {"pass": 1, "reject": 1, "unknown": 0},
        "4H:5": {"pass": 1, "reject": 1, "unknown": 0},
    }
    assert batch["certified"] is False
    assert "winner" not in batch

    with pytest.raises(ValueError, match="exactly four unique"):
        annotate_h3_events(books[:3], policy_id=LC_POLICY, child_events=events)
    with pytest.raises(ValueError, match="exactly four unique"):
        annotate_h3_events(books[:3] + [books[0]], policy_id=LC_POLICY, child_events=events)


def test_batch_counts_invalid_event_as_unknown_for_every_configuration():
    """Break caught: indexing a missing result config instead of emitting unknown rows."""
    books = [
        ledger(anchor_timeframe=timeframe, pivot_n=n)
        for timeframe, n in [("4H", 3), ("4H", 5), ("1D", 3), ("1D", 5)]
    ]

    batch = annotate_h3_events(books, policy_id=LC_POLICY, child_events=[{}])

    assert len(batch["rows"]) == 4
    assert all(row["status"] == "unknown" for row in batch["rows"])
    assert all(row["parent_config"] is not None for row in batch["rows"])
    assert batch["counts"] == {
        "1D:3": {"pass": 0, "reject": 0, "unknown": 1},
        "1D:5": {"pass": 0, "reject": 0, "unknown": 1},
        "4H:3": {"pass": 0, "reject": 0, "unknown": 1},
        "4H:5": {"pass": 0, "reject": 0, "unknown": 1},
    }


def test_batch_counts_valid_configs_with_malformed_coverage_as_unknown():
    """Break caught: malformed ledger containers crashing a valid four-config batch."""
    books = [
        ledger(anchor_timeframe=timeframe, pivot_n=n)
        for timeframe, n in [("4H", 3), ("4H", 5), ("1D", 3), ("1D", 5)]
    ]
    for book in books:
        book["coverage"] = None

    batch = annotate_h3_events(books, policy_id=LC_POLICY, child_events=[lc_event()])

    assert len(batch["rows"]) == 4
    assert all(row["status"] == "unknown" for row in batch["rows"])
    assert all(row["reasons"] == ["invalid_parent_ledger"] for row in batch["rows"])
    assert sum(statuses["unknown"] for statuses in batch["counts"].values()) == 4


def test_batch_rejects_unhashable_configuration_with_documented_value_error():
    """Break caught: unhashable config types escaping the batch as raw TypeError."""
    books = [
        ledger(anchor_timeframe=timeframe, pivot_n=n)
        for timeframe, n in [("4H", 3), ("4H", 5), ("1D", 3), ("1D", 5)]
    ]
    books[0]["manifest"]["parameters"]["anchor_timeframe"] = []

    with pytest.raises(ValueError, match="exactly four unique parent configurations required"):
        annotate_h3_events(books, policy_id=LC_POLICY, child_events=[lc_event()])


def _real_bars(periods=96):
    index = pd.date_range("2026-01-01", periods=periods, freq="h", tz="UTC")
    x = np.arange(len(index))
    center = 110 + 7 * np.sin(x / 35.0) + 2 * np.sin(x / 7.0)
    return pd.DataFrame(
        {
            "open": center,
            "high": center + 3,
            "low": center - 3,
            "close": center,
            "volume": np.ones(len(x)),
            # Independent fixture input: constant 2.0, available at each hour close.
            "atr_14": np.full(len(x), 2.0),
        },
        index=index,
    )


def _real_ledger(bars, anchor_timeframe="4H", pivot_n=3):
    return build_parent_ledger(
        bars,
        instrument="BTC-USD",
        data_stream_id="same-hourly-fixture",
        anchor_timeframe=anchor_timeframe,
        pivot_n=pivot_n,
        atr_contract=ATR_CONTRACT,
        source_paths=SOURCE_PATHS,
        expected_hashes=EXPECTED_HASHES,
    )


@pytest.mark.skipif(not LOCAL_SOURCE_AVAILABLE, reason="hashed recovered sibling source unavailable")
def test_recovered_source_prefix_restart_nonmutation_and_batch_multiplicity():
    """Break caught: future input or rebuild state changing a fixed real-ledger decision."""
    bars = _real_bars()
    event = lc_event(low=110.0, close=115.0, event_id="real-lc")
    event.update(
        data_stream_id="same-hourly-fixture",
        first_sweep_open="2026-01-02 22:00:00+00:00",
        reclaim_bar_open="2026-01-02 22:00:00+00:00",
        decision_time="2026-01-02 23:00:00+00:00",
        available_at="2026-01-02 23:00:00+00:00",
    )
    original_event = deepcopy(event)
    full = _real_ledger(bars)
    original_full = deepcopy(full)
    prefix = _real_ledger(bars.iloc[:48])
    restarted = _real_ledger(bars)

    full_result = evaluate_h3_permission(full, policy_id=LC_POLICY, child_event=event)
    prefix_result = evaluate_h3_permission(prefix, policy_id=LC_POLICY, child_event=event)
    restart_result = evaluate_h3_permission(restarted, policy_id=LC_POLICY, child_event=event)

    assert full_result == prefix_result == restart_result
    assert full_result["status"] == "pass"
    assert full_result["binding"]["parent_version_id"] == (
        "9014565abbe5ebde6d37687bbf6137a451d7a5218610b632f82e559fa2aef789"
    )
    assert full_result["evaluated_values"]["parent_low"] == 110.47113366740557
    assert full_result["evaluated_values"]["parent_high"] == 117.59999709143082
    assert full == original_full
    assert event == original_event

    real_books = [
        _real_ledger(bars, timeframe, n)
        for timeframe, n in [("4H", 3), ("4H", 5), ("1D", 3), ("1D", 5)]
    ]
    batch = annotate_h3_events(real_books, policy_id=LC_POLICY, child_events=[event])
    assert len(batch["rows"]) == 4
    assert sum(sum(statuses.values()) for statuses in batch["counts"].values()) == 4
    assert all(row["certified"] is False for row in batch["rows"])
