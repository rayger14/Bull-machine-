import copy
import json
import math

import pytest

from scripts.research.fusion_live_scorecard import build_fusion_scorecard


META = {
    "server_time": "2026-01-02T01:00:00Z",
    "heartbeat_updated_at": "2026-01-02T00:00:00Z",
    "source_hashes": {"trades": "abc", "status": "def", "signals": "ghi"},
}


def exit_row(**changes):
    row = {
        "position_id": "p1",
        "archetype": "test_long",
        "direction": "long",
        "timestamp_entry": "2026-01-01T00:00:00Z",
        "timestamp_exit": "2026-01-01T01:00:00Z",
        "entry_price": 100.0,
        "exit_price": 110.0,
        "quantity": 1.0,
        "position_size_usd": 100.0,
        "pnl_usd": 10.0,
        "pnl": 10.0,
        "pnl_pct": 10.0,
        "fusion_score": 0.30,
        "threshold_at_entry": 0.25,
        "threshold_margin": 0.05,
        "stop_loss": 90.0,
        "take_profit": 120.0,
        "exit_reason": "synthetic_exit",
        "duration_hours": 1.0,
        "entry_regime": "synthetic_regime",
        "atr_at_entry": 2.0,
        "crisis_prob_at_entry": 0.1,
        "instability_at_entry": 0.2,
        "leverage_applied": 1.0,
        "risk_temp_at_entry": 0.3,
        "factor_attribution": {"entry_conditions": {"dynamic_threshold": 0.27}},
        "source_version": "epoch1",
    }
    row.update(changes)
    return row


def build(rows, *, opens=None, signals=None, archetypes=None, meta=None):
    return build_fusion_scorecard(
        rows,
        open_positions=[] if opens is None else opens,
        signal_rows=[] if signals is None else signals,
        archetypes=["test_long", "test_short", "unused"]
        if archetypes is None
        else archetypes,
        snapshot_meta=META if meta is None else meta,
    )


def open_row(**changes):
    row = {
        "id": "p1",
        "archetype": "test_long",
        "direction": "long",
        "entry_time": "2026-01-01T00:00:00Z",
        "entry_price": 100.0,
        "current_quantity": 1.0,
        "original_quantity": 2.0,
    }
    row.update(changes)
    return row


def test_partial_exits_are_one_observation():
    rows = [
        exit_row(),
        exit_row(
            timestamp_exit="2026-01-01T02:00:00Z", pnl_usd=-5.0, pnl=-5.0
        ),
    ]

    report = build_fusion_scorecard(
        rows,
        open_positions=[],
        signal_rows=[],
        archetypes=["test_long", "unused"],
        snapshot_meta=META,
    )

    assert report["certified"] is False
    assert len(report["groups"]) == 1
    assert report["groups"][0]["row_indices"] == [0, 1]
    assert report["groups"][0]["recorded_exit_pnl_usd"] == 5.0
    assert report["groups"][0]["quantity_sum"] == 2.0
    assert report["groups"][0]["displayed_stop_risk_proxy_usd"] == 20.0
    assert report["groups"][0]["recorded_pnl_over_displayed_stop_risk_proxy"] == 0.25
    assert report["groups"][0]["source_label_duration_hours"] == 2.0
    assert report["summary"]["n"] == 1
    assert report["summary"]["wins"] == 1
    assert report["by_archetype"]["unused"]["n"] == 0


def test_full_shape_raw_api_exit_schema_maps_to_conceptual_output_fields():
    report = build([exit_row()])

    assert len(report["groups"]) == 1
    group = report["groups"][0]
    assert group["entry_time"] == "2026-01-01T00:00:00Z"
    assert group["first_exit_time"] == "2026-01-01T01:00:00Z"
    assert group["last_exit_time"] == "2026-01-01T01:00:00Z"
    assert group["displayed_stop_loss"] == 90.0


def test_invented_aliases_do_not_satisfy_missing_raw_exit_fields():
    row = exit_row()
    row["entry_time"] = row.pop("timestamp_entry")
    row["exit_time"] = row.pop("timestamp_exit")
    row["displayed_stop_loss"] = row.pop("stop_loss")

    report = build([row])

    assert report["groups"] == []
    assert report["quarantined_groups"][0]["row_indices"] == [0]
    assert {
        "invalid_entry_time",
        "invalid_exit_time",
        "invalid_displayed_stop_loss",
    }.issubset(report["quarantined_groups"][0]["reasons"])


def test_summary_and_logged_margin_cohorts_use_group_subtotals():
    rows = [
        exit_row(),
        exit_row(
            timestamp_exit="2026-01-01T02:00:00Z", pnl_usd=-5.0, pnl=-5.0
        ),
        exit_row(
            position_id="p2",
            archetype="test_short",
            direction="short",
            timestamp_exit="2026-01-01T03:00:00Z",
            stop_loss=110.0,
            pnl_usd=-5.0,
            pnl=-5.0,
            fusion_score=0.2,
            threshold_at_entry=0.3,
            threshold_margin=-0.1,
        ),
    ]

    report = build(rows)

    assert report["summary"]["n"] == 2
    assert report["summary"]["recorded_exit_pnl_usd_sum"] == 0.0
    assert report["summary"]["wins"] == 1
    assert report["summary"]["losses"] == 1
    assert report["summary"]["breakevens"] == 0
    assert report["summary"]["win_fraction"] == 0.5
    assert report["summary"]["gross_positive_pnl_usd"] == 5.0
    assert report["summary"]["gross_negative_pnl_usd_abs"] == 5.0
    assert report["summary"]["recorded_subtotal_profit_factor"] == 1.0
    assert report["summary"]["zero_loss"] is False
    assert report["summary"]["margin_cohorts"]["logged_nonnegative_margin"]["n"] == 1
    assert report["summary"]["margin_cohorts"]["logged_negative_margin"]["n"] == 1


def test_blank_ids_are_excluded_and_rows_reconcile_once():
    report = build([exit_row(position_id="  "), exit_row(position_id=None)])

    assert report["groups"] == []
    assert report["quarantined_groups"] == []
    assert [row["row_index"] for row in report["excluded_rows"]] == [0, 1]
    assert report["coverage"]["raw_exit_rows"] == 2
    assert report["coverage"]["ungrouped_rows"] == 2
    assert report["coverage"]["accounted_exit_rows"] == 2


def test_duplicates_and_conflicting_metadata_quarantine_whole_groups():
    duplicate = exit_row(position_id="dup")
    conflict_rows = [
        exit_row(position_id="conflict"),
        exit_row(position_id="conflict", fusion_score=0.31),
    ]

    report = build([duplicate, copy.deepcopy(duplicate), *conflict_rows])

    assert report["groups"] == []
    assert [group["position_id"] for group in report["quarantined_groups"]] == [
        "conflict",
        "dup",
    ]
    assert report["quarantined_groups"][0]["row_indices"] == [2, 3]
    assert "conflicting_fusion_score" in report["quarantined_groups"][0]["reasons"]
    assert "exact_duplicate_row" in report["quarantined_groups"][1]["reasons"]
    assert report["coverage"]["quarantined_rows"] == 4


@pytest.mark.parametrize(
    ("changes", "reason"),
    [
        ({"pnl_usd": float("inf")}, "invalid_pnl_usd"),
        ({"quantity": "1"}, "invalid_quantity"),
        ({"fusion_score": True}, "invalid_fusion_score"),
        ({"entry_price": 0}, "invalid_entry_price"),
    ],
)
def test_invalid_numeric_row_contaminates_its_whole_id_group(changes, reason):
    report = build([exit_row(), exit_row(**changes)])

    assert report["groups"] == []
    assert report["quarantined_groups"][0]["row_indices"] == [0, 1]
    assert reason in report["quarantined_groups"][0]["reasons"]


def test_utc_equivalent_entry_clocks_group_but_invalid_clock_is_quarantined():
    equivalent = build(
        [
            exit_row(timestamp_entry="2026-01-01T00:00:00Z"),
            exit_row(
                timestamp_entry="2025-12-31T19:00:00-05:00",
                timestamp_exit="2026-01-01T02:00:00+00:00",
            ),
        ]
    )
    invalid = build(
        [
            exit_row(
                position_id="late", timestamp_entry="2026-01-01T04:00:00Z"
            ),
            exit_row(position_id="late"),
        ]
    )

    assert len(equivalent["groups"]) == 1
    assert equivalent["groups"][0]["entry_time"] == "2026-01-01T00:00:00Z"
    assert invalid["groups"] == []
    assert "entry_after_exit" in invalid["quarantined_groups"][0]["reasons"]


@pytest.mark.parametrize(
    "meta",
    [
        {**META, "server_time": "2026-01-02T01:00:00"},
        {**META, "heartbeat_updated_at": "2026-01-03T00:00:00Z"},
        {"server_time": META["server_time"], "heartbeat_updated_at": META["heartbeat_updated_at"]},
    ],
)
def test_snapshot_clocks_and_source_hashes_are_required(meta):
    with pytest.raises(ValueError):
        build([], meta=meta)


def test_zero_score_is_real_but_zero_threshold_is_an_unknown_sentinel():
    report = build(
        [
            exit_row(
                position_id="zero-score",
                fusion_score=0.0,
                threshold_at_entry=0.25,
                threshold_margin=-0.25,
            ),
            exit_row(
                position_id="zero-threshold",
                fusion_score=0.4,
                threshold_at_entry=0.0,
                threshold_margin=0.4,
            ),
        ]
    )

    assert len(report["groups"]) == 2
    assert report["coverage"]["threshold_zero_groups"] == 1
    assert report["summary"]["n"] == 1
    assert report["groups"][0]["fusion_score"] == 0.0
    assert report["groups"][1]["score_margin_arithmetic_consistent"] is False


def test_stored_threshold_is_not_replaced_by_display_threshold_and_rounding_is_flagged():
    report = build(
        [
            exit_row(
                fusion_score=0.3,
                threshold_at_entry=0.30005,
                threshold_margin=0.0,
                factor_attribution={"entry_conditions": {"dynamic_threshold": 0.9}},
            ),
            exit_row(
                timestamp_exit="2026-01-01T02:00:00Z",
                fusion_score=0.3,
                threshold_at_entry=0.30005,
                threshold_margin=0.0,
                factor_attribution={"entry_conditions": {"dynamic_threshold": 0.8}},
            ),
        ]
    )
    group = report["groups"][0]

    assert group["threshold_at_entry"] == 0.30005
    assert group["stored_score_minus_entry_threshold"] == pytest.approx(-0.00005)
    assert group["score_margin_arithmetic_consistent"] is True
    assert group["boundary_rounding_ambiguous"] is True
    assert group["display_threshold_varied"] is True
    assert group["display_vs_entry_threshold_difference"] is True
    assert report["coverage"]["display_threshold_varied_groups"] == 1


def test_invalid_display_threshold_is_coverage_only_and_high_stored_threshold_is_flagged():
    report = build(
        [
            exit_row(
                fusion_score=0.9,
                threshold_at_entry=1.1,
                threshold_margin=-0.2,
                factor_attribution={"entry_conditions": {"dynamic_threshold": "unknown"}},
            )
        ]
    )

    assert len(report["groups"]) == 1
    assert report["groups"][0]["display_threshold_values"] == []
    assert report["groups"][0]["entry_threshold_above_score_range"] is True
    assert report["coverage"]["display_threshold_missing_or_invalid_groups"] == 1


def test_contradictory_pnl_alias_quarantines_group():
    report = build([exit_row(pnl_usd=1.0, pnl=1.006)])

    assert report["groups"] == []
    assert "pnl_alias_mismatch" in report["quarantined_groups"][0]["reasons"]


def test_open_inventory_is_retained_and_open_exit_subtotal_is_not_summarized():
    report = build([exit_row()], opens=[open_row(fusion_score=None)])

    assert report["open_inventory"]["usable"] is True
    assert report["open_inventory"]["positions"][0]["fusion_score"] is None
    assert report["groups"][0]["open_in_snapshot"] is True
    assert report["groups"][0]["completion_certified"] is False
    assert report["summary"]["n"] == 0
    assert report["coverage"]["groups_with_open_id_match"] == 1
    assert report["coverage"]["groups_without_open_id_match"] == 0
    assert "recorded_exit_groups_not_open_in_snapshot" in report["limitations"]


def test_open_without_exit_is_retained_and_identity_conflict_quarantines_exit():
    report = build(
        [exit_row()],
        opens=[
            open_row(direction="short"),
            open_row(id="open-only", archetype="unused", current_quantity=2.0),
        ],
    )

    assert [row["id"] for row in report["open_inventory"]["positions"]] == [
        "open-only",
        "p1",
    ]
    assert report["groups"] == []
    assert "open_inventory_identity_conflict" in report["quarantined_groups"][0]["reasons"]


@pytest.mark.parametrize(
    "opens",
    [
        [open_row(), open_row()],
        [open_row(id="")],
        [open_row(current_quantity=3.0, original_quantity=2.0)],
        [open_row(entry_time="2026-01-03T00:00:00Z")],
        [open_row(fusion_score="missing")],
    ],
)
def test_duplicate_or_malformed_open_inventory_fails_closed(opens):
    with pytest.raises(ValueError):
        build([], opens=opens)


def test_short_and_adverse_side_risk_proxy_rules():
    report = build(
        [
            exit_row(
                position_id="short",
                archetype="test_short",
                direction="short",
                stop_loss=110,
                pnl_usd=-5,
                pnl=-5,
            ),
            exit_row(position_id="bad-long-stop", stop_loss=110),
            exit_row(position_id="zero-stop", stop_loss=0),
        ]
    )
    groups = {group["position_id"]: group for group in report["groups"]}

    assert groups["short"]["displayed_stop_risk_proxy_usd"] == 10.0
    assert groups["short"]["recorded_pnl_over_displayed_stop_risk_proxy"] == -0.5
    assert groups["bad-long-stop"]["displayed_stop_risk_proxy_usd"] is None
    assert groups["zero-stop"]["displayed_stop_risk_proxy_usd"] is None
    assert report["coverage"]["risk_proxy_groups"] == 1


def test_source_version_conflict_quarantines_and_missing_label_is_unknown():
    report = build(
        [
            exit_row(position_id="mixed", source_version="epoch1"),
            exit_row(position_id="mixed", source_version="epoch2"),
            exit_row(position_id="unknown", source_version=None),
        ]
    )

    assert [group["position_id"] for group in report["groups"]] == ["unknown"]
    assert report["groups"][0]["source_version"] == "unknown"
    assert report["coverage"]["source_version_known_groups"] == 0
    assert report["coverage"]["source_version_unknown_groups"] == 1
    assert "conflicting_source_version" in report["quarantined_groups"][0]["reasons"]


def test_spearman_uses_average_tied_ranks_and_reports_sample_size():
    rows = [
        exit_row(position_id="a", fusion_score=0.1, threshold_at_entry=0.05, threshold_margin=0.05, pnl_usd=1, pnl=1),
        exit_row(position_id="b", fusion_score=0.1, threshold_at_entry=0.05, threshold_margin=0.05, pnl_usd=2, pnl=2),
        exit_row(position_id="c", fusion_score=0.2, threshold_at_entry=0.05, threshold_margin=0.15, pnl_usd=3, pnl=3),
    ]

    correlation = build(rows)["summary"]["correlations"][
        "fusion_score_vs_recorded_exit_pnl_usd"
    ]

    assert correlation["n"] == 3
    assert correlation["spearman"] == pytest.approx(math.sqrt(3) / 2)


def test_spearman_negative_monotonic_and_constant_operands():
    rows = [
        exit_row(position_id="a", fusion_score=0.1, threshold_at_entry=0.05, threshold_margin=0.05, pnl_usd=3, pnl=3),
        exit_row(position_id="b", fusion_score=0.2, threshold_at_entry=0.05, threshold_margin=0.15, pnl_usd=2, pnl=2),
        exit_row(position_id="c", fusion_score=0.3, threshold_at_entry=0.05, threshold_margin=0.25, pnl_usd=1, pnl=1),
    ]
    correlations = build(rows)["summary"]["correlations"]

    assert correlations["fusion_score_vs_recorded_exit_pnl_usd"] == {
        "n": 3,
        "spearman": -1.0,
        "sparse": True,
    }
    assert correlations["threshold_at_entry_vs_recorded_exit_pnl_usd"] == {
        "n": 3,
        "spearman": None,
        "sparse": True,
    }


def test_empty_groups_and_zero_losses_return_null_profit_factor():
    empty = build([])
    winning = build([exit_row()])

    assert empty["summary"]["n"] == 0
    assert empty["summary"]["recorded_subtotal_profit_factor"] is None
    assert empty["summary"]["zero_loss"] is True
    assert winning["summary"]["recorded_subtotal_profit_factor"] is None
    assert winning["summary"]["zero_loss"] is True


def test_signal_coverage_is_descriptive_and_does_not_join_outcomes():
    signals = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "status": "selected",
            "fusion_score": 0.3,
            "threshold": 0.25,
            "margin": 0.05,
        },
        {
            "timestamp": "2025-12-31T20:00:00-05:00",
            "status": "discarded",
            "fusion_score": 0.2,
            "threshold": 0.3,
            "margin": -0.1,
        },
        {
            "timestamp": "bad",
            "status": "selected",
            "fusion_score": True,
            "threshold": 0.3,
            "margin": 0.1,
        },
    ]

    coverage = build([], signals=signals)["signal_coverage"]

    assert coverage["raw_rows"] == 3
    assert coverage["valid_rows"] == 2
    assert coverage["status_counts"] == {"discarded": 1, "selected": 2}
    assert coverage["valid_status_counts"] == {"discarded": 1, "selected": 1}
    assert coverage["time_range"] == {
        "first": "2026-01-01T00:00:00Z",
        "last": "2026-01-01T01:00:00Z",
    }
    assert coverage["arithmetic_consistent_rows"] == 2
    assert coverage["arithmetic_inconsistent_rows"] == 0
    assert coverage["malformed_rows"][0]["row_index"] == 2


@pytest.mark.parametrize(
    "kwargs",
    [
        {"trades": {}},
        {"open_positions": {}},
        {"signal_rows": {}},
        {"archetypes": []},
        {"archetypes": ["dup", "dup"]},
        {"archetypes": [""]},
        {"snapshot_meta": []},
    ],
)
def test_invalid_top_level_inputs_raise_value_error(kwargs):
    arguments = {
        "trades": [],
        "open_positions": [],
        "signal_rows": [],
        "archetypes": ["test_long"],
        "snapshot_meta": META,
    }
    arguments.update(kwargs)

    with pytest.raises(ValueError):
        build_fusion_scorecard(**arguments)


def test_report_is_repeatable_json_safe_and_does_not_mutate_inputs():
    trades = [exit_row()]
    opens = [open_row(id="open-only", archetype="unused")]
    signals = [
        {
            "timestamp": "2026-01-01T00:00:00Z",
            "status": "selected",
            "fusion_score": 0.3,
            "threshold": 0.25,
            "margin": 0.05,
        }
    ]
    originals = copy.deepcopy((trades, opens, signals, META))

    first = build(trades, opens=opens, signals=signals)
    second = build(copy.deepcopy(trades), opens=copy.deepcopy(opens), signals=copy.deepcopy(signals))

    assert first == second
    assert (trades, opens, signals, META) == originals
    json.dumps(first, allow_nan=False, sort_keys=True)
