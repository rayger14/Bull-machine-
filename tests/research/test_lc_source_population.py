from copy import deepcopy
import json

import pytest

from scripts.research import lc_source_population as module


def row(
    decision_time,
    *,
    direction=None,
    selected=True,
    bb_width=0.04,
    features_present=True,
):
    setup_open = module._utc(decision_time, "decision_time") - module.ONE_HOUR
    features = (
        {
            "timestamp": setup_open.isoformat(),
            "open": 100.0,
            "high": 102.0,
            "low": 99.0,
            "close": 101.0,
            "volume": 10.0,
            "bb_width": bb_width,
            "volume_zscore": 0.1,
            "rsi_14": 50.0,
            "chop_score": 0.9,
        }
        if features_present
        else None
    )
    native_signal = None if direction is None else {"direction": direction, "strength": 0.7}
    signal = (
        {
            "archetypes": {
                "liquidity_compression": {
                    "native_signal": native_signal,
                    "selected": selected,
                    "structural": {"passed": False, "reason": "fixture_h2_failure"},
                    "other_winners": ["not-lc"],
                }
            }
        }
        if features_present
        else None
    )
    return {
        "decision_time": decision_time,
        "observation_ids": {"macro": "obs-1"},
        "output": {
            "features": features,
            "features_available_at": decision_time if features_present else None,
            "feature_observation_ids": {"macro": "obs-1"},
            "feature_observation_visible_at": {"macro": decision_time},
            "engine_signal": signal,
        },
    }


def collect(rows):
    return module.collect_native_lc(
        rows,
        "2026-01-01T00:00:00Z",
        "2026-02-01T00:00:00Z",
    )


def test_native_long_survives_prior_compression_failure():
    rows = [
        row("2026-01-02T00:00:00Z", bb_width=0.2),
        row("2026-01-02T01:00:00Z", direction="long", selected=False),
    ]

    result = collect(rows)

    assert len(result) == 1
    assert result[0]["previous_features"]["bb_width"] == 0.2
    assert result[0]["native_emitted"] is False
    assert result[0]["candidate_id"] == "hourly-lc:2026-01-02T01:00:00+00:00"
    assert result[0]["setup_open"] == "2026-01-02T00:00:00+00:00"


def test_missing_previous_features_remains_none():
    result = collect([row("2026-01-02T01:00:00Z", direction="long")])

    assert result[0]["previous_features"] is None


@pytest.mark.parametrize("direction", [None, "short", "LONG", "flat"])
def test_only_exact_native_long_direction_is_retained(direction):
    assert collect([row("2026-01-02T01:00:00Z", direction=direction)]) == []


def test_window_is_start_inclusive_and_end_exclusive():
    rows = [
        row("2025-12-31T23:00:00Z"),
        row("2026-01-01T00:00:00Z", direction="long"),
        row("2026-02-01T00:00:00Z", direction="long"),
    ]

    result = collect(rows)

    assert [candidate["decision_time"] for candidate in result] == [
        "2026-01-01T00:00:00+00:00"
    ]


@pytest.mark.parametrize(
    "rows",
    [
        [row("2026-01-02T01:00:00Z"), row("2026-01-02T01:00:00Z")],
        [row("2026-01-02T02:00:00Z"), row("2026-01-02T01:00:00Z")],
    ],
    ids=["duplicate", "out-of-order"],
)
def test_duplicate_or_out_of_order_decision_clocks_are_rejected(rows):
    with pytest.raises(ValueError, match="strictly increasing"):
        collect(rows)


def test_naive_decision_clock_is_rejected():
    with pytest.raises(ValueError, match="timezone-aware"):
        collect([row("2026-01-02T01:00:00Z") | {"decision_time": "2026-01-02T01:00:00"}])


def test_absent_feature_row_interrupts_previous_hour_continuity():
    rows = [
        row("2026-01-02T00:00:00Z", bb_width=0.03),
        row("2026-01-02T01:00:00Z", features_present=False),
        row("2026-01-02T02:00:00Z", direction="long"),
    ]

    assert collect(rows)[0]["previous_features"] is None


def test_gap_interrupts_previous_hour_continuity():
    rows = [
        row("2026-01-02T00:00:00Z", bb_width=0.03),
        row("2026-01-02T02:00:00Z", direction="long"),
    ]

    assert collect(rows)[0]["previous_features"] is None


def test_candidate_and_nested_values_are_deep_copied_from_replay_rows():
    rows = [
        row("2026-01-02T00:00:00Z", bb_width=0.03),
        row("2026-01-02T01:00:00Z", direction="long"),
    ]
    original = deepcopy(rows)

    result = collect(rows)
    result[0]["features"]["bb_width"] = 99
    result[0]["previous_features"]["bb_width"] = 98
    result[0]["native_diagnostic"]["native_signal"]["direction"] = "short"
    result[0]["feature_observation_ids"]["macro"] = "changed"

    assert rows == original
    rows[1]["output"]["features"]["close"] = 0
    assert result[0]["features"]["close"] == 101.0


def source_payload(month):
    return {
        "schema": "lc-source-population-v1",
        "certified": False,
        "scope": "source-only native LC population; no outcomes or market roles",
        "month": month,
        "source_path": "/fixture/minute.parquet",
        "source_sha256": "a" * 64,
        "seed": "2025-12-02T00:00:00+00:00",
        "start": "2026-01-01T00:00:00+00:00",
        "end_exclusive": "2026-02-01T00:00:00+00:00",
        "hourly_input_hash": "b" * 64,
        "candidates": [],
        "parent_ledgers": {"4H_N3": {}, "1D_N3": {}},
        "source_manifest": {},
        "code_manifest": {},
        "config_manifest": {},
        "runtime": {},
        "missing_input_limits": ["fixture limitation"],
    }


@pytest.mark.parametrize("month", ["2026-01", "2026-02", "2026-03"])
def test_cli_accepts_only_fixed_q1_months(monkeypatch, tmp_path, month):
    seen = []
    monkeypatch.setattr(module, "prepare_month", lambda chosen, out: seen.append((chosen, out)))

    assert module.main(["--month", month, "--out", str(tmp_path / month)]) == 0
    assert seen == [(month, tmp_path / month)]


def test_cli_rejects_month_outside_fixed_q1(tmp_path):
    with pytest.raises(SystemExit) as exc:
        module.main(["--month", "2026-04", "--out", str(tmp_path / "april")])

    assert exc.value.code == 2


def test_prepare_month_writes_only_source_artifact(monkeypatch, tmp_path):
    expected = source_payload("2026-01")
    monkeypatch.setattr(module, "_build_month_source", lambda month: deepcopy(expected))
    out = tmp_path / "january"

    returned = module.prepare_month("2026-01", out)

    assert returned == expected
    assert json.loads((out / "source.json").read_text()) == expected
    assert sorted(path.name for path in out.iterdir()) == ["source.json"]
    forbidden = {"outcomes", "assessment", "market_roles", "selected_candidates"}
    assert forbidden.isdisjoint(returned)


def test_prepare_month_rejects_existing_output_before_replay(monkeypatch, tmp_path):
    out = tmp_path / "existing"
    out.mkdir()

    def should_not_run(month):
        raise AssertionError("source replay must not run")

    monkeypatch.setattr(module, "_build_month_source", should_not_run)
    with pytest.raises(ValueError, match="new output directory"):
        module.prepare_month("2026-01", out)


def test_immutable_writer_allows_equal_replay_and_rejects_changed_value(tmp_path):
    target = tmp_path / "source.json"
    module._save_immutable(target, {"value": 1})
    module._save_immutable(target, {"value": 1})

    with pytest.raises(ValueError, match="unequal overwrite"):
        module._save_immutable(target, {"value": 2})


def test_prepare_month_rejects_invalid_month_before_creating_output(tmp_path):
    out = tmp_path / "invalid"

    with pytest.raises(ValueError, match="fixed Q1 month"):
        module.prepare_month("2025-01", out)

    assert not out.exists()
