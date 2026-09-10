from copy import deepcopy
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from scripts.research.causal_parent_ledger import (
    _annotate_source_outputs,
    _pivot_records_from_source,
    bind_parent,
    build_parent_ledger,
    parent_asof,
)
from scripts.research.replay_clock import context_at
from scripts.research.virtual_book_replay import side_effect_guard


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
LOCAL_SOURCE_REASON = "recovered sibling source is not installed at the explicit local paths"


def hourly_bars(periods=1200, start="2026-01-01"):
    index = pd.date_range(start, periods=periods, freq="h", tz="UTC")
    x = np.arange(len(index))
    center = 110 + 7 * np.sin(x / 35.0) + 2 * np.sin(x / 7.0)
    return pd.DataFrame(
        {
            "open": center,
            "high": center + 3,
            "low": center - 3,
            "close": center,
            "volume": np.ones(len(x)),
            "atr_14": np.full(len(x), 2.0),
        },
        index=index,
    )


def build(bars, anchor_timeframe="4H", pivot_n=3, **overrides):
    kwargs = dict(
        instrument="BTC-USD",
        data_stream_id="same-hourly-fixture",
        anchor_timeframe=anchor_timeframe,
        pivot_n=pivot_n,
        atr_contract=ATR_CONTRACT,
        source_paths=SOURCE_PATHS,
        expected_hashes=EXPECTED_HASHES,
    )
    kwargs.update(overrides)
    return build_parent_ledger(bars, **kwargs)


def _load_source(path, name, expected_hash):
    namespace = {"__name__": name, "__file__": str(path)}
    source = Path(path).read_bytes()
    assert hashlib.sha256(source).hexdigest() == expected_hash
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


def _source_result(bars, timeframe, n):
    records = []
    with side_effect_guard(records):
        htf = _load_source(
            SOURCE_PATHS["htf"], "test_recovered_htf", EXPECTED_HASHES["htf"]
        )
        range_source = _load_source(
            SOURCE_PATHS["range"], "test_recovered_range", EXPECTED_HASHES["range"]
        )
        private = bars.copy()
        private.index = private.index.tz_convert("UTC").tz_localize(None)
        anchor = htf["resample_htf"](private, timeframe)
        pivots = htf["detect_fractal_pivots"](anchor, n)
        low = htf["_broadcast"](pivots, private.index, "pivot_low_level", "is_swing_low")
        high = htf["_broadcast"](pivots, private.index, "pivot_high_level", "is_swing_high")
        private["swing_low_50"] = low
        private["swing_high_50"] = high
        result = range_source["build_structural_range"](private)
    assert records == []
    return pivots, result


def _plain(value):
    if isinstance(value, dict) and "__nonfinite__" in value:
        return float(value["__nonfinite__"])
    return value


@pytest.mark.skipif(not LOCAL_SOURCE_AVAILABLE, reason=LOCAL_SOURCE_REASON)
@pytest.mark.parametrize("anchor_timeframe,pivot_n", [("4H", 3), ("4H", 5), ("1D", 3), ("1D", 5)])
def test_real_source_parity_and_causal_confirmation(anchor_timeframe, pivot_n):
    """Break caught: adapter drift from any recovered pivot/range branch or early visibility."""
    bars = hourly_bars()
    original = bars.copy(deep=True)
    source_pivots, source_range = _source_result(bars, anchor_timeframe, pivot_n)

    ledger = build(bars, anchor_timeframe, pivot_n)

    assert ledger["certified"] is False
    assert len(ledger["transitions"]) == len(bars)
    for transition, (_, expected) in zip(ledger["transitions"], source_range.iterrows()):
        assert transition["source_range_state"] == expected.struct_range_state
        for ledger_key, source_key in (
            ("source_range_low", "struct_range_low"),
            ("source_range_high", "struct_range_high"),
            ("source_range_width_atr", "struct_range_width_atr"),
        ):
            actual = _plain(transition[ledger_key])
            wanted = expected[source_key]
            assert (pd.isna(actual) and pd.isna(wanted)) or actual == pytest.approx(wanted)
        assert transition["source_sweep_low"] == int(expected.struct_sweep_low)
        assert transition["source_sweep_high"] == int(expected.struct_sweep_high)

    delta = pd.Timedelta(anchor_timeframe.lower())
    for pivot in ledger["pivots"]:
        assert pd.Timestamp(pivot["confirming_close"]) == (
            pd.Timestamp(pivot["pivot_close"]) + pivot_n * delta
        )
        assert pd.Timestamp(pivot["available_at"]) == pd.Timestamp(pivot["confirming_close"])
    for version in ledger["versions"]:
        assert pd.Timestamp(version["available_at"]) == pd.Timestamp(version["formation_hour"]) + pd.Timedelta("1h")
    pd.testing.assert_frame_equal(bars, original)


@pytest.mark.skipif(not LOCAL_SOURCE_AVAILABLE, reason=LOCAL_SOURCE_REASON)
def test_prefix_restart_and_ids_do_not_depend_on_future_input():
    """Break caught: future input hash leaking into event IDs or batch-only state mutation."""
    bars = hourly_bars()
    full = build(bars, "4H", 3)
    prefix = build(bars.iloc[:800], "4H", 3)
    rebuilt = build(bars, "4H", 3)
    cutoff = bars.index[799] + pd.Timedelta("1h")

    for key in ("pivots", "versions", "transitions"):
        visible = [row for row in full[key] if pd.Timestamp(row["available_at"]) <= cutoff]
        assert visible == prefix[key]
    assert rebuilt == full
    assert full["manifest"]["input_hash"] != prefix["manifest"]["input_hash"]


@pytest.mark.skipif(not LOCAL_SOURCE_AVAILABLE, reason=LOCAL_SOURCE_REASON)
def test_context_at_minute_aggregation_has_identical_parent_identity():
    """Break caught: a separate minute parent clock or aggregation-dependent IDs."""
    hourly = hourly_bars(periods=480)
    minute_rows = []
    minute_index = []
    for opened, row in hourly.iterrows():
        for offset in range(60):
            minute_index.append(opened + pd.Timedelta(minutes=offset))
            value = float(row.close)
            minute_rows.append(
                dict(
                    open=float(row.open) if offset == 0 else value,
                    high=float(row.high) if offset == 10 else value,
                    low=float(row.low) if offset == 20 else value,
                    close=float(row.close),
                    volume=float(row.volume) / 60.0,
                )
            )
    minutes = pd.DataFrame(minute_rows, index=pd.DatetimeIndex(minute_index))
    aggregated_rows = context_at(
        minutes,
        minutes.index[-1] + pd.Timedelta("1min"),
        "1min",
        "1h",
    )["completed"]
    aggregated = pd.DataFrame(aggregated_rows).set_index("open_time")
    aggregated.index = pd.to_datetime(aggregated.index, utc=True)
    aggregated = aggregated[["open", "high", "low", "close", "volume"]]
    aggregated["volume"] = 1.0
    aggregated["atr_14"] = hourly["atr_14"].to_numpy()

    direct_ledger = build(hourly, "4H", 3)
    minute_ledger = build(aggregated, "4H", 3)

    for key in ("pivots", "versions", "transitions"):
        assert minute_ledger[key] == direct_ledger[key]


@pytest.mark.skipif(not LOCAL_SOURCE_AVAILABLE, reason=LOCAL_SOURCE_REASON)
def test_equal_epoch_non_utc_input_has_identical_parent_identity():
    """Break caught: rejecting or re-identifying an aware UTC-normalizable candle index."""
    utc_bars = hourly_bars(periods=480)
    local_bars = utc_bars.copy()
    local_bars.index = local_bars.index.tz_convert("America/Los_Angeles")

    utc_ledger = build(utc_bars, "4H", 3)
    local_ledger = build(local_bars, "4H", 3)

    assert local_ledger["manifest"]["contract_id"] == utc_ledger["manifest"]["contract_id"]
    assert local_ledger["manifest"]["input_hash"] == utc_ledger["manifest"]["input_hash"]
    for key in ("pivots", "versions", "transitions"):
        assert local_ledger[key] == utc_ledger[key]


SAFE_HTF = '''
import numpy as np
import pandas as pd

def resample_htf(df, freq):
    delta = pd.Timedelta(freq.lower())
    rows = []
    for start, group in df.groupby(df.index.floor(delta)):
        if group.index[0] == start and len(group) == int(delta / pd.Timedelta("1h")):
            rows.append(dict(open_time=start, open=group.open.iloc[0], high=group.high.max(),
                             low=group.low.min(), close=group.close.iloc[-1],
                             volume=group.volume.sum(), close_time=start + delta))
    if not rows:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume", "close_time"])
    return pd.DataFrame(rows).set_index("open_time")

def detect_fractal_pivots(htf, N):
    return pd.DataFrame({"is_swing_high": False, "is_swing_low": False,
                         "confirm_time": pd.NaT, "pivot_high_level": np.nan,
                         "pivot_low_level": np.nan}, index=htf.index)

def _broadcast(piv, one_h_index, level_col, flag_col):
    return pd.Series(np.nan, index=one_h_index)
'''

SAFE_RANGE = '''
import numpy as np
import pandas as pd

def build_structural_range(df, break_buffer_atr=0.0, break_confirm_bars=1):
    n = len(df)
    return pd.DataFrame({
        "struct_range_high": np.nan, "struct_range_low": np.nan,
        "struct_range_age_bars": np.nan, "struct_range_pos": np.nan,
        "struct_range_state": ["forming"] * n,
        "struct_sweep_low": np.zeros(n, dtype=np.int8),
        "struct_sweep_high": np.zeros(n, dtype=np.int8),
        "struct_range_width_atr": np.nan,
    }, index=df.index)
'''


def local_sources(tmp_path, htf=SAFE_HTF, range_source=SAFE_RANGE):
    htf_path = tmp_path / "htf.py"
    range_path = tmp_path / "range.py"
    htf_path.write_text(htf)
    range_path.write_text(range_source)
    paths = {"htf": str(htf_path), "range": str(range_path)}
    hashes = {
        "htf": hashlib.sha256(htf.encode()).hexdigest(),
        "range": hashlib.sha256(range_source.encode()).hexdigest(),
    }
    return paths, hashes


def portable_build(tmp_path, bars=None, **overrides):
    if "source_paths" in overrides or "expected_hashes" in overrides:
        paths = overrides.pop("source_paths")
        hashes = overrides.pop("expected_hashes")
    else:
        paths, hashes = local_sources(tmp_path)
    kwargs = dict(
        instrument="BTC-USD",
        data_stream_id="portable-stream",
        anchor_timeframe="4H",
        pivot_n=3,
        atr_contract=ATR_CONTRACT,
        source_paths=paths,
        expected_hashes=hashes,
    )
    kwargs.update(overrides)
    return build_parent_ledger(hourly_bars(16) if bars is None else bars, **kwargs)


@pytest.mark.parametrize(
    "mutator,match",
    [
        (lambda bars: bars.set_axis(bars.index + pd.Timedelta("30min")), "grid/continuity"),
        (lambda bars: bars.drop(bars.index[5]), "grid/continuity"),
        (lambda bars: bars.assign(high=np.nan), "Invalid OHLCV"),
    ],
)
def test_invalid_hourly_evidence_rejects_portably(tmp_path, mutator, match):
    """Break caught: silently shifting, filling, or accepting invalid hourly evidence."""
    with pytest.raises(ValueError, match=match):
        portable_build(tmp_path, bars=mutator(hourly_bars(16)))


@pytest.mark.parametrize("bad_atr", [-1.0, np.inf, "not-a-number", "2.0"])
def test_invalid_atr_rejects_without_substitution(tmp_path, bad_atr):
    """Break caught: invalid ATR coercion or replacement by an adapter-computed value."""
    bars = hourly_bars(16)
    bars["atr_14"] = bad_atr
    with pytest.raises(ValueError, match="ATR"):
        portable_build(tmp_path, bars=bars)


@pytest.mark.parametrize(
    "contract",
    [
        None,
        {},
        {"source": "x", "formula_id": "f", "version": "1"},
        {"source": "x", "formula_id": "f", "version": "1", "availability_policy": "receipt"},
    ],
)
def test_missing_or_incompatible_atr_contract_rejects(tmp_path, contract):
    """Break caught: creating a ledger without an explicit hour-close ATR contract."""
    with pytest.raises(ValueError, match="ATR contract"):
        portable_build(tmp_path, atr_contract=contract)


@pytest.mark.parametrize("pivot_n", [-1, True, 0, 4, 3.0, "3"])
def test_invalid_pivot_n_rejects(tmp_path, pivot_n):
    """Break caught: bool/coercible/non-registered pivot hypotheses entering the contract."""
    with pytest.raises(ValueError, match="pivot_n"):
        portable_build(tmp_path, pivot_n=pivot_n)


def test_source_hash_mismatch_rejects_before_execution(tmp_path):
    """Break caught: executing unverified recovered code."""
    paths, hashes = local_sources(tmp_path)
    hashes["htf"] = "0" * 64
    with pytest.raises(ValueError, match="source hash mismatch"):
        portable_build(tmp_path, source_paths=paths, expected_hashes=hashes)


def test_batch_cap_rejects_instead_of_truncating(tmp_path):
    """Break caught: silently truncating input beyond the registered 2,048-hour cap."""
    with pytest.raises(ValueError, match="2,048"):
        portable_build(tmp_path, bars=hourly_bars(2049))


def test_supplied_atr_availability_must_equal_hour_close(tmp_path):
    """Break caught: treating open-time or receipt-free ATR as close-available."""
    bars = hourly_bars(16)
    bars["atr_available_at"] = bars.index
    with pytest.raises(ValueError, match="ATR availability"):
        portable_build(tmp_path, bars=bars)


def test_nan_and_zero_atr_are_preserved_and_flagged(tmp_path):
    """Break caught: backward-filling warmup ATR or hiding zero-ATR quality."""
    bars = hourly_bars(16)
    bars.iloc[3, bars.columns.get_loc("atr_14")] = np.nan
    bars.iloc[4, bars.columns.get_loc("atr_14")] = 0.0
    ledger = portable_build(tmp_path, bars=bars)

    assert ledger["transitions"][3]["atr_14"] == {"__nonfinite__": "nan"}
    assert "atr_unavailable" in ledger["transitions"][3]["quality_flags"]
    assert ledger["transitions"][4]["atr_14"] == 0.0
    assert "atr_zero" in ledger["transitions"][4]["quality_flags"]


def test_leading_partial_anchor_bucket_is_excluded(tmp_path):
    """Break caught: accepting the 03:00 row as part of a fake complete 00:00 4H bucket."""
    bars = hourly_bars(9, start="2026-01-01 03:00")
    ledger = portable_build(tmp_path, bars=bars)

    assert [row["open_time"] for row in ledger["anchor_buckets"]["completed"]] == [
        "2026-01-01 04:00:00+00:00",
        "2026-01-01 08:00:00+00:00",
    ]
    assert ledger["anchor_buckets"]["incomplete"][0]["open_time"] == "2026-01-01 00:00:00+00:00"


def test_empty_complete_anchor_buckets_reject(tmp_path):
    """Break caught: invoking recovered pivot logic with only a developing anchor bucket."""
    with pytest.raises(ValueError, match="complete anchor buckets"):
        portable_build(tmp_path, bars=hourly_bars(1))


def test_source_side_effect_is_rejected_by_existing_guard(tmp_path):
    """Break caught: source top-level code writing to disk while being loaded."""
    unsafe = "open('/tmp/causal-parent-ledger-forbidden', 'w').write('x')\n" + SAFE_HTF
    paths, hashes = local_sources(tmp_path, htf=unsafe)
    with pytest.raises(RuntimeError, match="Prohibited side effect"):
        portable_build(tmp_path, source_paths=paths, expected_hashes=hashes)
    assert not Path("/tmp/causal-parent-ledger-forbidden").exists()


def _version(version_id="v1", available_at="2026-01-01 04:00:00+00:00"):
    return {
        "id": version_id,
        "lineage_id": "lineage-1",
        "predecessor_version_id": None,
        "creation_reason": "formation",
        "range_low": 90.0,
        "range_high": 120.0,
        "low_pivot_id": "low-1",
        "high_pivot_id": "high-1",
        "formation_hour": "2026-01-01 03:00:00+00:00",
        "available_at": available_at,
    }


def query_ledger():
    return {
        "coverage": {
            "first_open": "2026-01-01 00:00:00+00:00",
            "last_processed_close": "2026-01-01 05:00:00+00:00",
            "query_exclusive_end": "2026-01-01 06:00:00+00:00",
        },
        "versions": [_version()],
        "transitions": [
            {
                "id": "t1",
                "available_at": "2026-01-01 04:00:00+00:00",
                "post_state": "active",
                "post_version_id": "v1",
            },
            {
                "id": "t2",
                "available_at": "2026-01-01 05:00:00+00:00",
                "post_state": "active",
                "post_version_id": "v1",
            },
        ],
    }


def test_parent_asof_is_inclusive_or_strict_and_returns_a_copy():
    """Break caught: equal-time future leakage or caller mutation of ledger versions."""
    ledger = query_ledger()
    assert parent_asof(ledger, "2026-01-01 04:00:00+00:00")["id"] == "v1"
    assert parent_asof(ledger, "2026-01-01 04:00:00+00:00", strict=True) is None
    selected = parent_asof(ledger, "2026-01-01 04:30:00+00:00")
    selected["range_low"] = -1
    assert ledger["versions"][0]["range_low"] == 90.0


def test_query_rejects_naive_and_out_of_coverage_time():
    """Break caught: local-time ambiguity or indefinite carry beyond the next update."""
    ledger = query_ledger()
    with pytest.raises(ValueError, match="timezone-aware"):
        parent_asof(ledger, "2026-01-01 04:30:00")
    with pytest.raises(ValueError, match="out_of_coverage"):
        parent_asof(ledger, "2026-01-01 06:00:00+00:00")
    assert parent_asof(ledger, "2026-01-01 05:59:59+00:00")["id"] == "v1"


def test_binding_strictly_excludes_equal_time_parent_and_is_immutable():
    """Break caught: binding a parent confirmed at sweep-open or rewriting fixed geometry."""
    ledger = query_ledger()
    rejected = bind_parent(
        ledger,
        child_event_id="child-equal",
        child_timeframe="1min",
        first_sweep_open="2026-01-01 04:00:00+00:00",
    )
    assert rejected == {
        "status": "rejected",
        "reason": "absent_parent",
        "child_event_id": "child-equal",
        "child_timeframe": "1min",
        "first_sweep_open": "2026-01-01 04:00:00+00:00",
        "binding_policy": "parent_available_strictly_before_first_sweep_open",
    }
    bound = bind_parent(
        ledger,
        child_event_id="child-later",
        child_timeframe="1h",
        first_sweep_open="2026-01-01 04:30:00+00:00",
    )
    frozen = deepcopy(bound)
    ledger["versions"][0]["range_low"] = 100.0
    ledger["transitions"].append(
        {"id": "break", "available_at": "2026-01-01 05:30:00+00:00", "post_state": "broken_up", "post_version_id": None}
    )
    assert bound == frozen
    assert bound["parent_range_low"] == 90.0


def _source_frame(index, states, lows, highs, sweeps=None):
    sweeps = sweeps or [0] * len(index)
    return pd.DataFrame(
        {
            "struct_range_high": highs,
            "struct_range_low": lows,
            "struct_range_age_bars": list(range(len(index))),
            "struct_range_pos": [0.5] * len(index),
            "struct_range_state": states,
            "struct_sweep_low": sweeps,
            "struct_sweep_high": [0] * len(index),
            "struct_range_width_atr": [15.0] * len(index),
        },
        index=index,
    )


def _pivot(pid, side, level, available_at):
    return {
        "id": pid,
        "side": side,
        "level": level,
        "pivot_open": "2025-12-31 00:00:00+00:00",
        "pivot_close": "2025-12-31 04:00:00+00:00",
        "confirming_close": str(pd.Timestamp(available_at)),
        "available_at": str(pd.Timestamp(available_at)),
        "anchor_timeframe": "4H",
        "pivot_n": 3,
        "evidence_id": "evidence-" + pid,
    }


def test_tightening_is_a_new_version_but_sweep_is_evaluated_against_old_floor():
    """Break caught: reinterpreting the 95 wick against newly emitted floor 100."""
    bars = hourly_bars(2)
    bars.loc[bars.index[0], ["low", "close"]] = [100.0, 110.0]
    bars.loc[bars.index[1], ["low", "close"]] = [95.0, 110.0]
    pivots = [
        _pivot("low-90", "low", 90.0, bars.index[0]),
        _pivot("high-120", "high", 120.0, bars.index[0]),
        _pivot("low-100", "low", 100.0, bars.index[1]),
    ]
    source = _source_frame(bars.index, ["active", "active"], [90.0, 100.0], [120.0, 120.0])

    versions, transitions = _annotate_source_outputs(
        bars, source, pivots, contract_id="contract", data_stream_id="stream"
    )

    assert [version["creation_reason"] for version in versions] == ["formation", "floor_tightening"]
    assert versions[1]["lineage_id"] == versions[0]["lineage_id"]
    assert versions[1]["predecessor_version_id"] == versions[0]["id"]
    assert versions[1]["low_pivot_id"] == "low-100"
    assert versions[1]["high_pivot_id"] == "high-120"
    assert transitions[1]["evaluated_version_id"] == versions[0]["id"]
    assert transitions[1]["pre_range_low"] == 90.0
    assert transitions[1]["post_range_low"] == 100.0
    assert transitions[1]["source_sweep_low"] == 0


def test_old_anchor_reformation_creates_new_lineage_and_version():
    """Break caught: suppressing active reformation because old 90/120 anchors were reused."""
    bars = hourly_bars(3)
    pivots = [
        _pivot("low-90", "low", 90.0, bars.index[0]),
        _pivot("high-120", "high", 120.0, bars.index[0]),
    ]
    source = _source_frame(
        bars.index,
        ["active", "broken_up", "active"],
        [90.0, 90.0, 90.0],
        [120.0, 120.0, 120.0],
    )

    versions, transitions = _annotate_source_outputs(
        bars, source, pivots, contract_id="contract", data_stream_id="stream"
    )

    assert len(versions) == 2
    assert versions[0]["lineage_id"] != versions[1]["lineage_id"]
    assert versions[0]["low_pivot_id"] == versions[1]["low_pivot_id"] == "low-90"
    assert versions[0]["high_pivot_id"] == versions[1]["high_pivot_id"] == "high-120"
    assert transitions[1]["post_version_id"] is None
    assert transitions[1]["post_range_low"] == 90.0
    assert transitions[2]["post_state"] == "active"


def test_equal_level_pivots_have_distinct_ids_without_silent_anchor_adoption():
    """Break caught: level-based pivot identity collapse or anchor rewrite without a new version."""
    bars = hourly_bars(2)
    pivots = [
        _pivot("low-occurrence-a", "low", 90.0, bars.index[0]),
        _pivot("high-120", "high", 120.0, bars.index[0]),
        _pivot("low-occurrence-b", "low", 90.0, bars.index[1]),
    ]
    source = _source_frame(bars.index, ["active", "active"], [90.0, 90.0], [120.0, 120.0])

    versions, transitions = _annotate_source_outputs(
        bars, source, pivots, contract_id="contract", data_stream_id="stream"
    )

    assert pivots[0]["id"] != pivots[2]["id"]
    assert len(versions) == 1
    assert versions[0]["low_pivot_id"] == "low-occurrence-a"
    assert transitions[1]["latest_low_pivot_id"] == "low-occurrence-b"
    assert transitions[1]["post_version_id"] == versions[0]["id"]


def test_generated_pivot_ids_distinguish_equal_priced_occurrences():
    """Break caught: constructing pivot identity from price rather than occurrence evidence."""
    index = pd.date_range("2026-01-01", periods=11, freq="4h", tz="UTC")
    anchors = pd.DataFrame(
        {
            "open": np.arange(11) + 100.0,
            "high": np.arange(11) + 103.0,
            "low": np.arange(11) + 97.0,
            "close": np.arange(11) + 101.0,
            "volume": np.ones(11),
            "close_time": index + pd.Timedelta("4h"),
        },
        index=index,
    )
    pivots = pd.DataFrame(
        {
            "is_swing_high": [False] * 11,
            "is_swing_low": [False, False, False, True, False, False, False, True, False, False, False],
            "confirm_time": [pd.NaT, pd.NaT, pd.NaT, index[7], pd.NaT, pd.NaT, pd.NaT,
                             index[10] + pd.Timedelta("4h"), pd.NaT, pd.NaT, pd.NaT],
            "pivot_high_level": [np.nan] * 11,
            "pivot_low_level": [np.nan, np.nan, np.nan, 90.0, np.nan, np.nan, np.nan,
                                90.0, np.nan, np.nan, np.nan],
        },
        index=index,
    )

    records = _pivot_records_from_source(
        pivots,
        anchor_buckets=anchors,
        contract_id="contract",
        data_stream_id="stream",
        instrument="BTC-USD",
        anchor_timeframe="4H",
        pivot_n=3,
    )

    assert [record["level"] for record in records] == [90.0, 90.0]
    assert records[0]["pivot_open"] != records[1]["pivot_open"]
    assert records[0]["id"] != records[1]["id"]


def test_pivot_evidence_binds_confirming_window_and_instrument():
    """Break caught: price-only IDs surviving changed support evidence or instrument identity."""
    index = pd.date_range("2026-01-01", periods=7, freq="4h", tz="UTC")
    anchors = pd.DataFrame(
        {
            "open": [100.0] * 7,
            "high": [101.0, 102.0, 103.0, 110.0, 104.0, 103.0, 102.0],
            "low": [99.0] * 7,
            "close": [100.0] * 7,
            "volume": [1.0] * 7,
            "close_time": index + pd.Timedelta("4h"),
        },
        index=index,
    )
    pivots = pd.DataFrame(
        {
            "is_swing_high": [False, False, False, True, False, False, False],
            "is_swing_low": [False] * 7,
            "confirm_time": [pd.NaT, pd.NaT, pd.NaT, index[6] + pd.Timedelta("4h"), pd.NaT, pd.NaT, pd.NaT],
            "pivot_high_level": [np.nan, np.nan, np.nan, 110.0, np.nan, np.nan, np.nan],
            "pivot_low_level": [np.nan] * 7,
        },
        index=index,
    )

    base = _pivot_records_from_source(
        pivots, anchor_buckets=anchors, contract_id="contract", data_stream_id="stream",
        instrument="BTC-USD", anchor_timeframe="4H", pivot_n=3,
    )[0]
    changed_anchors = anchors.copy()
    changed_anchors.loc[index[6], "volume"] = 2.0
    changed_support = _pivot_records_from_source(
        pivots, anchor_buckets=changed_anchors, contract_id="contract", data_stream_id="stream",
        instrument="BTC-USD", anchor_timeframe="4H", pivot_n=3,
    )[0]
    changed_instrument = _pivot_records_from_source(
        pivots, anchor_buckets=anchors, contract_id="contract", data_stream_id="stream",
        instrument="ETH-USD", anchor_timeframe="4H", pivot_n=3,
    )[0]

    assert changed_support["level"] == base["level"] == 110.0
    assert changed_support["evidence_id"] != base["evidence_id"]
    assert changed_support["id"] != base["id"]
    assert changed_instrument["id"] != base["id"]
