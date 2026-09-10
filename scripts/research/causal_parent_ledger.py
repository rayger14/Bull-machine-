"""Guarded, append-only research ledger for recovered parent structure.

This module is an offline reference adapter.  It does not select a winning
parent hypothesis, certify source receipts, or integrate with native signals.
"""
from copy import deepcopy
import hashlib
from pathlib import Path
import platform

import numpy as np
import pandas as pd

from scripts.research.replay_clock import context_at, digest, json_safe, utc, validate_bars
from scripts.research.virtual_book_replay import side_effect_guard


MAX_INPUT_HOURS = 2048
ANCHOR_TIMEFRAMES = {"4H": pd.Timedelta("4h"), "1D": pd.Timedelta("1d")}
ATR_POLICY = "hour_close_assumed"
CONTRACT_SCHEMA = "causal_parent_ledger.v1"


def _nonempty(value, label):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("%s required" % label)
    return value


def _sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def _file_hash(path):
    return _sha256_bytes(Path(path).read_bytes())


def _utc_from_source(value, label):
    timestamp = pd.Timestamp(value)
    if pd.isna(timestamp):
        raise ValueError("%s is unknown" % label)
    if timestamp.tz is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return timestamp


def _finite_or_none(value):
    if value is None or pd.isna(value):
        return None
    return float(value)


def _validate_atr_contract(contract):
    if not isinstance(contract, dict):
        raise ValueError("ATR contract required")
    for key in ("source", "formula_id", "version"):
        value = contract.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError("ATR contract requires nonempty %s" % key)
    if contract.get("availability_policy") != ATR_POLICY:
        raise ValueError("ATR contract requires availability_policy='hour_close_assumed'")
    return deepcopy(contract)


def _validate_atr(bars):
    if "atr_14" not in bars.columns:
        raise ValueError("ATR column atr_14 required")
    series = bars["atr_14"]
    if pd.api.types.is_bool_dtype(series.dtype) or not pd.api.types.is_numeric_dtype(series.dtype):
        raise ValueError("ATR must be numeric, nonnegative, and finite or NaN")
    try:
        numeric = pd.to_numeric(series, errors="raise").to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("ATR must be numeric, nonnegative, and finite or NaN") from exc
    if np.isinf(numeric).any() or (numeric[~np.isnan(numeric)] < 0).any():
        raise ValueError("ATR must be nonnegative and finite or NaN")
    return numeric


def _validate_atr_availability(bars):
    expected = bars.index.tz_convert("UTC") + pd.Timedelta("1h")
    for column in ("available_at", "atr_available_at"):
        if column not in bars.columns:
            continue
        supplied = []
        for value in bars[column]:
            try:
                supplied.append(utc(value, column))
            except ValueError as exc:
                raise ValueError("ATR availability must equal each UTC hour close") from exc
        actual = pd.DatetimeIndex(supplied)
        if not actual.equals(expected):
            raise ValueError("ATR availability must equal each UTC hour close")


def _validate_sources(source_paths, expected_hashes):
    if not isinstance(source_paths, dict) or not isinstance(expected_hashes, dict):
        raise ValueError("source_paths and expected_hashes mappings required")
    source_bytes = {}
    actual_hashes = {}
    for key in ("htf", "range"):
        raw_path = source_paths.get(key)
        expected = expected_hashes.get(key)
        if not isinstance(raw_path, (str, Path)) or not isinstance(expected, str) or not expected:
            raise ValueError("source path and expected hash required for %s" % key)
        path = Path(raw_path)
        try:
            value = path.read_bytes()
        except OSError as exc:
            raise ValueError("source unavailable: %s" % key) from exc
        actual = _sha256_bytes(value)
        if actual != expected:
            raise ValueError("source hash mismatch: %s" % key)
        source_bytes[key] = value
        actual_hashes[key] = actual
    return source_bytes, actual_hashes


def _compile_source(source, path, name):
    namespace = {"__name__": name, "__file__": str(path), "__package__": None}
    exec(compile(source, str(path), "exec"), namespace)
    return namespace


def _independent_anchor_buckets(bars, anchor_timeframe):
    decision_time = bars.index[-1] + pd.Timedelta("1h")
    buckets = context_at(bars, decision_time, "1h", anchor_timeframe.lower())
    if not buckets["completed"]:
        raise ValueError("no complete anchor buckets")
    completed = pd.DataFrame(buckets["completed"])
    completed.index = pd.to_datetime(completed.pop("open_time"), utc=True)
    frame = completed[["open", "high", "low", "close", "volume"]].copy()
    frame["close_time"] = pd.to_datetime(completed["close_time"], utc=True)
    return buckets, frame


def _private_complete_hours(bars, completed):
    opens = []
    for record in completed:
        opens.extend(pd.Timestamp(value) for value in record["source_open_times"])
    wanted = pd.DatetimeIndex(opens).tz_convert("UTC")
    private = bars.loc[wanted, ["open", "high", "low", "close", "volume"]].copy()
    private.index = private.index.tz_convert("UTC").tz_localize(None)
    return private


def _assert_aggregation_parity(recovered, independent):
    recovered_utc = recovered.copy()
    recovered_utc.index = pd.DatetimeIndex(recovered_utc.index).tz_localize("UTC")
    recovered_utc["close_time"] = pd.to_datetime(recovered_utc["close_time"], utc=True)
    expected = independent[recovered_utc.columns]
    if not recovered_utc.index.equals(expected.index):
        raise ValueError("recovered aggregation mismatch")
    for column in ("open", "high", "low", "close", "volume"):
        if not np.array_equal(
            recovered_utc[column].to_numpy(dtype=float),
            expected[column].to_numpy(dtype=float),
        ):
            raise ValueError("recovered aggregation mismatch")
    if not recovered_utc["close_time"].equals(expected["close_time"]):
        raise ValueError("recovered aggregation mismatch")


def _runtime_manifest():
    versions = {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "pandas": pd.__version__,
        "numpy": np.__version__,
    }
    return {
        "versions": versions,
        "hashes": {key: digest({"runtime": key, "version": value}) for key, value in versions.items()},
    }


def _helper_hashes():
    root = Path(__file__).resolve().parent
    return {
        "adapter": _file_hash(__file__),
        "replay_clock": _file_hash(root / "replay_clock.py"),
        "side_effect_guard": _file_hash(root / "virtual_book_replay.py"),
    }


def _contract_manifest(instrument, anchor_timeframe, pivot_n, atr_contract, source_hashes):
    runtime = _runtime_manifest()
    contract = {
        "schema": CONTRACT_SCHEMA,
        "instrument": instrument,
        "source_hashes": deepcopy(source_hashes),
        "helper_hashes": _helper_hashes(),
        "runtime_hashes": runtime["hashes"],
        "runtime_versions": runtime["versions"],
        "parameters": {
            "anchor_timeframe": anchor_timeframe,
            "pivot_n": pivot_n,
            "range_min_width_atr": 1.5,
            "tighten_min_width_atr": 0.75,
            "break_buffer_atr": 0.0,
            "break_confirm_bars": 1,
            "range_update_timeframe": "1h",
        },
        "atr_contract": deepcopy(atr_contract),
        "scope": "guarded recovered source; offline parent annotation only",
    }
    contract["contract_id"] = digest(contract)
    return contract


def _input_hash(bars, instrument, data_stream_id, atr_contract):
    rows = []
    for opened, record in zip(bars.index, bars.to_dict("records")):
        rows.append({"open_time": str(opened.tz_convert("UTC")), **json_safe(record)})
    return digest(
        {
            "instrument": instrument,
            "data_stream_id": data_stream_id,
            "atr_contract": atr_contract,
            "bars": rows,
        }
    )


def _pivot_records_from_source(
    pivots,
    *,
    anchor_buckets,
    contract_id,
    data_stream_id,
    instrument,
    anchor_timeframe,
    pivot_n,
):
    """Translate source pivot occurrences without collapsing equal price levels."""
    delta = ANCHOR_TIMEFRAMES[anchor_timeframe]
    records = []
    for opened, row in pivots.iterrows():
        pivot_open = _utc_from_source(opened, "pivot_open")
        try:
            position = pivots.index.get_loc(opened)
        except KeyError as exc:
            raise ValueError("pivot occurrence missing from anchor buckets") from exc
        if not isinstance(position, (int, np.integer)):
            raise ValueError("duplicate pivot occurrence")
        for side, flag, level_column in (
            ("low", "is_swing_low", "pivot_low_level"),
            ("high", "is_swing_high", "pivot_high_level"),
        ):
            if not bool(row[flag]):
                continue
            confirming = _utc_from_source(row["confirm_time"], "confirm_time")
            pivot_close = pivot_open + delta
            start = int(position) - pivot_n
            stop = int(position) + pivot_n + 1
            if start < 0 or stop > len(anchor_buckets):
                raise ValueError("pivot lacks complete supporting anchor window")
            support = anchor_buckets.iloc[start:stop]
            supporting_anchor_ids = []
            for support_open, support_row in support.iterrows():
                support_open = _utc_from_source(support_open, "support_open")
                support_close = _utc_from_source(support_row["close_time"], "support_close")
                evidence = {
                    "kind": "complete_anchor_bucket",
                    "contract_id": contract_id,
                    "data_stream_id": data_stream_id,
                    "instrument": instrument,
                    "anchor_timeframe": anchor_timeframe,
                    "open_time": str(support_open),
                    "close_time": str(support_close),
                    "open": float(support_row["open"]),
                    "high": float(support_row["high"]),
                    "low": float(support_row["low"]),
                    "close": float(support_row["close"]),
                    "volume": float(support_row["volume"]),
                }
                supporting_anchor_ids.append(digest(evidence))
            evidence_id = digest(
                {
                    "kind": "pivot_support_window",
                    "contract_id": contract_id,
                    "data_stream_id": data_stream_id,
                    "instrument": instrument,
                    "pivot_open": str(pivot_open),
                    "pivot_n": pivot_n,
                    "supporting_anchor_ids": supporting_anchor_ids,
                }
            )
            causal = {
                "kind": "anchor_pivot",
                "contract_id": contract_id,
                "data_stream_id": data_stream_id,
                "instrument": instrument,
                "side": side,
                "pivot_open": str(pivot_open),
                "pivot_close": str(pivot_close),
                "confirming_close": str(confirming),
                "level": float(row[level_column]),
                "evidence_id": evidence_id,
            }
            record_id = digest(causal)
            records.append(
                {
                    "id": record_id,
                    "side": side,
                    "level": float(row[level_column]),
                    "pivot_open": str(pivot_open),
                    "pivot_close": str(pivot_close),
                    "confirming_close": str(confirming),
                    "available_at": str(confirming),
                    "instrument": instrument,
                    "data_stream_id": data_stream_id,
                    "anchor_timeframe": anchor_timeframe,
                    "pivot_n": pivot_n,
                    "evidence_id": evidence_id,
                    "supporting_anchor_ids": supporting_anchor_ids,
                }
            )
    return sorted(
        records,
        key=lambda record: (record["available_at"], record["pivot_open"], record["side"]),
    )


def _new_lineage_id(contract_id, data_stream_id, source_hour, low_id, high_id):
    return digest(
        {
            "kind": "parent_lineage_formation",
            "contract_id": contract_id,
            "data_stream_id": data_stream_id,
            "source_hour": str(source_hour),
            "low_pivot_id": low_id,
            "high_pivot_id": high_id,
        }
    )


def _new_version(
    *,
    contract_id,
    data_stream_id,
    lineage_id,
    predecessor,
    reason,
    source_hour,
    range_low,
    range_high,
    low_pivot_id,
    high_pivot_id,
):
    available_at = source_hour + pd.Timedelta("1h")
    causal = {
        "kind": "parent_version",
        "contract_id": contract_id,
        "data_stream_id": data_stream_id,
        "lineage_id": lineage_id,
        "predecessor_version_id": predecessor,
        "creation_reason": reason,
        "source_hour": str(source_hour),
        "range_low": float(range_low),
        "range_high": float(range_high),
        "low_pivot_id": low_pivot_id,
        "high_pivot_id": high_pivot_id,
    }
    return {
        "id": digest(causal),
        "lineage_id": lineage_id,
        "predecessor_version_id": predecessor,
        "creation_reason": reason,
        "range_low": float(range_low),
        "range_high": float(range_high),
        "low_pivot_id": low_pivot_id,
        "high_pivot_id": high_pivot_id,
        "formation_hour": str(source_hour),
        "available_at": str(available_at),
    }


def _quality_flags(atr):
    if pd.isna(atr):
        return ["atr_unavailable"]
    if float(atr) == 0.0:
        return ["atr_zero"]
    return []


def _annotate_source_outputs(bars, source_range, pivots, *, contract_id, data_stream_id):
    """Add causal identities to recovered hourly output.

    This seam deliberately annotates source output rather than reimplementing
    the recovered range machine.  Sweep and break diagnostics are attached to
    the pre-version; newly emitted geometry becomes a later immutable version.
    """
    if len(bars) != len(source_range) or not bars.index.equals(source_range.index):
        raise ValueError("source range index mismatch")
    ordered_pivots = sorted(
        deepcopy(pivots),
        key=lambda record: (record["available_at"], record.get("pivot_open", ""), record["side"]),
    )
    pivot_position = 0
    latest = {"low": None, "high": None}
    versions = []
    transitions = []
    current_version = None
    current_lineage = None
    previous_state = "forming"

    for (source_hour, bar), (_, source) in zip(bars.iterrows(), source_range.iterrows()):
        source_hour = utc(source_hour, "source_hour")
        while pivot_position < len(ordered_pivots):
            pivot = ordered_pivots[pivot_position]
            if utc(pivot["available_at"], "pivot available_at") > source_hour:
                break
            latest[pivot["side"]] = pivot
            pivot_position += 1

        pre_version = current_version
        pre_lineage = current_lineage
        pre_low = pre_version["range_low"] if pre_version is not None else None
        pre_high = pre_version["range_high"] if pre_version is not None else None
        post_state = str(source["struct_range_state"])
        post_low = _finite_or_none(source["struct_range_low"])
        post_high = _finite_or_none(source["struct_range_high"])
        reason = None
        if post_state == "active" and pre_version is None:
            reason = "formation"
        elif post_state == "active" and post_low != pre_low:
            reason = "floor_tightening"

        if reason == "formation":
            if latest["low"] is None or latest["high"] is None:
                raise ValueError("active source range lacks visible anchor pivots")
            current_lineage = _new_lineage_id(
                contract_id,
                data_stream_id,
                source_hour,
                latest["low"]["id"],
                latest["high"]["id"],
            )
            current_version = _new_version(
                contract_id=contract_id,
                data_stream_id=data_stream_id,
                lineage_id=current_lineage,
                predecessor=None,
                reason=reason,
                source_hour=source_hour,
                range_low=post_low,
                range_high=post_high,
                low_pivot_id=latest["low"]["id"],
                high_pivot_id=latest["high"]["id"],
            )
            versions.append(current_version)
        elif reason == "floor_tightening":
            if latest["low"] is None or float(latest["low"]["level"]) != float(post_low):
                raise ValueError("tightened source range lacks visible low pivot")
            current_version = _new_version(
                contract_id=contract_id,
                data_stream_id=data_stream_id,
                lineage_id=current_lineage,
                predecessor=pre_version["id"],
                reason=reason,
                source_hour=source_hour,
                range_low=post_low,
                range_high=post_high,
                low_pivot_id=latest["low"]["id"],
                high_pivot_id=pre_version["high_pivot_id"],
            )
            versions.append(current_version)
        elif post_state != "active":
            current_version = None
            current_lineage = None

        source_sweep_low = int(source["struct_sweep_low"])
        source_sweep_high = int(source["struct_sweep_high"])
        break_direction = None
        if post_state == "broken_up":
            break_direction = "up"
        elif post_state == "broken_down":
            break_direction = "down"
        causal_transition = {
            "kind": "parent_transition",
            "contract_id": contract_id,
            "data_stream_id": data_stream_id,
            "source_hour": str(source_hour),
            "pre_version_id": pre_version["id"] if pre_version else None,
            "post_version_id": current_version["id"] if current_version else None,
            "post_state": post_state,
            "source_sweep_low": source_sweep_low,
            "source_sweep_high": source_sweep_high,
            "source_break_direction": break_direction,
        }
        transition = {
            "id": digest(causal_transition),
            "source_hour": str(source_hour),
            "available_at": str(source_hour + pd.Timedelta("1h")),
            "pre_state": previous_state,
            "pre_range_low": pre_low,
            "pre_range_high": pre_high,
            "pre_lineage_id": pre_lineage,
            "pre_version_id": pre_version["id"] if pre_version else None,
            "evaluated_version_id": pre_version["id"] if pre_version else None,
            "source_break_direction": break_direction,
            "source_sweep_low": source_sweep_low,
            "source_sweep_high": source_sweep_high,
            "source_range_state": post_state,
            "source_range_low": source["struct_range_low"],
            "source_range_high": source["struct_range_high"],
            "source_range_age_bars": source["struct_range_age_bars"],
            "source_range_pos": source["struct_range_pos"],
            "source_range_width_atr": source["struct_range_width_atr"],
            "post_state": post_state,
            "post_range_low": post_low,
            "post_range_high": post_high,
            "post_lineage_id": current_lineage,
            "post_version_id": current_version["id"] if current_version else None,
            "latest_low_pivot_id": latest["low"]["id"] if latest["low"] else None,
            "latest_high_pivot_id": latest["high"]["id"] if latest["high"] else None,
            "adopted_low_pivot_id": current_version["low_pivot_id"] if current_version else None,
            "adopted_high_pivot_id": current_version["high_pivot_id"] if current_version else None,
            "atr_14": bar["atr_14"],
            "quality_flags": _quality_flags(bar["atr_14"]),
        }
        transitions.append(transition)
        previous_state = post_state
    return json_safe(versions), json_safe(transitions)


def build_parent_ledger(
    bars,
    *,
    instrument,
    data_stream_id,
    anchor_timeframe,
    pivot_n,
    atr_contract,
    source_paths,
    expected_hashes,
):
    """Build a serializable, restartable parent ledger from frozen source."""
    _nonempty(instrument, "instrument")
    _nonempty(data_stream_id, "data_stream_id")
    if anchor_timeframe not in ANCHOR_TIMEFRAMES:
        raise ValueError("anchor_timeframe must be '4H' or '1D'")
    if isinstance(pivot_n, bool) or not isinstance(pivot_n, int) or pivot_n not in (3, 5):
        raise ValueError("pivot_n must be integer 3 or 5")
    if len(bars) > MAX_INPUT_HOURS:
        raise ValueError("input exceeds 2,048 hourly bar cap")
    validate_bars(bars, "1h")
    normalized_bars = bars.copy(deep=True)
    normalized_bars.index = normalized_bars.index.tz_convert("UTC")
    bars = normalized_bars
    normalized_atr_contract = _validate_atr_contract(atr_contract)
    _validate_atr(bars)
    _validate_atr_availability(bars)
    source_bytes, source_hashes = _validate_sources(source_paths, expected_hashes)
    bucket_diagnostics, independent_anchor = _independent_anchor_buckets(bars, anchor_timeframe)
    contract = _contract_manifest(
        instrument,
        anchor_timeframe,
        pivot_n,
        normalized_atr_contract,
        source_hashes,
    )

    records = []
    with side_effect_guard(records):
        htf = _compile_source(
            source_bytes["htf"], source_paths["htf"], "_causal_parent_recovered_htf"
        )
        range_source = _compile_source(
            source_bytes["range"], source_paths["range"], "_causal_parent_recovered_range"
        )
        for namespace, functions in (
            (htf, ("resample_htf", "detect_fractal_pivots", "_broadcast")),
            (range_source, ("build_structural_range",)),
        ):
            if any(not callable(namespace.get(name)) for name in functions):
                raise ValueError("recovered source function missing")

        complete_hours = _private_complete_hours(bars, bucket_diagnostics["completed"])
        recovered_anchor = htf["resample_htf"](complete_hours, anchor_timeframe)
        _assert_aggregation_parity(recovered_anchor, independent_anchor)

        source_anchor = independent_anchor.copy()
        aware_anchor_index = source_anchor.index.copy()
        source_anchor.index = source_anchor.index.tz_convert("UTC").tz_localize(None)
        source_anchor["close_time"] = source_anchor["close_time"].dt.tz_convert("UTC").dt.tz_localize(None)
        if not np.array_equal(
            aware_anchor_index.asi8,
            source_anchor.index.tz_localize("UTC").asi8,
        ):
            raise ValueError("UTC epoch identity failure")
        source_pivots = htf["detect_fractal_pivots"](source_anchor, pivot_n)

        private = bars.copy(deep=True)
        aware_hour_index = private.index.tz_convert("UTC")
        private.index = aware_hour_index.tz_localize(None)
        if not np.array_equal(aware_hour_index.asi8, private.index.tz_localize("UTC").asi8):
            raise ValueError("UTC epoch identity failure")
        private["swing_low_50"] = htf["_broadcast"](
            source_pivots,
            private.index,
            "pivot_low_level",
            "is_swing_low",
        )
        private["swing_high_50"] = htf["_broadcast"](
            source_pivots,
            private.index,
            "pivot_high_level",
            "is_swing_high",
        )
        source_range = range_source["build_structural_range"](
            private,
            break_buffer_atr=0.0,
            break_confirm_bars=1,
        )
    if records:
        raise ValueError("source side effects recorded")

    source_range.index = source_range.index.tz_localize("UTC")
    source_pivots.index = source_pivots.index.tz_localize("UTC")
    source_pivots["confirm_time"] = pd.to_datetime(source_pivots["confirm_time"], utc=True)
    pivot_records = _pivot_records_from_source(
        source_pivots,
        anchor_buckets=independent_anchor,
        contract_id=contract["contract_id"],
        data_stream_id=data_stream_id,
        instrument=instrument,
        anchor_timeframe=anchor_timeframe,
        pivot_n=pivot_n,
    )
    versions, transitions = _annotate_source_outputs(
        bars,
        source_range,
        pivot_records,
        contract_id=contract["contract_id"],
        data_stream_id=data_stream_id,
    )

    first_open = bars.index[0].tz_convert("UTC")
    last_processed_close = bars.index[-1].tz_convert("UTC") + pd.Timedelta("1h")
    coverage = {
        "first_open": str(first_open),
        "first_processed_close": str(first_open + pd.Timedelta("1h")),
        "last_processed_close": str(last_processed_close),
        "query_exclusive_end": str(last_processed_close + pd.Timedelta("1h")),
        "input_hours": len(bars),
    }
    manifest = {
        **contract,
        "instrument": instrument,
        "data_stream_id": data_stream_id,
        "input_hash": _input_hash(bars, instrument, data_stream_id, normalized_atr_contract),
        "input_rows": len(bars),
        "source_paths": {key: str(source_paths[key]) for key in ("htf", "range")},
        "receipt_certified": False,
    }
    result = {
        "certified": False,
        "issues": [
            "recovered_strategy_source_only",
            "no_source_receipt_certification",
            "no_native_signal_or_execution_integration",
            "anchor_parameters_are_unselected_hypotheses",
        ],
        "manifest": manifest,
        "coverage": coverage,
        "anchor_buckets": bucket_diagnostics,
        "pivots": pivot_records,
        "versions": versions,
        "transitions": transitions,
    }
    return json_safe(result)


def parent_asof(ledger, decision_time, strict=False):
    """Return a copied active parent version visible at ``decision_time``."""
    if not isinstance(strict, bool):
        raise ValueError("strict must be boolean")
    decision = utc(decision_time, "decision_time")
    coverage = ledger.get("coverage", {})
    exclusive_end = utc(coverage.get("query_exclusive_end"), "query_exclusive_end")
    if decision >= exclusive_end:
        raise ValueError("out_of_coverage")
    chosen = None
    for transition in ledger.get("transitions", []):
        available = utc(transition.get("available_at"), "transition available_at")
        if available < decision or (not strict and available == decision):
            chosen = transition
        elif available >= decision:
            break
    if chosen is None or chosen.get("post_state") != "active":
        return None
    version_id = chosen.get("post_version_id")
    for version in ledger.get("versions", []):
        if version.get("id") == version_id:
            return deepcopy(version)
    raise ValueError("ledger references unknown parent version")


def bind_parent(ledger, *, child_event_id, child_timeframe, first_sweep_open):
    """Freeze the parent visible strictly before a child candidate's first sweep."""
    _nonempty(child_event_id, "child_event_id")
    _nonempty(child_timeframe, "child_timeframe")
    try:
        child_delta = pd.Timedelta(child_timeframe)
    except (TypeError, ValueError) as exc:
        raise ValueError("child_timeframe must be a positive duration") from exc
    if pd.isna(child_delta) or child_delta <= pd.Timedelta(0) or child_delta > pd.Timedelta("1d"):
        raise ValueError("child_timeframe must be a positive duration at most one day")
    first_sweep = utc(first_sweep_open, "first_sweep_open")
    policy = "parent_available_strictly_before_first_sweep_open"
    parent = parent_asof(ledger, first_sweep, strict=True)
    if parent is None:
        return {
            "status": "rejected",
            "reason": "absent_parent",
            "child_event_id": child_event_id,
            "child_timeframe": child_timeframe,
            "first_sweep_open": str(first_sweep),
            "binding_policy": policy,
        }
    causal = {
        "kind": "child_parent_binding",
        "child_event_id": child_event_id,
        "child_timeframe": child_timeframe,
        "first_sweep_open": str(first_sweep),
        "parent_version_id": parent["id"],
        "binding_policy": policy,
    }
    return {
        "id": digest(causal),
        "status": "bound",
        "child_event_id": child_event_id,
        "child_timeframe": child_timeframe,
        "first_sweep_open": str(first_sweep),
        "binding_policy": policy,
        "parent_lineage_id": parent["lineage_id"],
        "parent_version_id": parent["id"],
        "parent_available_at": parent["available_at"],
        "parent_range_low": parent["range_low"],
        "parent_range_high": parent["range_high"],
        "parent_low_pivot_id": parent["low_pivot_id"],
        "parent_high_pivot_id": parent["high_pivot_id"],
    }
