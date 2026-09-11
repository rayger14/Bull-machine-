"""Pure research-side annotation of fixed events with parent permission."""

from copy import deepcopy
import math

import pandas as pd

from scripts.research.causal_parent_ledger import bind_parent
from scripts.research.replay_clock import digest, json_safe, utc


LC_POLICY = "lc_fixed_parent_reclaim_v1"
MINUTE_POLICY = "minute_child_sweep_parent_location_v1"
POLICIES = {LC_POLICY, MINUTE_POLICY}
PARENT_CONFIGS = {("4H", 3), ("4H", 5), ("1D", 3), ("1D", 5)}
LIMITATIONS = [
    "research_only_fixed_event_annotation",
    "event_evidence_ids_are_not_source_receipt_authentication",
    "parent_break_is_a_lineage_lifecycle_veto_not_an_old_floor_counterfactual",
    "no_native_selection_execution_or_performance_claim",
]
COMPARISON_CONTRACT = {
    "geometry": "frozen_parent_version_strictly_before_first_sweep",
    "lineage_lifecycle": "any_bound_lineage_break_through_decision_rejects",
    "old_floor_break_counterfactual": False,
    "lc": "low < parent_low < close < parent_high",
    "minute_level": "parent_low <= child_level <= parent_midpoint",
    "minute_reclaim": "parent_low < reclaim_close < parent_high",
}


class _UnknownEvidence(ValueError):
    pass


def _nonempty(value):
    return isinstance(value, str) and bool(value.strip())


def _positive_number(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise _UnknownEvidence("invalid_child_event")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric <= 0:
        raise _UnknownEvidence("invalid_child_event")
    return value


def _on_grid(value, step):
    return value == value.floor(step)


def _validate_child_event(policy_id, child_event):
    if not isinstance(child_event, dict):
        raise _UnknownEvidence("invalid_child_event")
    event = deepcopy(child_event)
    for field in ("id", "contract_id", "kind", "instrument", "data_stream_id", "evidence_id"):
        if not _nonempty(event.get(field)):
            raise _UnknownEvidence("invalid_child_event")
    if not isinstance(event.get("values"), dict):
        raise _UnknownEvidence("invalid_child_event")
    clocks = {}
    try:
        for field in ("first_sweep_open", "reclaim_bar_open", "decision_time", "available_at"):
            clocks[field] = utc(event.get(field), field)
    except (TypeError, ValueError):
        raise _UnknownEvidence("invalid_child_event")

    values = event["values"]
    if policy_id == LC_POLICY:
        if event["kind"] != "hourly_lc":
            raise _UnknownEvidence("invalid_child_event")
        if not _on_grid(clocks["first_sweep_open"], "h"):
            raise _UnknownEvidence("invalid_child_event")
        if clocks["reclaim_bar_open"] != clocks["first_sweep_open"]:
            raise _UnknownEvidence("invalid_child_event")
        derived = clocks["reclaim_bar_open"] + pd.Timedelta("1h")
        low = _positive_number(values.get("low"))
        close = _positive_number(values.get("close"))
        if low > close:
            raise _UnknownEvidence("invalid_child_event")
        child_timeframe = "1h"
    else:
        if event["kind"] != "minute_equal_low_sweep":
            raise _UnknownEvidence("invalid_child_event")
        if not _on_grid(clocks["first_sweep_open"], "min") or not _on_grid(
            clocks["reclaim_bar_open"], "min"
        ):
            raise _UnknownEvidence("invalid_child_event")
        if clocks["reclaim_bar_open"] < clocks["first_sweep_open"]:
            raise _UnknownEvidence("invalid_child_event")
        derived = clocks["reclaim_bar_open"] + pd.Timedelta("1min")
        child_level = _positive_number(values.get("child_level"))
        sweep_low = _positive_number(values.get("sweep_low"))
        reclaim_close = _positive_number(values.get("reclaim_close"))
        if not sweep_low < child_level or sweep_low > reclaim_close:
            raise _UnknownEvidence("invalid_child_event")
        child_timeframe = "1min"
    if clocks["decision_time"] != derived or clocks["available_at"] != derived:
        raise _UnknownEvidence("invalid_child_event")
    for field, value in clocks.items():
        event[field] = str(value)
    event["values"] = deepcopy(values)
    return event, clocks, child_timeframe


def _parent_config(ledger):
    if not isinstance(ledger, dict):
        raise _UnknownEvidence("invalid_parent_ledger")
    manifest = ledger.get("manifest")
    if not isinstance(manifest, dict):
        raise _UnknownEvidence("invalid_parent_ledger")
    parameters = manifest.get("parameters")
    if not isinstance(parameters, dict):
        raise _UnknownEvidence("invalid_parent_ledger")
    anchor_timeframe = parameters.get("anchor_timeframe")
    pivot_n = parameters.get("pivot_n")
    if (
        not isinstance(anchor_timeframe, str)
        or isinstance(pivot_n, bool)
        or not isinstance(pivot_n, int)
    ):
        raise _UnknownEvidence("invalid_parent_ledger")
    config = (anchor_timeframe, pivot_n)
    if config not in PARENT_CONFIGS:
        raise _UnknownEvidence("invalid_parent_ledger")
    return {"anchor_timeframe": config[0], "pivot_n": config[1]}


def _version_index(ledger):
    versions = {}
    raw_versions = ledger.get("versions")
    if not isinstance(raw_versions, list):
        raise _UnknownEvidence("malformed_causal_prefix")
    for version in raw_versions:
        if not isinstance(version, dict) or not _nonempty(version.get("id")):
            continue
        versions.setdefault(version["id"], []).append(version)
    return versions


def _validated_version(versions, version_id):
    matches = versions.get(version_id, [])
    if len(matches) != 1:
        raise _UnknownEvidence("malformed_causal_prefix")
    version = matches[0]
    for field in ("id", "lineage_id", "creation_reason", "low_pivot_id", "high_pivot_id"):
        if not _nonempty(version.get(field)):
            raise _UnknownEvidence("malformed_causal_prefix")
    low = version.get("range_low")
    high = version.get("range_high")
    if (
        isinstance(low, bool)
        or isinstance(high, bool)
        or not isinstance(low, (int, float))
        or not isinstance(high, (int, float))
        or not math.isfinite(float(low))
        or not math.isfinite(float(high))
        or float(low) <= 0
        or float(low) >= float(high)
    ):
        raise _UnknownEvidence("malformed_causal_prefix")
    try:
        formed = utc(version.get("formation_hour"), "formation_hour")
        available = utc(version.get("available_at"), "version available_at")
    except (TypeError, ValueError):
        raise _UnknownEvidence("malformed_causal_prefix")
    if not _on_grid(formed, "h") or available != formed + pd.Timedelta("1h"):
        raise _UnknownEvidence("malformed_causal_prefix")
    return version


def _same_bound(actual, expected):
    return (
        isinstance(actual, (int, float))
        and not isinstance(actual, bool)
        and float(actual) == float(expected)
    )


def _validate_anchor_references(ledger, versions, referenced_ids, manifest, config):
    raw_pivots = ledger.get("pivots")
    if not isinstance(raw_pivots, list):
        raise _UnknownEvidence("malformed_causal_prefix")
    pivots = {}
    for pivot in raw_pivots:
        if isinstance(pivot, dict) and _nonempty(pivot.get("id")):
            pivots.setdefault(pivot["id"], []).append(pivot)
    for version_id in referenced_ids:
        version = _validated_version(versions, version_id)
        formed = utc(version["formation_hour"], "formation_hour")
        for side, field, bound in (
            ("low", "low_pivot_id", version["range_low"]),
            ("high", "high_pivot_id", version["range_high"]),
        ):
            matches = pivots.get(version[field], [])
            if len(matches) != 1:
                raise _UnknownEvidence("malformed_causal_prefix")
            pivot = matches[0]
            try:
                available = utc(pivot.get("available_at"), "pivot available_at")
                confirming = utc(pivot.get("confirming_close"), "confirming_close")
            except (TypeError, ValueError):
                raise _UnknownEvidence("malformed_causal_prefix")
            if (
                pivot.get("side") != side
                or not _nonempty(pivot.get("evidence_id"))
                or pivot.get("instrument") != manifest["instrument"]
                or pivot.get("data_stream_id") != manifest["data_stream_id"]
                or pivot.get("anchor_timeframe") != config["anchor_timeframe"]
                or pivot.get("pivot_n") != config["pivot_n"]
                or not _same_bound(pivot.get("level"), bound)
                or available != confirming
                or available > formed
            ):
                raise _UnknownEvidence("malformed_causal_prefix")


def _validate_lifecycle(prefix, versions):
    first = prefix[0]
    if (
        first.get("pre_state") != "forming"
        or first.get("pre_range_low") is not None
        or first.get("pre_range_high") is not None
        or first.get("pre_lineage_id") is not None
        or first.get("pre_version_id") is not None
        or first.get("evaluated_version_id") is not None
    ):
        raise _UnknownEvidence("malformed_causal_prefix")
    prior = None
    for transition in prefix:
        transition_available = utc(
            transition["available_at"], "transition available_at"
        )
        pre_state = transition.get("pre_state")
        post_state = transition.get("post_state")
        if not _nonempty(pre_state) or not _nonempty(post_state):
            raise _UnknownEvidence("malformed_causal_prefix")
        if transition.get("source_range_state") != post_state:
            raise _UnknownEvidence("malformed_causal_prefix")
        pre_version_id = transition.get("pre_version_id")
        post_version_id = transition.get("post_version_id")
        pre_lineage_id = transition.get("pre_lineage_id")
        post_lineage_id = transition.get("post_lineage_id")
        if transition.get("evaluated_version_id") != pre_version_id:
            raise _UnknownEvidence("malformed_causal_prefix")
        if prior is not None and (
            pre_state != prior.get("post_state")
            or pre_version_id != prior.get("post_version_id")
            or pre_lineage_id != prior.get("post_lineage_id")
        ):
            raise _UnknownEvidence("malformed_causal_prefix")

        pre_version = None
        if pre_state == "active":
            if not _nonempty(pre_version_id) or not _nonempty(pre_lineage_id):
                raise _UnknownEvidence("malformed_causal_prefix")
            pre_version = _validated_version(versions, pre_version_id)
            if (
                pre_version["lineage_id"] != pre_lineage_id
                or utc(pre_version["available_at"], "version available_at")
                > transition_available
                or not _same_bound(transition.get("pre_range_low"), pre_version["range_low"])
                or not _same_bound(transition.get("pre_range_high"), pre_version["range_high"])
            ):
                raise _UnknownEvidence("malformed_causal_prefix")
        elif pre_version_id is not None or pre_lineage_id is not None:
            raise _UnknownEvidence("malformed_causal_prefix")

        post_version = None
        if post_state == "active":
            if not _nonempty(post_version_id) or not _nonempty(post_lineage_id):
                raise _UnknownEvidence("malformed_causal_prefix")
            post_version = _validated_version(versions, post_version_id)
            if (
                post_version["lineage_id"] != post_lineage_id
                or utc(post_version["available_at"], "version available_at")
                > transition_available
                or not _same_bound(transition.get("post_range_low"), post_version["range_low"])
                or not _same_bound(transition.get("post_range_high"), post_version["range_high"])
            ):
                raise _UnknownEvidence("malformed_causal_prefix")
        elif post_version_id is not None or post_lineage_id is not None:
            raise _UnknownEvidence("malformed_causal_prefix")

        direction = transition.get("source_break_direction")
        if pre_state == "active" and post_state != "active":
            if direction not in ("up", "down") or post_state != "broken_" + direction:
                raise _UnknownEvidence("malformed_causal_prefix")
        elif direction is not None:
            raise _UnknownEvidence("malformed_causal_prefix")
        elif pre_state == "active" and post_state == "active":
            if post_lineage_id != pre_lineage_id:
                raise _UnknownEvidence("malformed_causal_prefix")
            if post_version_id != pre_version_id:
                if (
                    post_version.get("creation_reason") != "floor_tightening"
                    or post_version.get("predecessor_version_id") != pre_version_id
                ):
                    raise _UnknownEvidence("malformed_causal_prefix")
                if (
                    utc(post_version["available_at"], "version available_at")
                    != transition_available
                ):
                    raise _UnknownEvidence("malformed_causal_prefix")
        elif pre_state != "active" and post_state == "active":
            if (
                post_version.get("creation_reason") != "formation"
                or post_version.get("predecessor_version_id") is not None
            ):
                raise _UnknownEvidence("malformed_causal_prefix")
            if (
                utc(post_version["available_at"], "version available_at")
                != transition_available
            ):
                raise _UnknownEvidence("malformed_causal_prefix")
        prior = transition


def _validate_prefix(ledger, decision, config):
    if not isinstance(ledger, dict):
        raise _UnknownEvidence("invalid_parent_ledger")
    manifest = ledger.get("manifest")
    coverage = ledger.get("coverage")
    if not isinstance(manifest, dict) or not isinstance(coverage, dict):
        raise _UnknownEvidence("invalid_parent_ledger")
    try:
        if manifest.get("schema") != "causal_parent_ledger.v1":
            raise _UnknownEvidence("invalid_parent_ledger")
        for field in ("contract_id", "instrument", "data_stream_id"):
            if not _nonempty(manifest.get(field)):
                raise _UnknownEvidence("invalid_parent_ledger")
        first_open = utc(coverage.get("first_open"), "first_open")
        first_close = utc(coverage.get("first_processed_close"), "first_processed_close")
        last_close = utc(coverage.get("last_processed_close"), "last_processed_close")
        exclusive_end = utc(coverage.get("query_exclusive_end"), "query_exclusive_end")
        input_hours = coverage.get("input_hours")
    except (KeyError, TypeError, ValueError):
        raise _UnknownEvidence("invalid_parent_ledger")
    if (
        not _on_grid(first_open, "h")
        or first_close != first_open + pd.Timedelta("1h")
        or isinstance(input_hours, bool)
        or not isinstance(input_hours, int)
        or input_hours < 1
        or input_hours > 2048
        or last_close != first_open + input_hours * pd.Timedelta("1h")
        or exclusive_end != last_close + pd.Timedelta("1h")
    ):
        raise _UnknownEvidence("invalid_parent_ledger")
    latest_completed = decision.floor("h")
    if latest_completed < first_close or last_close < latest_completed:
        raise _UnknownEvidence("insufficient_causal_coverage")
    required = int((latest_completed - first_open) / pd.Timedelta("1h"))
    raw_transitions = ledger.get("transitions")
    if not isinstance(raw_transitions, list) or len(raw_transitions) < required:
        raise _UnknownEvidence("insufficient_causal_coverage")
    versions = _version_index(ledger)
    prefix = []
    referenced_ids = set()
    transition_ids = set()
    for position in range(required):
        transition = raw_transitions[position]
        if not isinstance(transition, dict) or not _nonempty(transition.get("id")):
            raise _UnknownEvidence("malformed_causal_prefix")
        if transition["id"] in transition_ids:
            raise _UnknownEvidence("malformed_causal_prefix")
        transition_ids.add(transition["id"])
        try:
            source = utc(transition.get("source_hour"), "source_hour")
            available = utc(transition.get("available_at"), "transition available_at")
        except (TypeError, ValueError):
            raise _UnknownEvidence("malformed_causal_prefix")
        expected_source = first_open + position * pd.Timedelta("1h")
        if source != expected_source or available != source + pd.Timedelta("1h"):
            raise _UnknownEvidence("malformed_causal_prefix")
        if available > decision:
            raise _UnknownEvidence("malformed_causal_prefix")
        for key in ("pre_version_id", "post_version_id"):
            if transition.get(key) is not None:
                _validated_version(versions, transition[key])
                referenced_ids.add(transition[key])
        prefix.append(deepcopy(transition))
    for transition in raw_transitions[required:]:
        if not isinstance(transition, dict):
            continue
        try:
            available = utc(transition.get("available_at"), "transition available_at")
        except (TypeError, ValueError):
            continue
        if available <= decision:
            raise _UnknownEvidence("malformed_causal_prefix")
    _validate_lifecycle(prefix, versions)
    _validate_anchor_references(ledger, versions, referenced_ids, manifest, config)
    return manifest, prefix, versions, referenced_ids


def _base_result(policy_id, event, ledger, parent_config, *, status, reasons, decision):
    manifest = ledger.get("manifest", {}) if isinstance(ledger, dict) else {}
    if not isinstance(manifest, dict):
        manifest = {}
    return {
        "policy_id": policy_id,
        "status": status,
        "would_allow": status == "pass",
        "reasons": list(reasons),
        "child_event_id": event.get("id") if isinstance(event, dict) else None,
        "child_event_contract_id": event.get("contract_id") if isinstance(event, dict) else None,
        "decision_time": str(decision) if decision is not None else None,
        "parent_config": deepcopy(parent_config),
        "parent_contract_id": manifest.get("contract_id"),
        "parent_data_stream_id": manifest.get("data_stream_id"),
        "binding": None,
        "interval_transition_ids": [],
        "bound_lineage_broken": False,
        "bound_version_superseded": False,
        "evaluated_values": {},
        "comparison_contract": deepcopy(COMPARISON_CONTRACT),
        "certified": False,
        "limitations": list(LIMITATIONS),
    }


def _finish(result, *, event_evidence):
    causal = {
        "kind": "h3_parent_permission",
        "policy_id": result["policy_id"],
        "comparison_contract": result["comparison_contract"],
        "child_event": event_evidence,
        "parent_contract_id": result["parent_contract_id"],
        "parent_config": result["parent_config"],
        "binding": result["binding"],
        "interval_transition_ids": result["interval_transition_ids"],
        "status": result["status"],
        "reasons": result["reasons"],
    }
    result["id"] = digest(causal)
    return json_safe(result)


def evaluate_h3_permission(ledger, *, policy_id, child_event):
    """Return copied, causal permission evidence; never mutate a trading engine."""
    if policy_id not in POLICIES:
        raise ValueError("unknown parent permission policy_id")
    raw_event = deepcopy(child_event)
    event_evidence = raw_event
    try:
        parent_config = _parent_config(ledger)
    except _UnknownEvidence:
        parent_config = None
    decision = None
    try:
        event, clocks, child_timeframe = _validate_child_event(policy_id, raw_event)
        event_evidence = event
        decision = clocks["decision_time"]
        if parent_config is None:
            raise _UnknownEvidence("invalid_parent_ledger")
        manifest, prefix, versions, validated_version_ids = _validate_prefix(
            ledger, decision, parent_config
        )
        if event["instrument"] != manifest["instrument"]:
            raise _UnknownEvidence("instrument_mismatch")
        if event["data_stream_id"] != manifest["data_stream_id"]:
            raise _UnknownEvidence("data_stream_mismatch")
        causal_ledger = deepcopy(ledger)
        causal_ledger["transitions"] = prefix
        binding = bind_parent(
            causal_ledger,
            child_event_id=event["id"],
            child_timeframe=child_timeframe,
            first_sweep_open=clocks["first_sweep_open"],
        )
        if (
            binding.get("status") == "bound"
            and binding.get("parent_version_id") not in validated_version_ids
        ):
            raise _UnknownEvidence("malformed_causal_prefix")
        if binding.get("parent_available_at") is not None:
            parent_available = utc(
                binding["parent_available_at"], "parent_available_at"
            )
            if parent_available >= clocks["first_sweep_open"]:
                raise _UnknownEvidence("malformed_causal_prefix")
            binding["parent_available_at"] = str(parent_available)
    except _UnknownEvidence as exc:
        result = _base_result(
            policy_id,
            raw_event,
            ledger,
            parent_config,
            status="unknown",
            reasons=[str(exc)],
            decision=decision,
        )
        return _finish(result, event_evidence=json_safe(event_evidence))
    except (KeyError, TypeError, ValueError):
        result = _base_result(
            policy_id,
            raw_event,
            ledger,
            parent_config,
            status="unknown",
            reasons=["invalid_parent_ledger"],
            decision=decision,
        )
        return _finish(result, event_evidence=json_safe(event_evidence))

    interval = [
        transition
        for transition in prefix
        if clocks["first_sweep_open"]
        <= utc(transition["available_at"], "transition available_at")
        <= decision
    ]
    status = "reject" if binding["status"] == "rejected" else "pass"
    reasons = [binding["reason"]] if status == "reject" else []
    result = _base_result(
        policy_id,
        event,
        ledger,
        parent_config,
        status=status,
        reasons=reasons,
        decision=decision,
    )
    result["binding"] = deepcopy(binding)
    result["interval_transition_ids"] = [row["id"] for row in interval]
    if binding["status"] == "bound":
        _validated_version(versions, binding["parent_version_id"])
        bound_lineage = binding["parent_lineage_id"]
        for transition in interval:
            if transition.get("pre_lineage_id") != bound_lineage:
                continue
            if transition.get("source_break_direction") in ("up", "down"):
                result["bound_lineage_broken"] = True
            elif transition.get("post_version_id") != transition.get("pre_version_id"):
                result["bound_version_superseded"] = True
        low = binding["parent_range_low"]
        high = binding["parent_range_high"]
        if policy_id == LC_POLICY:
            child_low = event["values"]["low"]
            close = event["values"]["close"]
            allowed = child_low < low and low < close < high
            result["evaluated_values"] = {
                "low": child_low,
                "close": close,
                "parent_low": low,
                "parent_high": high,
            }
        else:
            level = event["values"]["child_level"]
            reclaim = event["values"]["reclaim_close"]
            midpoint = (low + high) / 2
            allowed = low <= level <= midpoint and low < reclaim < high
            result["evaluated_values"] = {
                **event["values"],
                "parent_low": low,
                "parent_high": high,
                "parent_midpoint": midpoint,
            }
        if not allowed:
            result["status"] = "reject"
            result["would_allow"] = False
            result["reasons"] = ["frozen_geometry_rejected"]
        if result["bound_lineage_broken"]:
            result["status"] = "reject"
            result["would_allow"] = False
            result["reasons"] = ["bound_lineage_broken"]
    return _finish(result, event_evidence=event)


def annotate_h3_events(ledgers, *, policy_id, child_events):
    """Require all four hypotheses, annotate every event, summarize status counts."""
    if policy_id not in POLICIES:
        raise ValueError("unknown parent permission policy_id")
    if not isinstance(ledgers, (list, tuple)):
        raise ValueError("exactly four unique parent configurations required")
    configs = []
    ledger_configs = []
    try:
        for ledger in ledgers:
            config = _parent_config(ledger)
            ledger_configs.append(config)
            configs.append((config["anchor_timeframe"], config["pivot_n"]))
    except _UnknownEvidence as exc:
        raise ValueError("exactly four unique parent configurations required") from exc
    if len(configs) != 4 or len(set(configs)) != 4 or set(configs) != PARENT_CONFIGS:
        raise ValueError("exactly four unique parent configurations required")
    try:
        events = list(deepcopy(child_events))
    except TypeError as exc:
        raise ValueError("child_events must be iterable") from exc
    rows = []
    counts = {
        "%s:%d" % config: {"pass": 0, "reject": 0, "unknown": 0}
        for config in sorted(PARENT_CONFIGS)
    }
    for event in events:
        for ledger, config in zip(ledgers, ledger_configs):
            row = evaluate_h3_permission(ledger, policy_id=policy_id, child_event=event)
            rows.append(row)
            key = "%s:%d" % (config["anchor_timeframe"], config["pivot_n"])
            counts[key][row["status"]] += 1
    return json_safe(
        {
            "policy_id": policy_id,
            "rows": rows,
            "counts": counts,
            "certified": False,
            "limitations": list(LIMITATIONS),
        }
    )
