"""Build deterministic descriptive evidence from recorded fusion exits.

This module deliberately has no loading, networking, CLI, or production-engine
dependencies. Its output describes supplied records; it does not certify fills,
position closure, profitability, or historical decision-stage parity.
"""

from collections import Counter, defaultdict
from copy import deepcopy
from datetime import datetime, timezone
from math import fsum, isfinite, sqrt
from statistics import median


_ROUNDING_TOLERANCE = 0.00015 + 1e-12
_PNL_ALIAS_TOLERANCE = 0.005 + 1e-12


def _is_number(value):
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and isfinite(value)
    )


def _format_utc(value):
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _json_safe(value):
    """Copy JSON-shaped evidence while replacing nonfinite floats with null."""
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, float) and not isfinite(value):
        return None
    if isinstance(value, datetime):
        return _format_utc(value.astimezone(timezone.utc)) if value.tzinfo else None
    return deepcopy(value)


def _parse_aware_utc(value):
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, str):
        normalized = value[:-1] + "+00:00" if value.endswith("Z") else value
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return None
    else:
        return None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        return None
    return parsed.astimezone(timezone.utc)


def _validate_top_level(
    trades, open_positions, signal_rows, archetypes, snapshot_meta
):
    if not isinstance(trades, list):
        raise ValueError("trades must be a list")
    if not isinstance(open_positions, list):
        raise ValueError("open_positions must be a list")
    if not isinstance(signal_rows, list):
        raise ValueError("signal_rows must be a list")
    if not isinstance(archetypes, list) or not archetypes:
        raise ValueError("archetypes must be a nonempty list")
    if any(not isinstance(name, str) or not name.strip() for name in archetypes):
        raise ValueError("archetypes must contain nonempty strings")
    if len(set(archetypes)) != len(archetypes):
        raise ValueError("archetypes must be unique")
    if not isinstance(snapshot_meta, dict):
        raise ValueError("snapshot_meta must be a mapping")

    server_time = _parse_aware_utc(snapshot_meta.get("server_time"))
    heartbeat = _parse_aware_utc(snapshot_meta.get("heartbeat_updated_at"))
    if server_time is None or heartbeat is None:
        raise ValueError("snapshot clocks must be timezone-aware")
    if heartbeat > server_time:
        raise ValueError("heartbeat_updated_at must not exceed server_time")
    if not isinstance(snapshot_meta.get("source_hashes"), dict):
        raise ValueError("snapshot_meta.source_hashes must be a mapping")

    normalized_meta = _json_safe(snapshot_meta)
    normalized_meta["server_time"] = _format_utc(server_time)
    normalized_meta["heartbeat_updated_at"] = _format_utc(heartbeat)
    return server_time, heartbeat, normalized_meta


def _optional_open_number(row, canonical, aliases, validator):
    value = None
    present = False
    for name in (canonical,) + tuple(aliases):
        if name in row:
            value = row[name]
            present = True
            break
    if not present or value is None:
        return None
    if not _is_number(value) or not validator(float(value)):
        raise ValueError("invalid open inventory {}".format(canonical))
    return float(value)


def _validate_open_inventory(open_positions, archetypes, heartbeat):
    positions = []
    by_id = {}
    for index, raw in enumerate(open_positions):
        if not isinstance(raw, dict):
            raise ValueError("open inventory rows must be mappings")
        position_id = raw.get("id")
        if not isinstance(position_id, str) or not position_id.strip():
            raise ValueError("open inventory id must be nonempty")
        if position_id in by_id:
            raise ValueError("duplicate open inventory id")
        archetype = raw.get("archetype")
        direction = raw.get("direction")
        entry_time = _parse_aware_utc(raw.get("entry_time"))
        if archetype not in archetypes:
            raise ValueError("open inventory archetype must be known")
        if direction not in ("long", "short"):
            raise ValueError("open inventory direction must be long or short")
        if entry_time is None or entry_time > heartbeat:
            raise ValueError("invalid open inventory entry_time")

        numeric = {}
        for field in ("entry_price", "current_quantity", "original_quantity"):
            value = raw.get(field)
            if not _is_number(value) or float(value) <= 0:
                raise ValueError("invalid open inventory {}".format(field))
            numeric[field] = float(value)
        if numeric["current_quantity"] > numeric["original_quantity"]:
            raise ValueError("open current_quantity exceeds original_quantity")

        fusion_score = _optional_open_number(
            raw, "fusion_score", ("score",), lambda value: 0 <= value <= 1
        )
        threshold = _optional_open_number(
            raw,
            "threshold_at_entry",
            ("threshold",),
            lambda value: value >= 0,
        )
        margin = _optional_open_number(
            raw, "threshold_margin", ("margin",), lambda _value: True
        )
        position = {
            "row_index": index,
            "id": position_id,
            "archetype": archetype,
            "direction": direction,
            "entry_time": _format_utc(entry_time),
            "entry_price": numeric["entry_price"],
            "current_quantity": numeric["current_quantity"],
            "original_quantity": numeric["original_quantity"],
            "fusion_score": fusion_score,
            "threshold_at_entry": threshold,
            "threshold_margin": margin,
            "record": _json_safe(raw),
        }
        positions.append(position)
        by_id[position_id] = {
            "archetype": archetype,
            "direction": direction,
            "entry_time_value": entry_time,
            "entry_price": numeric["entry_price"],
        }
    positions.sort(key=lambda item: item["id"])
    return {"usable": True, "count": len(positions), "positions": positions}, by_id


def _validate_exit_row(row, archetypes, server_time):
    reasons = []
    normalized = {}

    archetype = row.get("archetype")
    if not isinstance(archetype, str) or not archetype.strip():
        reasons.append("invalid_archetype")
    elif archetype not in archetypes:
        reasons.append("unknown_archetype")
    normalized["archetype"] = archetype

    direction = row.get("direction")
    if direction not in ("long", "short"):
        reasons.append("invalid_direction")
    normalized["direction"] = direction

    entry_time = _parse_aware_utc(row.get("entry_time"))
    exit_time = _parse_aware_utc(row.get("exit_time"))
    if entry_time is None:
        reasons.append("invalid_entry_time")
    if exit_time is None:
        reasons.append("invalid_exit_time")
    if entry_time is not None and exit_time is not None and entry_time > exit_time:
        reasons.append("entry_after_exit")
    if exit_time is not None and exit_time > server_time:
        reasons.append("exit_after_snapshot_server_time")
    normalized["entry_time_value"] = entry_time
    normalized["exit_time_value"] = exit_time

    numeric_rules = {
        "entry_price": lambda value: value > 0,
        "quantity": lambda value: value > 0,
        "pnl_usd": lambda _value: True,
        "fusion_score": lambda value: 0 <= value <= 1,
        "threshold_at_entry": lambda value: value >= 0,
        "threshold_margin": lambda _value: True,
        "displayed_stop_loss": lambda value: value >= 0,
    }
    for field, rule in numeric_rules.items():
        value = row.get(field)
        if not _is_number(value) or not rule(float(value)):
            reasons.append("invalid_{}".format(field))
            normalized[field] = None
        else:
            normalized[field] = float(value)

    if "pnl" in row:
        pnl = row.get("pnl")
        if not _is_number(pnl):
            reasons.append("invalid_pnl")
        elif (
            normalized["pnl_usd"] is not None
            and abs(float(pnl) - normalized["pnl_usd"])
            > _PNL_ALIAS_TOLERANCE
        ):
            reasons.append("pnl_alias_mismatch")

    source_version = row.get("source_version")
    if source_version in (None, ""):
        normalized["source_version"] = None
    elif not isinstance(source_version, str) or not source_version.strip():
        reasons.append("invalid_source_version")
        normalized["source_version"] = None
    else:
        normalized["source_version"] = source_version
    return sorted(set(reasons)), normalized


def _duplicates_exist(indexed_rows):
    for left in range(len(indexed_rows)):
        for right in range(left + 1, len(indexed_rows)):
            if indexed_rows[left][1] == indexed_rows[right][1]:
                return True
    return False


def _group_conflicts(validated_rows):
    reasons = []
    conflict_fields = (
        "archetype",
        "direction",
        "entry_time_value",
        "entry_price",
        "fusion_score",
        "threshold_at_entry",
        "threshold_margin",
        "displayed_stop_loss",
    )
    for field in conflict_fields:
        values = [
            normalized.get(field)
            for _index, _row, _reasons, normalized in validated_rows
        ]
        if all(value is not None for value in values) and any(
            value != values[0] for value in values[1:]
        ):
            public_name = "entry_time" if field == "entry_time_value" else field
            reasons.append("conflicting_{}".format(public_name))
    versions = {
        normalized["source_version"]
        for _index, _row, _reasons, normalized in validated_rows
        if normalized.get("source_version") is not None
    }
    if len(versions) > 1:
        reasons.append("conflicting_source_version")
    return reasons


def _display_threshold(row):
    factor = row.get("factor_attribution")
    if not isinstance(factor, dict):
        return None
    conditions = factor.get("entry_conditions")
    if not isinstance(conditions, dict):
        return None
    value = conditions.get("dynamic_threshold")
    return float(value) if _is_number(value) else None


def _derive_group(position_id, validated_rows, open_ids):
    indexed_rows = [
        (index, row) for index, row, _reasons, _normalized in validated_rows
    ]
    first = validated_rows[0][3]
    entry_time = first["entry_time_value"]
    exit_times = [
        normalized["exit_time_value"]
        for _index, _row, _reasons, normalized in validated_rows
    ]
    quantity_sum = fsum(
        normalized["quantity"]
        for _index, _row, _reasons, normalized in validated_rows
    )
    pnl_sum = fsum(
        normalized["pnl_usd"]
        for _index, _row, _reasons, normalized in validated_rows
    )
    score = first["fusion_score"]
    threshold = first["threshold_at_entry"]
    margin = first["threshold_margin"]
    stored_difference = score - threshold
    arithmetic_consistent = threshold > 0 and abs(
        stored_difference - margin
    ) <= _ROUNDING_TOLERANCE

    stop = first["displayed_stop_loss"]
    entry_price = first["entry_price"]
    adverse_stop = stop > 0 and (
        (first["direction"] == "long" and stop < entry_price)
        or (first["direction"] == "short" and stop > entry_price)
    )
    risk_proxy = abs(entry_price - stop) * quantity_sum if adverse_stop else None

    display_values_by_row = [
        _display_threshold(row) for _index, row in indexed_rows
    ]
    display_values = sorted(
        set(value for value in display_values_by_row if value is not None)
    )
    display_missing = any(value is None for value in display_values_by_row)
    versions = sorted(
        {
            normalized["source_version"]
            for _index, _row, _reasons, normalized in validated_rows
            if normalized["source_version"] is not None
        }
    )
    source_version = versions[0] if versions else "unknown"
    return {
        "position_id": position_id,
        "row_indices": [index for index, _row in indexed_rows],
        "exit_rows": [
            {"row_index": index, "record": _json_safe(row)}
            for index, row in indexed_rows
        ],
        "archetype": first["archetype"],
        "direction": first["direction"],
        "entry_time": _format_utc(entry_time),
        "first_exit_time": _format_utc(min(exit_times)),
        "last_exit_time": _format_utc(max(exit_times)),
        "entry_price": entry_price,
        "quantity_sum": quantity_sum,
        "recorded_exit_pnl_usd": pnl_sum,
        "fusion_score": score,
        "threshold_at_entry": threshold,
        "threshold_margin": margin,
        "stored_score_minus_entry_threshold": stored_difference,
        "score_margin_arithmetic_consistent": arithmetic_consistent,
        "boundary_rounding_ambiguous": abs(margin) <= _ROUNDING_TOLERANCE,
        "entry_threshold_zero_sentinel": threshold == 0,
        "entry_threshold_above_score_range": threshold > 1,
        "displayed_stop_loss": stop,
        "displayed_stop_risk_proxy_usd": risk_proxy,
        "recorded_pnl_over_displayed_stop_risk_proxy": (
            pnl_sum / risk_proxy
            if risk_proxy is not None and risk_proxy > 0
            else None
        ),
        "display_threshold_values": display_values,
        "display_threshold_missing_or_invalid": display_missing,
        "display_threshold_varied": len(display_values) > 1,
        "display_vs_entry_threshold_difference": any(
            value != threshold for value in display_values
        ),
        "source_version": source_version,
        "source_version_explicit": bool(versions),
        "source_label_duration_hours": (
            max(exit_times) - entry_time
        ).total_seconds()
        / 3600.0,
        "open_in_snapshot": position_id in open_ids,
        "completion_certified": False,
    }


def _average_ranks(values):
    ordered = sorted(range(len(values)), key=lambda index: values[index])
    ranks = [0.0] * len(values)
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and values[ordered[end]] == values[ordered[start]]:
            end += 1
        average = ((start + 1) + end) / 2.0
        for position in range(start, end):
            ranks[ordered[position]] = average
        start = end
    return ranks


def _spearman(left, right, sparse_threshold):
    pairs = [
        (x, y) for x, y in zip(left, right) if x is not None and y is not None
    ]
    n = len(pairs)
    result = {"n": n, "spearman": None, "sparse": n < sparse_threshold}
    if n < 3:
        return result
    left_ranks = _average_ranks([pair[0] for pair in pairs])
    right_ranks = _average_ranks([pair[1] for pair in pairs])
    left_mean = fsum(left_ranks) / n
    right_mean = fsum(right_ranks) / n
    left_delta = [value - left_mean for value in left_ranks]
    right_delta = [value - right_mean for value in right_ranks]
    denominator = sqrt(
        fsum(value * value for value in left_delta)
        * fsum(value * value for value in right_delta)
    )
    if denominator == 0:
        return result
    value = fsum(x * y for x, y in zip(left_delta, right_delta)) / denominator
    result["spearman"] = max(-1.0, min(1.0, value))
    return result


def _correlations(groups, sparse_threshold):
    definitions = (
        ("fusion_score", "recorded_exit_pnl_usd"),
        ("fusion_score", "recorded_pnl_over_displayed_stop_risk_proxy"),
        ("threshold_at_entry", "recorded_exit_pnl_usd"),
        (
            "threshold_at_entry",
            "recorded_pnl_over_displayed_stop_risk_proxy",
        ),
        ("threshold_margin", "recorded_exit_pnl_usd"),
        ("threshold_margin", "recorded_pnl_over_displayed_stop_risk_proxy"),
    )
    return {
        "{}_vs_{}".format(left, right): _spearman(
            [group[left] for group in groups],
            [group[right] for group in groups],
            sparse_threshold,
        )
        for left, right in definitions
    }


def _mean(values):
    return fsum(values) / len(values) if values else None


def _stats(groups, sparse_threshold):
    pnls = [group["recorded_exit_pnl_usd"] for group in groups]
    positive = [value for value in pnls if value > 0]
    negative = [value for value in pnls if value < 0]
    ratios = [
        group["recorded_pnl_over_displayed_stop_risk_proxy"]
        for group in groups
        if group["recorded_pnl_over_displayed_stop_risk_proxy"] is not None
    ]
    proxies = [
        group["displayed_stop_risk_proxy_usd"]
        for group in groups
        if group["displayed_stop_risk_proxy_usd"] is not None
    ]
    durations = [group["source_label_duration_hours"] for group in groups]
    gross_positive = fsum(positive)
    gross_negative_abs = abs(fsum(negative))
    n = len(groups)
    return {
        "n": n,
        "recorded_exit_pnl_usd_sum": fsum(pnls),
        "wins": len(positive),
        "losses": len(negative),
        "breakevens": n - len(positive) - len(negative),
        "win_fraction": len(positive) / n if n else None,
        "gross_positive_pnl_usd": gross_positive,
        "gross_negative_pnl_usd_abs": gross_negative_abs,
        "recorded_subtotal_profit_factor": (
            gross_positive / gross_negative_abs if gross_negative_abs else None
        ),
        "zero_loss": gross_negative_abs == 0,
        "mean_recorded_group_pnl_usd": _mean(pnls),
        "median_recorded_group_pnl_usd": median(pnls) if pnls else None,
        "risk_proxy_ratio_n": len(ratios),
        "mean_recorded_pnl_over_displayed_stop_risk_proxy": _mean(ratios),
        "median_recorded_pnl_over_displayed_stop_risk_proxy": (
            median(ratios) if ratios else None
        ),
        "risk_proxy_usd_n": len(proxies),
        "mean_displayed_stop_risk_proxy_usd": _mean(proxies),
        "source_label_duration_n": len(durations),
        "mean_source_label_duration_hours": _mean(durations),
        "sparse": n < sparse_threshold,
        "correlations": _correlations(groups, sparse_threshold),
    }


def _time_range(values):
    if not values:
        return {"first": None, "last": None}
    return {"first": _format_utc(min(values)), "last": _format_utc(max(values))}


def _signal_coverage(signal_rows, server_time):
    malformed = []
    valid = []
    raw_statuses = Counter()
    for index, row in enumerate(signal_rows):
        reasons = []
        if not isinstance(row, dict):
            malformed.append(
                {
                    "row_index": index,
                    "reasons": ["invalid_row_type"],
                    "record": _json_safe(row),
                }
            )
            continue
        status = row.get("status")
        if isinstance(status, str) and status.strip():
            raw_statuses[status] += 1
        else:
            reasons.append("invalid_status")
        timestamp = _parse_aware_utc(row.get("timestamp"))
        if timestamp is None:
            reasons.append("invalid_timestamp")
        elif timestamp > server_time:
            reasons.append("timestamp_after_snapshot_server_time")

        numeric = {}
        for field, rule in (
            ("fusion_score", lambda value: 0 <= value <= 1),
            ("threshold", lambda value: value >= 0),
            ("margin", lambda _value: True),
        ):
            value = row.get(field)
            if not _is_number(value) or not rule(float(value)):
                reasons.append("invalid_{}".format(field))
                numeric[field] = None
            else:
                numeric[field] = float(value)
        if reasons:
            malformed.append(
                {
                    "row_index": index,
                    "reasons": sorted(set(reasons)),
                    "record": _json_safe(row),
                }
            )
            continue
        threshold = numeric["threshold"]
        valid.append(
            {
                "row_index": index,
                "status": status,
                "timestamp_value": timestamp,
                "threshold_zero_sentinel": threshold == 0,
                "arithmetic_consistent": threshold > 0
                and abs(
                    numeric["fusion_score"] - threshold - numeric["margin"]
                )
                <= _ROUNDING_TOLERANCE,
            }
        )
    valid_statuses = Counter(row["status"] for row in valid)
    timestamps = [row["timestamp_value"] for row in valid]
    threshold_zero = sum(row["threshold_zero_sentinel"] for row in valid)
    consistent = sum(row["arithmetic_consistent"] for row in valid)
    comparable = len(valid) - threshold_zero
    return {
        "raw_rows": len(signal_rows),
        "valid_rows": len(valid),
        "malformed_row_count": len(malformed),
        "malformed_rows": malformed,
        "status_counts": dict(sorted(raw_statuses.items())),
        "valid_status_counts": dict(sorted(valid_statuses.items())),
        "time_range": _time_range(timestamps),
        "threshold_zero_rows": threshold_zero,
        "arithmetic_consistent_rows": consistent,
        "arithmetic_inconsistent_rows": comparable - consistent,
        "arithmetic_unavailable_rows": threshold_zero,
        "candidate_outcome_join_performed": False,
        "coverage_truncated": True,
    }


def _partition_exit_rows(trades):
    excluded_rows = []
    raw_groups = defaultdict(list)
    for index, raw in enumerate(trades):
        if not isinstance(raw, dict):
            excluded_rows.append(
                {
                    "row_index": index,
                    "reasons": ["invalid_row_type", "missing_position_id"],
                    "record": _json_safe(raw),
                }
            )
            continue
        position_id = raw.get("position_id")
        if not isinstance(position_id, str) or not position_id.strip():
            excluded_rows.append(
                {
                    "row_index": index,
                    "reasons": ["blank_or_invalid_position_id"],
                    "record": _json_safe(raw),
                }
            )
            continue
        raw_groups[position_id].append((index, raw))
    return excluded_rows, raw_groups


def _open_identity_conflicts(first, open_identity):
    required = ("archetype", "direction", "entry_time_value", "entry_price")
    if not all(first.get(field) is not None for field in required):
        return False
    return any(
        (
            first["archetype"] != open_identity["archetype"],
            first["direction"] != open_identity["direction"],
            first["entry_time_value"] != open_identity["entry_time_value"],
            first["entry_price"] != open_identity["entry_price"],
        )
    )


def build_fusion_scorecard(
    trades, *, open_positions, signal_rows, archetypes, snapshot_meta
):
    """Return deterministic JSON-safe descriptive evidence; no side effects."""
    server_time, heartbeat, normalized_meta = _validate_top_level(
        trades, open_positions, signal_rows, archetypes, snapshot_meta
    )
    open_inventory, open_by_id = _validate_open_inventory(
        open_positions, archetypes, heartbeat
    )
    excluded_rows, raw_groups = _partition_exit_rows(trades)

    groups = []
    quarantined_groups = []
    for position_id in sorted(raw_groups):
        indexed_rows = raw_groups[position_id]
        validated = []
        reasons = []
        for index, row in indexed_rows:
            row_reasons, normalized = _validate_exit_row(
                row, archetypes, server_time
            )
            reasons.extend(row_reasons)
            validated.append((index, row, row_reasons, normalized))
        if _duplicates_exist(indexed_rows):
            reasons.append("exact_duplicate_row")
        reasons.extend(_group_conflicts(validated))

        if position_id in open_by_id and validated:
            first = validated[0][3]
            if _open_identity_conflicts(first, open_by_id[position_id]):
                reasons.append("open_inventory_identity_conflict")

        if reasons:
            quarantined_groups.append(
                {
                    "position_id": position_id,
                    "row_indices": [index for index, _row in indexed_rows],
                    "reasons": sorted(set(reasons)),
                    "exit_rows": [
                        {"row_index": index, "record": _json_safe(row)}
                        for index, row in indexed_rows
                    ],
                }
            )
        else:
            groups.append(_derive_group(position_id, validated, open_by_id))

    eligible = [
        group
        for group in groups
        if group["score_margin_arithmetic_consistent"]
        and not group["open_in_snapshot"]
    ]
    summary = _stats(eligible, 100)
    summary["margin_cohorts"] = {
        "logged_nonnegative_margin": _stats(
            [group for group in eligible if group["threshold_margin"] >= 0], 30
        ),
        "logged_negative_margin": _stats(
            [group for group in eligible if group["threshold_margin"] < 0], 30
        ),
    }

    by_archetype = {
        archetype: _stats(
            [group for group in eligible if group["archetype"] == archetype],
            100,
        )
        for archetype in archetypes
    }
    by_direction = {
        direction: _stats(
            [group for group in eligible if group["direction"] == direction],
            100,
        )
        for direction in ("long", "short")
    }
    months = sorted({group["entry_time"][:7] for group in eligible})
    by_month = {
        month: _stats(
            [
                group
                for group in eligible
                if group["entry_time"].startswith(month)
            ],
            100,
        )
        for month in months
    }
    versions = sorted({group["source_version"] for group in eligible})
    by_source_version = {
        version: _stats(
            [
                group
                for group in eligible
                if group["source_version"] == version
            ],
            100,
        )
        for version in versions
    }

    quarantine_reason_counts = Counter(
        reason for group in quarantined_groups for reason in group["reasons"]
    )
    entry_times = [_parse_aware_utc(group["entry_time"]) for group in groups]
    exit_times = [
        _parse_aware_utc(group["last_exit_time"]) for group in groups
    ]
    coverage = {
        "raw_exit_rows": len(trades),
        "raw_signal_rows": len(signal_rows),
        "explicit_id_rows": sum(len(rows) for rows in raw_groups.values()),
        "explicit_groups": len(raw_groups),
        "retained_groups": len(groups),
        "grouped_rows": sum(len(group["row_indices"]) for group in groups),
        "ungrouped_rows": len(excluded_rows),
        "quarantined_groups": len(quarantined_groups),
        "quarantined_rows": sum(
            len(group["row_indices"]) for group in quarantined_groups
        ),
        "accounted_exit_rows": len(excluded_rows)
        + sum(len(group["row_indices"]) for group in groups)
        + sum(len(group["row_indices"]) for group in quarantined_groups),
        "quarantine_reason_group_counts": dict(
            sorted(quarantine_reason_counts.items())
        ),
        "open_inventory_rows": open_inventory["count"],
        "groups_with_open_id_match": sum(
            group["open_in_snapshot"] for group in groups
        ),
        "groups_without_open_id_match": sum(
            not group["open_in_snapshot"] for group in groups
        ),
        "eligible_arithmetic_consistent_non_open_groups": len(eligible),
        "threshold_zero_groups": sum(
            group["entry_threshold_zero_sentinel"] for group in groups
        ),
        "score_margin_inconsistent_groups": sum(
            not group["score_margin_arithmetic_consistent"] for group in groups
        ),
        "boundary_rounding_ambiguous_groups": sum(
            group["boundary_rounding_ambiguous"] for group in groups
        ),
        "entry_threshold_above_score_range_groups": sum(
            group["entry_threshold_above_score_range"] for group in groups
        ),
        "display_threshold_missing_or_invalid_groups": sum(
            group["display_threshold_missing_or_invalid"] for group in groups
        ),
        "display_threshold_varied_groups": sum(
            group["display_threshold_varied"] for group in groups
        ),
        "display_vs_entry_threshold_difference_groups": sum(
            group["display_vs_entry_threshold_difference"] for group in groups
        ),
        "source_version_known_groups": sum(
            group["source_version_explicit"] for group in groups
        ),
        "source_version_unknown_groups": sum(
            not group["source_version_explicit"] for group in groups
        ),
        "risk_proxy_groups": sum(
            group["displayed_stop_risk_proxy_usd"] is not None
            for group in groups
        ),
        "source_entry_time_range": _time_range(entry_times),
        "source_exit_time_range": _time_range(exit_times),
        "completion_certified": False,
    }

    return {
        "certified": False,
        "limitations": [
            "paper_shadow_records",
            "snapshots_not_atomic",
            "recorded_exit_groups_not_open_in_snapshot",
            "completion_not_certified",
            "candidate_capture_truncated",
            "historical_source_versions_and_decision_stages_unauthenticated",
            "costs_and_full_net_returns_not_reconstructed",
            "displayed_stop_risk_proxy_not_certified_initial_risk",
            "descriptive_association_not_calibration_or_backtest",
        ],
        "snapshot_meta": normalized_meta,
        "coverage": coverage,
        "groups": groups,
        "quarantined_groups": quarantined_groups,
        "excluded_rows": excluded_rows,
        "open_inventory": open_inventory,
        "summary": summary,
        "by_archetype": by_archetype,
        "by_direction": by_direction,
        "by_month": by_month,
        "by_source_version": by_source_version,
        "signal_coverage": _signal_coverage(signal_rows, server_time),
    }
