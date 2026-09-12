"""Deterministic evidence-ID contract for bounded, research-only assessments.

Catalog membership proves only that a non-null packet value existed at a typed
path.  It does not prove provenance, claim atomicity, or semantic entailment.
"""
from copy import deepcopy
from datetime import datetime, timezone
import json
import math

from scripts.research.assessment_evidence_guard import resolve_evidence


_CATALOG_ROOTS = {
    "evidence", "rulecard", "plan", "indicative_economics", "input_status",
    "limitations", "specialist_instructions", "instruction", "candle_columns",
}
_CANDLE_KEYS = {"1m", "5m", "15m", "1h", "4h", "1d"}
_RESPONSE_KEYS = {
    "case_id", "decision", "probability_net_positive", "facts", "trade_plan", "claims",
}
_FACT_KEYS = {
    "decision_time", "last_1m_close", "parent_4h_status", "parent_1d_status",
    "minute_child_level",
}
_PLAN_KEYS = {
    "direction", "entry", "stop", "target_formula", "horizon_minutes", "notional",
    "roundtrip_cost", "actual_fill_known", "deadline_exit",
}
_CLAIM_KEYS = {"category", "claim", "status", "evidence_ids"}
_CATEGORIES = {"detector", "structure", "sequence", "economics", "reason", "missing"}
_STATUSES = {"supported", "contradicted", "unresolved", "not_applicable"}
_DECISIONS = {"accept", "reject", "insufficient_evidence"}
_PARENT_STATUSES = {"verified_present", "verified_absent", "unknown"}
_DEADLINE_EXIT = "deadline minute OPEN if no earlier stop/target"


def _contains_null(value, active=None):
    if value is None:
        return True
    if active is None:
        active = set()
    if type(value) in (dict, list):
        identity = id(value)
        if identity in active:
            raise ValueError("packet cannot contain cycles")
        active.add(identity)
        try:
            children = value.values() if type(value) is dict else value
            return any(_contains_null(item, active) for item in children)
        finally:
            active.remove(identity)
    return False


def build_catalog(packet):
    """Return stable IDs mapped to precise, non-null typed packet paths."""
    if not isinstance(packet, dict):
        raise ValueError("packet must be an object")
    paths = []

    def visit(value, path):
        if value is None:
            return
        if type(value) is dict:
            if any(not isinstance(key, str) for key in value):
                raise ValueError("packet object keys must be strings")
            for key in sorted(value):
                visit(value[key], [*path, key])
            return
        if type(value) is list:
            if len(path) == 2 and path[0] == "evidence" and path[1] in _CANDLE_KEYS:
                for index, row in enumerate(value):
                    if not _contains_null(row):
                        paths.append([*path, index])
            elif not _contains_null(value):
                paths.append(path)
            else:
                for index, item in enumerate(value):
                    visit(item, [*path, index])
            return
        paths.append(path)

    for root in sorted(_CATALOG_ROOTS & packet.keys()):
        visit(packet[root], [root])
    return {f"E{index:04d}": path for index, path in enumerate(paths, 1)}


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _finite_number(value):
    if type(value) not in (int, float):
        return False
    try:
        return math.isfinite(value)
    except OverflowError:
        return False


def _instant(value):
    if not isinstance(value, str):
        raise ValueError("timestamp must be text")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timestamp must include an offset")
    return parsed.astimezone(timezone.utc)


def _type_exact(left, right):
    """JSON-tree equality that does not equate bool with numeric zero/one."""
    if type(left) is not type(right):
        return False
    if type(left) is dict:
        return (set(left) == set(right)
                and all(_type_exact(left[key], right[key]) for key in left))
    if type(left) is list:
        return len(left) == len(right) and all(
            _type_exact(a, b) for a, b in zip(left, right)
        )
    return left == right


def expected_trade_plan(packet):
    """Detach the packet's fixed registered plan without deriving an alternative."""
    try:
        plan = packet["plan"]
        economics = packet["indicative_economics"]
        if not isinstance(packet, dict) or not isinstance(plan, dict) or not isinstance(economics, dict):
            raise ValueError
        if not all(_text(plan.get(key)) for key in ("direction", "entry", "target")):
            raise ValueError
        if type(plan.get("horizon_minutes")) is not int or plan["horizon_minutes"] <= 0:
            raise ValueError
        if (not _finite_number(plan.get("stop")) or not _finite_number(plan.get("notional"))
                or not _finite_number(plan.get("roundtrip_cost"))
                or plan["notional"] <= 0 or plan["roundtrip_cost"] < 0):
            raise ValueError
        if economics.get("actual_fill_known") is not False:
            raise ValueError
        return {
            "direction": plan["direction"],
            "entry": plan["entry"],
            "stop": deepcopy(plan["stop"]),
            "target_formula": plan["target"],
            "horizon_minutes": plan["horizon_minutes"],
            "notional": deepcopy(plan["notional"]),
            "roundtrip_cost": deepcopy(plan["roundtrip_cost"]),
            "actual_fill_known": False,
            "deadline_exit": _DEADLINE_EXIT,
        }
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("packet fixed trade plan is invalid") from exc
    except ValueError as exc:
        raise ValueError("packet fixed trade plan is invalid") from exc


def _number_equal(actual, expected):
    return (_finite_number(actual) and _finite_number(expected)
            and abs(actual - expected) <= 1e-6)


def _fact_errors(packet, facts):
    if not isinstance(facts, dict):
        return ["facts_object"]
    errors = []
    if set(facts) != _FACT_KEYS:
        errors.append("facts_keys")
    try:
        if _instant(facts.get("decision_time")) != _instant(packet["decision_time"]):
            errors.append("fact:decision_time")
    except (KeyError, TypeError, ValueError, OverflowError):
        errors.append("fact:decision_time")
    try:
        if not _number_equal(facts.get("last_1m_close"), packet["plan"]["indicative_close"]):
            errors.append("fact:last_1m_close")
    except (KeyError, TypeError, OverflowError):
        errors.append("fact:last_1m_close")
    for suffix in ("4h", "1d"):
        key = "parent_" + suffix + "_status"
        try:
            expected = packet["evidence"]["parent_" + suffix]["status"]
            if facts.get(key) not in _PARENT_STATUSES or facts.get(key) != expected:
                errors.append("fact:" + key)
        except (KeyError, TypeError, AttributeError):
            errors.append("fact:" + key)
    try:
        child = facts.get("minute_child_level")
        if packet.get("track") == "minute":
            good = _number_equal(child, packet["evidence"]["setup"]["child_level"])
        elif packet.get("track") == "hourly":
            good = child is None
        else:
            good = False
        if not good:
            errors.append("fact:minute_child_level")
    except (KeyError, TypeError, AttributeError, OverflowError):
        errors.append("fact:minute_child_level")
    return errors


def _trade_plan_errors(packet, supplied):
    if not isinstance(supplied, dict):
        return ["trade_plan_object"]
    errors = []
    if set(supplied) != _PLAN_KEYS:
        errors.append("trade_plan_keys")
    try:
        expected = expected_trade_plan(packet)
    except ValueError:
        return errors + ["packet_trade_plan"]
    for key in ("stop", "notional", "roundtrip_cost"):
        actual = supplied.get(key)
        if not (_finite_number(actual) and actual == expected[key]):
            errors.append("trade_plan:" + key)
    if (type(supplied.get("horizon_minutes")) is not int
            or supplied.get("horizon_minutes") != expected["horizon_minutes"]):
        errors.append("trade_plan:horizon_minutes")
    if supplied.get("actual_fill_known") is not False:
        errors.append("trade_plan:actual_fill_known")
    for key in ("direction", "entry", "target_formula", "deadline_exit"):
        if type(supplied.get(key)) is not str or supplied.get(key) != expected[key]:
            errors.append("trade_plan:" + key)
    return errors


def _claim_errors(claims, catalog):
    if not isinstance(claims, list):
        return ["claims_list"]
    errors = []
    if not 6 <= len(claims) <= 12:
        errors.append("claims_count")
    seen_categories = set()
    for index, claim in enumerate(claims):
        prefix = f"claim:{index}:"
        if not isinstance(claim, dict):
            errors.append(prefix + "object")
            continue
        if set(claim) != _CLAIM_KEYS:
            errors.append(prefix + "keys")
        category = claim.get("category")
        if category not in _CATEGORIES:
            errors.append(prefix + "category")
        else:
            seen_categories.add(category)
        text = claim.get("claim")
        if not _text(text) or len(text) > 400:
            errors.append(prefix + "claim")
        if claim.get("status") not in _STATUSES:
            errors.append(prefix + "status")
        ids = claim.get("evidence_ids")
        if not isinstance(ids, list) or not ids:
            errors.append(prefix + "evidence_ids")
        elif (any(type(item) is not str for item in ids)
              or len(ids) != len(set(ids))
              or any(item not in catalog for item in ids)):
            errors.append(prefix + "evidence_ids")
    if seen_categories != _CATEGORIES:
        errors.append("claims_categories")
    return errors


def grade_assessment(packet, response):
    """Return deterministic contract errors; never grade atomicity or entailment."""
    if not isinstance(packet, dict):
        return ["packet_object"]
    if not isinstance(response, dict):
        return ["response_object"]
    errors = []
    try:
        canonical_catalog = build_catalog(packet)
        if not _type_exact(packet.get("evidence_catalog"), canonical_catalog):
            errors.append("evidence_catalog")
        if set(response) != _RESPONSE_KEYS:
            errors.append("response_keys")
        case_id = packet.get("case_id")
        if not _text(case_id) or type(response.get("case_id")) is not str or response.get("case_id") != case_id:
            errors.append("case_id")
        if response.get("decision") not in _DECISIONS:
            errors.append("decision")
        probability = response.get("probability_net_positive")
        if probability is not None and (
            not _finite_number(probability) or not 0 <= probability <= 1
        ):
            errors.append("probability_net_positive")
        errors.extend(_fact_errors(packet, response.get("facts")))
        errors.extend(_trade_plan_errors(packet, response.get("trade_plan")))
        errors.extend(_claim_errors(response.get("claims"), canonical_catalog))
        try:
            serialized = json.dumps(response, ensure_ascii=False, allow_nan=False)
            if len(serialized.split()) > 1400:
                errors.append("word_limit")
        except (TypeError, ValueError, OverflowError):
            errors.append("json_serialization")
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError):
        errors.append("invalid_assessment")
    return list(dict.fromkeys(errors))


def resolve_claims(packet, response):
    """Flatten valid claim/evidence links for human semantic review.

    The returned records expose the cited values.  They do not assert that a
    claim is atomic or that a value supports the claim; those remain review work.
    """
    errors = grade_assessment(packet, response)
    if errors:
        raise ValueError("invalid assessment: " + ", ".join(errors))
    records = []
    catalog = build_catalog(packet)
    try:
        for claim_index, claim in enumerate(response["claims"]):
            for evidence_id in claim["evidence_ids"]:
                path = deepcopy(catalog[evidence_id])
                records.append({
                    "claim_index": claim_index,
                    "category": claim["category"],
                    "claim": claim["claim"],
                    "status": claim["status"],
                    "evidence_id": evidence_id,
                    "path": path,
                    "resolved_value": resolve_evidence(packet, path),
                })
    except (KeyError, TypeError, ValueError, AttributeError, IndexError) as exc:
        raise ValueError("invalid assessment: evidence resolution failed") from exc
    return records
