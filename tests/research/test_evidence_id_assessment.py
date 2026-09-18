"""Contract tests for bounded evidence-ID assessment records."""
from copy import deepcopy
import importlib
from pathlib import Path

import pytest


def module():
    assert Path("scripts/research/evidence_id_assessment.py").exists(), "assessment contract missing"
    return importlib.import_module("scripts.research.evidence_id_assessment")


def catalog_packet():
    return {
        "case_id": "ignored-case-id",
        "track": "minute",
        "decision_time": "2026-02-04T00:00:00+00:00",
        "candle_columns": ["open_time", "open", "high", "low", "close", "volume"],
        "evidence": {
            "1m": [["2026-02-03T23:59:00Z", 0, False],
                   ["2026-02-04T00:00:00Z", None, 4]],
            "5m": [],
            "features": {
                "bad_list": [1, None],
                "empty": [],
                "nested": {"omit": None, "x": 2},
                "off": False,
                "ordinary_list": [0, False, []],
                "zero": 0,
            },
            "null_branch": None,
        },
        "indicative_economics": {"actual_fill_known": False, "basis": 100.0},
        "input_status": None,
        "instruction": "Use only this packet.",
        "limitations": [],
        "plan": {"direction": "long"},
        "rulecard": {"z": "last", "a": "first"},
        "specialist_instructions": "Bounded specialist.",
        "response_contract": {"must_not": "be catalogued"},
        "evidence_catalog": {"E9999": ["must_not", "be_reused"]},
    }


def test_build_catalog_has_literal_stable_paths_without_null_targets():
    source = catalog_packet()
    before = deepcopy(source)

    result = module().build_catalog(source)

    assert result == {
        "E0001": ["candle_columns"],
        "E0002": ["evidence", "1m", 0],
        "E0003": ["evidence", "features", "bad_list", 0],
        "E0004": ["evidence", "features", "empty"],
        "E0005": ["evidence", "features", "nested", "x"],
        "E0006": ["evidence", "features", "off"],
        "E0007": ["evidence", "features", "ordinary_list"],
        "E0008": ["evidence", "features", "zero"],
        "E0009": ["indicative_economics", "actual_fill_known"],
        "E0010": ["indicative_economics", "basis"],
        "E0011": ["instruction"],
        "E0012": ["limitations"],
        "E0013": ["plan", "direction"],
        "E0014": ["rulecard", "a"],
        "E0015": ["rulecard", "z"],
        "E0016": ["specialist_instructions"],
    }
    assert module().build_catalog(deepcopy(source)) == result
    assert source == before


def test_build_catalog_uses_one_id_per_nonnull_candle_row_not_series_or_scalar():
    source = {"evidence": {
        "1m": [["t1", 1, 2], ["t2", 3, 4]],
        "5m": [["t5", 5]], "15m": [["t15", 15]],
        "1h": [["th", 60]], "4h": [["t4h", 240]], "1d": [["td", 1440]],
    }}
    assert module().build_catalog(source) == {
        "E0001": ["evidence", "15m", 0],
        "E0002": ["evidence", "1d", 0],
        "E0003": ["evidence", "1h", 0],
        "E0004": ["evidence", "1m", 0],
        "E0005": ["evidence", "1m", 1],
        "E0006": ["evidence", "4h", 0],
        "E0007": ["evidence", "5m", 0],
    }


def test_build_catalog_keeps_column_mapping_and_nonnull_parent_update_fields():
    source = {
        "candle_columns": ["open_time", "open", "high", "low", "close", "volume"],
        "evidence": {
            "parent_4h": {
                "updates": [{
                    "available_at": "2026-02-03T20:00:00+00:00",
                    "source_break_direction": None,
                    "post_state": "active",
                    "lineage": {"parent_id": "P4", "revision": 0},
                }],
            },
        },
    }
    assert module().build_catalog(source) == {
        "E0001": ["candle_columns"],
        "E0002": ["evidence", "parent_4h", "updates", 0, "available_at"],
        "E0003": ["evidence", "parent_4h", "updates", 0, "lineage", "parent_id"],
        "E0004": ["evidence", "parent_4h", "updates", 0, "lineage", "revision"],
        "E0005": ["evidence", "parent_4h", "updates", 0, "post_state"],
    }


@pytest.mark.parametrize("bad", [None, [], "packet"])
def test_build_catalog_rejects_nonobject_packets(bad):
    with pytest.raises(ValueError, match="packet must be an object"):
        module().build_catalog(bad)


def packet():
    value = {
        "case_id": "P01",
        "track": "minute",
        "decision_time": "2026-02-04T00:00:00+00:00",
        "evidence": {
            "1m": [["2026-02-03T23:59:00+00:00", 99.0, 101.0, 98.0, 100.125, 10.0]],
            "features": {"enabled": False, "zero": 0},
            "parent_1d": {"status": "verified_absent"},
            "parent_4h": {"status": "verified_present"},
            "setup": {"child_level": 98.25},
        },
        "indicative_economics": {"actual_fill_known": False, "basis": 100.125},
        "input_status": {"feed": "present"},
        "instruction": "Assess only this packet.",
        "limitations": ["Historical reconstruction."],
        "plan": {
            "direction": "long",
            "entry": "same-source minute OPEN at decision; actual fill withheld",
            "horizon_minutes": 240,
            "indicative_close": 100.125,
            "notional": 50000.0,
            "roundtrip_cost": 60.0,
            "stop": 98.0,
            "target": "actual entry+2*(actual entry-stop)",
        },
        "rulecard": {"identity": "detector source"},
        "specialist_instructions": "Bounded specialist instructions.",
    }
    value["evidence_catalog"] = {
        "E0001": ["evidence", "1m", 0],
        "E0002": ["evidence", "features", "enabled"],
        "E0003": ["evidence", "features", "zero"],
        "E0004": ["evidence", "parent_1d", "status"],
        "E0005": ["evidence", "parent_4h", "status"],
        "E0006": ["evidence", "setup", "child_level"],
        "E0007": ["indicative_economics", "actual_fill_known"],
        "E0008": ["indicative_economics", "basis"],
        "E0009": ["input_status", "feed"],
        "E0010": ["instruction"],
        "E0011": ["limitations"],
        "E0012": ["plan", "direction"],
        "E0013": ["plan", "entry"],
        "E0014": ["plan", "horizon_minutes"],
        "E0015": ["plan", "indicative_close"],
        "E0016": ["plan", "notional"],
        "E0017": ["plan", "roundtrip_cost"],
        "E0018": ["plan", "stop"],
        "E0019": ["plan", "target"],
        "E0020": ["rulecard", "identity"],
        "E0021": ["specialist_instructions"],
    }
    return value


def expected_plan():
    return {
        "direction": "long",
        "entry": "same-source minute OPEN at decision; actual fill withheld",
        "stop": 98.0,
        "target_formula": "actual entry+2*(actual entry-stop)",
        "horizon_minutes": 240,
        "notional": 50000.0,
        "roundtrip_cost": 60.0,
        "actual_fill_known": False,
        "deadline_exit": "deadline minute OPEN if no earlier stop/target",
    }


def response():
    claims = []
    for category in ("detector", "structure", "sequence", "economics", "reason", "missing"):
        claims.append({
            "category": category,
            "claim": f"Atomic {category} assessment from the cited packet value.",
            "status": "unresolved" if category == "missing" else "supported",
            "evidence_ids": ["E0001"],
        })
    return {
        "case_id": "P01",
        "decision": "accept",
        "probability_net_positive": 0.51,
        "facts": {
            "decision_time": "2026-02-04T00:00:00+00:00",
            "last_1m_close": 100.1250005,
            "parent_4h_status": "verified_present",
            "parent_1d_status": "verified_absent",
            "minute_child_level": 98.2500005,
        },
        "trade_plan": expected_plan(),
        "claims": claims,
    }


def test_expected_trade_plan_is_a_literal_detached_fixed_contract():
    source = packet()
    before = deepcopy(source)
    actual = module().expected_trade_plan(source)
    assert actual == expected_plan()
    actual["direction"] = "changed"
    assert source == before


def test_grade_assessment_accepts_exact_bounded_schema_and_preserves_inputs():
    source, answer = packet(), response()
    before = deepcopy((source, answer))
    assert module().grade_assessment(source, answer) == []
    assert (source, answer) == before


@pytest.mark.parametrize("field,bad", [
    ("stop", 97.0),
    ("target_formula", "entry plus one R"),
    ("horizon_minutes", 241),
    ("roundtrip_cost", 59.0),
    ("actual_fill_known", True),
])
def test_grade_assessment_rejects_changed_stop_target_expiry_cost_or_fill_status(field, bad):
    answer = response()
    answer["trade_plan"][field] = bad
    assert f"trade_plan:{field}" in module().grade_assessment(packet(), answer)


@pytest.mark.parametrize("field,bad", [
    ("stop", 98.0000005),
    ("notional", 50000.0000005),
    ("roundtrip_cost", 60.0000005),
])
def test_grade_assessment_requires_exact_numeric_trade_plan_echo_within_fact_tolerance(field, bad):
    answer = response()
    answer["trade_plan"][field] = bad
    assert f"trade_plan:{field}" in module().grade_assessment(packet(), answer)


@pytest.mark.parametrize("field", list(expected_plan()))
def test_grade_assessment_rejects_every_missing_trade_plan_field(field):
    answer = response()
    answer["trade_plan"].pop(field)
    assert "trade_plan_keys" in module().grade_assessment(packet(), answer)


@pytest.mark.parametrize("location,field", [
    ("response", "extra"), ("facts", "extra"), ("trade_plan", "extra"), ("claim", "extra"),
])
def test_grade_assessment_rejects_schema_extras(location, field):
    answer = response()
    target = answer if location == "response" else (
        answer["claims"][0] if location == "claim" else answer[location]
    )
    target[field] = "not allowed"
    errors = module().grade_assessment(packet(), answer)
    expected = "response_keys" if location == "response" else (
        "claim:0:keys" if location == "claim" else f"{location}_keys"
    )
    assert expected in errors


@pytest.mark.parametrize("field,bad", [
    ("decision_time", "2026-02-04T00:00:01+00:00"),
    ("last_1m_close", 100.2),
    ("parent_4h_status", "verified_absent"),
    ("parent_1d_status", "unknown"),
    ("minute_child_level", None),
])
def test_grade_assessment_rejects_wrong_source_facts(field, bad):
    answer = response()
    answer["facts"][field] = bad
    assert f"fact:{field}" in module().grade_assessment(packet(), answer)


@pytest.mark.parametrize("field", ["last_1m_close", "minute_child_level"])
def test_grade_assessment_rejects_boolean_numeric_facts(field):
    answer = response()
    answer["facts"][field] = True
    assert f"fact:{field}" in module().grade_assessment(packet(), answer)


@pytest.mark.parametrize("value", [True, float("nan"), float("inf"), -0.01, 1.01, "0.5"])
def test_grade_assessment_rejects_invalid_probability(value):
    answer = response()
    answer["probability_net_positive"] = value
    assert "probability_net_positive" in module().grade_assessment(packet(), answer)


def test_grade_assessment_allows_null_probability():
    answer = response()
    answer["probability_net_positive"] = None
    assert module().grade_assessment(packet(), answer) == []


@pytest.mark.parametrize("mutation", ["missing", "path", "id", "boolean_index"])
def test_grade_assessment_recomputes_and_type_checks_compiler_catalog(mutation):
    source = packet()
    if mutation == "missing":
        source.pop("evidence_catalog")
    elif mutation == "path":
        source["evidence_catalog"]["E0002"] = ["evidence", "features", "zero"]
    elif mutation == "id":
        source["evidence_catalog"]["E9999"] = source["evidence_catalog"].pop("E0001")
    else:
        source["evidence_catalog"]["E0001"][-1] = False
    assert "evidence_catalog" in module().grade_assessment(source, response())


@pytest.mark.parametrize("mutation", [
    "too_few", "too_many", "missing_category", "bad_category", "bad_status",
    "blank_claim", "long_claim", "empty_ids", "duplicate_id", "unknown_id", "id_not_text",
])
def test_grade_assessment_rejects_claim_list_category_text_status_and_id_errors(mutation):
    answer = response()
    if mutation == "too_few":
        answer["claims"].pop()
    elif mutation == "too_many":
        answer["claims"] = answer["claims"] * 3
    elif mutation == "missing_category":
        answer["claims"][5]["category"] = "reason"
    elif mutation == "bad_category":
        answer["claims"][0]["category"] = "prediction"
    elif mutation == "bad_status":
        answer["claims"][0]["status"] = "maybe"
    elif mutation == "blank_claim":
        answer["claims"][0]["claim"] = " "
    elif mutation == "long_claim":
        answer["claims"][0]["claim"] = "x" * 401
    elif mutation == "empty_ids":
        answer["claims"][0]["evidence_ids"] = []
    elif mutation == "duplicate_id":
        answer["claims"][0]["evidence_ids"] = ["E0001", "E0001"]
    elif mutation == "unknown_id":
        answer["claims"][0]["evidence_ids"] = ["E9999"]
    else:
        answer["claims"][0]["evidence_ids"] = [1]
    errors = module().grade_assessment(packet(), answer)
    assert errors and any(error.startswith("claims") or error.startswith("claim:") for error in errors)


def test_grade_assessment_enforces_json_total_word_limit():
    answer = response()
    answer["claims"][0]["claim"] = "word " * 1390
    errors = module().grade_assessment(packet(), answer)
    assert "claim:0:claim" in errors and "word_limit" in errors


@pytest.mark.parametrize("bad", [None, [], {}, {"case_id": "P01"}])
def test_grade_assessment_fails_closed_with_controlled_errors_for_malformed_response(bad):
    errors = module().grade_assessment(packet(), bad)
    assert isinstance(errors, list) and errors


def test_resolve_claims_returns_flat_detached_human_inspectable_records():
    source, answer = packet(), response()
    answer["claims"][0]["evidence_ids"] = ["E0001", "E0002"]
    records = module().resolve_claims(source, answer)
    assert records[0] == {
        "claim_index": 0,
        "category": "detector",
        "claim": "Atomic detector assessment from the cited packet value.",
        "status": "supported",
        "evidence_id": "E0001",
        "path": ["evidence", "1m", 0],
        "resolved_value": ["2026-02-03T23:59:00+00:00", 99.0, 101.0, 98.0, 100.125, 10.0],
    }
    assert records[1]["evidence_id"] == "E0002"
    assert records[1]["resolved_value"] is False
    assert len(records) == 7
    records[0]["resolved_value"][1] = -1
    assert source["evidence"]["1m"][0][1] == 99.0


@pytest.mark.parametrize("bad", [None, {}, {"case_id": "P01"}])
def test_resolve_claims_explicitly_rejects_invalid_assessments(bad):
    with pytest.raises(ValueError, match="invalid assessment"):
        module().resolve_claims(packet(), bad)
