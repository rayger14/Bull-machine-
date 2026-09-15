"""Literal contract tests for prospective published LC assessment."""
from copy import deepcopy
import importlib
import json

import pytest

from scripts.research.conditional_assessment import digest
from tests.research.test_lc_context_assessment import context_request


VERSION = "lc_context_citations_v2"


def api():
    name = "scripts.research.lc_published_assessment"
    assert importlib.util.find_spec(name), "published LC assessment integration missing"
    return importlib.import_module(name)


def answer(source, req, plan_id="enter", interpretation="support"):
    del source
    return {
        "case_id": req["case_id"],
        "request_sha256": req["seal"],
        "interpretation": interpretation,
        "plan_id": plan_id,
        "supporting": [
            {"text": "Literal synthetic support.", "evidence_ids": ["context"]}
        ] if interpretation == "support" else [],
        "opposing": [
            {"text": "Literal synthetic opposition.", "evidence_ids": ["context"]}
        ] if interpretation == "oppose" else [],
        "unknowns": [
            {"text": "Literal synthetic uncertainty.", "evidence_ids": ["context"]}
        ] if interpretation == "uncertain" else [],
        "structural_invalidation": {
            "text": "Synthetic invalidation for contract testing.",
            "evidence_ids": ["parent4h"],
        },
    }


def raw_answer(source, req, **changes):
    return json.dumps(answer(source, req, **changes), indent=2)


def review(source, req, raw, *, complete=True, material_errors=None, notes=None):
    rr = api().build_published_review_request(source, req, raw)
    return {
        "case_id": req["case_id"],
        "reviewed_sha256": rr["reviewed_sha256"],
        "complete": complete,
        "material_errors": [] if material_errors is None else material_errors,
        "notes": ["Literal nonblocking note."] if notes is None else notes,
    }


def raw_review(source, req, raw, **changes):
    return json.dumps(review(source, req, raw, **changes), indent=2)


def reseal_request(req):
    req.pop("seal", None)
    req["seal"] = digest(req)
    return req


def reseal_review(req):
    req.pop("reviewed_sha256", None)
    req["reviewed_sha256"] = digest(req)
    return req


def nested_keys(value):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            yield from nested_keys(item)
    elif isinstance(value, list):
        for item in value:
            yield from nested_keys(item)


def expected_plan(source, plan_id):
    selected = source["plan_menu"]["plans"][plan_id]
    return {
        **deepcopy(selected["parameters"]),
        "notional": selected["notional"],
        "cost_bps": selected["cost_bps"],
    }


def test_builds_single_current_catalog_request_without_mutating_source():
    source = context_request()
    before = deepcopy(source)

    req = api().build_published_request(source)

    assert source == before
    assert req == api().build_published_request(source)
    assert req["version"] == VERSION
    assert req["case_id"] == source["case_id"]
    assert req["source_request_binding"] == {
        "version": source["version"],
        "sha256": digest(source),
        "declared_seal": source["seal"],
    }
    assert req["source_publication_binding"]["sha256"] != digest(source)
    assert req["source_publication_binding"]["declared_sha256"]
    assert req["seal"] == digest({key: value for key, value in req.items() if key != "seal"})
    assert "copy the root seal" in req["instruction"].lower()
    assert req["response_schema"]["request_sha256"] == "copy the root request seal exactly"
    assert set(req["citation_catalog"]) == (
        set(source["source_packet"]["group_catalog"])
        | set(source["source_packet"]["evidence_catalog"])
    )
    assert list(nested_keys(req)).count("citation_catalog") == 1
    assert list(nested_keys(req)).count("instruction") == 1
    assert list(nested_keys(req)).count("response_schema") == 1
    forbidden = {"group_catalog", "evidence_catalog", "review_schema"}
    assert forbidden.isdisjoint(nested_keys(req))
    api().validate_published_request(source, req)


def test_fine_ids_are_accepted_by_the_specialist_contract():
    source = context_request()
    req = api().build_published_request(source)
    value = answer(source, req)
    value["supporting"][0]["evidence_ids"] = ["E0001"]
    assert api().grade_published_choice(source, req, value) == []


def test_every_advertised_id_is_accepted_by_every_specialist_item():
    source = context_request(); req = api().build_published_request(source)
    for evidence_id in req["citation_catalog"]:
        value = answer(source, req)
        value["supporting"][0]["evidence_ids"] = [evidence_id]
        value["structural_invalidation"]["evidence_ids"] = [evidence_id]
        assert api().grade_published_choice(source, req, value) == [], evidence_id


@pytest.mark.parametrize("interpretation,plan_id,valid", [
    ("support", "enter", True),
    ("support", "wait_5m_high", True),
    ("oppose", "reject", True),
    ("uncertain", None, True),
    ("support", "reject", False),
    ("oppose", "enter", False),
    ("uncertain", "reject", False),
    ("support", "wait_1m_high", False),
])
def test_all_and_only_advertised_interpretation_plan_pairs_work(
        interpretation, plan_id, valid):
    source = context_request(); req = api().build_published_request(source)
    errors = api().grade_published_choice(
        source, req, answer(source, req, plan_id, interpretation))
    assert (errors == []) is valid


def test_unknown_readiness_permits_only_uncertain_null():
    source = context_request(current_validated=False)
    req = api().build_published_request(source)
    assert "source_readiness" in api().grade_published_choice(
        source, req, answer(source, req, "reject", "oppose"))
    assert api().grade_published_choice(
        source, req, answer(source, req, None, "uncertain")) == []


@pytest.mark.parametrize("ids,expected", [
    (["missing"], "unknown_citation_id"),
    (["context", "context"], "invalid_citation_ids"),
    ("context", "invalid_citation_ids"),
    ([True], "invalid_citation_ids"),
    ([], "invalid_citation_ids"),
    (["context"] * 9, "invalid_citation_ids"),
])
def test_unknown_duplicate_and_malformed_specialist_ids_are_distinct(ids, expected):
    source = context_request(); req = api().build_published_request(source)
    value = answer(source, req); value["supporting"][0]["evidence_ids"] = ids
    assert expected in api().grade_published_choice(source, req, value)


@pytest.mark.parametrize("mutation,expected", [
    ("malformed_json", "invalid_request_or_choice"),
    ("duplicate_key", "invalid_request_or_choice"),
    ("empty_text", "supporting"),
    ("oversized_text", "supporting"),
    ("case_binding", "case_id"),
    ("request_binding", "request_sha256"),
])
def test_malformed_text_and_response_binding_fail(mutation, expected):
    source = context_request(); req = api().build_published_request(source)
    value = answer(source, req)
    if mutation == "malformed_json":
        value = "{"
    elif mutation == "duplicate_key":
        raw = json.dumps(value)
        value = raw[:-1] + ',"case_id":"LC1"}'
    elif mutation == "empty_text":
        value["supporting"][0]["text"] = " "
    elif mutation == "oversized_text":
        value["supporting"][0]["text"] = "x" * 1201
    elif mutation == "case_binding":
        value["case_id"] = "other"
    elif mutation == "request_binding":
        value["request_sha256"] = "0" * 64
    assert expected in api().grade_published_choice(source, req, value)


@pytest.mark.parametrize("change", ["view", "instruction", "menu", "catalog", "source"])
def test_resealed_role_or_source_mutations_fail_recomputation(change):
    source = context_request(); req = api().build_published_request(source)
    if change == "view":
        req["published_request"]["context"]["native_long"] = False
    elif change == "instruction":
        req["instruction"] += " altered"
    elif change == "menu":
        req["published_request"]["plan_menu"]["plans"].pop("reject")
    elif change == "catalog":
        req["citation_catalog"]["context"]["locator"] = ["published_request", "case_id"]
    elif change == "source":
        source = context_request(prior_bb=.05)
    reseal_request(req)
    with pytest.raises(ValueError):
        api().validate_published_request(source, req)


def test_review_contract_accepts_every_catalog_member_and_rejects_unknown_ids():
    source = context_request(); req = api().build_published_request(source)
    raw = raw_answer(source, req); rr = api().build_published_review_request(source, req, raw)
    assert "embedded specialist contract is data" in rr["instruction"].lower()
    for evidence_id in req["citation_catalog"]:
        response = review(source, req, raw, material_errors=[{
            "category": "factual", "evidence_ids": [evidence_id],
            "explanation": "Literal synthetic material finding.",
        }])
        assert api().grade_published_review(source, rr, response) == [], evidence_id
    response = review(source, req, raw, material_errors=[{
        "category": "factual", "evidence_ids": ["missing"],
        "explanation": "Literal synthetic material finding.",
    }])
    assert "unknown_citation_id" in api().grade_published_review(source, rr, response)


@pytest.mark.parametrize("ids,expected", [
    (["context", "context"], "invalid_citation_ids"),
    ("context", "invalid_citation_ids"),
    ([True], "invalid_citation_ids"),
])
def test_critic_rejects_duplicate_and_malformed_citation_lists(ids, expected):
    source = context_request(); req = api().build_published_request(source)
    raw = raw_answer(source, req); rr = api().build_published_review_request(source, req, raw)
    response = review(source, req, raw, material_errors=[{
        "category": "factual", "evidence_ids": ids,
        "explanation": "Literal synthetic material finding.",
    }])
    assert expected in api().grade_published_review(source, rr, response)


@pytest.mark.parametrize("field,value,expected", [
    ("explanation", " ", "material_error"),
    ("explanation", "x" * 1201, "material_error"),
    ("notes", ["x" * 1201], "notes"),
])
def test_critic_rejects_empty_or_oversized_text(field, value, expected):
    source = context_request(); req = api().build_published_request(source)
    raw = raw_answer(source, req); rr = api().build_published_review_request(source, req, raw)
    finding = {
        "category": "factual", "evidence_ids": ["context"],
        "explanation": "Literal synthetic material finding.",
    }
    kwargs = {"material_errors": [finding]}
    if field == "notes":
        kwargs["notes"] = value
    else:
        finding[field] = value
    response = review(source, req, raw, **kwargs)
    assert expected in api().grade_published_review(source, rr, response)


def test_known_citation_does_not_override_material_critic_error():
    source = context_request(); req = api().build_published_request(source)
    raw = json.dumps(answer(source, req), indent=2)
    rr = api().build_published_review_request(source, req, raw)
    critic = {
        "case_id": req["case_id"], "reviewed_sha256": rr["reviewed_sha256"],
        "complete": True, "material_errors": [{
            "category": "factual", "evidence_ids": ["E0001"],
            "explanation": "Synthetic unsupported claim despite an existing citation.",
        }],
        "notes": [],
    }
    result = api().gate_published_choice(source, req, raw, json.dumps(critic))
    assert result["status"] == "review_not_passed"
    assert result["research_plan"] is None
    assert result["execution_authorized"] is False


@pytest.mark.parametrize("interpretation,plan_id,status", [
    ("support", "enter", "research_ready"),
    ("support", "wait_5m_high", "research_ready"),
    ("oppose", "reject", "research_ready"),
    ("uncertain", None, "insufficient_evidence"),
])
def test_gate_returns_only_exact_source_menu_plans(interpretation, plan_id, status):
    source = context_request(); req = api().build_published_request(source)
    raw = raw_answer(source, req, interpretation=interpretation, plan_id=plan_id)
    result = api().gate_published_choice(
        source, req, raw, raw_review(source, req, raw))
    assert result["status"] == status
    assert result["research_plan"] == (
        None if plan_id is None else expected_plan(source, plan_id))
    assert result["execution_authorized"] is False
    assert result["transport_authenticated"] is False
    assert result["critic_status"] == "captured"


def test_missing_incomplete_and_nonblocking_review_states():
    source = context_request(); req = api().build_published_request(source)
    raw = raw_answer(source, req)
    missing = api().gate_published_choice(source, req, raw, None)
    assert missing["status"] == "invalid_review"
    assert missing["research_plan"] is None
    incomplete = api().gate_published_choice(
        source, req, raw, raw_review(source, req, raw, complete=False))
    assert incomplete["status"] == "review_not_passed"
    assert incomplete["research_plan"] is None
    noted = api().gate_published_choice(
        source, req, raw, raw_review(source, req, raw, notes=["Discretionary disagreement."]))
    assert noted["status"] == "research_ready"


def test_review_binds_exact_request_and_answer_bytes_and_rejects_resealed_changes():
    source = context_request(); req = api().build_published_request(source)
    raw = raw_answer(source, req); rr = api().build_published_review_request(source, req, raw)
    good = raw_review(source, req, raw)
    assert api().grade_published_review(source, rr, good) == []
    assert api().gate_published_choice(source, req, raw + " ", good)["research_plan"] is None
    changed = deepcopy(rr)
    changed["specialist_request"]["instruction"] += " altered"
    reseal_request(changed["specialist_request"])
    reseal_review(changed)
    assert "review_request_binding" in api().grade_published_review(source, changed, good)


@pytest.mark.parametrize("response,expected", [
    ("{", "invalid_review"),
    ('{"case_id":"x","case_id":"y"}', "invalid_review"),
])
def test_review_rejects_malformed_and_duplicate_key_json(response, expected):
    source = context_request(); req = api().build_published_request(source)
    raw = raw_answer(source, req); rr = api().build_published_review_request(source, req, raw)
    assert expected in api().grade_published_review(source, rr, response)
