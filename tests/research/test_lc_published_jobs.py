"""Immutable-job tests for the published LC assessment namespace."""
from copy import deepcopy
import hashlib
import importlib
import json

import pytest

from scripts.research.assessment_evidence_guard import _canonical
from scripts.research.conditional_assessment import digest
from tests.research.test_lc_context_assessment import context_request
from tests.research.test_lc_published_assessment import (
    answer,
    api as assessment,
    raw_answer,
    raw_review,
)


def jobs():
    name = "scripts.research.lc_published_jobs"
    assert importlib.util.find_spec(name), "published LC research jobs missing"
    return importlib.import_module(name)


def job(path):
    return jobs().PublishedContextResearchJob(path)


def valid_transport(role, valid=True):
    return {
        "role": role,
        "agent_id": role + "-fixture",
        "requested_model": "fixture-model",
        "actual_model": "fixture-model",
        "capture_sha256": "a" * 64,
        "transport_valid": valid,
    }


def rehash_artifact(path):
    value = json.loads(path.read_text())
    value["sha256"] = digest({key: item for key, item in value.items() if key != "sha256"})
    path.write_text(_canonical(value))


def complete_captures(path, *, specialist_transport=True, reviewer_transport=True,
                      review_changes=None):
    source = context_request(); runner = job(path); runner.prepare(source)
    req = runner.role_request("specialist")
    raw = raw_answer(source, req)
    runner.capture("specialist", raw, valid_transport("specialist", specialist_transport))
    reviewer_request = runner.role_request("reviewer")
    critique = raw_review(source, req, raw, **(review_changes or {}))
    runner.capture("reviewer", critique, valid_transport("reviewer", reviewer_transport))
    return source, req, raw, reviewer_request, critique


def test_complete_job_reopens_recomputes_and_orders_reveal_outcome(tmp_path):
    path = tmp_path / "case"
    source = context_request(); runner = job(path)
    runner.prepare(source)
    specialist = runner.role_request("specialist")
    specialist["case_id"] = "detached-mutation"
    req = runner.role_request("specialist")
    assert req["case_id"] == source["case_id"]
    assert "source_request" not in req
    raw = raw_answer(source, req)
    runner.capture("specialist", raw, valid_transport("specialist"))
    reviewer = runner.role_request("reviewer")
    assert reviewer == assessment().build_published_review_request(source, req, raw)
    reviewer["case_id"] = "detached-mutation"
    assert runner.role_request("reviewer")["case_id"] == source["case_id"]
    critique = raw_review(source, req, raw)
    runner.capture("reviewer", critique, valid_transport("reviewer"))
    grade = job(path).lock_grade()
    assert grade["status"] == "research_ready"
    assert grade["critic_status"] == "captured"
    assert grade["execution_authorized"] is False
    job(path).authorize_reveal()
    outcome = {"status": "literal_fixture_closed", "net_pnl": 0.0}
    job(path).save_outcome(outcome)
    assert job(path).save_outcome(outcome) == outcome
    assert job(path).state == "outcome"


def test_equal_retries_are_idempotent_and_unequal_completed_stages_fail(tmp_path):
    path = tmp_path / "case"
    source, req, raw, _, critique = complete_captures(path)
    runner = job(path)
    runner.prepare(source)
    runner.capture("specialist", raw, valid_transport("specialist"))
    runner.capture("reviewer", critique, valid_transport("reviewer"))
    grade = runner.lock_grade(); runner.lock_grade(deepcopy(grade))
    reveal = runner.authorize_reveal(); runner.authorize_reveal()
    outcome = {"status": "literal"}; runner.save_outcome(outcome)
    with pytest.raises(ValueError):
        runner.capture("specialist", raw + " ", valid_transport("specialist"))
    with pytest.raises(ValueError):
        runner.capture("reviewer", critique + " ", valid_transport("reviewer"))
    changed_source = context_request(prior_bb=.05)
    with pytest.raises(ValueError):
        runner.prepare(changed_source)
    forged = deepcopy(grade); forged["status"] = "forged"
    with pytest.raises(ValueError):
        runner.lock_grade(forged)
    with pytest.raises(ValueError):
        runner.save_outcome({"status": "different"})
    assert reveal["authorized"] is True


def test_valid_specialist_cannot_skip_critic(tmp_path):
    source = context_request(); runner = job(tmp_path)
    runner.prepare(source); req = runner.role_request("specialist")
    raw = json.dumps(answer(source, req))
    runner.capture("specialist", raw, valid_transport("specialist"))
    with pytest.raises(ValueError):
        runner.skip_review()
    with pytest.raises(ValueError):
        runner.lock_grade()


def test_invalid_specialist_uses_truthful_skip_event_and_null_grade(tmp_path):
    path = tmp_path / "invalid"; source = context_request(); runner = job(path)
    runner.prepare(source); req = runner.role_request("specialist")
    value = answer(source, req); value["supporting"][0]["evidence_ids"] = ["missing"]
    raw = json.dumps(value)
    runner.capture("specialist", raw, valid_transport("specialist"))
    with pytest.raises(ValueError):
        runner.role_request("reviewer")
    event = runner.skip_review()
    empty_hash = hashlib.sha256(b"").hexdigest()
    assert event == {
        "kind": "review_not_run",
        "reason": "invalid_assessment",
        "raw_response": "",
        "raw_sha256": empty_hash,
        "provenance": {
            "role": "reviewer",
            "agent_id": "controller-not-invoked",
            "requested_model": "not-invoked",
            "actual_model": None,
            "capture_sha256": None,
            "transport_valid": False,
        },
    }
    assert job(path).skip_review() == event
    grade = job(path).lock_grade()
    assert grade["status"] == "invalid_assessment"
    assert grade["critic_status"] == "not_invoked"
    assert grade["research_plan"] is None
    assert "unknown_citation_id" in grade["errors"]


def test_failed_specialist_transport_uses_explicit_skip_and_transport_grade(tmp_path):
    path = tmp_path / "transport"; source = context_request(); runner = job(path)
    runner.prepare(source); req = runner.role_request("specialist")
    raw = raw_answer(source, req)
    runner.capture("specialist", raw, valid_transport("specialist", False))
    event = runner.skip_review()
    assert event["reason"] == "invalid_transport"
    assert event["provenance"]["actual_model"] is None
    assert event["provenance"]["transport_valid"] is False
    grade = job(path).lock_grade()
    assert grade["status"] == "invalid_transport"
    assert grade["critic_status"] == "not_invoked"
    assert grade["research_plan"] is None
    assert "unverified_declared_transport" in grade["errors"]


def test_actual_critic_capture_reports_capture_not_semantic_approval(tmp_path):
    path = tmp_path / "critic"
    complete_captures(path, review_changes={"complete": False})
    grade = job(path).lock_grade()
    assert grade["status"] == "review_not_passed"
    assert grade["critic_status"] == "captured"
    assert grade["research_plan"] is None


def test_actual_reviewer_transport_failure_nulls_plan(tmp_path):
    path = tmp_path / "reviewer"
    complete_captures(
        path,
        reviewer_transport=False,
    )
    grade = job(path).lock_grade()
    assert grade["status"] == "invalid_transport"
    assert grade["critic_status"] == "captured"
    assert grade["research_plan"] is None


def test_declared_valid_transport_allows_unknown_actual_model_without_authentication(tmp_path):
    path = tmp_path / "declared"; source = context_request(); runner = job(path)
    runner.prepare(source); req = runner.role_request("specialist")
    raw = raw_answer(source, req)
    specialist_meta = valid_transport("specialist")
    specialist_meta.update(actual_model=None, capture_sha256=None)
    runner.capture("specialist", raw, specialist_meta)
    critique = raw_review(source, req, raw)
    reviewer_meta = valid_transport("reviewer")
    reviewer_meta.update(actual_model=None, capture_sha256=None)
    runner.capture("reviewer", critique, reviewer_meta)
    grade = runner.lock_grade()
    assert grade["status"] == "research_ready"
    assert grade["transport_authenticated"] is False


def test_reopen_rejects_forged_skip_reason_even_when_rehashed(tmp_path):
    path = tmp_path / "skip"; source = context_request(); runner = job(path)
    runner.prepare(source); req = runner.role_request("specialist")
    value = answer(source, req); value["case_id"] = "other"
    runner.capture("specialist", json.dumps(value), valid_transport("specialist"))
    runner.skip_review()
    artifact_path = path / "reviewer.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["payload"]["reason"] = "invalid_transport"
    artifact_path.write_text(_canonical(artifact))
    rehash_artifact(artifact_path)
    with pytest.raises(ValueError, match="skip"):
        job(path)


@pytest.mark.parametrize("stage", ["empty", "request", "specialist", "reviewer"])
def test_reveal_and_outcome_are_refused_before_grade(tmp_path, stage):
    path = tmp_path / stage; runner = job(path); source = context_request()
    if stage != "empty":
        runner.prepare(source); req = runner.role_request("specialist")
    if stage in ("specialist", "reviewer"):
        raw = raw_answer(source, req)
        runner.capture("specialist", raw, valid_transport("specialist"))
    if stage == "reviewer":
        runner.capture("reviewer", raw_review(source, req, raw), valid_transport("reviewer"))
    with pytest.raises(ValueError):
        runner.authorize_reveal()
    with pytest.raises(ValueError):
        runner.save_outcome({"status": "too_early"})


def test_unknown_roles_and_out_of_order_reviewer_fail(tmp_path):
    runner = job(tmp_path); source = context_request()
    with pytest.raises(ValueError):
        runner.role_request("specialist")
    runner.prepare(source)
    with pytest.raises(ValueError):
        runner.role_request("unknown")
    with pytest.raises(ValueError):
        runner.role_request("reviewer")
    with pytest.raises(ValueError):
        runner.capture("reviewer", "{}", valid_transport("reviewer"))


def test_cross_namespace_directory_is_rejected(tmp_path):
    from scripts.research.lc_context_jobs import ContextResearchJob

    path = tmp_path / "context"
    ContextResearchJob(path).prepare(context_request())
    with pytest.raises(ValueError, match="published"):
        job(path)


def test_reopen_rejects_rehashed_bundle_grade_and_reveal_forgery(tmp_path):
    bundle_path = tmp_path / "bundle"; source = context_request(); runner = job(bundle_path)
    runner.prepare(source)
    artifact_path = bundle_path / "request.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["payload"]["role_request"]["case_id"] = "forged"
    artifact_path.write_text(_canonical(artifact)); rehash_artifact(artifact_path)
    with pytest.raises(ValueError, match="published"):
        job(bundle_path)

    grade_path = tmp_path / "grade"
    complete_captures(grade_path)
    job(grade_path).lock_grade()
    artifact_path = grade_path / "grade.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["payload"]["status"] = "forged"
    artifact_path.write_text(_canonical(artifact)); rehash_artifact(artifact_path)
    with pytest.raises(ValueError, match="grade"):
        job(grade_path)

    reveal_path = tmp_path / "reveal"
    complete_captures(reveal_path)
    job(reveal_path).lock_grade(); job(reveal_path).authorize_reveal()
    artifact_path = reveal_path / "reveal.json"
    artifact = json.loads(artifact_path.read_text())
    artifact["payload"]["grade_sha256"] = "0" * 64
    artifact_path.write_text(_canonical(artifact)); rehash_artifact(artifact_path)
    with pytest.raises(ValueError, match="reveal"):
        job(reveal_path)


def test_changed_raw_answer_bytes_are_rejected_on_reopen(tmp_path):
    path = tmp_path / "changed"
    complete_captures(path)
    artifact_path = path / "specialist.json"
    artifact_path.write_bytes(artifact_path.read_bytes().replace(
        b"Literal synthetic support.", b"Literal synthetic altered."))
    with pytest.raises(ValueError):
        job(path)
