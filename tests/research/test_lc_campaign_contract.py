"""Focused contract tests for the offline consolidated-campaign ledger."""
from copy import deepcopy
import hashlib
import json

import pytest

from scripts.research.assessment_evidence_guard import _canonical


def manifest(case_ids, root="/synthetic"):
    return {
        "schema_version": "lc_campaign_contract_v1",
        "campaign_id": "fixture-campaign",
        "cases": [
            {
                "case_id": case_id,
                "job_directory": f"{root}/{case_id}",
                "source_request_sha256": "1" * 64,
                "role_request_sha256": "2" * 64,
            }
            for case_id in case_ids
        ],
    }


class SyntheticJob:
    def __init__(self, path, *, source="1" * 64, role="2" * 64,
                 captures=None, grade=None, reviewer_eligible=True):
        self.path = str(path)
        self.source = source
        self.role = role
        self.captures = captures or {}
        self.grade = grade
        self._reviewer_eligible = reviewer_eligible

    def campaign_binding(self):
        return {
            "source_request_sha256": self.source,
            "role_request_sha256": self.role,
        }

    def capture_binding(self, role):
        return self.captures[role]

    def grade_binding(self):
        return self.grade

    def reviewer_eligible(self):
        return self._reviewer_eligible


def ledger(tmp_path, now=[0], jobs=None):
    from scripts.research.lc_campaign_contract import CampaignLedger

    jobs = jobs or {}
    return CampaignLedger(
        tmp_path / "ledger",
        clock=lambda: now[0],
        job_loader=lambda path: jobs[str(path)],
    )


def delivery(case_id, role):
    return {
        "kind": "delivered",
        "job_directory": f"/synthetic/{case_id}",
        "raw_response_sha256": "3" * 64,
        "capture_sha256": "4" * 64,
    }


def rewrite_rehashed_state(path, mutate):
    value = json.loads(path.read_text())
    mutate(value)
    body = {key: item for key, item in value.items() if key != "state_sha256"}
    value["state_sha256"] = hashlib.sha256(_canonical(body).encode("ascii")).hexdigest()
    path.write_text(_canonical(value))


def test_exclusion_splits_run_instead_of_deleting_interior():
    from scripts.research.lc_campaign_contract import select_block

    assert select_block(["a", "b", "c", "d", "e"], {"c"}, 3) == ["a", "b"]
    assert select_block(["a", "b", "c", "d", "e"], {"a"}, 3) == ["b", "c", "d"]


def test_freeze_requires_exact_pinned_manifest_and_rejects_restart_drift(tmp_path):
    jobs = {"/synthetic/a": SyntheticJob("/synthetic/a")}
    first = ledger(tmp_path, jobs=jobs)
    frozen = manifest(["a"])
    assert first.freeze(frozen) == frozen
    assert ledger(tmp_path, jobs=jobs).freeze(deepcopy(frozen)) == frozen
    changed = deepcopy(frozen); changed["cases"][0]["source_request_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="immutable|binding"):
        ledger(tmp_path, jobs=jobs).freeze(changed)


def test_attempt_must_follow_freeze_and_duplicate_is_rejected_after_restart(tmp_path):
    jobs = {"/synthetic/a": SyntheticJob("/synthetic/a")}
    with pytest.raises(ValueError, match="freeze"):
        ledger(tmp_path, jobs=jobs).start_attempt("a", "specialist")
    ledger(tmp_path, jobs=jobs).freeze(manifest(["a"]))
    ledger(tmp_path, jobs=jobs).start_attempt("a", "specialist")
    with pytest.raises(ValueError, match="already invoked"):
        ledger(tmp_path, jobs=jobs).start_attempt("a", "specialist")


def test_budget_caps_are_thirty_per_role_and_active_slots_are_reserved(tmp_path):
    ids = [f"c{i}" for i in range(30)]
    jobs = {f"/synthetic/{case_id}": SyntheticJob(f"/synthetic/{case_id}") for case_id in ids}
    value = ledger(tmp_path, jobs=jobs); value.freeze(manifest(ids))
    for case_id in ids[:3]:
        value.start_attempt(case_id, "specialist")
    with pytest.raises(ValueError, match="active"):
        value.start_attempt(ids[3], "specialist")
    for case_id in ids[:3]:
        value.finish_attempt(case_id, "specialist", {"kind": "external_failure", "reason": "missing"})
    for case_id in ids[3:]:
        value.start_attempt(case_id, "specialist")
        value.finish_attempt(case_id, "specialist", {"kind": "external_failure", "reason": "missing"})
    assert value.state()["budgets"]["specialist"] == 30
    too_many = manifest(ids + ["overflow"])
    jobs["/synthetic/overflow"] = SyntheticJob("/synthetic/overflow")
    with pytest.raises(ValueError, match="30"):
        ledger(tmp_path / "overflow", jobs=jobs).freeze(too_many)


def test_invalid_or_missing_specialist_cannot_route_to_reviewer(tmp_path):
    jobs = {"/synthetic/a": SyntheticJob("/synthetic/a")}
    value = ledger(tmp_path, jobs=jobs); value.freeze(manifest(["a"]))
    value.start_attempt("a", "specialist")
    value.finish_attempt("a", "specialist", {"kind": "external_failure", "reason": "invalid"})
    with pytest.raises(ValueError, match="specialist"):
        value.start_attempt("a", "reviewer")


def test_invalid_delivered_published_specialist_cannot_route_to_reviewer(tmp_path):
    jobs = {"/synthetic/a": SyntheticJob("/synthetic/a", captures={
        "specialist": {"raw_response_sha256": "3" * 64, "capture_sha256": "4" * 64},
    }, reviewer_eligible=False)}
    value = ledger(tmp_path, jobs=jobs); value.freeze(manifest(["a"]))
    value.start_attempt("a", "specialist")
    value.finish_attempt("a", "specialist", delivery("a", "specialist"))
    with pytest.raises(ValueError, match="specialist"):
        value.start_attempt("a", "reviewer")


def test_timeout_is_terminal_and_late_delivery_is_an_immutable_side_record(tmp_path):
    now = [0]
    jobs = {"/synthetic/a": SyntheticJob("/synthetic/a", captures={
        "specialist": {"raw_response_sha256": "3" * 64, "capture_sha256": "4" * 64},
    })}
    value = ledger(tmp_path, now, jobs); value.freeze(manifest(["a"]))
    value.start_attempt("a", "specialist")
    now[0] = 601
    result = value.finish_attempt("a", "specialist", delivery("a", "specialist"))
    assert result["kind"] == "late_delivery"
    state = value.state()
    assert state["cases"]["a"]["roles"]["specialist"]["status"] == "timeout"
    assert state["late_deliveries"][0]["result"]["kind"] == "delivered"


def test_exact_600_seconds_times_out_and_accepts_only_one_late_result(tmp_path):
    now = [0.25]
    jobs = {"/synthetic/a": SyntheticJob("/synthetic/a", captures={
        "specialist": {"raw_response_sha256": "3" * 64, "capture_sha256": "4" * 64},
    })}
    value = ledger(tmp_path, now, jobs); value.freeze(manifest(["a"])); value.start_attempt("a", "specialist")
    now[0] = 600.25
    failure = {"kind": "external_failure", "reason": "controller_crash"}
    first = value.finish_attempt("a", "specialist", failure)
    assert first["kind"] == "late_external_failure"
    now[0] = 601.25
    assert value.finish_attempt("a", "specialist", deepcopy(failure)) == first
    with pytest.raises(ValueError, match="late result"):
        value.finish_attempt("a", "specialist", delivery("a", "specialist"))
    state = value.state()
    record = state["cases"]["a"]["roles"]["specialist"]
    assert record["status"] == "timeout"
    assert record["deadline_at"] == 600.25
    assert len(state["late_deliveries"]) == 1


def test_reopen_rejects_rehashed_malformed_nested_state_and_budget_invariant(tmp_path):
    jobs = {"/synthetic/a": SyntheticJob("/synthetic/a")}
    value = ledger(tmp_path, jobs=jobs); value.freeze(manifest(["a"]))
    state_path = tmp_path / "ledger" / "ledger.json"
    rewrite_rehashed_state(state_path, lambda state: state["cases"]["a"].update(roles={}))
    with pytest.raises(ValueError, match="role|case"):
        ledger(tmp_path, jobs=jobs).state()

    other = tmp_path / "budget"; value = ledger(other, jobs=jobs); value.freeze(manifest(["a"]))
    value.start_attempt("a", "specialist")
    rewrite_rehashed_state(other / "ledger" / "ledger.json", lambda state: state["budgets"].update(specialist=0))
    with pytest.raises(ValueError, match="budget"):
        ledger(other, jobs=jobs).state()


def test_reopen_rejects_rehashed_impossible_finish_time_and_forged_attempt_id(tmp_path):
    jobs = {"/synthetic/a": SyntheticJob("/synthetic/a")}
    now = [10]
    value = ledger(tmp_path, now, jobs); value.freeze(manifest(["a"])); value.start_attempt("a", "specialist")
    value.finish_attempt("a", "specialist", {"kind": "external_failure", "reason": "missing"})
    state_path = tmp_path / "ledger" / "ledger.json"
    rewrite_rehashed_state(state_path, lambda state: state["cases"]["a"]["roles"]["specialist"].update(finished_at=5))
    with pytest.raises(ValueError, match="precedes"):
        ledger(tmp_path, jobs=jobs).state()

    other = tmp_path / "attempt"; value = ledger(other, jobs=jobs); value.freeze(manifest(["a"])); value.start_attempt("a", "specialist")
    rewrite_rehashed_state(other / "ledger" / "ledger.json", lambda state: state["cases"]["a"]["roles"]["specialist"].update(attempt_id="0" * 64))
    with pytest.raises(ValueError, match="attempt id"):
        ledger(other, jobs=jobs).state()


def test_delivered_attempt_requires_job_capture_hash_binding(tmp_path):
    captures = {"specialist": {"raw_response_sha256": "3" * 64, "capture_sha256": "4" * 64}}
    jobs = {"/synthetic/a": SyntheticJob("/synthetic/a", captures=captures)}
    value = ledger(tmp_path, jobs=jobs); value.freeze(manifest(["a"]))
    value.start_attempt("a", "specialist")
    assert value.finish_attempt("a", "specialist", delivery("a", "specialist"))["kind"] == "delivered"
    bad = ledger(tmp_path / "bad", jobs={"/synthetic/a": SyntheticJob("/synthetic/a", captures=captures)})
    bad.freeze(manifest(["a"])); bad.start_attempt("a", "specialist")
    forged = delivery("a", "specialist"); forged["capture_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="capture"):
        bad.finish_attempt("a", "specialist", forged)


def test_lock_requires_every_case_and_published_grade_is_recomputable(tmp_path):
    grade = {"grade_sha256": "5" * 64, "status": "research_ready", "research_plan": {"action": "wait_5m_high"}}
    jobs = {
        "/synthetic/a": SyntheticJob("/synthetic/a", grade=grade),
        "/synthetic/b": SyntheticJob("/synthetic/b"),
    }
    value = ledger(tmp_path, jobs=jobs); value.freeze(manifest(["a", "b"]))
    terminals = {
        "a": {"kind": "published_grade", "job_directory": "/synthetic/a", "grade_sha256": "5" * 64},
    }
    with pytest.raises(ValueError, match="every"):
        value.lock_terminals(terminals)
    terminals["b"] = {"kind": "external_failure", "reason": "specialist_timeout"}
    value.finalize_case("a", terminals["a"])
    value.finalize_case("b", terminals["b"])
    value.lock_terminals(terminals)
    assert value.assert_reveal_allowed() is True


def test_real_published_job_artifacts_bind_delivery_and_grade(tmp_path):
    from scripts.research.lc_published_jobs import PublishedContextResearchJob
    from tests.research.test_lc_context_assessment import context_request
    from tests.research.test_lc_published_assessment import raw_answer, raw_review
    from tests.research.test_lc_published_jobs import valid_transport
    from scripts.research.lc_campaign_contract import CampaignLedger

    path = tmp_path / "published"; source = context_request(); job = PublishedContextResearchJob(path)
    job.prepare(source); request = job.role_request("specialist")
    specialist = raw_answer(source, request); job.capture("specialist", specialist, valid_transport("specialist"))
    review = raw_review(source, request, specialist); job.capture("reviewer", review, valid_transport("reviewer"))
    job.lock_grade(); stages = job._read(); bundle = stages["request"]["payload"]
    campaign = CampaignLedger(tmp_path / "campaign", clock=lambda: 0)
    frozen = {"schema_version": "lc_campaign_contract_v1", "campaign_id": "actual-job", "cases": [{
        "case_id": source["case_id"], "job_directory": str(path),
        "source_request_sha256": bundle["source_request_sha256"],
        "role_request_sha256": bundle["role_request_sha256"],
    }]}
    campaign.freeze(frozen)
    for role in ("specialist", "reviewer"):
        campaign.start_attempt(source["case_id"], role)
        payload = stages[role]["payload"]
        campaign.finish_attempt(source["case_id"], role, {
            "kind": "delivered", "job_directory": str(path),
            "raw_response_sha256": payload["raw_sha256"], "capture_sha256": stages[role]["sha256"],
        })
    terminal = {
        "kind": "published_grade", "job_directory": str(path), "grade_sha256": stages["grade"]["sha256"],
    }
    campaign.finalize_case(source["case_id"], terminal)
    campaign.lock_terminals({source["case_id"]: terminal})
    assert campaign.assert_reveal_allowed() is True


def test_reveal_is_refused_before_terminals_and_empty_campaign_is_not_launchable(tmp_path):
    value = ledger(tmp_path); value.freeze(manifest([]))
    with pytest.raises(ValueError, match="terminal"):
        value.assert_reveal_allowed()
    with pytest.raises(ValueError, match="unknown|empty"):
        value.start_attempt("a", "specialist")
    value.lock_terminals({})
    assert value.assert_reveal_allowed() is True


def test_decision_path_seals_specialist_start_and_same_runtime_elapsed_with_ceiling(tmp_path):
    from scripts.research.lc_campaign_contract import ceil_elapsed_seconds

    wall = [100]; mono = [200]; now = [0]
    jobs = {"/synthetic/a": SyntheticJob("/synthetic/a")}
    value = ledger(tmp_path, now, jobs)
    value._wall_ns = lambda: wall[0]; value._monotonic_ns = lambda: mono[0]; value._runtime_id = "timer-a"
    value.freeze(manifest(["a"])); value.start_attempt("a", "specialist")
    path = value.state()["cases"]["a"]["decision_path"]
    assert path["start"] == {"runtime_id": "timer-a", "wall_ns": 100, "monotonic_ns": 200}
    value.finish_attempt("a", "specialist", {"kind": "external_failure", "reason": "missing"})
    wall[0] = 999; mono[0] = 1_000_000_201
    terminal = {"kind": "external_failure", "reason": "specialist_missing"}
    sealed = value.finalize_case("a", terminal)
    assert sealed["timing_valid"] is True
    assert sealed["elapsed_ns"] == 1_000_000_001
    assert ceil_elapsed_seconds(sealed["elapsed_ns"]) == 2
    assert value.finalize_case("a", terminal) == sealed
    value.lock_terminals({"a": terminal})


def test_decision_path_runtime_restart_invalidates_elapsed_and_terminal_mismatch_blocks_lock(tmp_path):
    wall = [1]; mono = [2]; now = [0]
    jobs = {"/synthetic/a": SyntheticJob("/synthetic/a")}
    first = ledger(tmp_path, now, jobs)
    first._wall_ns = lambda: wall[0]; first._monotonic_ns = lambda: mono[0]; first._runtime_id = "timer-a"
    first.freeze(manifest(["a"])); first.start_attempt("a", "specialist")
    first.finish_attempt("a", "specialist", {"kind": "external_failure", "reason": "missing"})
    resumed = ledger(tmp_path, now, jobs)
    resumed._wall_ns = lambda: wall[0]; resumed._monotonic_ns = lambda: mono[0]; resumed._runtime_id = "timer-b"
    terminal = {"kind": "external_failure", "reason": "specialist_missing"}
    sealed = resumed.finalize_case("a", terminal)
    assert sealed["timing_valid"] is False
    assert sealed["reason"] == "timer_runtime_changed"
    assert sealed["elapsed_ns"] is None
    with pytest.raises(ValueError, match="finalized|hash"):
        resumed.lock_terminals({"a": {"kind": "external_failure", "reason": "different"}})
    resumed.lock_terminals({"a": terminal})


def test_lock_requires_finalized_decision_path_for_timeout_and_skipped_reviewer(tmp_path):
    now = [0]; jobs = {"/synthetic/a": SyntheticJob("/synthetic/a")}
    value = ledger(tmp_path, now, jobs); value.freeze(manifest(["a"])); value.start_attempt("a", "specialist")
    now[0] = 600
    value.state()
    terminal = {"kind": "external_failure", "reason": "specialist_timeout"}
    with pytest.raises(ValueError, match="finalized"):
        value.lock_terminals({"a": terminal})
    value.finalize_case("a", terminal)
    value.lock_terminals({"a": terminal})
