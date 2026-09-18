"""End-to-end synthetic tests for the full LC judgment execution lifecycle."""
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.assessment_evidence_guard import _canonical, build_envelope
from scripts.research.conditional_assessment import digest
from scripts.research.lc_campaign_contract import CampaignLedger
from scripts.research.lc_context_assessment import build_context_request
from scripts.research.lc_judgment_execution import JudgmentExecution
from scripts.research.lc_master_assessment import build_lc_packet
from scripts.research.lc_published_assessment import build_published_request
from tests.research.test_lc_context_assessment import context_brief, empty_snapshot, SETTINGS
from tests.research.test_lc_master_assessment import inputs
from tests.research.test_lc_published_assessment import raw_answer, raw_review


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value).encode("ascii")); return path


def freeze_inputs(root):
    evidence = root / "evidence"
    roster = [f"LCX{index:02d}" for index in range(20)]
    files = [save(evidence / "roster.json", roster)]
    raw, bars, parents, provenance = inputs()
    base_packet = build_lc_packet(raw, bars, parents, provenance, roster[0])
    for case_id in roster:
        packet = deepcopy(base_packet)
        packet.pop("seal"); packet["case_id"] = case_id
        packet["seal"] = digest(packet)
        request = build_context_request(packet, empty_snapshot(), context_brief(), SETTINGS)
        role_request = build_published_request(request)
        wrapper = {"case_id": case_id, "plan": request["plan"], "request": role_request}
        files.extend([
            save(evidence / f"{case_id}_source_request.json", request),
            save(evidence / f"{case_id}_specialist_packet.json", wrapper),
            save(evidence / f"{case_id}_specialist_envelope.json", build_envelope(wrapper)),
        ])
    prepared_body = {"schema_version": "lc_judgment_preparation_v1",
                     "roster": roster, "roles_enabled": False}
    prepared = save(root / "judgment_prepare.json",
                    dict(prepared_body, sha256=digest(prepared_body)))
    lock_body = {"files": {str(p.resolve()): sha(p) for p in files},
                 "roles_enabled": False, "outcome_reveal_authorized": False,
                 "scope": "synthetic20", "roster": roster,
                 "expected_fixed_hashes": {}}
    evidence_lock = save(evidence / "evidence_lock.json",
                         dict(lock_body, sha256=digest(lock_body)))
    binding_body = {
        "files": {str(p.resolve()): sha(p) for p in (prepared, evidence_lock)},
        "evidence_files": {str(p.resolve()): sha(p) for p in files},
        "roles_enabled": False, "outcome_reveal_authorized": False,
        "budget_status": "synthetic", "roster": roster,
    }
    save(root / "preparation_binding.json",
         dict(binding_body, sha256=digest(binding_body)))
    return roster


def returns(reservation):
    return [{"role": reservation["role"], "case_id": reservation["case_id"],
             "packet_sha256": reservation["envelope"]["packet_sha256"],
             "chunk_index": chunk["index"],
             "exec_result": {"exit_code": 0, "output": chunk["text"]}}
            for chunk in reservation["envelope"]["chunks"]]


def metadata(role, agent):
    return {"agent_id": agent, "requested_model": "gpt-6-astra",
            "actual_model": None, "actual_runtime_verified": True}


@pytest.fixture
def run(tmp_path):
    preparation = tmp_path / "judgment_v1"; roster = freeze_inputs(preparation)
    seconds = [0.0]; wall = [100]; mono = [200]
    def ledger_factory(path, **kwargs):
        return CampaignLedger(path, clock=lambda: seconds[0],
                              wall_ns=lambda: wall[0], monotonic_ns=lambda: mono[0],
                              runtime_id=kwargs["runtime_id"])
    def make(runtime_id="execution-a"):
        return JudgmentExecution(
            preparation / "runtime_v2", preparation_dir=preparation,
            ledger_factory=ledger_factory, runtime_id=runtime_id,
            utc_now=lambda: datetime(2026, 9, 16, 7, 0, tzinfo=timezone.utc))
    return make, roster, seconds, wall, mono


def snapshot(balance):
    return {"balance_credits": balance,
            "observed_at": "2026-09-16T07:00:00+00:00",
            "source": "synthetic-usage"}


def readiness():
    return {"approved": True, "reviewer": "independent-fixture",
            "reviewed_at": "2026-09-16T06:59:00+00:00",
            "scope": "lc_judgment_execution_v1", "outcomes_accessed": False}


def test_valid_specialist_critic_and_all20_terminals_unlock_reveal(run):
    make, roster, _, _, mono = run
    runner = make(); assert runner.prepare()["roster_count"] == 20
    token = runner.authorize_start(snapshot(2000), readiness(),
                                   max_observed_balance_drop=500)
    cid = roster[0]
    specialist = runner.reserve_role(cid, "specialist", snapshot(2000),
                                     authorization_token=token)
    source = json.loads((runner.evidence_dir / f"{cid}_source_request.json").read_bytes())
    raw = raw_answer(source, specialist["wrapper"]["request"])
    runner.capture_role(cid, "specialist", raw.encode(), metadata("specialist", "agent-s"),
                        returns(specialist))
    review = runner.prepare_reviewer(cid)
    assert review["reviewer_required"] is True
    critic = runner.reserve_role(cid, "reviewer", snapshot(1990))
    review_raw = raw_review(source, specialist["wrapper"]["request"], raw)
    runner.capture_role(cid, "reviewer", review_raw.encode(), metadata("reviewer", "agent-r"),
                        returns(critic))
    mono[0] += 1_500_000_001
    grade = runner.finalize_grade(cid)
    assert grade["kind"] == "published_grade"
    assert grade["decision_path"]["timing_valid"] is True
    for other in roster[1:]:
        runner.finalize_failure(other, "not_invoked_test_terminal")
    with pytest.raises(ValueError, match="all case terminals"):
        runner.assert_reveal_allowed()
    with pytest.raises(ValueError, match="finalized case"):
        runner.reserve_role(roster[1], "specialist", snapshot(1990))
    assert len(runner.lock_terminals()) == 20
    assert runner.assert_reveal_allowed() is True
    assert runner.status()["phase"] == "reveal_allowed"
    runner.close()


def test_invalid_specialist_skips_critic_and_reused_agent_is_invalid_transport(run):
    make, roster, _, _, _ = run
    runner = make(); runner.prepare()
    token = runner.authorize_start(snapshot(2000), readiness(),
                                   max_observed_balance_drop=100)
    first = runner.reserve_role(roster[0], "specialist", snapshot(2000),
                                authorization_token=token)
    runner.capture_role(roster[0], "specialist", b"not-json",
                        metadata("specialist", "reused-agent"), returns(first))
    skipped = runner.prepare_reviewer(roster[0])
    assert skipped["reviewer_required"] is False
    assert skipped["terminal"]["kind"] == "published_grade"

    second = runner.reserve_role(roster[1], "specialist", snapshot(1999))
    source = json.loads((runner.evidence_dir / f"{roster[1]}_source_request.json").read_bytes())
    raw = raw_answer(source, second["wrapper"]["request"])
    captured = runner.capture_role(roster[1], "specialist", raw.encode(),
                                   metadata("specialist", "reused-agent"), returns(second))
    assert captured["transport_valid"] is False
    terminal = runner.prepare_reviewer(roster[1])
    assert terminal["reviewer_required"] is False

    third = runner.reserve_role(roster[2], "specialist", snapshot(1998))
    failed = runner.finish_failure(roster[2], "specialist", "dispatch_bridge_failed")
    assert third["dispatch_authorized"] is True
    assert failed["kind"] == "external_failure"
    assert runner.finish_failure(
        roster[2], "specialist", "dispatch_bridge_failed") == failed
    with pytest.raises(ValueError, match="immutable dispatch failure"):
        runner.finish_failure(roster[2], "specialist", "different")
    with pytest.raises(ValueError, match="finalized case"):
        runner.reserve_role(roster[2], "specialist", snapshot(1998))
    assert runner.ledger_state()["cases"][roster[2]]["decision_path"]["end"] is not None
    runner.close()


def test_timeout_restart_no_retry_and_observed_drop_blocks_only_new_work(run):
    make, roster, seconds, _, _ = run
    first = make(); first.prepare()
    token = first.authorize_start(snapshot(2000), readiness(),
                                  max_observed_balance_drop=50)
    reservation = first.reserve_role(roster[0], "specialist", snapshot(2000),
                                     authorization_token=token)
    seconds[0] = 601; first.close()
    resumed = make("execution-b")
    assert resumed.ledger_state()["cases"][roster[0]]["roles"]["specialist"]["status"] == "timeout"
    with pytest.raises(ValueError, match="already invoked"):
        resumed.reserve_role(roster[0], "specialist", snapshot(1999))
    resumed.finalize_failure(roster[0], "specialist_timeout")
    path = resumed.ledger_state()["cases"][roster[0]]["decision_path"]
    assert path["timing_valid"] is False and path["reason"] == "timer_runtime_changed"
    with pytest.raises(ValueError, match="allowance"):
        resumed.reserve_role(roster[1], "specialist", snapshot(1949))
    assert resumed.ledger_state()["budgets"]["specialist"] == 1

    for other in roster[1:]:
        resumed.finalize_failure(other, "not_invoked_test_terminal")
    resumed.lock_terminals()
    with pytest.raises(ValueError, match="terminal lock"):
        resumed.capture_role(roster[0], "specialist", b"late",
                             metadata("specialist", "late-agent"),
                             returns(reservation))
    assert resumed.assert_reveal_allowed() is True
    resumed.close()
