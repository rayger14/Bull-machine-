"""Focused tests for the fail-closed LC judgment runner foundation."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.assessment_evidence_guard import _canonical, build_envelope
from scripts.research.lc_campaign_contract import CampaignLedger
from scripts.research.lc_judgment_runner import JudgmentRunner


def digest(value):
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical(value).encode("ascii"))
    return path


def lock(path, files, **extra):
    body = {"files": {str(Path(item).resolve()): sha(item) for item in files}, **extra}
    return save(path, dict(body, sha256=digest(body)))


class FakePublishedJob:
    def __init__(self, path):
        self.path = Path(path)

    def prepare(self, source_request):
        path = self.path / "request.json"
        payload = {"source_request": deepcopy(source_request),
                   "source_request_sha256": digest(source_request),
                   "role_request": deepcopy(source_request["role_request"]),
                   "role_request_sha256": digest(source_request["role_request"])}
        if path.exists() and json.loads(path.read_bytes()) != payload:
            raise ValueError("immutable fake job differs")
        save(path, payload)
        return payload

    def _payload(self):
        return json.loads((self.path / "request.json").read_bytes())

    def role_request(self, role):
        if role != "specialist":
            raise ValueError("specialist capture required before reviewer")
        return deepcopy(self._payload()["role_request"])

    def campaign_binding(self):
        value = self._payload()
        return {key: value[key] for key in
                ("source_request_sha256", "role_request_sha256")}

    def reviewer_eligible(self):
        return False


@pytest.fixture
def frozen(tmp_path):
    preparation = tmp_path / "judgment_v1"
    evidence = preparation / "evidence"
    roster = [f"case-{index:02d}" for index in range(20)]
    save(evidence / "roster.json", roster)
    evidence_files = [evidence / "roster.json"]
    wrappers = {}
    for case_id in roster:
        plan = {"indicative_close": 100.0, "stop": 95.0, "notional": 50000.0,
                "roundtrip_cost": 60.0}
        role_request = {"case_id": case_id, "instruction": "frozen"}
        source_request = {"case_id": case_id, "plan": plan,
                          "role_request": role_request}
        wrapper = {"case_id": case_id, "plan": plan, "request": role_request}
        wrappers[case_id] = wrapper
        evidence_files.extend([
            save(evidence / f"{case_id}_source_request.json", source_request),
            save(evidence / f"{case_id}_specialist_packet.json", wrapper),
            save(evidence / f"{case_id}_specialist_envelope.json",
                 build_envelope(wrapper, max_chunk_bytes=4096)),
        ])
    prepared_body = {"schema_version": "lc_judgment_preparation_v1",
                     "roster": roster, "roles_enabled": False}
    prepared = save(preparation / "judgment_prepare.json",
                    dict(prepared_body, sha256=digest(prepared_body)))
    evidence_lock = lock(
        evidence / "evidence_lock.json", evidence_files,
        roles_enabled=False, outcome_reveal_authorized=False,
        scope="synthetic judgment evidence", roster=roster,
        expected_fixed_hashes={},
    )
    binding_files = [prepared, evidence_lock]
    binding_body = {
        "files": {str(path.resolve()): sha(path) for path in binding_files},
        "evidence_files": {str(path.resolve()): sha(path) for path in evidence_files},
        "roles_enabled": False,
        "outcome_reveal_authorized": False,
        "budget_status": "below_existing_1900_start_floor",
        "roster": roster,
    }
    save(preparation / "preparation_binding.json",
         dict(binding_body, sha256=digest(binding_body)))

    now = [100.0]
    wall = [1_000]
    mono = [2_000]
    factory_calls = []

    def ledger_factory(path, **kwargs):
        factory_calls.append(str(path))
        return CampaignLedger(
            path, clock=lambda: now[0], wall_ns=lambda: wall[0],
            monotonic_ns=lambda: mono[0],
            job_loader=lambda job_path: FakePublishedJob(job_path),
            runtime_id=kwargs["runtime_id"],
        )

    def make_runner(runtime=None, runtime_id="runtime-a", dry_run=True):
        return JudgmentRunner(
            runtime or preparation / "runtime_v1",
            preparation_dir=preparation,
            job_factory=FakePublishedJob,
            ledger_factory=ledger_factory,
            runtime_id=runtime_id,
            dry_run=dry_run,
            utc_now=lambda: datetime(2026, 9, 16, 7, 0, tzinfo=timezone.utc),
        )

    return make_runner, preparation, roster, wrappers, now, factory_calls


def fresh_budget(balance=1900.0, minutes_old=0):
    observed = datetime(2026, 9, 16, 7, 0, tzinfo=timezone.utc) - timedelta(minutes=minutes_old)
    return {"balance_credits": balance, "observed_at": observed.isoformat(),
            "source": "synthetic-local-usage"}


def test_prepare_verifies_cross_bound_roster_clones_jobs_and_reuses_one_ledger(frozen):
    """Catches bypassed evidence bindings or a fresh ledger per method call."""
    make_runner, preparation, roster, _, _, factory_calls = frozen
    runner = make_runner()
    try:
        result = runner.prepare()
        assert result == {"prepared": True, "roster_count": 20,
                          "launch_ready": False, "roles_enabled": False}
        assert len(factory_calls) == 1
        assert runner.verify()["roster_count"] == 20
        assert runner.status()["phase"] == "prepared"
        assert len(factory_calls) == 1
        for case_id in roster:
            assert (preparation / "runtime_v1" / "jobs" / case_id / "request.json").exists()
        runtime_lock = json.loads(
            (preparation / "runtime_v1" / "runtime_lock.json").read_bytes())
        assert runtime_lock["preparation_binding_sha256"] == sha(
            preparation / "preparation_binding.json")
        assert runtime_lock["launch_ready"] is False
    finally:
        runner.close()
    with pytest.raises(ValueError, match="ownership"):
        runner.status()


def test_fresh_campaign_start_gate_and_reservation_precedes_return(frozen):
    """Catches stale/low campaign start or wrappers returned before durable consumption."""
    make_runner, preparation, roster, wrappers, now, _ = frozen
    runner = make_runner(); runner.prepare()
    try:
        with pytest.raises(ValueError, match="1900"):
            runner.authorize_start(fresh_budget(1899.99))
        with pytest.raises(ValueError, match="fresh"):
            runner.authorize_start(fresh_budget(2000, minutes_old=6))
        token = runner.authorize_start(fresh_budget())
        reserved = runner.reserve_role(roster[0], "specialist",
                                       authorization_token=token)
        assert reserved["wrapper"] == wrappers[roster[0]]
        assert reserved["dispatch_authorized"] is False
        assert reserved["reason"] == "capture_lifecycle_not_implemented"
        journal = preparation / "runtime_v1" / "reservations" / reserved["attempt_id"] / "reservation.json"
        assert journal.exists()
        state = runner.ledger_state()
        role = state["cases"][roster[0]]["roles"]["specialist"]
        assert role["status"] == "active"
        assert role["deadline_at"] == now[0] + 600
        with pytest.raises(ValueError, match="already invoked"):
            runner.reserve_role(roster[0], "specialist")
    finally:
        runner.close()


def test_process_ownership_and_explicit_restart_recovery_never_retry(frozen):
    """Catches concurrent controllers and redispatch after an uncertain reservation."""
    make_runner, _, roster, _, _, _ = frozen
    first = make_runner(); first.prepare()
    token = first.authorize_start(fresh_budget())
    reserved = first.reserve_role(roster[0], "specialist",
                                  authorization_token=token)
    with pytest.raises(ValueError, match="owned"):
        make_runner(runtime_id="runtime-b")
    first.close()

    resumed = make_runner(runtime_id="runtime-b")
    try:
        status = resumed.status()
        assert status["recovery_required"] == [{
            "case_id": roster[0], "role": "specialist",
            "attempt_id": reserved["attempt_id"],
        }]
        assert status["timing_continuity"] is False
        with pytest.raises(ValueError, match="already invoked"):
            resumed.reserve_role(roster[0], "specialist")
        recovered = resumed.recover_uncertain(roster[0], "specialist")
        assert recovered == {"kind": "external_failure",
                             "reason": "controller_restart_after_reservation"}
        assert resumed.ledger_state()["cases"][roster[0]]["roles"]["specialist"][
            "status"] == "external_failure"
        with pytest.raises(ValueError, match="already invoked"):
            resumed.reserve_role(roster[0], "specialist")
    finally:
        resumed.close()


def test_closed_runner_rejects_every_operation_after_new_owner_acquires(frozen):
    """Catches stale controller use after its runtime lease passes to a new owner."""
    make_runner, _, roster, _, _, _ = frozen
    stale = make_runner(runtime_id="runtime-stale")
    stale.prepare()
    stale.close()
    current = make_runner(runtime_id="runtime-current")
    try:
        operations = [
            stale.prepare,
            stale.verify,
            stale.status,
            lambda: stale.authorize_start(fresh_budget()),
            lambda: stale.reserve_role(roster[0], "specialist"),
            lambda: stale.recover_uncertain(roster[0], "specialist"),
        ]
        for operation in operations:
            with pytest.raises(ValueError, match="ownership"):
                operation()
        assert current.status()["budgets"] == {"specialist": 0, "reviewer": 0}
    finally:
        current.close()


def test_prepare_rejects_changed_cross_binding_without_writing_runtime(frozen):
    """Catches launch preparation against altered frozen evidence."""
    make_runner, preparation, roster, _, _, _ = frozen
    source = preparation / "evidence" / f"{roster[0]}_source_request.json"
    source.write_text("{}")
    runner = make_runner()
    try:
        with pytest.raises(ValueError, match="locked|binding"):
            runner.prepare()
        assert not (preparation / "runtime_v1" / "runtime_lock.json").exists()
    finally:
        runner.close()


def test_real_namespace_refuses_reservation_before_consuming_attempt(frozen):
    """Catches accidental paid-role enablement while capture lifecycle is absent."""
    make_runner, _, roster, _, _, _ = frozen
    runner = make_runner(runtime_id="runtime-real", dry_run=False)
    runner.prepare()
    try:
        token = runner.authorize_start(fresh_budget())
        with pytest.raises(ValueError, match="launch-ready"):
            runner.reserve_role(roster[0], "specialist",
                                authorization_token=token)
        assert runner.ledger_state()["budgets"] == {"specialist": 0, "reviewer": 0}
    finally:
        runner.close()


def test_verify_binds_current_preparation_and_evidence_lock_hashes(frozen):
    """Catches a runtime lock that records but never checks its upstream bindings."""
    make_runner, preparation, _, _, _, _ = frozen
    runner = make_runner(); runner.prepare()
    try:
        path = preparation / "runtime_v1" / "runtime_lock.json"
        value = json.loads(path.read_bytes())
        value["preparation_binding_sha256"] = "0" * 64
        body = {key: item for key, item in value.items() if key != "sha256"}
        value["sha256"] = digest(body)
        path.write_bytes(_canonical(value).encode("ascii"))
        with pytest.raises(ValueError, match="upstream binding"):
            runner.verify()
    finally:
        runner.close()
