"""Immutable, offline roster and terminal ledger for the LC campaign.

The public manifest/result schemas intentionally accept no extensible fields.
This keeps the controller's freeze and audit boundary explicit: role delivery
is recorded separately from a recomputable published grade, and controller
failures never become grades.
"""
from copy import deepcopy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

from scripts.research.assessment_evidence_guard import _canonical


SCHEMA_VERSION = "lc_campaign_contract_v1"
ROLES = ("specialist", "reviewer")
MAX_CASES = MAX_ROLE_ATTEMPTS = 30
MAX_ACTIVE = 3
DEADLINE_SECONDS = 600
_HEX = set("0123456789abcdef")


def select_block(candidate_ids: list[str], excluded: set[str], cap: int = 30) -> list[str]:
    """Choose earliest full contiguous unexcluded block, else longest earliest run."""
    if type(cap) is not int or cap < 1 or cap > MAX_CASES:
        raise ValueError("cap must be an integer from 1 through 30")
    if not isinstance(candidate_ids, list) or not isinstance(excluded, set):
        raise ValueError("candidate_ids must be a list and excluded a set")
    if any(not isinstance(case_id, str) or not case_id for case_id in candidate_ids):
        raise ValueError("candidate IDs must be nonempty strings")
    if len(set(candidate_ids)) != len(candidate_ids):
        raise ValueError("candidate IDs must be unique")
    if any(not isinstance(case_id, str) for case_id in excluded):
        raise ValueError("excluded IDs must be strings")
    runs, run = [], []
    for case_id in candidate_ids:
        if case_id in excluded:
            if run:
                runs.append(run)
                run = []
        else:
            run.append(case_id)
    if run:
        runs.append(run)
    for run in runs:
        if len(run) >= cap:
            return run[:cap]
    return list(max(runs, key=len)) if runs else []


def _hash(value):
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _sha(value, label):
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise ValueError(label + " must be a lowercase SHA-256")


def _now_seconds(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or value < 0:
        raise ValueError("clock must return nonnegative seconds")
    return int(value)


def _default_job_loader(path):
    from scripts.research.lc_published_jobs import PublishedContextResearchJob
    return _PublishedJobAdapter(path)


class _PublishedJobAdapter:
    """Small read-only adapter; construction rechecks the published job chain."""
    def __init__(self, path):
        from scripts.research.lc_published_jobs import PublishedContextResearchJob
        self.path = str(path)
        self._job = PublishedContextResearchJob(path)

    def _stages(self):
        return self._job._read()

    def campaign_binding(self):
        bundle = self._stages()["request"]["payload"]
        return {"source_request_sha256": bundle["source_request_sha256"],
                "role_request_sha256": bundle["role_request_sha256"]}

    def capture_binding(self, role):
        stage = self._stages()[role]
        payload = stage["payload"]
        return {"raw_response_sha256": payload["raw_sha256"],
                "capture_sha256": stage["sha256"]}

    def grade_binding(self):
        stage = self._stages()["grade"]
        payload = stage["payload"]
        return {"grade_sha256": stage["sha256"], "status": payload["status"],
                "research_plan": deepcopy(payload["research_plan"])}

    def reviewer_eligible(self):
        try:
            self._job.role_request("reviewer")
        except ValueError:
            return False
        return True


class CampaignLedger:
    """Atomic immutable campaign ledger; it dispatches nothing and reads no market data.

    `freeze` requires the exact manifest schema below. `finish_attempt` accepts
    a delivered capture bound to its published job, or a controller failure.
    `lock_terminals` accepts one terminal per frozen case: a recomputable
    published grade or an external controller failure with its null plan.
    """
    def __init__(self, directory: Path, *, clock=time.time, job_loader=_default_job_loader):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._clock = clock
        self._job_loader = job_loader
        self._lock_path = self.directory / ".lock"

    def _locked(self):
        stream = self._lock_path.open("a+")
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        return stream

    def _read(self):
        path = self.directory / "ledger.json"
        if not path.exists():
            return None
        raw = path.read_bytes()
        try:
            value = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid campaign ledger bytes") from exc
        if _canonical(value).encode("ascii") != raw or value.get("state_sha256") != _hash(
                {key: item for key, item in value.items() if key != "state_sha256"}):
            raise ValueError("campaign ledger hash or bytes changed")
        self._validate_state(value)
        return value

    def _write(self, state):
        body = deepcopy(state); body.pop("state_sha256", None)
        state = dict(body, state_sha256=_hash(body))
        raw = _canonical(state).encode("ascii")
        fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.directory)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self.directory / "ledger.json")
            directory_fd = os.open(self.directory, os.O_RDONLY)
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
        finally:
            if os.path.exists(temporary):
                os.unlink(temporary)
        return deepcopy(state)

    def _validate_manifest(self, manifest):
        expected = {"schema_version", "campaign_id", "cases"}
        if not isinstance(manifest, dict) or set(manifest) != expected:
            raise ValueError("exact campaign manifest schema required")
        if manifest["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported campaign manifest schema")
        if not isinstance(manifest["campaign_id"], str) or not manifest["campaign_id"]:
            raise ValueError("campaign_id must be nonempty")
        cases = manifest["cases"]
        if not isinstance(cases, list) or len(cases) > MAX_CASES:
            raise ValueError("campaign may contain at most 30 cases")
        required = {"case_id", "job_directory", "source_request_sha256", "role_request_sha256"}
        seen = set()
        for case in cases:
            if not isinstance(case, dict) or set(case) != required:
                raise ValueError("exact case manifest schema required")
            if not isinstance(case["case_id"], str) or not case["case_id"] or case["case_id"] in seen:
                raise ValueError("case IDs must be unique nonempty strings")
            if not isinstance(case["job_directory"], str) or not case["job_directory"]:
                raise ValueError("job_directory must be nonempty")
            _sha(case["source_request_sha256"], "source request hash")
            _sha(case["role_request_sha256"], "role request hash")
            binding = self._job_loader(case["job_directory"]).campaign_binding()
            if (not isinstance(binding, dict) or set(binding) != {"source_request_sha256", "role_request_sha256"}
                    or binding != {key: case[key] for key in binding}):
                raise ValueError("published job binding differs from manifest")
            seen.add(case["case_id"])

    def _validate_state(self, state):
        required = {"schema_version", "manifest", "cases", "budgets", "late_deliveries", "terminals", "terminals_locked", "state_sha256"}
        if not isinstance(state, dict) or set(state) != required or state["schema_version"] != SCHEMA_VERSION:
            raise ValueError("invalid campaign ledger schema")
        self._validate_manifest(state["manifest"])
        ids = [case["case_id"] for case in state["manifest"]["cases"]]
        if set(state["cases"]) != set(ids) or set(state["budgets"]) != set(ROLES):
            raise ValueError("campaign state cases or budgets differ from manifest")
        for role, count in state["budgets"].items():
            if type(count) is not int or not 0 <= count <= MAX_ROLE_ATTEMPTS:
                raise ValueError("invalid role budget")
        if (not isinstance(state["late_deliveries"], list) or not isinstance(state["terminals"], dict)
                or type(state["terminals_locked"]) is not bool):
            raise ValueError("invalid campaign collections")

    def _expire(self, state, now):
        for case in state["cases"].values():
            for role in ROLES:
                record = case["roles"][role]
                if record["status"] == "active" and now > record["deadline_at"]:
                    record["status"] = "timeout"
                    record["finished_at"] = record["deadline_at"]
                    record["result"] = {"kind": "deadline_timeout", "deadline_at": record["deadline_at"]}

    def freeze(self, manifest):
        with self._locked() as stream:
            self._validate_manifest(manifest)
            state = self._read()
            if state is not None:
                if state["manifest"] != manifest:
                    raise ValueError("immutable campaign manifest differs")
                return deepcopy(manifest)
            cases = {}
            for case in manifest["cases"]:
                cases[case["case_id"]] = {"roles": {
                    role: {"status": "not_invoked", "attempt_id": None, "started_at": None,
                           "deadline_at": None, "finished_at": None, "result": None}
                    for role in ROLES}}
            self._write({"schema_version": SCHEMA_VERSION, "manifest": deepcopy(manifest), "cases": cases,
                         "budgets": {role: 0 for role in ROLES}, "late_deliveries": [], "terminals": {},
                         "terminals_locked": False})
            return deepcopy(manifest)

    def _state_for_write(self):
        state = self._read()
        if state is None:
            raise ValueError("campaign manifest must freeze before attempts")
        self._expire(state, _now_seconds(self._clock()))
        return state

    def start_attempt(self, case_id, role):
        with self._locked() as stream:
            state = self._state_for_write()
            if role not in ROLES or case_id not in state["cases"]:
                raise ValueError("unknown case or role")
            if state["terminals_locked"]:
                raise ValueError("terminal lock forbids attempts")
            record = state["cases"][case_id]["roles"][role]
            if record["attempt_id"] is not None:
                raise ValueError("role already invoked")
            if (role == "reviewer" and (state["cases"][case_id]["roles"]["specialist"]["status"] != "delivered"
                    or self._job_loader(next(item for item in state["manifest"]["cases"] if item["case_id"] == case_id)["job_directory"]).reviewer_eligible() is not True)):
                raise ValueError("valid delivered specialist required before reviewer")
            if state["budgets"][role] >= MAX_ROLE_ATTEMPTS:
                raise ValueError("role budget exhausted")
            active = sum(record["status"] == "active" for case in state["cases"].values() for record in case["roles"].values())
            if active >= MAX_ACTIVE:
                raise ValueError("maximum active roles reached")
            now = _now_seconds(self._clock())
            record.update({"status": "active", "attempt_id": _hash({"case_id": case_id, "role": role, "started_at": now}),
                           "started_at": now, "deadline_at": now + DEADLINE_SECONDS})
            state["budgets"][role] += 1
            self._write(state)
            return deepcopy(record)

    def _validate_result(self, case, role, result):
        if not isinstance(result, dict) or not isinstance(result.get("kind"), str):
            raise ValueError("exact attempt result schema required")
        if result["kind"] == "external_failure":
            if set(result) != {"kind", "reason"} or not isinstance(result["reason"], str) or not result["reason"]:
                raise ValueError("exact external failure schema required")
            return deepcopy(result)
        if result["kind"] != "delivered" or set(result) != {"kind", "job_directory", "raw_response_sha256", "capture_sha256"}:
            raise ValueError("exact delivered result schema required")
        manifest_case = next(item for item in self._read()["manifest"]["cases"] if item["case_id"] == case)
        if result["job_directory"] != manifest_case["job_directory"]:
            raise ValueError("delivered job differs from frozen case")
        _sha(result["raw_response_sha256"], "raw response hash"); _sha(result["capture_sha256"], "capture hash")
        binding = self._job_loader(result["job_directory"]).capture_binding(role)
        if binding != {"raw_response_sha256": result["raw_response_sha256"], "capture_sha256": result["capture_sha256"]}:
            raise ValueError("delivered capture differs from published job")
        return deepcopy(result)

    def finish_attempt(self, case_id, role, result):
        with self._locked() as stream:
            state = self._state_for_write()
            if role not in ROLES or case_id not in state["cases"]:
                raise ValueError("unknown case or role")
            record = state["cases"][case_id]["roles"][role]
            if record["attempt_id"] is None:
                raise ValueError("attempt was not invoked")
            validated = self._validate_result(case_id, role, result)
            now = _now_seconds(self._clock())
            if record["status"] == "timeout":
                late = {"case_id": case_id, "role": role, "attempt_id": record["attempt_id"], "received_at": now, "result": validated}
                if late not in state["late_deliveries"]:
                    state["late_deliveries"].append(late); self._write(state)
                return {"kind": "late_delivery", **deepcopy(late)}
            if record["status"] != "active":
                if record["result"] != validated:
                    raise ValueError("immutable attempt result differs")
                return deepcopy(record["result"])
            record.update({"status": "delivered" if validated["kind"] == "delivered" else "external_failure",
                           "finished_at": now, "result": validated})
            self._write(state)
            return deepcopy(validated)

    def _validate_terminal(self, state, case_id, terminal):
        if not isinstance(terminal, dict) or not isinstance(terminal.get("kind"), str):
            raise ValueError("exact terminal schema required")
        if terminal["kind"] == "external_failure":
            if set(terminal) != {"kind", "reason"} or not isinstance(terminal["reason"], str) or not terminal["reason"]:
                raise ValueError("exact external terminal schema required")
            return deepcopy(terminal)
        if terminal["kind"] != "published_grade" or set(terminal) != {"kind", "job_directory", "grade_sha256"}:
            raise ValueError("exact published terminal schema required")
        manifest_case = next(item for item in state["manifest"]["cases"] if item["case_id"] == case_id)
        if terminal["job_directory"] != manifest_case["job_directory"]:
            raise ValueError("terminal job differs from frozen case")
        _sha(terminal["grade_sha256"], "grade hash")
        grade = self._job_loader(terminal["job_directory"]).grade_binding()
        if (not isinstance(grade, dict) or set(grade) != {"grade_sha256", "status", "research_plan"}
                or grade["grade_sha256"] != terminal["grade_sha256"]):
            raise ValueError("published grade is not recomputable")
        return dict(terminal, status=grade["status"], research_plan=deepcopy(grade["research_plan"]))

    def lock_terminals(self, terminals):
        with self._locked() as stream:
            state = self._state_for_write()
            if not isinstance(terminals, dict) or set(terminals) != set(state["cases"]):
                raise ValueError("terminal required for every frozen case")
            if any(record["status"] == "active" for case in state["cases"].values() for record in case["roles"].values()):
                raise ValueError("active attempt prevents terminal lock")
            expected = {case_id: self._validate_terminal(state, case_id, item) for case_id, item in terminals.items()}
            if state["terminals_locked"]:
                if state["terminals"] != expected:
                    raise ValueError("immutable terminals differ")
                return deepcopy(expected)
            for case in state["cases"].values():
                for record in case["roles"].values():
                    if record["status"] == "not_invoked":
                        record["result"] = {"kind": "not_invoked"}
            state["terminals"] = expected
            state["terminals_locked"] = True
            self._write(state)
            return deepcopy(expected)

    def assert_reveal_allowed(self):
        with self._locked() as stream:
            state = self._state_for_write()
            if not state["terminals_locked"] or set(state["terminals"]) != set(state["cases"]):
                raise ValueError("all case terminals must lock before reveal")
            self._write(state)
            return True

    def state(self):
        with self._locked() as stream:
            state = self._state_for_write()
            self._write(state)
            return deepcopy(state)
