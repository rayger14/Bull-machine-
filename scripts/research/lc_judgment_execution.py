"""Versioned, fail-closed execution controller for the frozen LC sample.

The controller owns persistence and validation around an external role bridge;
it never invokes a model.  Reservations are durably recorded before their
packet is returned, captures preserve exact bytes and runtime-return records,
and outcomes stay unavailable until every frozen case has a terminal.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import math
import os
from pathlib import Path
import tempfile

from scripts.research.assessment_evidence_guard import (
    _canonical,
    build_envelope,
    validate_runtime_returns,
)
from scripts.research.lc_judgment_runner import (
    EXPECTED_CASES,
    START_FLOOR_CREDITS,
    START_SNAPSHOT_MAX_AGE_SECONDS,
    JudgmentRunner,
    _digest,
    _load,
    _save_equal,
    _sha,
    _verify_lock,
)


VERSION = "lc_judgment_execution_v1"
CAMPAIGN_ID = "lc_judgment_filtered20_runtime_v2"
REQUESTED_MODEL = "gpt-6-astra"
ROLE_INSTRUCTIONS = Path(
    "results/lc_context_discrimination_2026_09_15/role_launch_instructions.md"
).resolve()
_ROLES = ("specialist", "reviewer")


def _save_bytes_equal(path, raw):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    if not isinstance(raw, bytes):
        raise ValueError("raw response must be bytes")
    if path.exists():
        if path.read_bytes() != raw:
            raise ValueError("immutable runtime artifact differs: " + str(path))
        return path
    fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(raw); stream.flush(); os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != raw:
                raise ValueError("concurrent unequal runtime artifact")
    finally:
        os.unlink(temporary)
    return path


def _artifact_lock(path, files, **fields):
    body = dict(fields)
    body["files"] = {str(Path(item).resolve()): _sha(item) for item in files}
    value = dict(body, sha256=_digest(body))
    _save_equal(path, value)
    return value


class JudgmentExecution(JudgmentRunner):
    """One-owner controller for the complete frozen assessment lifecycle."""

    def _runtime_dependencies(self):
        paths = super()._runtime_dependencies() + [Path(__file__).resolve(), ROLE_INSTRUCTIONS]
        return list(dict.fromkeys(Path(path).resolve() for path in paths))

    def prepare(self):
        """Clone all 20 jobs and freeze executable runtime dependencies."""
        self._require_owner()
        _, _, _, roster = self._inputs()
        cases = []; runtime_files = self._runtime_dependencies()
        for case_id in roster:
            source_path = self.evidence_dir / f"{case_id}_source_request.json"
            packet_path = self.evidence_dir / f"{case_id}_specialist_packet.json"
            envelope_path = self.evidence_dir / f"{case_id}_specialist_envelope.json"
            source = _load(source_path)
            if source.get("case_id") != case_id:
                raise ValueError("source request case differs from roster")
            job = self._job(case_id); job.prepare(source)
            wrapper = {"case_id": case_id, "plan": deepcopy(source["plan"]),
                       "request": job.role_request("specialist")}
            envelope = build_envelope(wrapper, max_chunk_bytes=4096)
            if (_canonical(_load(packet_path)) != _canonical(wrapper)
                    or _canonical(_load(envelope_path)) != _canonical(envelope)):
                raise ValueError("frozen specialist wrapper or envelope differs")
            cases.append({"case_id": case_id,
                          "job_directory": str((self.runtime_dir / "jobs" / case_id).resolve()),
                          **self._binding(job)})
            runtime_files.extend([source_path, packet_path, envelope_path,
                                  self.runtime_dir / "jobs" / case_id / "request.json"])
        manifest = {"schema_version": "lc_campaign_contract_v1",
                    "campaign_id": CAMPAIGN_ID, "cases": cases}
        body = {
            "schema_version": VERSION,
            "preparation_binding_sha256": _sha(
                self.preparation_dir / "preparation_binding.json"),
            "evidence_lock_sha256": _sha(self.evidence_dir / "evidence_lock.json"),
            "roster": roster,
            "manifest": manifest,
            "files": {str(path.resolve()): _sha(path) for path in runtime_files},
            "roles_enabled": False,
            "software_ready": True,
            "capture_supported": True,
            "role_instructions": str(ROLE_INSTRUCTIONS),
        }
        _save_equal(self.runtime_dir / "runtime_lock.json",
                    dict(body, sha256=_digest(body)))
        self._ledger.freeze(manifest)
        return {"prepared": True, "roster_count": len(roster),
                "software_ready": True, "roles_enabled": False}

    def verify(self):
        self._require_owner()
        _, _, _, roster = self._inputs()
        runtime = _verify_lock(self.runtime_dir / "runtime_lock.json")
        if (runtime.get("schema_version") != VERSION
                or runtime.get("roster") != roster
                or runtime.get("roles_enabled") is not False
                or runtime.get("software_ready") is not True
                or runtime.get("capture_supported") is not True
                or runtime.get("role_instructions") != str(ROLE_INSTRUCTIONS)):
            raise ValueError("runtime lock differs from execution contract")
        if (runtime.get("preparation_binding_sha256") != _sha(
                self.preparation_dir / "preparation_binding.json")
                or runtime.get("evidence_lock_sha256") != _sha(
                    self.evidence_dir / "evidence_lock.json")):
            raise ValueError("runtime upstream binding differs")
        state = self._ledger._read()
        if state is None or state["manifest"] != runtime["manifest"]:
            raise ValueError("runtime ledger manifest differs")
        self._verify_generated_locks()
        return {"roster_count": len(roster), "software_ready": True,
                "roles_enabled": (self.runtime_dir / "start_authorization.json").exists()}

    def _verify_generated_locks(self):
        for pattern in ("reservations/*/reservation_lock.json",
                        "captures/*/*/capture_lock.json",
                        "captures/*/*/processing_lock.json",
                        "reviewers/*/review_preparation_lock.json"):
            for path in self.runtime_dir.glob(pattern):
                _verify_lock(path)
        for name in ("terminal_lock.json", "reveal_gate.json"):
            path = self.runtime_dir / name
            if path.exists():
                _verify_lock(path)

    def _budget_snapshot(self, snapshot, *, start=False):
        if (not isinstance(snapshot, dict)
                or set(snapshot) != {"balance_credits", "observed_at", "source"}):
            raise ValueError("exact budget snapshot required")
        balance = snapshot["balance_credits"]
        if (isinstance(balance, bool) or not isinstance(balance, (int, float))
                or not math.isfinite(balance) or balance <= 0):
            raise ValueError("positive observed balance required")
        if start and balance < START_FLOOR_CREDITS:
            raise ValueError("campaign start requires at least 1900 credits")
        try:
            observed = datetime.fromisoformat(snapshot["observed_at"].replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("aware budget observation time required") from exc
        now = self._utc_now()
        if observed.tzinfo is None or now.tzinfo is None:
            raise ValueError("aware budget clocks required")
        age = (now.astimezone(timezone.utc)
               - observed.astimezone(timezone.utc)).total_seconds()
        if age < 0 or age > START_SNAPSHOT_MAX_AGE_SECONDS:
            raise ValueError("fresh budget snapshot within five minutes required")
        if not isinstance(snapshot["source"], str) or not snapshot["source"].strip():
            raise ValueError("budget snapshot source required")
        return deepcopy(snapshot)

    def authorize_start(self, snapshot, readiness, *, max_observed_balance_drop=None):
        """Freeze independent readiness and the one-time 1900-credit start gate."""
        self.verify()
        budget = self._budget_snapshot(snapshot, start=True)
        required = {"approved", "reviewer", "reviewed_at", "scope", "outcomes_accessed"}
        if (not isinstance(readiness, dict) or set(readiness) != required
                or readiness["approved"] is not True
                or readiness["outcomes_accessed"] is not False
                or readiness["scope"] != VERSION
                or not isinstance(readiness["reviewer"], str)
                or not readiness["reviewer"].strip()):
            raise ValueError("exact independent readiness approval required")
        try:
            reviewed = datetime.fromisoformat(readiness["reviewed_at"].replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("aware readiness review time required") from exc
        if reviewed.tzinfo is None:
            raise ValueError("aware readiness review time required")
        if max_observed_balance_drop is not None:
            if (isinstance(max_observed_balance_drop, bool)
                    or not isinstance(max_observed_balance_drop, (int, float))
                    or not math.isfinite(max_observed_balance_drop)
                    or max_observed_balance_drop <= 0
                    or max_observed_balance_drop > budget["balance_credits"]):
                raise ValueError("positive observed-balance-drop allowance required")
        body = {"schema_version": VERSION, "budget_snapshot": budget,
                "readiness": deepcopy(readiness),
                "max_observed_balance_drop": max_observed_balance_drop}
        token = _digest(body)
        authorization = dict(body, authorization_token=token)
        _save_equal(self.runtime_dir / "start_authorization.json", authorization)
        return token

    def _authorization(self, token=None, *, require_token=False):
        path = self.runtime_dir / "start_authorization.json"
        if not path.exists():
            raise ValueError("campaign start is not authorized")
        value = _load(path)
        body = {key: item for key, item in value.items() if key != "authorization_token"}
        if value.get("authorization_token") != _digest(body):
            raise ValueError("start authorization binding changed")
        if require_token and token != value["authorization_token"]:
            raise ValueError("start authorization token required for first reservation")
        return value

    def _check_reservation_budget(self, snapshot, authorization):
        current = self._budget_snapshot(snapshot)
        start = authorization["budget_snapshot"]["balance_credits"]
        observed_drop = max(0.0, float(start) - float(current["balance_credits"]))
        ceiling = authorization["max_observed_balance_drop"]
        if ceiling is not None and observed_drop > ceiling:
            raise ValueError("observed balance-drop allowance exhausted")
        return current, observed_drop

    def _role_material(self, case_id, role):
        if role == "specialist":
            packet_path = self.evidence_dir / f"{case_id}_specialist_packet.json"
            envelope_path = self.evidence_dir / f"{case_id}_specialist_envelope.json"
        elif role == "reviewer":
            packet_path = self.runtime_dir / "reviewers" / case_id / "packet.json"
            envelope_path = self.runtime_dir / "reviewers" / case_id / "envelope.json"
            _verify_lock(self.runtime_dir / "reviewers" / case_id
                         / "review_preparation_lock.json")
        else:
            raise ValueError("unknown role")
        packet = _load(packet_path); envelope = _load(envelope_path)
        if _canonical(envelope) != _canonical(build_envelope(packet, max_chunk_bytes=4096)):
            raise ValueError("role packet envelope differs")
        return packet_path, envelope_path, packet, envelope

    def reserve_role(self, case_id, role, budget_snapshot, *, authorization_token=None):
        """Durably consume the sole role attempt before returning dispatch data."""
        self.verify()
        state = self._ledger._read()
        try:
            case = state["cases"][case_id]
        except KeyError as exc:
            raise ValueError("unknown case") from exc
        if (case["decision_path"]["end"] is not None
                or self._terminal_path(case_id).exists()):
            raise ValueError("finalized case forbids role reservation")
        first = sum(state["budgets"].values()) == 0
        authorization = self._authorization(authorization_token, require_token=first)
        if first:
            self._budget_snapshot(authorization["budget_snapshot"], start=True)
        current, observed_drop = self._check_reservation_budget(
            budget_snapshot, authorization)
        packet_path, envelope_path, packet, envelope = self._role_material(case_id, role)
        record = self._ledger.start_attempt(case_id, role)
        directory = self.runtime_dir / "reservations" / record["attempt_id"]
        reservation = {
            "schema_version": VERSION,
            "case_id": case_id,
            "role": role,
            "attempt_id": record["attempt_id"],
            "runtime_id": self.runtime_id,
            "budget_snapshot": current,
            "observed_balance_drop_since_start": observed_drop,
            "packet_path": str(packet_path.resolve()),
            "envelope_path": str(envelope_path.resolve()),
            "role_instructions_path": str(ROLE_INSTRUCTIONS),
            "wrapper": packet,
            "envelope": envelope,
            "dispatch_authorized": True,
        }
        reservation_path = _save_equal(directory / "reservation.json", reservation)
        _artifact_lock(directory / "reservation_lock.json", [reservation_path])
        return deepcopy(reservation)

    def _reservation(self, case_id, role, state):
        try:
            record = state["cases"][case_id]["roles"][role]
        except KeyError as exc:
            raise ValueError("unknown case or role") from exc
        if record["attempt_id"] is None:
            raise ValueError("role was not reserved")
        path = self.runtime_dir / "reservations" / record["attempt_id"]
        _verify_lock(path / "reservation_lock.json")
        value = _load(path / "reservation.json")
        if (value.get("case_id") != case_id or value.get("role") != role
                or value.get("attempt_id") != record["attempt_id"]):
            raise ValueError("reservation binding differs")
        return value, record

    @staticmethod
    def _metadata(role, metadata):
        required = {"agent_id", "requested_model", "actual_model",
                    "actual_runtime_verified"}
        if (not isinstance(metadata, dict) or set(metadata) != required
                or not isinstance(metadata["agent_id"], str)
                or not metadata["agent_id"].strip()
                or metadata["requested_model"] != REQUESTED_MODEL
                or type(metadata["actual_runtime_verified"]) is not bool
                or (metadata["actual_model"] is not None
                    and (not isinstance(metadata["actual_model"], str)
                         or not metadata["actual_model"].strip()))):
            raise ValueError("exact declared role metadata required")
        return deepcopy(metadata)

    def _agent_is_unique(self, agent_id, current_path):
        for path in self.runtime_dir.glob("captures/*/*/metadata.json"):
            if path.resolve() != current_path.resolve() and _load(path).get("agent_id") == agent_id:
                return False
        return True

    def capture_role(self, case_id, role, raw_bytes, metadata, runtime_returns):
        """Persist exact role returns, validate transport, and bind the job stage."""
        self.verify()
        if role not in _ROLES:
            raise ValueError("unknown role")
        state = self._ledger.state()
        if (state["terminals_locked"]
                or (self.runtime_dir / "reveal_gate.json").exists()):
            raise ValueError("terminal lock forbids capture")
        if (case_id in state["cases"]
                and state["cases"][case_id]["decision_path"]["end"] is not None):
            raise ValueError("finalized case forbids capture")
        reservation, record = self._reservation(case_id, role, state)
        meta = self._metadata(role, metadata)
        directory = self.runtime_dir / "captures" / case_id / role
        response_path = _save_bytes_equal(directory / "response.bin", raw_bytes)
        metadata_path = _save_equal(directory / "metadata.json", meta)
        returns_path = _save_equal(directory / "runtime_returns.json", runtime_returns)
        capture_lock_path = directory / "capture_lock.json"
        _artifact_lock(capture_lock_path, [response_path, metadata_path, returns_path],
                       case_id=case_id, role=role,
                       attempt_id=record["attempt_id"])
        validation = validate_runtime_returns(
            reservation["wrapper"], reservation["envelope"], runtime_returns,
            case_id=case_id, role=role)
        unique = self._agent_is_unique(meta["agent_id"], metadata_path)
        transport = bool(validation["valid"]
                         and meta["actual_runtime_verified"] and unique)
        try:
            raw = raw_bytes.decode("utf-8")
        except UnicodeError as exc:
            self._ledger.finish_attempt(
                case_id, role, {"kind": "external_failure",
                                "reason": "role_response_not_utf8"})
            raise ValueError("role response must be exact UTF-8") from exc
        processing = {
            "metadata": meta,
            "validation": validation,
            "declared_agent_unique_across_campaign": unique,
            "raw_response_present": True,
            "transport_valid": transport,
            "authority": ("Runtime validity and identity/model are declared transport truth; "
                          "outer rendering and model attention remain unverified."),
        }
        processing_path = _save_equal(directory / "processing.json", processing)
        provenance = {
            "role": role,
            "agent_id": meta["agent_id"],
            "requested_model": meta["requested_model"],
            "actual_model": meta["actual_model"],
            "capture_sha256": _sha(capture_lock_path),
            "transport_valid": transport,
        }
        job = self._job(case_id)
        job.capture(role, raw, provenance)
        job_stage = self.runtime_dir / "jobs" / case_id / f"{role}.json"
        processing_lock_path = directory / "processing_lock.json"
        _artifact_lock(processing_lock_path,
                       [capture_lock_path, processing_path, job_stage],
                       case_id=case_id, role=role)
        stage = job._read()[role]
        delivered = {"kind": "delivered",
                     "job_directory": str((self.runtime_dir / "jobs" / case_id).resolve()),
                     "raw_response_sha256": stage["payload"]["raw_sha256"],
                     "capture_sha256": stage["sha256"]}
        finish = self._ledger.finish_attempt(case_id, role, delivered)
        return {**deepcopy(processing), "ledger_result": finish}

    def prepare_reviewer(self, case_id):
        """Freeze a critic packet only for an on-time, eligible specialist."""
        self.verify()
        state = self._ledger.state()
        try:
            status = state["cases"][case_id]["roles"]["specialist"]["status"]
        except KeyError as exc:
            raise ValueError("unknown case") from exc
        if status != "delivered":
            terminal = self.finalize_failure(case_id, "specialist_" + status)
            return {"reviewer_required": False, "terminal": terminal}
        job = self._job(case_id)
        try:
            request = job.role_request("reviewer")
        except ValueError:
            job.skip_review(); job.lock_grade()
            terminal = self.finalize_grade(case_id)
            return {"reviewer_required": False, "terminal": terminal}
        source = _load(self.evidence_dir / f"{case_id}_source_request.json")
        packet = {"case_id": case_id, "plan": deepcopy(source["plan"]),
                  "request": request}
        envelope = build_envelope(packet, max_chunk_bytes=4096)
        directory = self.runtime_dir / "reviewers" / case_id
        packet_path = _save_equal(directory / "packet.json", packet)
        envelope_path = _save_equal(directory / "envelope.json", envelope)
        lock = _artifact_lock(directory / "review_preparation_lock.json",
                              [packet_path, envelope_path,
                               self.runtime_dir / "jobs" / case_id / "specialist.json",
                               self.runtime_dir / "captures" / case_id / "specialist"
                               / "processing_lock.json"],
                              case_id=case_id)
        return {"reviewer_required": True, "wrapper": packet,
                "envelope": envelope, "packet_path": str(packet_path.resolve()),
                "envelope_path": str(envelope_path.resolve()), "lock": lock}

    def _terminal_path(self, case_id):
        return self.runtime_dir / "terminals" / f"{case_id}.json"

    def finalize_grade(self, case_id):
        self.verify()
        state = self._ledger.state()
        case = state["cases"].get(case_id)
        if case is None:
            raise ValueError("unknown case")
        if any(item["status"] in {"active", "timeout", "external_failure"}
               for item in case["roles"].values()):
            raise ValueError("failed, timed-out, or active role cannot publish grade")
        job = self._job(case_id); job.lock_grade()
        grade_stage = job._read()["grade"]
        terminal = {"kind": "published_grade",
                    "job_directory": str((self.runtime_dir / "jobs" / case_id).resolve()),
                    "grade_sha256": grade_stage["sha256"]}
        _save_equal(self._terminal_path(case_id), terminal)
        path = self._ledger.finalize_case(case_id, terminal)
        return dict(terminal, decision_path=path)

    def finalize_failure(self, case_id, reason):
        self.verify()
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("failure reason required")
        state = self._ledger.state()
        if case_id not in state["cases"]:
            raise ValueError("unknown case")
        if any(item["status"] == "active"
               for item in state["cases"][case_id]["roles"].values()):
            raise ValueError("active role prevents failure finalization")
        terminal = {"kind": "external_failure", "reason": reason}
        _save_equal(self._terminal_path(case_id), terminal)
        path = self._ledger.finalize_case(case_id, terminal)
        return dict(terminal, decision_path=path)

    def finish_failure(self, case_id, role, reason):
        """Record a current dispatch failure and terminalize the affected case."""
        self.verify()
        if role not in _ROLES or not isinstance(reason, str) or not reason.strip():
            raise ValueError("role and failure reason required")
        state = self._ledger.state()
        try:
            record = state["cases"][case_id]["roles"][role]
        except KeyError as exc:
            raise ValueError("unknown case or role") from exc
        failure = {"kind": "external_failure", "reason": reason}
        if record["status"] == "active":
            reservation, _ = self._reservation(case_id, role, state)
            if reservation["runtime_id"] != self.runtime_id:
                raise ValueError("current runtime does not own active reservation")
            self._ledger.finish_attempt(case_id, role, failure)
        elif record["status"] == "external_failure":
            if record["result"] != failure:
                raise ValueError("immutable dispatch failure differs")
        else:
            raise ValueError("active or matching failed role required")
        return self.finalize_failure(case_id, role + "_" + reason)

    def lock_terminals(self):
        self.verify()
        _, _, _, roster = self._inputs()
        terminals = {case_id: _load(self._terminal_path(case_id)) for case_id in roster}
        locked = self._ledger.lock_terminals(terminals)
        files = [self._terminal_path(case_id) for case_id in roster]
        ledger_path = self.runtime_dir / "ledger" / "ledger.json"
        _artifact_lock(self.runtime_dir / "terminal_lock.json", [*files, ledger_path],
                       roster=roster)
        return locked

    def assert_reveal_allowed(self):
        self.verify()
        allowed = self._ledger.assert_reveal_allowed()
        terminal_lock = self.runtime_dir / "terminal_lock.json"
        if not terminal_lock.exists():
            raise ValueError("terminal artifact lock required before reveal")
        _verify_lock(terminal_lock)
        ledger_path = self.runtime_dir / "ledger" / "ledger.json"
        _artifact_lock(self.runtime_dir / "reveal_gate.json",
                       [terminal_lock, ledger_path], allowed=True)
        return allowed

    def ledger_state(self):
        self.verify()
        return deepcopy(self._ledger.state())

    def status(self):
        self._require_owner()
        if not (self.runtime_dir / "runtime_lock.json").exists():
            return {"phase": "unprepared", "software_ready": False,
                    "roles_enabled": False}
        verified = self.verify(); state = self._ledger.state()
        if (self.runtime_dir / "reveal_gate.json").exists():
            phase = "reveal_allowed"
        elif state["terminals_locked"]:
            phase = "terminals_locked"
        elif (self.runtime_dir / "start_authorization.json").exists():
            phase = "authorized"
        else:
            phase = "prepared"
        return {"phase": phase, **verified, "budgets": deepcopy(state["budgets"]),
                "terminals_locked": state["terminals_locked"],
                "terminal_count": len(state["terminals"])}
