"""Fail-closed runtime foundation for the frozen 20-case LC judgment sample.

This version verifies preparation, clones exact requests, freezes one persistent
ledger per controller instance, and proves durable reservation semantics.  It
does not implement capture, critic routing, terminal publication, or reveal, so
the real namespace cannot reserve or dispatch a role.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import uuid

from scripts.research.assessment_evidence_guard import _canonical, build_envelope
from scripts.research.lc_campaign_contract import CampaignLedger
from scripts.research.lc_published_jobs import PublishedContextResearchJob


VERSION = "lc_judgment_runner_foundation_v1"
START_FLOOR_CREDITS = 1900.0
START_SNAPSHOT_MAX_AGE_SECONDS = 300
EXPECTED_CASES = 20


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _sha(path):
    path = Path(path)
    if not path.exists():
        raise ValueError("locked file missing: " + str(path))
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _load(path):
    path = Path(path)
    try:
        raw = path.read_bytes(); value = json.loads(raw)
    except (OSError, ValueError) as exc:
        raise ValueError("missing or malformed artifact: " + str(path)) from exc
    if _canonical(value).encode("ascii") != raw:
        raise ValueError("artifact is not canonical: " + str(path))
    return value


def _verify_digest(path):
    value = _load(path)
    if not isinstance(value, dict) or "sha256" not in value:
        raise ValueError("hashed artifact required: " + str(path))
    body = {key: item for key, item in value.items() if key != "sha256"}
    if value["sha256"] != _digest(body):
        raise ValueError("artifact binding changed: " + str(path))
    return value


def _verify_lock(path):
    value = _verify_digest(path)
    for field in ("files", "evidence_files", "expected_fixed_hashes"):
        mapping = value.get(field, {})
        if not isinstance(mapping, dict):
            raise ValueError("invalid lock mapping: " + field)
        for name, expected in mapping.items():
            if _sha(name) != expected:
                raise ValueError("locked file changed: " + name)
    return value


def _save_equal(path, value):
    """Atomically publish once; exact equal replay is the only overwrite."""
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    raw = _canonical(value).encode("ascii")
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


class JudgmentRunner:
    """Own one runtime lease and one persistent ``CampaignLedger`` instance."""

    def __init__(self, runtime_dir, *, preparation_dir,
                 job_factory=PublishedContextResearchJob,
                 ledger_factory=CampaignLedger, runtime_id=None,
                 utc_now=lambda: datetime.now(timezone.utc), dry_run=False):
        self.runtime_dir = Path(runtime_dir).resolve()
        self.preparation_dir = Path(preparation_dir).resolve()
        self.evidence_dir = self.preparation_dir / "evidence"
        self.runtime_dir.mkdir(parents=True, exist_ok=True)
        self.runtime_id = runtime_id or uuid.uuid4().hex
        if not isinstance(self.runtime_id, str) or not self.runtime_id:
            raise ValueError("runtime_id must be nonempty")
        self._job_factory = job_factory
        self._utc_now = utc_now
        self._dry_run = dry_run
        self._owner = (self.runtime_dir / ".owner.lock").open("a+")
        try:
            fcntl.flock(self._owner.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            self._owner.close()
            raise ValueError("runtime is already owned by another controller") from exc
        try:
            self._ledger = ledger_factory(
                self.runtime_dir / "ledger", runtime_id=self.runtime_id)
        except Exception:
            self.close()
            raise

    def close(self):
        owner = getattr(self, "_owner", None)
        if owner is not None and not owner.closed:
            fcntl.flock(owner.fileno(), fcntl.LOCK_UN)
            owner.close()

    def _require_owner(self):
        if self._owner.closed:
            raise ValueError("runtime ownership has been released")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        self.close()

    def _inputs(self):
        prepared = _verify_digest(self.preparation_dir / "judgment_prepare.json")
        evidence = _verify_lock(self.evidence_dir / "evidence_lock.json")
        binding = _verify_lock(self.preparation_dir / "preparation_binding.json")
        roster = _load(self.evidence_dir / "roster.json")
        if (not isinstance(roster, list) or len(roster) != EXPECTED_CASES
                or len(set(roster)) != EXPECTED_CASES
                or prepared.get("roster") != roster
                or evidence.get("roster") != roster
                or binding.get("roster") != roster):
            raise ValueError("exact matching 20-case roster required")
        if (prepared.get("roles_enabled") is not False
                or evidence.get("roles_enabled") is not False
                or binding.get("roles_enabled") is not False
                or evidence.get("outcome_reveal_authorized") is not False
                or binding.get("outcome_reveal_authorized") is not False):
            raise ValueError("preparation must remain role/reveal disabled")
        return prepared, evidence, binding, roster

    def _job(self, case_id):
        return self._job_factory(self.runtime_dir / "jobs" / case_id)

    @staticmethod
    def _binding(job):
        if hasattr(job, "campaign_binding"):
            return job.campaign_binding()
        stages = job._read(); payload = stages["request"]["payload"]
        return {"source_request_sha256": payload["source_request_sha256"],
                "role_request_sha256": payload["role_request_sha256"]}

    def _runtime_dependencies(self):
        import scripts.research.assessment_evidence_guard as evidence_guard
        import scripts.research.lc_campaign_contract as campaign_contract
        import scripts.research.lc_published_jobs as published_jobs
        import scripts.research.master_research_jobs as research_jobs
        return [Path(__file__).resolve(), Path(evidence_guard.__file__).resolve(),
                Path(campaign_contract.__file__).resolve(),
                Path(published_jobs.__file__).resolve(),
                Path(research_jobs.__file__).resolve()]

    def prepare(self):
        """Verify frozen inputs, clone exact jobs, and freeze the runtime manifest."""
        self._require_owner()
        _, _, _, roster = self._inputs()
        cases = []; runtime_files = self._runtime_dependencies()
        for case_id in roster:
            source_path = self.evidence_dir / (case_id + "_source_request.json")
            packet_path = self.evidence_dir / (case_id + "_specialist_packet.json")
            envelope_path = self.evidence_dir / (case_id + "_specialist_envelope.json")
            source = _load(source_path)
            if source.get("case_id") != case_id:
                raise ValueError("source request case differs from roster")
            job = self._job(case_id); job.prepare(source)
            binding = self._binding(job)
            wrapper = {"case_id": case_id, "plan": deepcopy(source["plan"]),
                       "request": job.role_request("specialist")}
            if (_canonical(_load(packet_path)) != _canonical(wrapper)
                    or _canonical(_load(envelope_path)) != _canonical(
                        build_envelope(wrapper, max_chunk_bytes=4096))):
                raise ValueError("frozen specialist wrapper or envelope differs")
            cases.append({"case_id": case_id,
                          "job_directory": str((self.runtime_dir / "jobs" / case_id)),
                          **binding})
            runtime_files.extend([source_path, packet_path, envelope_path,
                                  self.runtime_dir / "jobs" / case_id / "request.json"])
        manifest = {"schema_version": "lc_campaign_contract_v1",
                    "campaign_id": "lc_judgment_filtered20_runtime_v1",
                    "cases": cases}
        body = {
            "schema_version": VERSION,
            "preparation_binding_sha256": _sha(
                self.preparation_dir / "preparation_binding.json"),
            "evidence_lock_sha256": _sha(self.evidence_dir / "evidence_lock.json"),
            "roster": roster,
            "manifest": manifest,
            "files": {str(path.resolve()): _sha(path) for path in runtime_files},
            "roles_enabled": False,
            "launch_ready": False,
            "capture_supported": False,
        }
        _save_equal(self.runtime_dir / "runtime_lock.json",
                    dict(body, sha256=_digest(body)))
        self._ledger.freeze(manifest)
        return {"prepared": True, "roster_count": len(roster),
                "launch_ready": False, "roles_enabled": False}

    def verify(self):
        """Reopen frozen preparation, runtime files, jobs, and ledger read-only."""
        self._require_owner()
        _, _, _, roster = self._inputs()
        runtime = _verify_lock(self.runtime_dir / "runtime_lock.json")
        if (runtime.get("schema_version") != VERSION or runtime.get("roster") != roster
                or runtime.get("roles_enabled") is not False
                or runtime.get("launch_ready") is not False
                or runtime.get("capture_supported") is not False):
            raise ValueError("runtime lock differs from fail-closed foundation")
        if (runtime.get("preparation_binding_sha256") != _sha(
                self.preparation_dir / "preparation_binding.json")
                or runtime.get("evidence_lock_sha256") != _sha(
                    self.evidence_dir / "evidence_lock.json")):
            raise ValueError("runtime upstream binding differs")
        state = self._ledger._read()
        if state is None or state["manifest"] != runtime["manifest"]:
            raise ValueError("runtime ledger manifest differs")
        return {"roster_count": len(roster), "launch_ready": False,
                "roles_enabled": False}

    def ledger_state(self):
        self.verify()
        return deepcopy(self._ledger._read())

    def authorize_start(self, snapshot):
        """Freeze a fresh campaign-start balance observation; authorize no dispatch."""
        self.verify()
        if (not isinstance(snapshot, dict)
                or set(snapshot) != {"balance_credits", "observed_at", "source"}):
            raise ValueError("exact budget snapshot required")
        balance = snapshot["balance_credits"]
        if (isinstance(balance, bool) or not isinstance(balance, (int, float))
                or not math.isfinite(balance) or balance < START_FLOOR_CREDITS):
            raise ValueError("campaign start requires at least 1900 credits")
        try:
            observed = datetime.fromisoformat(snapshot["observed_at"].replace("Z", "+00:00"))
        except (AttributeError, ValueError) as exc:
            raise ValueError("aware budget observation time required") from exc
        now = self._utc_now()
        if observed.tzinfo is None or now.tzinfo is None:
            raise ValueError("aware budget clocks required")
        age = (now.astimezone(timezone.utc) - observed.astimezone(timezone.utc)).total_seconds()
        if age < 0 or age > START_SNAPSHOT_MAX_AGE_SECONDS:
            raise ValueError("fresh budget snapshot within five minutes required")
        if not isinstance(snapshot["source"], str) or not snapshot["source"].strip():
            raise ValueError("budget snapshot source required")
        token = _digest(snapshot)
        _save_equal(self.runtime_dir / "authorizations" / (token + ".json"), snapshot)
        return token

    def _authorization(self, token):
        if not isinstance(token, str) or len(token) != 64:
            raise ValueError("fresh start authorization token required")
        snapshot = _load(self.runtime_dir / "authorizations" / (token + ".json"))
        if _digest(snapshot) != token:
            raise ValueError("start authorization binding changed")
        observed = datetime.fromisoformat(snapshot["observed_at"].replace("Z", "+00:00"))
        age = (self._utc_now().astimezone(timezone.utc)
               - observed.astimezone(timezone.utc)).total_seconds()
        if age < 0 or age > START_SNAPSHOT_MAX_AGE_SECONDS:
            raise ValueError("fresh start authorization expired")
        return snapshot

    def reserve_role(self, case_id, role, *, authorization_token=None):
        """Dry-run-only proof that ledger reservation precedes wrapper return."""
        self.verify()
        if not self._dry_run:
            raise ValueError("runner is not launch-ready; capture lifecycle is absent")
        if role != "specialist":
            raise ValueError("reviewer lifecycle is not implemented")
        state = self._ledger._read()
        budget_snapshot = None
        if sum(state["budgets"].values()) == 0:
            budget_snapshot = self._authorization(authorization_token)
        packet = _load(self.evidence_dir / (case_id + "_specialist_packet.json"))
        envelope = _load(self.evidence_dir / (case_id + "_specialist_envelope.json"))
        if _canonical(envelope) != _canonical(build_envelope(packet, max_chunk_bytes=4096)):
            raise ValueError("specialist envelope differs")
        record = self._ledger.start_attempt(case_id, role)
        reservation = {
            "schema_version": VERSION,
            "case_id": case_id,
            "role": role,
            "attempt_id": record["attempt_id"],
            "runtime_id": self.runtime_id,
            "budget_snapshot": budget_snapshot,
            "wrapper": packet,
            "envelope": envelope,
            "dispatch_authorized": False,
            "reason": "capture_lifecycle_not_implemented",
        }
        _save_equal(self.runtime_dir / "reservations" / record["attempt_id"]
                    / "reservation.json", reservation)
        return deepcopy(reservation)

    def _active_owner(self, case_id, role, record, state):
        path = self.runtime_dir / "reservations" / record["attempt_id"] / "reservation.json"
        if path.exists():
            return _load(path).get("runtime_id")
        if role == "specialist":
            start = state["cases"][case_id]["decision_path"]["start"]
            return None if start is None else start["runtime_id"]
        return None

    def status(self):
        self._require_owner()
        if not (self.runtime_dir / "runtime_lock.json").exists():
            return {"phase": "unprepared", "launch_ready": False,
                    "recovery_required": [], "timing_continuity": True}
        self.verify(); state = self._ledger._read()
        recovery = []
        continuity = True
        for case_id, case in state["cases"].items():
            start = case["decision_path"]["start"]
            if start is not None and start["runtime_id"] != self.runtime_id:
                continuity = False
            for role, record in case["roles"].items():
                if (record["status"] == "active"
                        and self._active_owner(case_id, role, record, state) != self.runtime_id):
                    recovery.append({"case_id": case_id, "role": role,
                                     "attempt_id": record["attempt_id"]})
        return {"phase": "prepared", "launch_ready": False,
                "recovery_required": recovery,
                "timing_continuity": continuity,
                "budgets": deepcopy(state["budgets"])}

    def recover_uncertain(self, case_id, role):
        """Explicitly consume an active reservation owned by an earlier runtime."""
        self.verify(); state = self._ledger._read()
        try:
            record = state["cases"][case_id]["roles"][role]
        except KeyError as exc:
            raise ValueError("unknown case or role") from exc
        if record["status"] != "active":
            raise ValueError("active uncertain reservation required")
        if self._active_owner(case_id, role, record, state) == self.runtime_id:
            raise ValueError("current runtime still owns active reservation")
        return self._ledger.finish_attempt(
            case_id, role,
            {"kind": "external_failure",
             "reason": "controller_restart_after_reservation"},
        )

    def capture_role(self, *args, **kwargs):
        self._require_owner()
        raise ValueError("capture lifecycle is not implemented; runner is not launch-ready")
