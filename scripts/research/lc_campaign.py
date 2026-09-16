"""Restartable offline controller for the consolidated LC campaign.

The controller never starts source replay or a model.  A lead process reserves
work before dispatch, publishes immutable Task-1/job artifacts, and calls the
explicit completion methods.  The only CLI operations are read-only status and
report; mutating operations require structured Python inputs.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import time

from scripts.research.assessment_evidence_guard import _canonical
from scripts.research.lc_campaign_contract import CampaignLedger, select_block


VERSION = "lc_campaign_controller_v2"
INVENTORY_VERSION = "lc_campaign_inventory_v1"
SOURCE_UNITS = tuple(f"{year:04d}-{month:02d}" for year in range(2024, 2027)
                     for month in range(1, 13) if (year, month) <= (2026, 7))
MAX_SOURCE_WORKERS = 2
MAX_SOURCE_HOURS = 18
SNAPSHOT_ID = "299193b77af10d1382bc02d537a8f919b1e2cd9a3fe5557543e4bb5184fb727d"
REQUIRED_REGISTRIES = (
    ("results/evidence_id_pilot_2026_09_12/prior_assessed_cases.json",
     "63beebb534ad387a667387b2a5e946fdf4072a8053d321b1b0b4c375c2472e71"),
    ("results/evidence_id_pilot_2026_09_12/hidden_controls.json",
     "49b272acfccaf216899f3c004ae19cb025a57b5911acf2ba90dc8a65f155344c"),
    ("results/isolated_entry_comparison_2026_09_14/cases.json",
     "1639efde6496dc3594e4809b4f82f50a189de03fa129015091a18e5add2f2e88"),
    ("results/lc_context_discrimination_2026_09_15/run_v1/cases.json",
     "800c4a97b9b030903fd16abc23cff481c75c96d0499b93e428d8c29939b080fd"),
    ("results/lc_jan19_validity_2026_09_15/run_v1/cases.json",
     "90a01d89462e52e6c7ea288b960bf1adac5c81162389359fd31d8539ca185255"),
)
_STATE_KEYS = {"version", "inventory", "sources", "source_attempts",
               "reserved_worker_hours", "prepared", "operations", "score", "sha256"}
_HEX = set("0123456789abcdef")


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _sha_file(path):
    path = Path(path)
    if not path.exists():
        return None
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _sha(value, label, *, nullable=False):
    if nullable and value is None:
        return
    if not isinstance(value, str) or len(value) != 64 or set(value) - _HEX:
        raise ValueError(label + " must be a lowercase SHA-256")


def _utc(value, label):
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (AttributeError, ValueError) as exc:
        raise ValueError(label + " must be an aware timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(label + " must be an aware timestamp")
    return parsed.astimezone(timezone.utc)


def _default_source_validator(source, month):
    from scripts.research.lc_campaign_source import _validate_source
    _validate_source(source, month)


def _default_job_loader(path):
    from scripts.research.lc_campaign_contract import _default_job_loader as load
    return load(path)


def _default_q1_projections():
    from scripts.research.lc_campaign_source import Q1_PROJECTIONS
    return deepcopy(Q1_PROJECTIONS)


def _manifest_files(source):
    """Merge top-level leaves plus nested replay leaves, rejecting conflicts."""
    expected = {}

    def merge(files, label):
        if not isinstance(files, dict):
            raise ValueError(label + " files must be a mapping")
        for path, wanted in files.items():
            _sha(wanted, label + " hash")
            if path in expected and expected[path] != wanted:
                raise ValueError("conflicting nested source_manifest.replay hash")
            expected[path] = wanted

    def nested(node):
        if isinstance(node, list):
            for child in node: nested(child)
        elif isinstance(node, dict):
            if "files" in node: merge(node["files"], "source_manifest.replay")
            for key, child in node.items():
                if key != "files": nested(child)

    for name in ("source_manifest", "code_manifest", "config_manifest"):
        manifest = source.get(name)
        if not isinstance(manifest, dict):
            raise ValueError("complete " + name + " required")
        merge(manifest.get("files"), name)
    nested(source["source_manifest"].get("replay"))
    return expected


class CampaignController:
    def __init__(self, run_dir, *, ledger_factory=CampaignLedger, scorer=None,
                 job_loader=_default_job_loader, source_validator=_default_source_validator,
                 file_hasher=_sha_file, required_registries=REQUIRED_REGISTRIES,
                 q1_projections=None, clock=time.time):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._ledger_factory, self._scorer = ledger_factory, scorer
        self._job_loader, self._source_validator = job_loader, source_validator
        self._file_hasher = file_hasher
        self._required_registries = tuple(required_registries)
        self._q1_projections = (_default_q1_projections() if q1_projections is None
                                else deepcopy(q1_projections))
        self._clock = clock
        self._path = self.run_dir / "controller.json"
        self._lock_path = self.run_dir / ".controller.lock"

    @contextmanager
    def _locked(self):
        with self._lock_path.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield

    def _blank(self):
        return {"version": VERSION, "inventory": None, "sources": {},
                "source_attempts": {}, "reserved_worker_hours": 0,
                "prepared": None, "operations": {}, "score": None}

    def _read(self, *, validate_external=True):
        if not self._path.exists():
            return None
        raw = self._path.read_bytes()
        try:
            state = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid controller state bytes") from exc
        body = {key: value for key, value in state.items() if key != "sha256"}
        if (_canonical(state).encode("ascii") != raw or set(state) != _STATE_KEYS
                or state.get("version") != VERSION or state.get("sha256") != _digest(body)):
            raise ValueError("controller state hash or canonical bytes changed")
        self._validate_state(state, validate_external=validate_external)
        return state

    def _write(self, state):
        body = deepcopy(state); body.pop("sha256", None)
        state = dict(body, sha256=_digest(body)); raw = _canonical(state).encode("ascii")
        fd, temporary = tempfile.mkstemp(prefix=".pending-controller-", dir=self.run_dir)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self._path)
            directory = os.open(self.run_dir, os.O_RDONLY)
            try: os.fsync(directory)
            finally: os.close(directory)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
        return deepcopy(state)

    def _state(self):
        return self._read() or self._blank()

    def _ledger(self):
        try:
            return self._ledger_factory(self.run_dir / "ledger", job_loader=self._job_loader)
        except TypeError:
            return self._ledger_factory(self.run_dir / "ledger")

    def _validate_inventory(self, inventory, *, external=True):
        keys = {"schema_version", "source_queue", "registries", "exposure_identities",
                "time_match_exclusions", "current_dependencies", "file_state_guards",
                "q1_receipts"}
        if not isinstance(inventory, dict) or set(inventory) != keys:
            raise ValueError("exact inventory schema required")
        if inventory["schema_version"] != INVENTORY_VERSION:
            raise ValueError("unsupported inventory schema")
        if inventory["source_queue"] != list(SOURCE_UNITS):
            raise ValueError("exact chronological 31-unit source queue required")
        registries = inventory["registries"]
        if (not isinstance(registries, list) or len(registries) != len(self._required_registries)
                or [(r.get("path"), r.get("sha256")) for r in registries]
                != list(self._required_registries)):
            raise ValueError("exact five registry paths and hashes required")
        for registry in registries:
            if set(registry) != {"path", "sha256", "exposures"} or not isinstance(registry["exposures"], list):
                raise ValueError("exact registry exposure schema required")
            if external and self._file_hasher(registry["path"]) != registry["sha256"]:
                raise ValueError("registry hash mismatch: " + registry["path"])
        identities = inventory["exposure_identities"]
        if not isinstance(identities, list): raise ValueError("exposure identities required")
        seen = {}
        for item in identities:
            if (not isinstance(item, dict) or set(item) != {"candidate_id", "decision_time", "provenance"}
                    or not isinstance(item["candidate_id"], str) or not item["candidate_id"]
                    or not isinstance(item["provenance"], list) or not item["provenance"]):
                raise ValueError("exact exposure identity and provenance required")
            decision = _utc(item["decision_time"], "exposure decision_time")
            if decision.minute or decision.second or decision.microsecond:
                raise ValueError("hourly exposure decision must be exact-hour")
            prior = seen.setdefault(item["candidate_id"], decision)
            if prior != decision: raise ValueError("conflicting exposure identity time")
        time_exclusions = inventory["time_match_exclusions"]
        required_times = {"2026-06-11T13:00:00+00:00", "2026-06-14T22:00:00+00:00"}
        if (not isinstance(time_exclusions, list) or
                {item.get("decision_time") for item in time_exclusions} != required_times):
            raise ValueError("exact two null-ID hourly time exclusions required")
        for item in time_exclusions:
            if set(item) != {"decision_time", "provenance"} or not item["provenance"]:
                raise ValueError("time exclusion provenance required")
            decision = _utc(item["decision_time"], "time exclusion")
            if decision.minute or decision.second or decision.microsecond:
                raise ValueError("time exclusion must be exact-hour")
        dependencies = inventory["current_dependencies"]
        if not isinstance(dependencies, list) or not dependencies:
            raise ValueError("current dependency identities required")
        receipts = inventory["q1_receipts"]
        if (not isinstance(receipts, list) or len(receipts) != 3
                or {r.get("month") for r in receipts} != {"2026-01", "2026-02", "2026-03"}):
            raise ValueError("exact three Q1 receipts required")
        guards = inventory["file_state_guards"]
        if not isinstance(guards, list) or len(guards) != 256:
            raise ValueError("exactly 256 file-state guards required")
        if len({item.get("path") for item in guards}) != 256:
            raise ValueError("file-state guard paths must be unique")
        if sum(item.get("sha256") is None for item in guards) != 2:
            raise ValueError("file-state guards must preserve exactly two expected-absent paths")
        for item, nullable in [(item, False) for item in dependencies] + [(item, True) for item in guards]:
            if not isinstance(item, dict) or set(item) != {"path", "sha256"} or not item["path"]:
                raise ValueError("exact file-state schema required")
            _sha(item["sha256"], "file-state hash", nullable=nullable)
            if external and self._file_hasher(item["path"]) != item["sha256"]:
                raise ValueError("file-state guard mismatch: " + item["path"])
        if external:
            for receipt in receipts: self._validate_q1_receipt(receipt)

    def _validate_q1_receipt(self, receipt):
        keys = {"month", "source_file", "source_file_sha256", "hourly_input_hash",
                "candidate_ids"}
        if (not isinstance(receipt, dict) or set(receipt) != keys
                or receipt["month"] not in {"2026-01", "2026-02", "2026-03"}):
            raise ValueError("exact saved-Q1 receipt required")
        _sha(receipt["source_file_sha256"], "saved Q1 source hash")
        _sha(receipt["hourly_input_hash"], "saved Q1 hourly input hash")
        projection = {key: receipt[key] for key in
                      ("source_file_sha256", "hourly_input_hash", "candidate_ids")}
        if projection != self._q1_projections.get(receipt["month"]):
            raise ValueError("saved Q1 receipt differs from reviewed projection")
        if self._file_hasher(receipt["source_file"]) != receipt["source_file_sha256"]:
            raise ValueError("saved Q1 source artifact hash mismatch")
        try: source = json.loads(Path(receipt["source_file"]).read_bytes())
        except (OSError, ValueError) as exc: raise ValueError("invalid saved Q1 source artifact") from exc
        self._source_validator(source, receipt["month"])
        ids = [row.get("candidate_id") for row in source.get("candidates", [])]
        if (source.get("hourly_input_hash") != receipt["hourly_input_hash"]
                or ids != receipt["candidate_ids"]):
            raise ValueError("saved Q1 source projection differs")
        for path, wanted in _manifest_files(source).items():
            if self._file_hasher(path) != wanted:
                raise ValueError("nested source manifest file hash mismatch: " + path)
        return source

    def _validate_source_receipt(self, receipt):
        keys = {"month", "directory", "source_sha256", "manifest_sha256"}
        if not isinstance(receipt, dict) or set(receipt) != keys or receipt["month"] not in SOURCE_UNITS:
            raise ValueError("exact source receipt required")
        _sha(receipt["source_sha256"], "source hash"); _sha(receipt["manifest_sha256"], "manifest hash")
        directory = Path(receipt["directory"])
        source_path, manifest_path = directory / "source.json", directory / "manifest.json"
        if self._file_hasher(str(source_path)) != receipt["source_sha256"]:
            raise ValueError("source artifact hash mismatch")
        if self._file_hasher(str(manifest_path)) != receipt["manifest_sha256"]:
            raise ValueError("source manifest hash mismatch")
        try:
            source, manifest = json.loads(source_path.read_bytes()), json.loads(manifest_path.read_bytes())
        except (OSError, ValueError) as exc:
            raise ValueError("invalid completed source artifacts") from exc
        self._source_validator(source, receipt["month"])
        expected_manifest = {"schema", "wrapper", "wrapper_sha256", "source_payload_schema",
                             "construction", "month", "seed", "start", "end_exclusive",
                             "source_file", "source_file_sha256", "candidate_count",
                             "input_limit_disclosures", "candidates"}
        if (set(manifest) != expected_manifest or manifest["schema"] != "lc-campaign-candidate-manifest-v1"
                or manifest["month"] != receipt["month"] or manifest["source_file"] != "source.json"
                or manifest["source_file_sha256"] != receipt["source_sha256"]
                or manifest["candidate_count"] != len(manifest["candidates"])
                or manifest["candidates"] != source.get("candidates")):
            raise ValueError("source candidate manifest differs from source payload")
        for path, wanted in _manifest_files(source).items():
            if self._file_hasher(path) != wanted:
                raise ValueError("nested source manifest file hash mismatch: " + path)
        return source, manifest

    def _validate_prepared(self, prepared):
        if not isinstance(prepared, dict): raise ValueError("invalid prepared state")
        keys = {"campaign_id", "roster", "selection", "manifest", "manifest_sha256",
                "packets", "packet_hashes", "requests", "request_hashes", "jobs",
                "curriculum", "curriculum_sha256"}
        if set(prepared) != keys or prepared["manifest_sha256"] != _digest(prepared["manifest"]):
            raise ValueError("prepared manifest binding changed")
        if prepared["curriculum_sha256"] != _digest(prepared["curriculum"]):
            raise ValueError("prepared curriculum binding changed")
        roster = prepared["roster"]
        if set(roster) != set(prepared["packets"]) or set(roster) != set(prepared["requests"]) or set(roster) != set(prepared["jobs"]):
            raise ValueError("prepared case artifacts differ from roster")
        for case_id in roster:
            if prepared["packet_hashes"].get(case_id) != _digest(prepared["packets"][case_id]):
                raise ValueError("prepared packet binding changed")
            if prepared["request_hashes"].get(case_id) != _digest(prepared["requests"][case_id]):
                raise ValueError("prepared request binding changed")
            job = self._job_loader(prepared["jobs"][case_id])
            binding = job.campaign_binding()
            case = next(item for item in prepared["manifest"]["cases"] if item["case_id"] == case_id)
            if binding != {"source_request_sha256": case["source_request_sha256"],
                           "role_request_sha256": case["role_request_sha256"]}:
                raise ValueError("published job binding differs from prepared manifest")
            if binding["source_request_sha256"] != prepared["request_hashes"][case_id]:
                raise ValueError("published source request binding differs")
            if _digest(job.role_request("specialist")) != binding["role_request_sha256"]:
                raise ValueError("published role request binding differs")
        ledger = self._ledger().state()
        if ledger["manifest"] != prepared["manifest"]:
            raise ValueError("ledger manifest differs from prepared controller state")
        return ledger

    def _validate_operations(self, state, ledger):
        for key, operation in state["operations"].items():
            if key == "terminal_lock":
                if (set(operation) != {"terminals_sha256"}
                        or operation["terminals_sha256"] != _digest(ledger["terminals"])):
                    raise ValueError("controller terminal-lock binding differs")
                continue
            if key.endswith(":finalize"):
                if set(operation) != {"terminal", "decision_path_sha256"}:
                    raise ValueError("invalid controller finalization binding")
                case_id = key[:-9]
                if (case_id not in ledger["cases"] or operation["decision_path_sha256"]
                        != ledger["cases"][case_id]["decision_path"]["path_sha256"]):
                    raise ValueError("controller finalization differs from ledger")
                continue
            if (not isinstance(operation, dict) or set(operation) != {"case_id", "role", "attempt_id",
                    "wrapper_bytes", "wrapper_sha256", "capture"}):
                raise ValueError("invalid controller role operation")
            case_id, role = operation["case_id"], operation["role"]
            if key != case_id + ":" + role or case_id not in ledger["cases"] or role not in ("specialist", "reviewer"):
                raise ValueError("controller role operation identity differs")
            if ledger["cases"][case_id]["roles"][role]["attempt_id"] != operation["attempt_id"]:
                raise ValueError("controller attempt differs from ledger")
            raw = operation["wrapper_bytes"]
            if (not isinstance(raw, str) or operation["wrapper_sha256"]
                    != hashlib.sha256(raw.encode("ascii")).hexdigest()):
                raise ValueError("controller exact-byte wrapper hash differs")
            try: wrapper = json.loads(raw)
            except ValueError as exc: raise ValueError("invalid controller wrapper bytes") from exc
            expected = {"case_id": case_id,
                        "plan": state["prepared"]["requests"][case_id].get("plan"),
                        "request": self._job_loader(state["prepared"]["jobs"][case_id]).role_request(role)}
            if _canonical(wrapper) != raw or wrapper != expected:
                raise ValueError("controller exact-byte wrapper binding differs")
            capture = operation["capture"]
            role_record = ledger["cases"][case_id]["roles"][role]
            if capture is None:
                continue
            if (isinstance(capture, dict) and capture.get("kind") == "delivered_bytes"):
                if (set(capture) != {"kind", "raw_response_sha256", "capture_sha256", "byte_length"}
                        or type(capture["byte_length"]) is not int or capture["byte_length"] < 0):
                    raise ValueError("invalid exact-byte capture schema")
                binding = self._job_loader(state["prepared"]["jobs"][case_id]).capture_binding(role)
                if (role_record["status"] != "delivered"
                        or any(capture.get(name) != value for name, value in binding.items())):
                    raise ValueError("controller capture differs from published job")
            elif (isinstance(capture, dict) and capture.get("kind") == "invalid_response_bytes"):
                if (set(capture) != {"kind", "raw_response_sha256", "byte_length", "reason"}
                        or type(capture["byte_length"]) is not int or capture["byte_length"] < 0
                        or capture["reason"] != "invalid_response_utf8"
                        or role_record["status"] != "external_failure"
                        or role_record["result"] != {"kind": "external_failure",
                                                     "reason": "invalid_response_utf8"}):
                    raise ValueError("invalid response-byte failure binding")
            elif (not isinstance(capture, dict) or set(capture) != {"kind", "reason"}
                  or capture["kind"] != "external_failure"
                  or role_record["status"] != "external_failure"
                  or role_record["result"] != capture):
                raise ValueError("invalid controller failure capture")

    def _validate_state(self, state, *, validate_external):
        if not isinstance(state["sources"], dict) or not isinstance(state["source_attempts"], dict):
            raise ValueError("invalid source state")
        if (isinstance(state["reserved_worker_hours"], bool)
                or not isinstance(state["reserved_worker_hours"], (int, float))
                or not math.isfinite(state["reserved_worker_hours"])
                or state["reserved_worker_hours"] < 0):
            raise ValueError("invalid reserved worker-hour total")
        if state["inventory"] is not None:
            self._validate_inventory(state["inventory"], external=validate_external)
        if set(state["sources"]) - set(SOURCE_UNITS): raise ValueError("unknown source completion")
        if validate_external:
            for receipt in state["sources"].values():
                (self._validate_q1_receipt(receipt) if "source_file" in receipt
                 else self._validate_source_receipt(receipt))
        active = 0; reserved = 0
        attempted_directories = set(); live_months = set()
        for attempt in state["source_attempts"].values():
            if set(attempt) != {"attempt_id", "month", "directory", "reserved_hours", "started_at",
                               "status", "finished_at", "elapsed_hours", "failure_reason"}:
                raise ValueError("invalid source attempt schema")
            if (attempt["month"] not in SOURCE_UNITS
                    or attempt["directory"] in attempted_directories):
                raise ValueError("duplicate or unknown source attempt identity")
            attempted_directories.add(attempt["directory"])
            if (isinstance(attempt["reserved_hours"], bool)
                    or not isinstance(attempt["reserved_hours"], (int, float))
                    or not math.isfinite(attempt["reserved_hours"]) or attempt["reserved_hours"] <= 0):
                raise ValueError("invalid source attempt reservation")
            for name in ("started_at", "finished_at"):
                value = attempt[name]
                if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value < 0):
                    raise ValueError("invalid source attempt clock")
            if attempt["status"] == "active":
                if (attempt["finished_at"] is not None or attempt["elapsed_hours"] is not None
                        or attempt["failure_reason"] is not None
                        or attempt["month"] in live_months):
                    raise ValueError("active source attempt has terminal timing")
                live_months.add(attempt["month"])
            elif attempt["status"] == "completed":
                elapsed = attempt["elapsed_hours"]
                if (attempt["finished_at"] is None or not isinstance(elapsed, (int, float))
                        or isinstance(elapsed, bool) or not 0 <= elapsed <= attempt["reserved_hours"]
                        or abs(elapsed - (attempt["finished_at"] - attempt["started_at"]) / 3600) > 1e-12
                        or attempt["month"] not in state["sources"]
                        or attempt["failure_reason"] is not None
                        or attempt["month"] in live_months):
                    raise ValueError("invalid completed source attempt timing")
                live_months.add(attempt["month"])
            elif attempt["status"] == "failed":
                elapsed = attempt["elapsed_hours"]
                if (attempt["finished_at"] is None or isinstance(elapsed, bool)
                        or not isinstance(elapsed, (int, float)) or not math.isfinite(elapsed)
                        or elapsed < 0
                        or abs(elapsed - (attempt["finished_at"] - attempt["started_at"]) / 3600) > 1e-12
                        or not isinstance(attempt["failure_reason"], str)
                        or not attempt["failure_reason"]):
                    raise ValueError("invalid failed source attempt timing")
            else:
                raise ValueError("invalid source attempt status")
            reserved += max(attempt["reserved_hours"], attempt["elapsed_hours"] or 0)
            active += attempt["status"] == "active"
        if active > MAX_SOURCE_WORKERS or abs(reserved - state["reserved_worker_hours"]) > 1e-9:
            raise ValueError("source capacity or budget ledger differs")
        if not isinstance(state["operations"], dict): raise ValueError("invalid operations state")
        if state["prepared"] is not None and validate_external:
            self._validate_operations(state, self._validate_prepared(state["prepared"]))
        score = state["score"]
        if score is not None:
            keys = {"terminal_state_sha256", "accounting_sha256", "outcome_sha256",
                    "bars_sha256", "result", "result_sha256"}
            if not isinstance(score, dict) or set(score) != keys:
                raise ValueError("invalid score binding schema")
            for name in keys - {"result"}:
                _sha(score[name], "score " + name)
            if score["result_sha256"] != _digest(score["result"]):
                raise ValueError("score result hash changed")

    def inventory(self, inventory):
        with self._locked():
            state = self._state(); self._validate_inventory(inventory)
            if state["inventory"] is not None and state["inventory"] != inventory:
                raise ValueError("immutable inventory differs")
            state["inventory"] = deepcopy(inventory)
            for receipt in inventory["q1_receipts"]:
                prior = state["sources"].get(receipt["month"])
                if prior is not None and prior != receipt: raise ValueError("immutable Q1 receipt differs")
                state["sources"][receipt["month"]] = deepcopy(receipt)
            self._write(state); return deepcopy(inventory)

    def reserve_source(self, month, directory, *, reserved_hours):
        with self._locked():
            state = self._state()
            if state["inventory"] is None: raise ValueError("inventory required before source")
            if month not in SOURCE_UNITS: raise ValueError("unknown source unit")
            if (month in state["sources"] or any(a["month"] == month and a["status"] == "active"
                                                 for a in state["source_attempts"].values())):
                raise ValueError("source unit already attempted; immutable collision/reuse forbidden")
            if (any(a["directory"] == str(directory) for a in state["source_attempts"].values())
                    or any(r.get("directory") == str(directory) for r in state["sources"].values())):
                raise ValueError("source output directory already bound to another unit")
            if (isinstance(reserved_hours, bool) or not isinstance(reserved_hours, (int, float))
                    or not math.isfinite(reserved_hours) or reserved_hours <= 0):
                raise ValueError("positive finite reserved worker hours required")
            if sum(a["status"] == "active" for a in state["source_attempts"].values()) >= MAX_SOURCE_WORKERS:
                raise ValueError("maximum two active source workers reached")
            if state["reserved_worker_hours"] + reserved_hours > MAX_SOURCE_HOURS:
                raise ValueError("source budget exhausted")
            if Path(directory).exists():
                raise ValueError("source output collision; immutable artifacts cannot be reused")
            started = self._clock()
            if (isinstance(started, bool) or not isinstance(started, (int, float))
                    or not math.isfinite(started) or started < 0):
                raise ValueError("source clock must be finite nonnegative seconds")
            attempt_id = _digest({"month": month, "directory": str(directory), "started_at": started})
            attempt = {"attempt_id": attempt_id, "month": month, "directory": str(directory),
                       "reserved_hours": reserved_hours, "started_at": started,
                       "status": "active", "finished_at": None, "elapsed_hours": None,
                       "failure_reason": None}
            state["source_attempts"][attempt_id] = attempt
            state["reserved_worker_hours"] += reserved_hours
            self._write(state); return deepcopy(attempt)

    def complete_source(self, month, attempt_id):
        with self._locked():
            state = self._state(); attempt = state["source_attempts"].get(attempt_id)
            if attempt is None or attempt["month"] != month or attempt["status"] != "active":
                raise ValueError("active matching source attempt required")
            directory = Path(attempt["directory"])
            receipt = {"month": month, "directory": str(directory),
                       "source_sha256": self._file_hasher(str(directory / "source.json")),
                       "manifest_sha256": self._file_hasher(str(directory / "manifest.json"))}
            self._validate_source_receipt(receipt)
            finished = self._clock(); elapsed = (finished - attempt["started_at"]) / 3600
            if elapsed < 0 or elapsed > attempt["reserved_hours"]:
                raise ValueError("source attempt exceeded its conservatively reserved budget")
            attempt["status"] = "completed"; attempt["finished_at"] = finished
            attempt["elapsed_hours"] = elapsed
            state["sources"][month] = receipt
            self._write(state); return deepcopy(receipt)

    def fail_source(self, month, attempt_id, reason):
        """Terminalize interrupted source work without refunding its reservation.

        A later attempt for the same month must use a new output directory and
        consumes a new reservation.  If interruption lasted beyond the original
        reservation, the overrun is charged too and blocks further reservations.
        """
        with self._locked():
            state = self._state(); attempt = state["source_attempts"].get(attempt_id)
            if (attempt is None or attempt["month"] != month or attempt["status"] != "active"):
                raise ValueError("active matching source attempt required")
            if not isinstance(reason, str) or not reason:
                raise ValueError("source failure reason required")
            finished = self._clock()
            if (isinstance(finished, bool) or not isinstance(finished, (int, float))
                    or not math.isfinite(finished) or finished < attempt["started_at"]):
                raise ValueError("source failure clock must not precede start")
            elapsed = (finished - attempt["started_at"]) / 3600
            extra = max(0, elapsed - attempt["reserved_hours"])
            attempt.update(status="failed", finished_at=finished,
                           elapsed_hours=elapsed, failure_reason=reason)
            state["reserved_worker_hours"] += extra
            self._write(state); return deepcopy(attempt)

    def _curriculum(self, curriculum):
        if (not isinstance(curriculum, dict) or curriculum.get("snapshot_id") != SNAPSHOT_ID
                or curriculum.get("as_of") != "2026-01-01"
                or curriculum.get("training_end") != "2026-01-01"
                or len(curriculum.get("records", [])) != 5):
            raise ValueError("unchanged five-record curriculum required")

    def _candidate_roster(self, state):
        candidates = []
        for month in SOURCE_UNITS:
            receipt = state["sources"][month]
            if "source_file" in receipt:
                source = self._validate_q1_receipt(receipt)
                candidates.extend(source["candidates"])
            else:
                _, manifest = self._validate_source_receipt(receipt)
                candidates.extend(manifest["candidates"])
        candidates.sort(key=lambda row: _utc(row["decision_time"], "candidate decision_time"))
        ids = [row["candidate_id"] for row in candidates]
        if len(ids) != len(set(ids)): raise ValueError("candidate identities must be unique")
        eligible = [row for row in candidates if datetime(2026, 1, 1, tzinfo=timezone.utc)
                    <= _utc(row["decision_time"], "candidate decision_time")
                    < datetime(2026, 8, 1, tzinfo=timezone.utc)]
        identities = {item["candidate_id"]: item for item in state["inventory"]["exposure_identities"]}
        clocks = {_utc(item["decision_time"], "time exclusion"): item
                  for item in state["inventory"]["time_match_exclusions"]}
        excluded = []
        for row in eligible:
            item = identities.get(row["candidate_id"])
            if item is not None: excluded.append(deepcopy(item))
            else:
                clock = _utc(row["decision_time"], "candidate decision_time")
                item = clocks.get(clock)
            if item is not None and row["candidate_id"] not in identities:
                excluded.append({"candidate_id": row["candidate_id"], "decision_time": row["decision_time"],
                                 "provenance": [deepcopy(item)]})
        chosen = select_block([row["candidate_id"] for row in eligible],
                              {item["candidate_id"] for item in excluded}, cap=30)
        return chosen, {"candidate_count": len(candidates), "agent_candidate_count": len(eligible),
                        "excluded": excluded, "rule": "select_block(cap=30)"}

    def prepare(self, *, campaign_id, packets, requests, jobs, curriculum):
        with self._locked():
            state = self._state()
            if set(state["sources"]) != set(SOURCE_UNITS):
                raise ValueError("all 31 source units must complete before prepare")
            roster, selection = self._candidate_roster(state)
            if any(not isinstance(value, dict) for value in (packets, requests, jobs)):
                raise ValueError("packet, request and job maps required")
            if set(roster) != set(packets) or set(roster) != set(requests) or set(roster) != set(jobs):
                raise ValueError("all and only roster packets, requests and jobs must freeze")
            self._curriculum(curriculum)
            cases = []
            for case_id in roster:
                request = requests[case_id]
                if request.get("case_id") != case_id:
                    raise ValueError("source request case differs from roster")
                if "packet_sha256" in request:
                    if (request["packet_sha256"] != _digest(packets[case_id])
                            or request.get("source_packet") != packets[case_id]):
                        raise ValueError("source packet differs from exact request binding")
                job = self._job_loader(jobs[case_id]); binding = job.campaign_binding()
                if (binding.get("source_request_sha256") != _digest(request)
                        or binding.get("role_request_sha256") != _digest(job.role_request("specialist"))):
                    raise ValueError("published job binding differs from exact request")
                cases.append({"case_id": case_id, "job_directory": jobs[case_id], **binding})
            manifest = {"schema_version": "lc_campaign_contract_v1",
                        "campaign_id": campaign_id, "cases": cases}
            prepared = {"campaign_id": campaign_id, "roster": roster, "selection": selection,
                        "manifest": manifest, "manifest_sha256": _digest(manifest),
                        "packets": deepcopy(packets),
                        "packet_hashes": {key: _digest(value) for key, value in packets.items()},
                        "requests": deepcopy(requests),
                        "request_hashes": {key: _digest(value) for key, value in requests.items()},
                        "jobs": deepcopy(jobs), "curriculum": deepcopy(curriculum),
                        "curriculum_sha256": _digest(curriculum)}
            if state["prepared"] is not None:
                if state["prepared"] != prepared: raise ValueError("immutable prepare differs")
                return deepcopy(prepared)
            self._ledger().freeze(manifest)
            state["prepared"] = prepared; self._write(state); return deepcopy(prepared)

    def request_role(self, case_id, role):
        """Reserve Task-2 attempt, then return exact bytes for lead dispatch."""
        with self._locked():
            state = self._state()
            if state["prepared"] is None or case_id not in state["prepared"]["roster"]:
                raise ValueError("prepared roster case required")
            key = case_id + ":" + role
            if key in state["operations"]: raise ValueError("role already invoked")
            record = self._ledger().start_attempt(case_id, role)
            job = self._job_loader(state["prepared"]["jobs"][case_id])
            wrapper = {"case_id": case_id,
                       "plan": deepcopy(state["prepared"]["requests"][case_id].get("plan")),
                       "request": job.role_request(role)}
            wrapper_bytes = _canonical(wrapper)
            operation = {"case_id": case_id, "role": role, "attempt_id": record["attempt_id"],
                         "wrapper_bytes": wrapper_bytes, "wrapper_sha256": hashlib.sha256(
                             wrapper_bytes.encode("ascii")).hexdigest(), "capture": None}
            state["operations"][key] = operation; self._write(state); return deepcopy(operation)

    def capture_role(self, case_id, role, raw_response, provenance):
        """Capture lead-returned bytes; this method never invokes or retries a model."""
        with self._locked():
            state = self._state(); key = case_id + ":" + role
            operation = state["operations"].get(key)
            if operation is None: raise ValueError("reserved role request required")
            if not isinstance(raw_response, bytes) or not isinstance(provenance, dict):
                raise ValueError("exact response bytes and provenance required")
            raw_sha256 = hashlib.sha256(raw_response).hexdigest()
            try:
                response_text = raw_response.decode("utf-8")
            except UnicodeDecodeError:
                result = {"kind": "external_failure", "reason": "invalid_response_utf8"}
                self._ledger().finish_attempt(case_id, role, result)
                capture = {"kind": "invalid_response_bytes",
                           "raw_response_sha256": raw_sha256,
                           "byte_length": len(raw_response),
                           "reason": "invalid_response_utf8"}
                if operation["capture"] is not None and operation["capture"] != capture:
                    raise ValueError("immutable capture differs")
                operation["capture"] = capture; self._write(state)
                return deepcopy(result)
            job = self._job_loader(state["prepared"]["jobs"][case_id])
            job.capture(role, response_text, provenance); binding = job.capture_binding(role)
            result = {"kind": "delivered", "job_directory": state["prepared"]["jobs"][case_id],
                      **binding}
            self._ledger().finish_attempt(case_id, role, result)
            capture = {"kind": "delivered_bytes", "byte_length": len(raw_response),
                       "raw_response_sha256": raw_sha256, **binding}
            if operation["capture"] is not None and operation["capture"] != capture:
                raise ValueError("immutable capture differs")
            operation["capture"] = capture; self._write(state); return deepcopy(result)

    def finish_failure(self, case_id, role, reason):
        with self._locked():
            state = self._state(); key = case_id + ":" + role
            if key not in state["operations"]: raise ValueError("reserved role request required")
            result = {"kind": "external_failure", "reason": reason}
            value = self._ledger().finish_attempt(case_id, role, result)
            state["operations"][key]["capture"] = deepcopy(result); self._write(state); return value

    def finalize_case(self, case_id, terminal):
        with self._locked():
            state = self._state(); result = self._ledger().finalize_case(case_id, terminal)
            operation = state["operations"].setdefault(case_id + ":finalize", {})
            binding = {"terminal": deepcopy(terminal), "decision_path_sha256": result["path_sha256"]}
            if operation and operation != binding: raise ValueError("immutable finalization differs")
            state["operations"][case_id + ":finalize"] = binding
            self._write(state); return result

    def lock_terminals(self, terminals):
        with self._locked():
            state = self._state(); result = self._ledger().lock_terminals(terminals)
            state["operations"]["terminal_lock"] = {"terminals_sha256": _digest(result)}
            self._write(state); return result

    def score(self, accounting_manifest, bars, *, outcome):
        with self._locked():
            state = self._state()
            if state["prepared"] is None: raise ValueError("prepare required")
            ledger = self._ledger()
            try: ledger.assert_reveal_allowed()
            except ValueError as exc: raise ValueError("reveal gate required before scoring") from exc
            if (not isinstance(accounting_manifest, dict)
                    or accounting_manifest.get("campaign_manifest") != state["prepared"]["manifest"]):
                raise ValueError("accounting manifest differs from frozen campaign manifest")
            if not isinstance(outcome, dict) or set(outcome) != {"sha256"}:
                raise ValueError("exact outcome hash binding required")
            _sha(outcome["sha256"], "outcome hash")
            terminal_state = ledger.state()
            binding = {"terminal_state_sha256": terminal_state["state_sha256"],
                       "accounting_sha256": _digest(accounting_manifest),
                       "outcome_sha256": outcome["sha256"], "bars_sha256": _digest(bars)}
            if state["score"] is not None:
                if {key: state["score"][key] for key in binding} != binding:
                    raise ValueError("immutable score inputs differ")
                return deepcopy(state["score"]["result"])
            if self._scorer is None: raise ValueError("explicit offline scorer required")
            result = self._scorer(bars, accounting_manifest, terminal_state)
            state["score"] = {**binding, "result": deepcopy(result),
                              "result_sha256": _digest(result)}
            self._write(state); return deepcopy(result)

    def status(self):
        with self._locked():
            state = self._read()
            if state is None:
                return {"phase": "planned", "inventory_frozen": False,
                        "source": {"completed": 0, "required": 31, "active": 0,
                                   "reserved_worker_hours": 0, "budget_hours": 18},
                        "roster_n": None, "roles": None, "terminal_count": 0,
                        "terminals_locked": False, "scored": False}
            active = sum(a["status"] == "active" for a in state["source_attempts"].values())
            terminal_locked = False
            if state["prepared"] is not None:
                ledger_state = self._ledger().state()
                terminal_locked = ledger_state["terminals_locked"]
            else:
                ledger_state = None
            if state["score"] is not None: phase = "completed"
            elif terminal_locked: phase = "ready_to_score"
            elif state["prepared"] is not None: phase = "roles"
            elif state["sources"]: phase = "source"
            elif state["inventory"] is not None: phase = "inventory"
            else: phase = "planned"
            role_counts = None
            terminal_count = 0
            if ledger_state is not None:
                role_counts = {role: {name: 0 for name in
                               ("not_invoked", "active", "delivered", "external_failure", "timeout")}
                               for role in ("specialist", "reviewer")}
                for case in ledger_state["cases"].values():
                    for role, record in case["roles"].items():
                        role_counts[role][record["status"]] += 1
                terminal_count = len(ledger_state["terminals"])
            return {"phase": phase, "inventory_frozen": state["inventory"] is not None,
                    "source": {"completed": len(state["sources"]), "required": 31,
                               "active": active, "reserved_worker_hours": state["reserved_worker_hours"],
                               "budget_hours": 18},
                    "roster_n": None if state["prepared"] is None else len(state["prepared"]["roster"]),
                    "roles": role_counts, "terminal_count": terminal_count,
                    "terminals_locked": terminal_locked, "scored": state["score"] is not None}

    def report(self):
        status = self.status()
        with self._locked():
            state = self._read()
        inventory = None if state is None else state["inventory"]
        return {"state": "completed" if status["scored"] else
                ("planned" if status["phase"] == "planned" else "running"),
                "status": status,
                "selection": None if state is None or state["prepared"] is None
                else deepcopy(state["prepared"]["selection"]),
                "inventory": None if inventory is None else {
                    "schema_version": inventory["schema_version"],
                    "source_queue": deepcopy(inventory["source_queue"]),
                    "registry_hashes": {item["path"]: item["sha256"] for item in inventory["registries"]},
                    "exposure_identity_count": len(inventory["exposure_identities"]),
                    "time_match_exclusions": deepcopy(inventory["time_match_exclusions"]),
                    "dependency_count": len(inventory["current_dependencies"]),
                    "file_state_guard_count": len(inventory["file_state_guards"]),
                    "expected_absent_count": sum(item["sha256"] is None for item in inventory["file_state_guards"]),
                    "q1_receipts": deepcopy(inventory["q1_receipts"])},
                "source_attempts": {} if state is None else deepcopy(state["source_attempts"]),
                "source_receipts": {} if state is None else deepcopy(state["sources"]),
                "prepared": None if state is None or state["prepared"] is None else {
                    "campaign_id": state["prepared"]["campaign_id"],
                    "roster": deepcopy(state["prepared"]["roster"]),
                    "manifest_sha256": state["prepared"]["manifest_sha256"],
                    "curriculum_sha256": state["prepared"]["curriculum_sha256"],
                    "packet_hashes": deepcopy(state["prepared"]["packet_hashes"]),
                    "request_hashes": deepcopy(state["prepared"]["request_hashes"])},
                "operations": {} if state is None else deepcopy(state["operations"]),
                "score": None if state is None else deepcopy(state["score"]),
                "limitations": ["offline reconstructed source receipts",
                                "controller invokes neither source replay nor market models"]}


def main(argv=None):
    parser = argparse.ArgumentParser(description="Read-only LC campaign controller status")
    parser.add_argument("phase", choices=("status", "report"))
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv); controller = CampaignController(args.run_dir)
    print(_canonical(controller.status() if args.phase == "status" else controller.report()))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
