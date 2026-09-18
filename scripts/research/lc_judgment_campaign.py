"""Versioned, source-only preparation for the amended LC judgment sample.

This module treats the completed ``lc_campaign_controller_v2`` census as an
immutable dependency.  It does not replay sources, publish jobs, dispatch
roles, or score outcomes.  The only mutation is an atomic preparation freeze
in a separate judgment run directory.
"""
from __future__ import annotations

from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import tempfile

from scripts.research.assessment_evidence_guard import _canonical
from scripts.research.lc_campaign import CampaignController, SOURCE_UNITS, _digest, _sha_file, _utc
from scripts.research.lc_judgment_sampling import POLICY, select_unassessed


VERSION = "lc_judgment_preparation_v1"
SOURCE_CONTROLLER_VERSION = "lc_campaign_controller_v2"
PREPARE_FILE = "judgment_prepare.json"
INTERVAL_START = datetime(2026, 1, 1, tzinfo=timezone.utc)
INTERVAL_END = datetime(2026, 8, 1, tzinfo=timezone.utc)
EXPECTED_COUNTS = {
    "census_candidate_count": 142,
    "interval_candidate_count": 36,
    "excluded_count": 16,
    "roster_count": 20,
}


class JudgmentCampaign:
    """Freeze and revalidate the approved chronological judgment roster.

    ``case_source`` is deliberately the last interface in this module.  Packet
    construction and role enablement belong to a separately reviewed stage.
    """

    def __init__(self, run_dir, *, source_run_dir, source_controller_factory=CampaignController,
                 file_hasher=_sha_file):
        self.run_dir = Path(run_dir)
        self.source_run_dir = Path(source_run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._source_controller_factory = source_controller_factory
        self._file_hasher = file_hasher
        self._path = self.run_dir / PREPARE_FILE
        self._lock_path = self.run_dir / ".judgment_prepare.lock"

    @contextmanager
    def _locked(self):
        with self._lock_path.open("a+") as stream:
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            yield

    def _source_controller(self):
        return self._source_controller_factory(self.source_run_dir)

    def _controller_path(self):
        return self.source_run_dir / "controller.json"

    def _read_prepared(self):
        if not self._path.exists():
            raise ValueError("judgment preparation has not been frozen")
        raw = self._path.read_bytes()
        try:
            prepared = json.loads(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("invalid judgment preparation bytes") from exc
        body = {key: value for key, value in prepared.items() if key != "sha256"}
        if (_canonical(prepared).encode("ascii") != raw
                or prepared.get("sha256") != _digest(body)):
            raise ValueError("judgment preparation hash or canonical bytes changed")
        return prepared

    def _validate_census_state(self):
        controller = self._source_controller()
        state = controller._read()
        if (not isinstance(state, dict) or state.get("version") != SOURCE_CONTROLLER_VERSION
                or state.get("prepared") is not None or state.get("operations") != {}
                or state.get("score") is not None):
            raise ValueError("completed source-only census state required")
        if set(state.get("sources", {})) != set(SOURCE_UNITS):
            raise ValueError("exact 31 completed source receipts required")
        inventory = state.get("inventory")
        if (not isinstance(inventory, dict)
                or inventory.get("source_queue") != list(SOURCE_UNITS)):
            raise ValueError("frozen chronological census inventory required")
        controller_sha256 = self._file_hasher(str(self._controller_path()))
        if controller_sha256 is None:
            raise ValueError("source controller artifact required")
        return controller, state, controller_sha256

    @staticmethod
    def _normalized_receipt(receipt):
        if "source_file" in receipt:
            return {
                "month": receipt["month"],
                "source_file": str(receipt["source_file"]),
                "source_file_sha256": receipt["source_file_sha256"],
                "manifest_file": None,
                "manifest_sha256": None,
                "receipt_sha256": _digest(receipt),
            }
        directory = Path(receipt["directory"])
        return {
            "month": receipt["month"],
            "source_file": str(directory / "source.json"),
            "source_file_sha256": receipt["source_sha256"],
            "manifest_file": str(directory / "manifest.json"),
            "manifest_sha256": receipt["manifest_sha256"],
            "receipt_sha256": _digest(receipt),
        }

    @staticmethod
    def _load_source(controller, receipt):
        if "source_file" in receipt:
            return controller._validate_q1_receipt(receipt)
        source, _ = controller._validate_source_receipt(receipt)
        return source

    def _projection(self):
        controller, state, controller_sha256 = self._validate_census_state()
        inventory = state["inventory"]
        all_candidates = []
        locations = {}
        parent_hashes = {}
        for month in SOURCE_UNITS:
            receipt = state["sources"][month]
            source = self._load_source(controller, receipt)
            ledgers = source.get("parent_ledgers")
            if not isinstance(ledgers, dict):
                raise ValueError("source parent ledgers required: " + month)
            parent_hashes[month] = _digest(ledgers)
            for candidate in source.get("candidates", []):
                case_id = candidate.get("candidate_id")
                if not isinstance(case_id, str) or not case_id or case_id in locations:
                    raise ValueError("candidate identities must be nonempty and unique")
                decision = _utc(candidate.get("decision_time"), "candidate decision_time")
                locations[case_id] = {
                    "month": month,
                    "candidate": deepcopy(candidate),
                    "decision": decision,
                    "receipt": receipt,
                }
                all_candidates.append(locations[case_id])
        all_candidates.sort(key=lambda item: item["decision"])
        eligible = [item for item in all_candidates
                    if INTERVAL_START <= item["decision"] < INTERVAL_END]

        identities = {item["candidate_id"]: item
                      for item in inventory["exposure_identities"]}
        clock_exclusions = {
            _utc(item["decision_time"], "time exclusion"): item
            for item in inventory["time_match_exclusions"]
        }
        exclusions = []
        for item in eligible:
            case_id = item["candidate"]["candidate_id"]
            identity = identities.get(case_id)
            if identity is not None:
                exclusions.append({
                    "candidate_id": case_id,
                    "decision_time": item["decision"].isoformat(),
                    "match": "candidate_id",
                    "provenance": deepcopy(identity["provenance"]),
                })
                continue
            clock = clock_exclusions.get(item["decision"])
            if clock is not None:
                exclusions.append({
                    "candidate_id": case_id,
                    "decision_time": item["decision"].isoformat(),
                    "match": "decision_time",
                    "provenance": [deepcopy(clock)],
                })

        excluded_ids = {item["candidate_id"] for item in exclusions}
        roster = select_unassessed(
            [item["candidate"]["candidate_id"] for item in eligible],
            excluded_ids,
            cap=30,
        )
        counts = {
            "census_candidate_count": len(all_candidates),
            "interval_candidate_count": len(eligible),
            "excluded_count": len(exclusions),
            "roster_count": len(roster),
        }
        if counts != EXPECTED_COUNTS:
            raise ValueError("amended census counts differ: " + repr(counts))

        cases = []
        for case_id in roster:
            item = locations[case_id]
            normalized = self._normalized_receipt(item["receipt"])
            cases.append({
                "candidate_id": case_id,
                "decision_time": item["decision"].isoformat(),
                "month": item["month"],
                "candidate_sha256": _digest(item["candidate"]),
                "parent_ledgers_sha256": parent_hashes[item["month"]],
                "source_file": normalized["source_file"],
                "source_file_sha256": normalized["source_file_sha256"],
                "source_manifest_sha256": normalized["manifest_sha256"],
                "source_receipt_sha256": normalized["receipt_sha256"],
            })

        exposure_metadata = {
            "registries": inventory["registries"],
            "exposure_identities": inventory["exposure_identities"],
            "time_match_exclusions": inventory["time_match_exclusions"],
        }
        return {
            "schema_version": VERSION,
            "policy": POLICY,
            "source_controller": {
                "path": str(self._controller_path().resolve()),
                "sha256": controller_sha256,
                "state_sha256": _digest(state),
            },
            "census_bindings": {
                "inventory_sha256": _digest(inventory),
                "source_receipts_sha256": _digest(state["sources"]),
                "exposure_metadata_sha256": _digest(exposure_metadata),
            },
            "selection": {
                **counts,
                "cap": 30,
                "interval_start": INTERVAL_START.isoformat(),
                "interval_end_exclusive": INTERVAL_END.isoformat(),
            },
            "exclusions": exclusions,
            "roster": roster,
            "cases": cases,
            "roles_enabled": False,
            "limitations": [
                "filtered judgment sample, not a continuous opportunity stream",
                "Q1 includes exposed development history",
                "source-only freeze; packets, requests, roles, and outcomes are not enabled",
            ],
        }

    def prepare(self):
        """Atomically freeze the approved roster without changing the census."""
        with self._locked():
            if self._path.exists():
                return self._verify_unlocked()
            body = self._projection()
            prepared = dict(body, sha256=_digest(body))
            raw = _canonical(prepared).encode("ascii")
            fd, temporary = tempfile.mkstemp(prefix=".pending-judgment-", dir=self.run_dir)
            try:
                with os.fdopen(fd, "wb") as stream:
                    stream.write(raw)
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(temporary, self._path)
                directory = os.open(self.run_dir, os.O_RDONLY)
                try:
                    os.fsync(directory)
                finally:
                    os.close(directory)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
            return deepcopy(prepared)

    def _verify_unlocked(self):
        prepared = self._read_prepared()
        current_controller_hash = self._file_hasher(str(self._controller_path()))
        if current_controller_hash != prepared.get("source_controller", {}).get("sha256"):
            raise ValueError("source controller hash changed")
        current = self._projection()
        frozen = {key: value for key, value in prepared.items() if key != "sha256"}
        if current != frozen:
            raise ValueError("frozen prepared projection differs from validated census")
        return deepcopy(prepared)

    def verify(self):
        """Reopen every pinned receipt and compare the exact frozen projection."""
        with self._locked():
            return self._verify_unlocked()

    def case_source(self, case_id):
        """Return full validated source provenance for one frozen roster case."""
        with self._locked():
            prepared = self._verify_unlocked()
            if case_id not in prepared["roster"]:
                raise ValueError("case is not in the frozen roster")
            case = next(item for item in prepared["cases"]
                        if item["candidate_id"] == case_id)
            controller, state, _ = self._validate_census_state()
            receipt = state["sources"][case["month"]]
            source = self._load_source(controller, receipt)
            candidate = next((row for row in source["candidates"]
                              if row.get("candidate_id") == case_id), None)
            ledgers = source.get("parent_ledgers")
            if (candidate is None or _digest(candidate) != case["candidate_sha256"]
                    or _digest(ledgers) != case["parent_ledgers_sha256"]):
                raise ValueError("case source differs from frozen preparation")
            return {
                "candidate": deepcopy(candidate),
                "parent_ledgers": deepcopy(ledgers),
                "source": deepcopy(source),
                "receipt": self._normalized_receipt(receipt),
            }
