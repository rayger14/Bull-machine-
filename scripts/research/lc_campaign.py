"""Offline, fail-closed controller for the consolidated LC campaign.

This module deliberately has no model client, source replay, or outcome reader.
The lead process supplies immutable source receipts and invokes role transport;
this controller only freezes their bindings and gates later phases.
"""
from __future__ import annotations

from copy import deepcopy
import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from scripts.research.assessment_evidence_guard import _canonical
from scripts.research.lc_campaign_contract import CampaignLedger


VERSION = "lc_campaign_controller_v1"
SOURCE_UNITS = tuple(f"{year:04d}-{month:02d}" for year in range(2024, 2027)
                     for month in range(1, 13)
                     if (year, month) >= (2024, 1) and (year, month) <= (2026, 7))
MAX_SOURCE_WORKERS = 2
MAX_SOURCE_HOURS = 18


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _sha(value, name):
    if not isinstance(value, str) or len(value) != 64 or set(value) - set("0123456789abcdef"):
        raise ValueError(name + " must be a lowercase SHA-256")


class CampaignController:
    def __init__(self, run_dir, *, ledger_factory=CampaignLedger, scorer=None):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._ledger_factory = ledger_factory
        self._scorer = scorer
        self._ledger = None

    @property
    def _path(self): return self.run_dir / "controller.json"

    def _read(self):
        if not self._path.exists(): return None
        raw = self._path.read_bytes(); state = json.loads(raw)
        body = {key: value for key, value in state.items() if key != "sha256"}
        if _canonical(state).encode("ascii") != raw or state.get("sha256") != _digest(body):
            raise ValueError("controller state hash changed")
        if set(state) != {"version", "inventory", "sources", "source_worker_hours", "prepared", "score", "sha256"} or state["version"] != VERSION:
            raise ValueError("invalid controller state schema")
        return state

    def _write(self, state):
        body = deepcopy(state); body.pop("sha256", None); state = dict(body, sha256=_digest(body))
        raw = _canonical(state).encode("ascii"); fd, temporary = tempfile.mkstemp(prefix=".pending-", dir=self.run_dir)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(raw); stream.flush(); os.fsync(stream.fileno())
            os.replace(temporary, self._path)
        finally:
            if os.path.exists(temporary): os.unlink(temporary)
        return deepcopy(state)

    def _state(self):
        state = self._read()
        return state if state is not None else {"version": VERSION, "inventory": None, "sources": {}, "source_worker_hours": 0, "prepared": None, "score": None}

    def inventory(self, *, dependencies, exclusions, q1_hashes):
        if not isinstance(dependencies, dict) or not dependencies or not isinstance(exclusions, list) or not isinstance(q1_hashes, dict):
            raise ValueError("exact inventory dependencies, exclusions and Q1 hashes required")
        for value in dependencies.values(): _sha(value, "dependency hash")
        for value in q1_hashes.values(): _sha(value, "Q1 hash")
        if any(not isinstance(item, str) or not item for item in exclusions) or len(set(exclusions)) != len(exclusions):
            raise ValueError("exclusions must be unique nonempty identities")
        state = self._state(); inventory = {"source_queue": list(SOURCE_UNITS), "dependencies": deepcopy(dependencies),
                                             "exclusions": list(exclusions), "q1_hashes": deepcopy(q1_hashes)}
        if state["inventory"] is not None and state["inventory"] != inventory: raise ValueError("immutable inventory differs")
        state["inventory"] = inventory; self._write(state); return deepcopy(inventory)

    def source(self, completed, *, workers, worker_hours):
        state = self._state()
        if state["inventory"] is None: raise ValueError("inventory required before source")
        if type(workers) is not int or not 1 <= workers <= MAX_SOURCE_WORKERS: raise ValueError("source workers exceed two")
        if isinstance(worker_hours, bool) or not isinstance(worker_hours, (int, float)) or worker_hours < 0: raise ValueError("worker hours required")
        if not isinstance(completed, dict) or any(unit not in state["inventory"]["source_queue"] for unit in completed): raise ValueError("unknown source unit")
        existing = state["sources"]
        for unit, receipt in completed.items():
            if not isinstance(receipt, dict) or set(receipt) != {"sha256"}: raise ValueError("exact source receipt required")
            _sha(receipt["sha256"], "source hash")
            if unit in existing and existing[unit]["sha256"] != receipt["sha256"]: raise ValueError("immutable source receipt differs")
        total = state.get("source_worker_hours", 0) + worker_hours
        if total > MAX_SOURCE_HOURS: raise ValueError("source budget exhausted")
        existing.update(deepcopy(completed)); state["sources"] = existing; state["source_worker_hours"] = total
        self._write(state); return deepcopy(existing)

    def prepare(self, manifest, *, requests):
        state = self._state()
        if state["inventory"] is None or set(state["sources"]) != set(state["inventory"]["source_queue"]):
            raise ValueError("all 31 source units must complete before prepare")
        if not isinstance(requests, dict): raise ValueError("frozen request map required")
        if state["prepared"] is not None:
            if state["prepared"]["manifest"] != manifest or state["prepared"]["requests"] != requests: raise ValueError("immutable prepare differs")
            return deepcopy(state["prepared"])
        self._ledger = self._ledger_factory(self.run_dir / "ledger")
        self._ledger.freeze(manifest)
        prepared = {"manifest": deepcopy(manifest), "requests": deepcopy(requests), "manifest_sha256": _digest(manifest)}
        state["prepared"] = prepared; self._write(state); return deepcopy(prepared)

    def _require_ledger(self):
        state = self._state()
        if state["prepared"] is None: raise ValueError("prepare required")
        if self._ledger is None: self._ledger = self._ledger_factory(self.run_dir / "ledger")
        return state, self._ledger

    # These are controller gates only. The lead invokes collaboration transport itself.
    def start_attempt(self, case_id, role): return self._require_ledger()[1].start_attempt(case_id, role)
    def finish_attempt(self, case_id, role, result): return self._require_ledger()[1].finish_attempt(case_id, role, result)
    def finalize_case(self, case_id, terminal): return self._require_ledger()[1].finalize_case(case_id, terminal)
    def lock_terminals(self, terminals): return self._require_ledger()[1].lock_terminals(terminals)

    def score(self, accounting_manifest, bars):
        state, ledger = self._require_ledger()
        try: ledger.assert_reveal_allowed()
        except ValueError as exc: raise ValueError("reveal gate required before scoring") from exc
        if not isinstance(accounting_manifest, dict) or accounting_manifest.get("campaign_manifest") != state["prepared"]["manifest"]:
            raise ValueError("accounting manifest differs from frozen campaign manifest")
        if self._scorer is None: raise ValueError("explicit offline scorer required")
        result = self._scorer(bars, accounting_manifest, ledger.state())
        state["score"] = {"manifest_sha256": _digest(accounting_manifest), "result": deepcopy(result)}
        self._write(state); return deepcopy(result)

    def status(self):
        state = self._read()
        if state is None: return {"phase": "planned", "source_worker_hours": 0}
        phase = "completed" if state["score"] is not None else ("prepared" if state["prepared"] is not None else ("source" if state["sources"] else ("inventory" if state["inventory"] else "planned")))
        return {"phase": phase, "source_worker_hours": state.get("source_worker_hours", 0),
                "sources_completed": len(state["sources"]), "sources_required": len(SOURCE_UNITS)}

    def report(self):
        phase = self.status()["phase"]
        return {"state": "planned" if phase == "planned" else ("completed" if phase == "completed" else "running"),
                "status": self.status()}


def main(argv=None):
    parser = argparse.ArgumentParser(); parser.add_argument("phase", choices=("inventory", "source", "prepare", "status", "score", "report")); parser.add_argument("--run-dir", required=True)
    args = parser.parse_args(argv); controller = CampaignController(args.run_dir)
    if args.phase == "status": value = controller.status()
    elif args.phase == "report": value = controller.report()
    else: raise SystemExit("mutating phases require the lead controller's explicit Python API inputs")
    print(_canonical(value)); return 0


if __name__ == "__main__": raise SystemExit(main())
