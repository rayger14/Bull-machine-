"""Synthetic state-machine tests for the offline LC campaign controller."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.assessment_evidence_guard import _canonical
from scripts.research.lc_campaign import CampaignController, SOURCE_UNITS


def test_nested_source_manifest_preserves_expected_absent_files():
    from scripts.research.lc_campaign import _manifest_files
    source = {
        "source_manifest": {"files": {}, "replay": {"models": {"files": {
            "missing-calibrator.pkl": None,
        }}}},
        "code_manifest": {"files": {}},
        "config_manifest": {"files": {}},
    }
    assert _manifest_files(source) == {"missing-calibrator.pkl": None}


def digest_bytes(value):
    return hashlib.sha256(value).hexdigest()


class Job:
    def __init__(self, path, store):
        self.path, self.store = str(path), store

    def campaign_binding(self):
        return deepcopy(self.store[self.path]["binding"])

    def role_request(self, role):
        if role == "reviewer" and "specialist" not in self.store[self.path]:
            raise ValueError("specialist required")
        return deepcopy(self.store[self.path][role + "_request"])

    def capture(self, role, raw, provenance):
        value = {"raw": raw, "provenance": deepcopy(provenance)}
        prior = self.store[self.path].get(role)
        if prior is not None and prior != value:
            raise ValueError("immutable capture differs")
        self.store[self.path][role] = value

    def capture_binding(self, role):
        value = self.store[self.path][role]
        return {"raw_response_sha256": digest_bytes(value["raw"].encode()),
                "capture_sha256": digest_bytes(_canonical(value).encode("ascii"))}

    def reviewer_eligible(self):
        return "specialist" in self.store[self.path]

    def grade_binding(self):
        return deepcopy(self.store[self.path]["grade"])


def write_source(root, month, candidates=(), replay_files=None):
    directory = root / month
    directory.mkdir(parents=True)
    rows = [{"candidate_id": case_id, "decision_time": when,
             "exposure_key": case_id, "track": "hourly"}
            for case_id, when in candidates]
    source = {"month": month, "candidates": rows, "hourly_input_hash": "1" * 64,
              "source_manifest": {"files": {}, "replay": {"files": replay_files or {}}},
              "code_manifest": {"files": {}}, "config_manifest": {"files": {}}}
    source_raw = (_canonical(source) + "\n").encode()
    (directory / "source.json").write_bytes(source_raw)
    manifest = {"schema": "lc-campaign-candidate-manifest-v1",
                "wrapper": "lc-campaign-source-wrapper-v1", "wrapper_sha256": "2" * 64,
                "source_payload_schema": "lc-source-population-v1",
                "construction": "lc-persistent-master-source-v1", "month": month,
                "seed": "synthetic", "start": "synthetic", "end_exclusive": "synthetic",
                "source_file": "source.json", "source_file_sha256": digest_bytes(source_raw),
                "candidate_count": len(rows), "input_limit_disclosures": [], "candidates": rows}
    manifest_raw = (_canonical(manifest) + "\n").encode()
    (directory / "manifest.json").write_bytes(manifest_raw)
    return {"month": month, "directory": str(directory),
            "source_sha256": digest_bytes(source_raw),
            "manifest_sha256": digest_bytes(manifest_raw)}


@pytest.fixture
def setup(tmp_path):
    case_id = "hourly-lc:2026-04-01T00:00:00+00:00"
    source_root = tmp_path / "sources"; source_root.mkdir()
    receipts = {}
    for month in SOURCE_UNITS:
        candidates = [(case_id, "2026-04-01T00:00:00+00:00")] if month == "2026-04" else []
        if month not in {"2026-01", "2026-02", "2026-03"}:
            receipts[month] = {"directory": str(source_root / month), "candidates": candidates}
        else:
            completed = write_source(source_root, month, candidates)
            receipts[month] = {
                "month": month,
                "source_file": str(source_root / month / "source.json"),
                "source_file_sha256": completed["source_sha256"],
                "hourly_input_hash": "1" * 64,
                "candidate_ids": [case for case, _ in candidates]}
    registry_specs = tuple((f"registry-{n}.json", chr(97 + n) * 64) for n in range(5))
    jobs = {}
    job_loader = lambda path: Job(path, jobs)
    guard_hashes = {f"guard-{n}": (None if n in (3, 19) else "f" * 64)
                    for n in range(256)}
    guard_hashes["dep"] = "d" * 64
    guard_hashes.update(dict(registry_specs))
    def file_hasher(path):
        if path in guard_hashes:
            return guard_hashes[path]
        value = Path(path)
        return digest_bytes(value.read_bytes()) if value.exists() else None
    controller = CampaignController(
        tmp_path / "run", job_loader=job_loader,
        source_validator=lambda source, month: None,
        file_hasher=file_hasher,
        required_registries=registry_specs,
        q1_projections={month: {key: value for key, value in receipts[month].items()
                        if key in {"source_file_sha256", "hourly_input_hash", "candidate_ids"}}
                        for month in ("2026-01", "2026-02", "2026-03")})
    inventory = {
        "schema_version": "lc_campaign_inventory_v1",
        "source_queue": list(SOURCE_UNITS),
        "registries": [{"path": path, "sha256": value, "exposures": []}
                       for path, value in registry_specs],
        "exposure_identities": [],
        "time_match_exclusions": [
            {"decision_time": "2026-06-11T13:00:00+00:00", "provenance": "hourly-null-id:C1"},
            {"decision_time": "2026-06-14T22:00:00+00:00", "provenance": "hourly-null-id:C2"}],
        "current_dependencies": [{"path": "dep", "sha256": "d" * 64}],
        "file_state_guards": [{"path": path, "sha256": guard_hashes[path]}
                              for path in (f"guard-{n}" for n in range(256))],
        "q1_receipts": [receipts[m] for m in ("2026-01", "2026-02", "2026-03")],
    }
    return controller, inventory, receipts, jobs, job_loader, case_id


def complete_sources(controller, inventory, receipts):
    controller.inventory(inventory)
    for month in SOURCE_UNITS:
        if month in {"2026-01", "2026-02", "2026-03"}:
            continue
        attempt = controller.reserve_source(month, receipts[month]["directory"], reserved_hours=.25)
        write_source(Path(receipts[month]["directory"]).parent, month,
                     receipts[month]["candidates"])
        controller.complete_source(month, attempt["attempt_id"])


def prepare_one(setup, *, excluded=False):
    controller, inventory, receipts, jobs, _, case_id = setup
    if excluded:
        inventory["exposure_identities"] = [
            {"candidate_id": case_id, "decision_time": "2026-04-01T00:00:00+00:00",
             "provenance": [{"registry": "registry-0.json", "record": "old"}]}]
    complete_sources(controller, inventory, receipts)
    source_request = {"case_id": case_id, "plan": {"entry": "wait_5m_high"}, "packet": "p"}
    specialist = {"contract": "specialist", "case_id": case_id}
    jobs["job-one"] = {
        "binding": {"source_request_sha256": digest_bytes(_canonical(source_request).encode("ascii")),
                    "role_request_sha256": digest_bytes(_canonical(specialist).encode("ascii"))},
        "specialist_request": specialist,
        "reviewer_request": {"contract": "reviewer", "case_id": case_id}}
    artifacts = {} if excluded else {case_id: {"packet": "p"}}
    request_artifacts = {} if excluded else {case_id: source_request}
    job_artifacts = {} if excluded else {case_id: "job-one"}
    prepared = controller.prepare(
        campaign_id="tiny", packets=artifacts,
        requests=request_artifacts, jobs=job_artifacts,
        curriculum={"snapshot_id": "299193b77af10d1382bc02d537a8f919b1e2cd9a3fe5557543e4bb5184fb727d",
                    "as_of": "2026-01-01", "training_end": "2026-01-01",
                    "records": [1, 2, 3, 4, 5]})
    return controller, prepared, jobs, case_id


def test_inventory_requires_exact_registries_256_guards_three_q1(setup):
    controller, inventory, _, _, _, _ = setup
    bad = deepcopy(inventory); bad["file_state_guards"].pop()
    with pytest.raises(ValueError, match="256"): controller.inventory(bad)
    bad = deepcopy(inventory); bad["registries"][0]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="registry"): controller.inventory(bad)
    bad = deepcopy(inventory); bad["q1_receipts"].pop()
    with pytest.raises(ValueError, match="Q1"): controller.inventory(bad)


def test_inventory_explicitly_rejects_nested_replay_hash_conflict(setup):
    controller, inventory, _, _, _, _ = setup
    receipt = inventory["q1_receipts"][0]
    source_path = Path(receipt["source_file"])
    source = json.loads(source_path.read_text())
    source["source_manifest"] = {
        "files": {"same": "a" * 64}, "replay": {"files": {"same": "b" * 64}}}
    source_raw = (_canonical(source) + "\n").encode()
    source_path.write_bytes(source_raw)
    receipt["source_file_sha256"] = digest_bytes(source_raw)
    controller._q1_projections[receipt["month"]]["source_file_sha256"] = digest_bytes(source_raw)
    with pytest.raises(ValueError, match="conflicting nested"):
        controller.inventory(inventory)


def test_source_reserves_capacity_and_budget_before_launch_across_restart(setup):
    controller, inventory, receipts, _, job_loader, _ = setup
    controller.inventory(inventory)
    first = controller.reserve_source("2024-01", receipts["2024-01"]["directory"], reserved_hours=8)
    controller.reserve_source("2024-02", receipts["2024-02"]["directory"], reserved_hours=8)
    with pytest.raises(ValueError, match="active"):
        controller.reserve_source("2024-03", receipts["2024-03"]["directory"], reserved_hours=1)
    reopened = CampaignController(controller.run_dir, job_loader=job_loader,
        source_validator=lambda source, month: None, file_hasher=controller._file_hasher,
        required_registries=controller._required_registries,
        q1_projections=controller._q1_projections)
    write_source(Path(receipts["2024-01"]["directory"]).parent, "2024-01")
    reopened.complete_source("2024-01", first["attempt_id"])
    with pytest.raises(ValueError, match="budget"):
        reopened.reserve_source("2024-03", receipts["2024-03"]["directory"], reserved_hours=3)
    assert reopened.status()["source"]["reserved_worker_hours"] == 16


def test_interrupted_source_is_terminal_charged_and_releases_slot_for_new_directory(setup):
    controller, inventory, receipts, _, job_loader, _ = setup
    now = [0.0]; controller._clock = lambda: now[0]
    controller.inventory(inventory)
    first = controller.reserve_source("2024-01", receipts["2024-01"]["directory"],
                                      reserved_hours=.5)
    reopened = CampaignController(controller.run_dir, job_loader=job_loader,
        source_validator=lambda source, month: None, file_hasher=controller._file_hasher,
        required_registries=controller._required_registries,
        q1_projections=controller._q1_projections, clock=lambda: now[0])
    now[0] = 900
    failed = reopened.fail_source("2024-01", first["attempt_id"], "worker_interrupted")
    assert failed["status"] == "failed" and failed["elapsed_hours"] == .25
    retry = reopened.reserve_source("2024-01", str(Path(receipts["2024-01"]["directory"]).with_name("2024-01-retry")),
                                    reserved_hours=.25)
    status = reopened.status()["source"]
    assert retry["status"] == "active" and status["active"] == 1
    assert status["reserved_worker_hours"] == .75


def test_source_reopens_artifacts_and_completed_bytes_are_immutable(setup):
    controller, inventory, receipts, _, _, _ = setup
    controller.inventory(inventory)
    attempt = controller.reserve_source("2024-01", receipts["2024-01"]["directory"], reserved_hours=1)
    write_source(Path(receipts["2024-01"]["directory"]).parent, "2024-01")
    controller.complete_source("2024-01", attempt["attempt_id"])
    with pytest.raises(ValueError, match="already attempted"):
        controller.reserve_source("2024-01", receipts["2024-01"]["directory"], reserved_hours=1)
    (Path(receipts["2024-01"]["directory"]) / "source.json").write_text("changed")
    with pytest.raises(ValueError, match="hash"): controller.status()


def test_source_receipt_compares_lean_manifest_projection_to_full_candidates(setup):
    controller, inventory, receipts, _, _, _ = setup
    controller.inventory(inventory)
    month = "2024-01"; directory = Path(receipts[month]["directory"])
    attempt = controller.reserve_source(month, directory, reserved_hours=1)
    write_source(directory.parent, month, [
        ("hourly-lc:2024-01-03T12:00:00+00:00", "2024-01-03T12:00:00+00:00")])
    source_path, manifest_path = directory / "source.json", directory / "manifest.json"
    source = json.loads(source_path.read_text())
    source["candidates"][0]["features"] = {"source_close": 100.0}
    source_raw = (_canonical(source) + "\n").encode(); source_path.write_bytes(source_raw)
    manifest = json.loads(manifest_path.read_text())
    manifest["source_file_sha256"] = digest_bytes(source_raw)
    manifest_path.write_bytes((_canonical(manifest) + "\n").encode())
    assert controller.complete_source(month, attempt["attempt_id"])["month"] == month


def test_source_reservation_rejects_preexisting_output_collision(setup):
    controller, inventory, receipts, _, _, _ = setup
    controller.inventory(inventory)
    write_source(Path(receipts["2024-01"]["directory"]).parent, "2024-01")
    with pytest.raises(ValueError, match="collision"):
        controller.reserve_source("2024-01", receipts["2024-01"]["directory"], reserved_hours=1)


def test_prepare_needs_all_31_and_preserves_exclusion_provenance(setup):
    controller, inventory, _, _, _, case_id = setup
    inventory["exposure_identities"] = [
        {"candidate_id": case_id, "decision_time": "2026-04-01T00:00:00+00:00",
         "provenance": [{"registry": "registry-0.json", "record": "old"}]}]
    controller.inventory(inventory)
    with pytest.raises(ValueError, match="all 31"):
        controller.prepare(campaign_id="x", packets={}, requests={}, jobs={}, curriculum={})
    controller, prepared, _, _ = prepare_one(setup, excluded=True)
    assert prepared["roster"] == []
    assert prepared["selection"]["excluded"][0]["provenance"][0]["record"] == "old"


def test_restart_revalidates_prepared_job_and_ledger_bindings(setup):
    controller, prepared, jobs, case_id = prepare_one(setup)
    assert prepared["roster"] == [case_id]
    jobs["job-one"]["binding"]["source_request_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="binding"): controller.status()


def test_lead_request_is_exact_bytes_and_crash_after_dispatch_is_consumed(setup):
    controller, _, _, case_id = prepare_one(setup)
    operation = controller.request_role(case_id, "specialist")
    wrapper = json.loads(operation["wrapper_bytes"])
    assert set(wrapper) == {"case_id", "plan", "request"}
    reopened = CampaignController(controller.run_dir, job_loader=controller._job_loader,
        source_validator=controller._source_validator, file_hasher=controller._file_hasher,
        required_registries=controller._required_registries,
        q1_projections=controller._q1_projections)
    with pytest.raises(ValueError, match="already invoked"):
        reopened.request_role(case_id, "specialist")


def test_capture_finalize_and_score_are_hash_bound_and_idempotent(setup):
    controller, prepared, jobs, case_id = prepare_one(setup)
    controller.request_role(case_id, "specialist")
    assert controller.capture_role(case_id, "specialist", b"raw answer",
                                   {"transport_valid": True})["kind"] == "delivered"
    jobs["job-one"]["grade"] = {"grade_sha256": "9" * 64, "status": "valid",
                                 "research_plan": {"id": "wait"}}
    terminal = {"kind": "published_grade", "job_directory": "job-one",
                "grade_sha256": "9" * 64}
    controller.finalize_case(case_id, terminal); controller.lock_terminals({case_id: terminal})
    calls = []; controller._scorer = lambda bars, manifest, state: calls.append(1) or {"net": 7}
    accounting = {"campaign_manifest": prepared["manifest"], "cases": []}
    result = controller.score(accounting, {"bars": []}, outcome={"sha256": "8" * 64})
    assert result == {"net": 7}
    assert controller.score(accounting, {"bars": []}, outcome={"sha256": "8" * 64}) == result
    assert calls == [1]
    with pytest.raises(ValueError, match="immutable score"):
        controller.score(accounting, {"bars": [1]}, outcome={"sha256": "7" * 64})
    assert controller.report()["score"]["result_sha256"] == digest_bytes(_canonical(result).encode("ascii"))


def test_non_utf8_role_bytes_are_hash_bound_terminal_failure(setup):
    controller, _, _, case_id = prepare_one(setup)
    controller.request_role(case_id, "specialist")
    raw = b"\xff\x00tool-return"
    assert controller.capture_role(case_id, "specialist", raw,
                                   {"transport_valid": True}) == {
        "kind": "external_failure", "reason": "invalid_response_utf8"}
    operation = controller._read()["operations"][case_id + ":specialist"]
    assert operation["capture"] == {
        "kind": "invalid_response_bytes",
        "raw_response_sha256": digest_bytes(raw),
        "byte_length": len(raw),
        "reason": "invalid_response_utf8",
    }


def test_status_is_read_only_and_cli_only_advertises_read_only_phases(tmp_path, capsys):
    controller = CampaignController(tmp_path / "empty", required_registries=())
    assert controller.status()["phase"] == "planned"
    assert not (controller.run_dir / "controller.json").exists()
    from scripts.research.lc_campaign import main
    with pytest.raises(SystemExit): main(["--help"])
    help_text = capsys.readouterr().out
    assert "status" in help_text and "report" in help_text
    assert "source" not in help_text and "score" not in help_text
