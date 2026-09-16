"""Synthetic-only state tests for the fail-closed campaign controller."""
import pytest


class Ledger:
    def __init__(self, directory): self.directory = directory; self.frozen = None; self.revealed = False
    def freeze(self, manifest): self.frozen = manifest; return manifest
    def state(self): return {"manifest": self.frozen, "terminals_locked": self.revealed}
    def assert_reveal_allowed(self):
        if not self.revealed: raise ValueError("terminals")
        return True


def controller(tmp_path):
    from scripts.research.lc_campaign import CampaignController
    return CampaignController(tmp_path / "run", ledger_factory=Ledger)


def test_status_never_starts_work_and_report_separates_planned_running_completed(tmp_path):
    value = controller(tmp_path)
    assert value.status()["phase"] == "planned"
    assert value.report()["state"] == "planned"
    value.inventory(dependencies={"registry": "a" * 64}, exclusions=["old"], q1_hashes={})
    assert value.status()["phase"] == "inventory"
    assert value.report()["state"] == "running"


def test_inventory_is_exact_31_units_and_source_budget_is_persistent(tmp_path):
    value = controller(tmp_path)
    inventory = value.inventory(dependencies={"registry": "a" * 64}, exclusions=[], q1_hashes={})
    assert len(inventory["source_queue"]) == 31
    with pytest.raises(ValueError, match="workers"):
        value.source({inventory["source_queue"][0]: {"sha256": "b" * 64}}, workers=3, worker_hours=1)
    with pytest.raises(ValueError, match="budget"):
        value.source({inventory["source_queue"][0]: {"sha256": "b" * 64}}, workers=1, worker_hours=19)
    value.source({inventory["source_queue"][0]: {"sha256": "b" * 64}}, workers=1, worker_hours=1)
    assert value.status()["source_worker_hours"] == 1


def test_prepare_refuses_incomplete_census_and_freezes_only_after_all_sources(tmp_path):
    value = controller(tmp_path); inv = value.inventory(dependencies={"registry": "a" * 64}, exclusions=["x"], q1_hashes={})
    manifest = {"schema_version": "lc_campaign_contract_v1", "campaign_id": "tiny", "cases": []}
    with pytest.raises(ValueError, match="all 31"):
        value.prepare(manifest, requests={})
    complete = {unit: {"sha256": "b" * 64} for unit in inv["source_queue"]}
    value.source(complete, workers=2, worker_hours=2)
    assert value.prepare(manifest, requests={})["manifest"] == manifest


def test_score_requires_task2_reveal_gate_and_shared_manifest(tmp_path):
    value = controller(tmp_path); inv = value.inventory(dependencies={"registry": "a" * 64}, exclusions=[], q1_hashes={})
    value.source({unit: {"sha256": "b" * 64} for unit in inv["source_queue"]}, workers=1, worker_hours=1)
    manifest = {"schema_version": "lc_campaign_contract_v1", "campaign_id": "tiny", "cases": []}; value.prepare(manifest, requests={})
    with pytest.raises(ValueError, match="reveal"):
        value.score({}, {})
    value._ledger.revealed = True
    with pytest.raises(ValueError, match="manifest"):
        value.score({}, {"campaign_manifest": {}})

