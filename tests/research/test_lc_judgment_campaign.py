"""Tests for the versioned, source-only LC judgment preparation freeze."""
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path

import pytest

from scripts.research.assessment_evidence_guard import _canonical
from scripts.research.lc_campaign import SOURCE_UNITS
from scripts.research.lc_judgment_campaign import JudgmentCampaign


def _candidate(case_id, decision_time):
    return {
        "candidate_id": case_id,
        "decision_time": decision_time,
        "track": "hourly",
        "features": {"close": 100.0},
    }


class FakeSourceController:
    def __init__(self, state, source_by_month):
        self.state = state
        self.source_by_month = source_by_month

    def _read(self):
        return deepcopy(self.state)

    def _validate_q1_receipt(self, receipt):
        return deepcopy(self.source_by_month[receipt["month"]])

    def _validate_source_receipt(self, receipt):
        return deepcopy(self.source_by_month[receipt["month"]]), {"month": receipt["month"]}


@pytest.fixture
def census(tmp_path):
    source_run = tmp_path / "source-run"
    source_run.mkdir()
    (source_run / "controller.json").write_text('{"frozen":"census"}')

    source_by_month = {
        month: {"month": month, "candidates": [],
                "parent_ledgers": {"4H_N3": {"month": month}, "1D_N3": {}}}
        for month in SOURCE_UNITS
    }
    old_start = datetime(2024, 1, 1, tzinfo=timezone.utc)
    source_by_month["2024-01"]["candidates"] = [
        _candidate(f"old-{index:03d}", (old_start + timedelta(hours=index)).isoformat())
        for index in range(106)
    ]
    interval_start = datetime(2026, 1, 1, tzinfo=timezone.utc)
    source_by_month["2026-01"]["candidates"] = [
        _candidate(f"case-{index:02d}", (interval_start + timedelta(hours=index)).isoformat())
        for index in range(36)
    ]

    sources = {}
    for month in SOURCE_UNITS:
        if month in {"2026-01", "2026-02", "2026-03"}:
            sources[month] = {
                "month": month,
                "source_file": str(tmp_path / f"{month}.json"),
                "source_file_sha256": str(int(month[-2:])) * 64,
                "hourly_input_hash": "a" * 64,
                "candidate_ids": [row["candidate_id"] for row in source_by_month[month]["candidates"]],
            }
        else:
            sources[month] = {
                "month": month,
                "directory": str(tmp_path / month),
                "source_sha256": "b" * 64,
                "manifest_sha256": "c" * 64,
            }

    exposure_identities = [
        {"candidate_id": f"case-{index:02d}",
         "decision_time": (interval_start + timedelta(hours=index)).isoformat(),
         "provenance": [{"registry": "prior.json", "record": index}]}
        for index in range(14)
    ]
    time_match_exclusions = [
        {"decision_time": (interval_start + timedelta(hours=index)).isoformat(),
         "provenance": f"null-id:{index}"}
        for index in (14, 15)
    ]
    state = {
        "version": "lc_campaign_controller_v2",
        "inventory": {
            "schema_version": "lc_campaign_inventory_v1",
            "source_queue": list(SOURCE_UNITS),
            "registries": [{"path": "prior.json", "sha256": "d" * 64, "exposures": []}],
            "exposure_identities": exposure_identities,
            "time_match_exclusions": time_match_exclusions,
            "current_dependencies": [],
            "file_state_guards": [],
            "q1_receipts": [],
        },
        "sources": sources,
        "source_attempts": {},
        "reserved_worker_hours": 0,
        "prepared": None,
        "operations": {},
        "score": None,
        "sha256": "e" * 64,
    }
    controller = FakeSourceController(state, source_by_month)
    campaign = JudgmentCampaign(
        tmp_path / "judgment", source_run_dir=source_run,
        source_controller_factory=lambda _: controller,
    )
    return campaign, controller, source_run


def test_prepare_freezes_the_approved_chronological_unassessed_roster(census):
    """Catches reuse of the old consecutive-block selector or a padded sample."""
    campaign, _, _ = census
    prepared = campaign.prepare()

    assert prepared["schema_version"] == "lc_judgment_preparation_v1"
    assert prepared["policy"] == "lc_unassessed_chronological_v1"
    assert prepared["selection"] == {
        "census_candidate_count": 142,
        "interval_candidate_count": 36,
        "excluded_count": 16,
        "cap": 30,
        "roster_count": 20,
        "interval_start": "2026-01-01T00:00:00+00:00",
        "interval_end_exclusive": "2026-08-01T00:00:00+00:00",
    }
    assert prepared["roster"] == [f"case-{index:02d}" for index in range(16, 36)]
    assert prepared["roles_enabled"] is False
    assert len(prepared["cases"]) == 20
    assert prepared["cases"][0]["decision_time"] == "2026-01-01T16:00:00+00:00"
    assert prepared == campaign.prepare()

    raw = (campaign.run_dir / "judgment_prepare.json").read_bytes()
    assert raw == _canonical(json.loads(raw)).encode("ascii")


def test_verify_rejects_changed_census_controller_or_source_projection(census):
    """Catches a preparation freeze that trusts stale receipt or exposure metadata."""
    campaign, controller, source_run = census
    campaign.prepare()

    (source_run / "controller.json").write_text('{"changed":true}')
    with pytest.raises(ValueError, match="source controller hash"):
        campaign.verify()

    (source_run / "controller.json").write_text('{"frozen":"census"}')
    controller.source_by_month["2026-01"]["candidates"][16]["features"]["close"] = 99.0
    with pytest.raises(ValueError, match="prepared projection"):
        campaign.verify()


def test_case_source_returns_validated_full_source_and_receipt_only_for_roster(census):
    """Catches source access that loses parent ledgers/provenance or bypasses roster scope."""
    campaign, controller, _ = census
    campaign.prepare()

    material = campaign.case_source("case-16")
    assert material["candidate"]["candidate_id"] == "case-16"
    assert material["parent_ledgers"] == {
        "4H_N3": {"month": "2026-01"}, "1D_N3": {}}
    assert material["source"]["month"] == "2026-01"
    assert material["receipt"]["month"] == "2026-01"
    assert material["receipt"]["source_file"].endswith("2026-01.json")
    assert material["receipt"]["source_file_sha256"] == "1" * 64

    material["candidate"]["features"]["close"] = 0
    assert controller.source_by_month["2026-01"]["candidates"][16]["features"]["close"] == 100.0
    with pytest.raises(ValueError, match="frozen roster"):
        campaign.case_source("case-00")


def test_prepare_refuses_contaminated_or_incomplete_census_state(census):
    """Catches accidental preparation from the old cohort path or partial receipts."""
    campaign, controller, _ = census
    controller.state["prepared"] = {"old": "consecutive-block"}
    with pytest.raises(ValueError, match="source-only census"):
        campaign.prepare()

    controller.state["prepared"] = None
    controller.state["sources"].pop("2025-12")
    with pytest.raises(ValueError, match="exact 31"):
        campaign.prepare()
