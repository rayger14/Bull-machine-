from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import importlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


EXPECTED_MONTHS = (
    "2024-01",
    "2024-02",
    "2024-03",
    "2024-04",
    "2024-05",
    "2024-06",
    "2024-07",
    "2024-08",
    "2024-09",
    "2024-10",
    "2024-11",
    "2024-12",
    "2025-01",
    "2025-02",
    "2025-03",
    "2025-04",
    "2025-05",
    "2025-06",
    "2025-07",
    "2025-08",
    "2025-09",
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
    "2026-07",
)


def campaign_source():
    return importlib.import_module("scripts.research.lc_campaign_source")


def test_allowlist_is_exactly_january_2024_through_july_2026():
    module = campaign_source()
    assert module.MONTHS == EXPECTED_MONTHS


def test_first_month_has_seed_and_exclusive_end():
    module = campaign_source()
    start, end, seed = module.month_bounds("2024-01")

    assert start == datetime(2024, 1, 1, tzinfo=timezone.utc)
    assert end == datetime(2024, 2, 1, tzinfo=timezone.utc)
    assert seed == datetime(2023, 12, 2, tzinfo=timezone.utc)


def test_july_has_seed_and_exclusive_end():
    module = campaign_source()
    start, end, seed = module.month_bounds("2026-07")

    assert start.isoformat() == "2026-07-01T00:00:00+00:00"
    assert end.isoformat() == "2026-08-01T00:00:00+00:00"
    assert seed.isoformat() == "2026-06-01T00:00:00+00:00"


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_payload(module, monkeypatch, tmp_path, candidates=None):
    archive = tmp_path / "minute.parquet"
    config = tmp_path / "liquidity_compression.yaml"
    archive.write_bytes(b"fixed minute archive fixture")
    config.write_text("fixture: true\n")
    monkeypatch.setattr(module, "ARCHIVE", archive)
    monkeypatch.setattr(module, "LC_CONFIG", config)
    monkeypatch.setattr(module, "SOURCE_SHA256", _sha(archive))
    candidates = candidates or [
        {
            "candidate_id": "hourly-lc:2024-01-03T04:00:00+00:00",
            "track": "hourly",
            "decision_time": "2024-01-03T04:00:00+00:00",
            "features": {"close": 100.0},
        }
    ]
    return {
        "schema": "lc-source-population-v1",
        "month": "2024-01",
        "source_path": str(archive),
        "source_sha256": _sha(archive),
        "seed": "2023-12-02 00:00:00+00:00",
        "start": "2024-01-01 00:00:00+00:00",
        "end_exclusive": "2024-02-01 00:00:00+00:00",
        "seed_days": 30,
        "minute_rows": 87840,
        "hourly_rows": 1464,
        "hourly_input_hash": "b" * 64,
        "candidate_count": len(candidates),
        "candidates": deepcopy(candidates),
        "parent_ledgers": {"4H_N3": {}, "1D_N3": {}},
        "atr_contract": {
            "source": (
                "lc-persistent-master-source-v1; same-minute-stream complete hours; "
                "independent cold start 2023-12-02 00:00:00+00:00; exactly 30 days "
                "before 2024-01-01 00:00:00+00:00"
            )
        },
        "source_manifest": {
            "files": {str(archive.resolve()): _sha(archive)},
            "replay": {
                "signals": {
                    "archetype_order": sorted(module.EXPECTED_ARCHETYPES),
                }
            },
        },
        "code_manifest": {
            "files": {
                str(module.OLD_COLLECTOR.resolve()): module.OLD_COLLECTOR_SHA256,
            }
        },
        "config_manifest": {"files": {str(config.resolve()): _sha(config)}},
        "runtime": module._runtime_manifest(),
        "missing_input_limits": [],
    }


def test_unsupported_month_is_rejected_before_creating_output(tmp_path):
    module = campaign_source()
    out = tmp_path / "invalid"

    with pytest.raises(ValueError, match="fixed campaign allowlist"):
        module.prepare_source("2023-12", out)

    assert not out.exists()


def test_duplicate_candidates_are_rejected_before_output(monkeypatch, tmp_path):
    module = campaign_source()
    first = {
        "candidate_id": "hourly-lc:2024-01-03T04:00:00+00:00",
        "track": "hourly",
        "decision_time": "2024-01-03T04:00:00+00:00",
    }
    payload = source_payload(module, monkeypatch, tmp_path, [first, deepcopy(first)])
    monkeypatch.setattr(module, "_source_for_month", lambda month: payload)
    out = tmp_path / "duplicate-output"

    with pytest.raises(ValueError, match="candidate IDs must be unique"):
        module.prepare_source("2024-01", out)

    assert not out.exists()


def test_missing_minute_is_rejected_by_tiny_injected_slice():
    module = campaign_source()
    seed = pd.Timestamp("2024-01-01T00:00:00Z")
    end = pd.Timestamp("2024-01-01T02:00:00Z")
    index = pd.date_range(seed, end, freq="1min", inclusive="left").delete(61)
    minute = pd.DataFrame(
        {name: 1.0 for name in ("open", "high", "low", "close", "volume")},
        index=index,
    )

    with pytest.raises(ValueError, match="minute coverage is incomplete"):
        module._hourly_from_minutes(minute, seed, end, max_input_hours=2048)


def test_new_source_uses_permanent_archive_when_historical_temp_link_is_gone(
    monkeypatch, tmp_path
):
    module = campaign_source()
    archive = tmp_path / "permanent.parquet"
    archive.write_bytes(b"permanent archive")
    monkeypatch.setattr(module, "ARCHIVE", archive)
    monkeypatch.setattr(module, "SOURCE_SHA256", _sha(archive))
    inventory = {
        "source_path": str(tmp_path / "deleted-historical-temp.parquet"),
        "source_sha256": _sha(archive),
    }

    module._verify_archive_inventory(inventory)


@pytest.mark.parametrize("manifest_name", ["source_manifest", "config_manifest"])
def test_altered_source_or_config_is_rejected(monkeypatch, tmp_path, manifest_name):
    module = campaign_source()
    payload = source_payload(module, monkeypatch, tmp_path)
    guarded_path = Path(next(iter(payload[manifest_name]["files"])))
    guarded_path.write_bytes(guarded_path.read_bytes() + b"altered")
    monkeypatch.setattr(module, "_source_for_month", lambda month: payload)
    out = tmp_path / (manifest_name + "-output")

    with pytest.raises(ValueError, match="hash mismatch"):
        module.prepare_source("2024-01", out)

    assert not out.exists()


@pytest.mark.parametrize(
    "source_key,wrong_manifest,expected_message",
    [
        ("ARCHIVE", "config_manifest", "archive is absent from source manifest"),
        ("LC_CONFIG", "source_manifest", "LC configuration is absent from config manifest"),
    ],
)
def test_source_and_config_files_cannot_be_misclassified(
    monkeypatch, tmp_path, source_key, wrong_manifest, expected_message
):
    module = campaign_source()
    payload = source_payload(module, monkeypatch, tmp_path)
    path = str(getattr(module, source_key).resolve())
    correct_manifest = (
        "source_manifest" if wrong_manifest == "config_manifest" else "config_manifest"
    )
    expected_hash = payload[correct_manifest]["files"].pop(path)
    payload[wrong_manifest]["files"][path] = expected_hash
    monkeypatch.setattr(module, "_source_for_month", lambda month: payload)

    with pytest.raises(ValueError, match=expected_message):
        module.prepare_source("2024-01", tmp_path / (source_key + "-misclassified"))


def test_existing_output_collision_does_not_start_replay(monkeypatch, tmp_path):
    module = campaign_source()
    out = tmp_path / "existing"
    out.mkdir()

    def should_not_run(month):
        raise AssertionError("source replay must not run")

    monkeypatch.setattr(module, "_source_for_month", should_not_run)
    with pytest.raises(ValueError, match="new output path"):
        module.prepare_source("2024-01", out)


def test_prepare_source_writes_atomic_source_and_lean_exposure_manifest(
    monkeypatch, tmp_path
):
    module = campaign_source()
    payload = source_payload(module, monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_source_for_month", lambda month: deepcopy(payload))
    out = tmp_path / "prepared"

    returned = module.prepare_source("2024-01", out)

    assert returned == payload
    assert sorted(path.name for path in out.iterdir()) == ["manifest.json", "source.json"]
    assert json.loads((out / "source.json").read_text()) == payload
    manifest = json.loads((out / "manifest.json").read_text())
    assert manifest == {
        "schema": "lc-campaign-candidate-manifest-v1",
        "wrapper": "lc-campaign-source-wrapper-v1",
        "wrapper_sha256": _sha(module.__file__),
        "source_payload_schema": "lc-source-population-v1",
        "construction": "lc-persistent-master-source-v1",
        "month": "2024-01",
        "seed": "2023-12-02T00:00:00+00:00",
        "start": "2024-01-01T00:00:00+00:00",
        "end_exclusive": "2024-02-01T00:00:00+00:00",
        "source_file": "source.json",
        "source_file_sha256": _sha(out / "source.json"),
        "candidate_count": 1,
        "input_limit_disclosures": [
            (
                "Native replay may contain defaulted derivative fallback values; "
                "those values are not historical observations."
            ),
            (
                "Historical macro, model, and calibrator inputs may be absent; "
                "the source replay blockers and error evidence remain authoritative."
            ),
        ],
        "candidates": [
            {
                "candidate_id": "hourly-lc:2024-01-03T04:00:00+00:00",
                "decision_time": "2024-01-03T04:00:00+00:00",
                "exposure_key": "hourly-lc:2024-01-03T04:00:00+00:00",
                "track": "hourly",
            }
        ],
    }
    assert {"outcomes", "prices", "assessment", "selected"}.isdisjoint(manifest)


def test_atomic_publish_rejects_a_racing_output_collision(monkeypatch, tmp_path):
    module = campaign_source()
    payload = source_payload(module, monkeypatch, tmp_path)
    monkeypatch.setattr(module, "_source_for_month", lambda month: payload)
    out = tmp_path / "racing-output"

    def collide(source, destination):
        Path(destination).mkdir()
        raise FileExistsError(destination)

    monkeypatch.setattr(os, "rename", collide)
    with pytest.raises(ValueError, match="output collision"):
        module.prepare_source("2024-01", out)

    assert out.is_dir()
    assert list(out.iterdir()) == []
    assert not list(tmp_path.glob(".racing-output.*"))


def test_nested_replay_manifest_conflict_is_rejected(monkeypatch, tmp_path):
    module = campaign_source()
    payload = source_payload(module, monkeypatch, tmp_path)
    archive_path = str(module.ARCHIVE.resolve())
    payload["source_manifest"]["replay"]["features"] = {
        "files": {archive_path: "0" * 64}
    }
    monkeypatch.setattr(module, "_source_for_month", lambda month: payload)

    with pytest.raises(ValueError, match="conflicting source replay manifest hash"):
        module.prepare_source("2024-01", tmp_path / "conflict")


def test_all_17_archetypes_are_required(monkeypatch, tmp_path):
    module = campaign_source()
    payload = source_payload(module, monkeypatch, tmp_path)
    payload["source_manifest"]["replay"]["signals"]["archetype_order"].pop()
    monkeypatch.setattr(module, "_source_for_month", lambda month: payload)

    with pytest.raises(ValueError, match="fixed all-17 inventory"):
        module.prepare_source("2024-01", tmp_path / "short-inventory")


def test_replay_manifest_cannot_be_omitted(monkeypatch, tmp_path):
    module = campaign_source()
    payload = source_payload(module, monkeypatch, tmp_path)
    del payload["source_manifest"]["replay"]
    monkeypatch.setattr(module, "_source_for_month", lambda month: payload)

    with pytest.raises(ValueError, match="complete replay manifest"):
        module.prepare_source("2024-01", tmp_path / "missing-replay")


def test_exposure_key_identity_cannot_diverge_from_decision(monkeypatch, tmp_path):
    module = campaign_source()
    payload = source_payload(module, monkeypatch, tmp_path)
    payload["candidates"][0]["candidate_id"] = (
        "hourly-lc:2024-01-03T05:00:00+00:00"
    )
    monkeypatch.setattr(module, "_source_for_month", lambda month: payload)

    with pytest.raises(ValueError, match="candidate_id must match decision_time"):
        module.prepare_source("2024-01", tmp_path / "wrong-exposure-key")


def test_tiny_injected_replay_keeps_causal_construction_and_manifests(
    monkeypatch, tmp_path
):
    module = campaign_source()
    archive = tmp_path / "minute.parquet"
    config = tmp_path / "liquidity_compression.yaml"
    archive.write_bytes(b"archive fixture")
    config.write_text("fixture: true\n")
    monkeypatch.setattr(module, "ARCHIVE", archive)
    monkeypatch.setattr(module, "LC_CONFIG", config)
    monkeypatch.setattr(module, "SOURCE_SHA256", _sha(archive))
    start, end, seed = module.month_bounds("2024-01")
    index = pd.date_range(seed, end, freq="1min", inclusive="left")
    minute = pd.DataFrame(
        {
            "open": np.full(len(index), 100.0),
            "high": np.full(len(index), 102.0),
            "low": np.full(len(index), 99.0),
            "close": np.full(len(index), 101.0),
            "volume": np.full(len(index), 2.0),
        },
        index=index,
    )
    calls = {"replay": 0, "collect": 0, "parents": []}
    candidate = {
        "candidate_id": "hourly-lc:2024-01-03T04:00:00+00:00",
        "track": "hourly",
        "decision_time": "2024-01-03T04:00:00+00:00",
    }

    def run_replay(hourly, month):
        calls["replay"] += 1
        assert month == "2024-01"
        assert len(hourly) == 1464
        return {
            "rows": [{"decision_time": candidate["decision_time"]}],
            "source_manifest": {
                "features": {"files": {}},
                "signals": {
                    "files": {},
                    "archetype_order": sorted(module.EXPECTED_ARCHETYPES),
                    "effective_config": {"fixture": True},
                },
            },
            "contract_id": "replay-contract-fixture",
            "blockers": ["cold_start_not_live_state"],
        }, [], 0

    def collect(rows, lower, upper):
        calls["collect"] += 1
        assert rows == [{"decision_time": candidate["decision_time"]}]
        assert module._utc(lower, "lower") == start
        assert module._utc(upper, "upper") == end
        return [deepcopy(candidate)]

    def parent(hourly, **kwargs):
        calls["parents"].append((len(hourly), kwargs["anchor_timeframe"]))
        assert kwargs["atr_contract"]["source"].startswith(
            "lc-persistent-master-source-v1;"
        )
        return {"anchor": kwargs["anchor_timeframe"]}

    def atr(high, low, close, timeperiod):
        assert timeperiod == 14
        values = np.full(len(close), 1.5)
        values[:14] = np.nan
        return values

    source_hashes = {str(archive.resolve()): _sha(archive)}
    code_hashes = {str(module.OLD_COLLECTOR.resolve()): module.OLD_COLLECTOR_SHA256}
    config_hashes = {str(config.resolve()): _sha(config)}
    context = {
        "load_minutes": lambda lower, upper: minute.copy(),
        "validate_bars": lambda bars, timeframe: None,
        "digest": lambda rows: hashlib.sha256(
            json.dumps(rows, sort_keys=True).encode()
        ).hexdigest(),
        "run_replay": run_replay,
        "collect_native_lc": collect,
        "build_parent_ledger": parent,
        "atr": atr,
        "max_input_hours": 2048,
        "stream": "btc_1m_2021_2026_saved_5b8a4533f70b8ccd",
        "reference_atr_contract": {"version": module._runtime_manifest()["talib"]},
        "parent_source_paths": {},
        "parent_source_hashes": {},
        "source_hashes": source_hashes,
        "code_hashes": code_hashes,
        "config_hashes": config_hashes,
        "runtime": module._runtime_manifest(),
    }

    result = module._build_month_source("2024-01", context=context)

    assert calls == {
        "replay": 1,
        "collect": 1,
        "parents": [(1464, "4H"), (1464, "1D")],
    }
    assert result["schema"] == "lc-source-population-v1"
    assert result["source_path"] == str(archive.resolve())
    assert result["source_sha256"] == _sha(archive)
    assert result["candidate_count"] == 1
    assert result["candidates"] == [candidate]
    assert result["parent_ledgers"] == {
        "4H_N3": {"anchor": "4H"},
        "1D_N3": {"anchor": "1D"},
    }
    assert result["source_manifest"]["files"] == source_hashes
    assert result["code_manifest"]["files"] == code_hashes
    assert result["config_manifest"]["files"] == config_hashes


def test_frozen_q1_collector_bytes_are_unchanged():
    module = campaign_source()

    assert _sha(module.OLD_COLLECTOR) == (
        "3ca33499bf96bdefd93581436c4232fdca326a6a4a6df984227cbf364f296b3f"
    )


def test_saved_q1_source_projections_match_reviewed_artifacts():
    module = campaign_source()
    expected = {
        "2026-01": {
            "source_file_sha256": "6ba371d2ded2fb152176b6e82ea2bd56f4ba5714e982f27ee04f44efaa0ad51e",
            "hourly_input_hash": "234a29353227c708648ff52802066f3c9dd301b68f8bc66492ef9fe4b256ca8e",
            "candidate_ids": [
                "hourly-lc:2026-01-02T04:00:00+00:00",
                "hourly-lc:2026-01-04T01:00:00+00:00",
                "hourly-lc:2026-01-05T01:00:00+00:00",
                "hourly-lc:2026-01-16T16:00:00+00:00",
                "hourly-lc:2026-01-19T01:00:00+00:00",
                "hourly-lc:2026-01-20T06:00:00+00:00",
                "hourly-lc:2026-01-25T09:00:00+00:00",
                "hourly-lc:2026-01-29T16:00:00+00:00",
                "hourly-lc:2026-01-31T15:00:00+00:00",
            ],
        },
        "2026-02": {
            "source_file_sha256": "8c703b3802e45fc351493dab238e370c753ad7d60d971507d0319531f7cd5ec7",
            "hourly_input_hash": "7cc34ea32df069f87e1484fc15025116245c0c50a3a448675aeaa883bb7d0c0d",
            "candidate_ids": [
                "hourly-lc:2026-02-02T04:00:00+00:00",
                "hourly-lc:2026-02-23T02:00:00+00:00",
                "hourly-lc:2026-02-25T02:00:00+00:00",
                "hourly-lc:2026-02-28T07:00:00+00:00",
            ],
        },
        "2026-03": {
            "source_file_sha256": "4de2cdbf7b8247aebac3e0c0d92129bac8bf4fa249686e26778ea8b83561ef14",
            "hourly_input_hash": "a8ac96df6fac5ed9a21f46105f45d6a69c9e5eecc3640c63e054a6c3535c2cd7",
            "candidate_ids": [
                "hourly-lc:2026-03-06T14:00:00+00:00",
                "hourly-lc:2026-03-07T20:00:00+00:00",
                "hourly-lc:2026-03-08T23:00:00+00:00",
                "hourly-lc:2026-03-15T23:00:00+00:00",
                "hourly-lc:2026-03-22T22:00:00+00:00",
            ],
        },
    }

    assert module.saved_q1_projections() == expected
