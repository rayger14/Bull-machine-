"""Immutable monthly LC source census for the consolidated campaign.

This module is a separately versioned wrapper around the reviewed causal source
construction.  It does not change or monkeypatch the frozen Q1 collector.
Heavy replay dependencies are imported only by the source builder.
"""

from copy import deepcopy
from datetime import datetime, timedelta, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile


ROOT = Path(__file__).resolve().parents[2]
ARCHIVE = ROOT / "data/recovered_binance_minute_2026_09_14/btc_1m_2021_2026.parquet"
PRIVATE_PREPARER = ROOT / "results/agent_layered_entry_2026_09_11/prepare_sources.py"
INVENTORY = ROOT / (
    "results/research_validation_2026_09_10/minute_parent_coverage/calendar_inventory.json"
)
REFERENCE = ROOT / (
    "results/research_validation_2026_09_10/h3_parent_permission/"
    "june_minute_frozen_reference.json"
)
LC_CONFIG = ROOT / "configs/archetypes/liquidity_compression.yaml"
OLD_COLLECTOR = ROOT / "scripts/research/lc_source_population.py"
OLD_COLLECTOR_SHA256 = "3ca33499bf96bdefd93581436c4232fdca326a6a4a6df984227cbf364f296b3f"
SOURCE_SHA256 = "5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035"
SOURCE_SCHEMA = "lc-source-population-v1"
MANIFEST_SCHEMA = "lc-campaign-candidate-manifest-v1"
WRAPPER_ID = "lc-campaign-source-wrapper-v1"
CONSTRUCTION_ID = "lc-persistent-master-source-v1"
ONE_MINUTE = timedelta(minutes=1)
EXPECTED_ARCHETYPES = frozenset(
    {
        "confluence_breakout",
        "exhaustion_reversal",
        "failed_continuation",
        "funding_divergence",
        "fvg_continuation",
        "liquidity_compression",
        "liquidity_sweep",
        "liquidity_vacuum",
        "long_squeeze",
        "oi_divergence",
        "order_block_retest",
        "retest_cluster",
        "spring",
        "trap_within_trend",
        "volume_fade_chop",
        "whipsaw",
        "wick_trap",
    }
)
MONTHS = (
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
SAVED_Q1 = {
    "2026-01": (
        ROOT / "results/lc_persistent_master_2026_09_14/source/2026-01/source.json",
        "6ba371d2ded2fb152176b6e82ea2bd56f4ba5714e982f27ee04f44efaa0ad51e",
    ),
    "2026-02": (
        ROOT / "results/lc_persistent_master_2026_09_14/source/2026-02/source.json",
        "8c703b3802e45fc351493dab238e370c753ad7d60d971507d0319531f7cd5ec7",
    ),
    "2026-03": (
        ROOT / "results/lc_persistent_master_2026_09_14/source/2026-03/source.json",
        "4de2cdbf7b8247aebac3e0c0d92129bac8bf4fa249686e26778ea8b83561ef14",
    ),
}
Q1_PROJECTIONS = {
    "2026-01": {
        "source_file_sha256": SAVED_Q1["2026-01"][1],
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
        "source_file_sha256": SAVED_Q1["2026-02"][1],
        "hourly_input_hash": "7cc34ea32df069f87e1484fc15025116245c0c50a3a448675aeaa883bb7d0c0d",
        "candidate_ids": [
            "hourly-lc:2026-02-02T04:00:00+00:00",
            "hourly-lc:2026-02-23T02:00:00+00:00",
            "hourly-lc:2026-02-25T02:00:00+00:00",
            "hourly-lc:2026-02-28T07:00:00+00:00",
        ],
    },
    "2026-03": {
        "source_file_sha256": SAVED_Q1["2026-03"][1],
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


def month_bounds(month: str) -> tuple[datetime, datetime, datetime]:
    """Return the fixed UTC month start, exclusive end, and 30-day seed."""
    if month not in MONTHS:
        raise ValueError("month must be in the fixed campaign allowlist: " + ", ".join(MONTHS))
    start = datetime.strptime(month, "%Y-%m").replace(tzinfo=timezone.utc)
    if start.month == 12:
        end = start.replace(year=start.year + 1, month=1)
    else:
        end = start.replace(month=start.month + 1)
    return start, end, start - timedelta(days=30)


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc(value, label):
    if not isinstance(value, (str, datetime)):
        raise ValueError(label + " must be a valid UTC timestamp")
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
    except ValueError as exc:
        raise ValueError(label + " must be a valid UTC timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(label + " must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _merge_hashes(target, additions, label):
    if not isinstance(additions, dict):
        raise ValueError(label + " files must be a mapping")
    for raw_path, expected in additions.items():
        path = str((ROOT / Path(raw_path)).resolve())
        if path in target and target[path] != expected:
            raise ValueError("conflicting " + label + " hash: " + path)
        target[path] = expected


def _nested_manifest_files(node, target, label):
    if isinstance(node, list):
        for child in node:
            _nested_manifest_files(child, target, label)
        return
    if not isinstance(node, dict):
        return
    if "files" in node:
        _merge_hashes(target, node["files"], label)
    for key, child in node.items():
        if key != "files":
            _nested_manifest_files(child, target, label)


def _manifest_hashes(source):
    expected = {}
    for name in ("source_manifest", "code_manifest", "config_manifest"):
        manifest = source.get(name)
        if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
            raise ValueError("complete " + name + " files are required")
        _merge_hashes(expected, manifest["files"], name)
    _nested_manifest_files(
        source["source_manifest"].get("replay"),
        expected,
        "source replay manifest",
    )
    return expected


def _verify_hashes(expected):
    for raw_path, wanted in expected.items():
        path = Path(raw_path)
        actual = _sha(path) if path.exists() else None
        if actual != wanted:
            raise ValueError("file hash mismatch: " + str(path))


def _runtime_manifest():
    import numpy as np
    import pandas as pd
    import talib

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "talib": "python-talib={};library={}".format(
            talib.__version__, talib.__ta_version__.decode()
        ),
    }


def _verify_archive_inventory(inventory):
    if not isinstance(inventory, dict):
        raise ValueError("archive inventory must be a mapping")
    if (
        inventory.get("source_sha256") != SOURCE_SHA256
        or not ARCHIVE.exists()
        or _sha(ARCHIVE) != SOURCE_SHA256
    ):
        raise ValueError("original minute archive hash mismatch")


def _hourly_from_minutes(minute, seed, end, *, max_input_hours):
    import pandas as pd

    required = ("open", "high", "low", "close", "volume")
    if list(minute.columns) != list(required):
        raise ValueError("minute source must have exact OHLCV columns")
    expected = pd.date_range(seed, end, freq="1min", inclusive="left")
    if not minute.index.equals(expected):
        raise ValueError("fixed seed/month minute coverage is incomplete")
    grouped = minute.groupby(minute.index.floor("h"))
    if not (grouped.size() == 60).all():
        raise ValueError("only complete source hours may enter replay")
    hourly = grouped.agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    if len(hourly) > max_input_hours:
        raise ValueError("hourly source exceeds parent-ledger input cap")
    return hourly


def _validate_candidates(candidates, start, end):
    if not isinstance(candidates, list):
        raise ValueError("candidates must be a list")
    ids = []
    clocks = []
    for candidate in candidates:
        if not isinstance(candidate, dict):
            raise ValueError("candidate must be a mapping")
        candidate_id = candidate.get("candidate_id")
        if not isinstance(candidate_id, str) or not candidate_id:
            raise ValueError("candidate_id must be a non-empty string")
        decision = _utc(candidate.get("decision_time"), "candidate decision_time")
        if not start <= decision < end:
            raise ValueError("candidate decision must fall inside its source month")
        if candidate.get("track") != "hourly":
            raise ValueError("campaign source candidates must remain hourly")
        if candidate_id != "hourly-lc:" + decision.isoformat():
            raise ValueError("candidate_id must match decision_time")
        ids.append(candidate_id)
        clocks.append(decision)
    if len(set(ids)) != len(ids):
        raise ValueError("candidate IDs must be unique")
    if clocks != sorted(clocks) or len(set(clocks)) != len(clocks):
        raise ValueError("candidate decisions must be unique and chronological")


def _is_pinned_saved_source(source, month):
    saved = SAVED_Q1.get(month)
    if saved is None:
        return False
    path, expected_hash = saved
    if not path.exists() or _sha(path) != expected_hash:
        return False
    rendered_hash = hashlib.sha256(_json_text(source).encode()).hexdigest()
    return rendered_hash == expected_hash


def _validate_source(source, month):
    if not isinstance(source, dict):
        raise ValueError("source must be a mapping")
    start, end, seed = month_bounds(month)
    if source.get("schema") != SOURCE_SCHEMA or source.get("month") != month:
        raise ValueError("source schema or month differs from campaign contract")
    for key, expected in (("start", start), ("end_exclusive", end), ("seed", seed)):
        if _utc(source.get(key), key) != expected:
            raise ValueError(key + " differs from fixed month bounds")
    if source.get("seed_days") != 30:
        raise ValueError("source must retain the reviewed 30-day seed")
    source_path = Path(source.get("source_path", ""))
    if (
        source_path.resolve() != ARCHIVE.resolve()
        and not _is_pinned_saved_source(source, month)
    ):
        raise ValueError("source must use the permanent minute archive")
    if source.get("source_sha256") != SOURCE_SHA256:
        raise ValueError("source archive identity differs")
    if not ARCHIVE.exists() or _sha(ARCHIVE) != SOURCE_SHA256:
        raise ValueError("source archive hash mismatch")
    expected_minutes = int((end - seed) / ONE_MINUTE)
    if source.get("minute_rows") != expected_minutes:
        raise ValueError("fixed seed/month minute coverage is incomplete")
    if source.get("hourly_rows") != expected_minutes // 60:
        raise ValueError("fixed seed/month hourly coverage is incomplete")
    if source["hourly_rows"] > 2048:
        raise ValueError("hourly source exceeds parent-ledger input cap")
    candidates = source.get("candidates")
    _validate_candidates(candidates, start, end)
    if source.get("candidate_count") != len(candidates):
        raise ValueError("candidate_count differs from candidates")
    if set(source.get("parent_ledgers", {})) != {"4H_N3", "1D_N3"}:
        raise ValueError("both causal parent ledgers are required")
    atr_source = source.get("atr_contract", {}).get("source")
    if not isinstance(atr_source, str) or not atr_source.startswith(CONSTRUCTION_ID + ";"):
        raise ValueError("legacy parent-ledger construction identity differs")
    replay = source.get("source_manifest", {}).get("replay")
    if not isinstance(replay, dict):
        raise ValueError("complete replay manifest is required")
    order = replay.get("signals", {}).get("archetype_order")
    if len(order or []) != 17 or set(order) != EXPECTED_ARCHETYPES:
        raise ValueError("native replay must evaluate the fixed all-17 inventory")
    if source.get("runtime") != _runtime_manifest():
        raise ValueError("source runtime differs from the frozen runtime manifest")
    source_files = {}
    _merge_hashes(source_files, source["source_manifest"].get("files"), "source_manifest")
    config_files = {}
    _merge_hashes(config_files, source["config_manifest"].get("files"), "config_manifest")
    if source_files.get(str(ARCHIVE.resolve())) != SOURCE_SHA256:
        raise ValueError("permanent minute archive is absent from source manifest")
    if str(LC_CONFIG.resolve()) not in config_files:
        raise ValueError("LC configuration is absent from config manifest")
    expected_hashes = _manifest_hashes(source)
    old_path = str(OLD_COLLECTOR.resolve())
    if expected_hashes.get(old_path) != OLD_COLLECTOR_SHA256:
        raise ValueError("frozen Q1 collector manifest identity differs")
    _verify_hashes(expected_hashes)


def saved_q1_projections():
    projections = {}
    for month, (path, expected_hash) in SAVED_Q1.items():
        if _sha(path) != expected_hash:
            raise ValueError("saved Q1 source hash mismatch: " + month)
        source = json.loads(path.read_text())
        projections[month] = {
            "source_file_sha256": expected_hash,
            "hourly_input_hash": source.get("hourly_input_hash"),
            "candidate_ids": [row.get("candidate_id") for row in source.get("candidates", [])],
        }
    return projections


def _load_saved_q1(month):
    projections = saved_q1_projections()
    if projections.get(month) != Q1_PROJECTIONS[month]:
        raise ValueError("saved Q1 source projection differs: " + month)
    source = json.loads(SAVED_Q1[month][0].read_text())
    _validate_source(source, month)
    return source


def _reviewed_q1_hashes():
    """Return file expectations authenticated by the pinned Q1 source payloads."""
    reviewed = {}
    for month, (path, expected_source_hash) in SAVED_Q1.items():
        if not path.exists() or _sha(path) != expected_source_hash:
            raise ValueError("saved Q1 source hash mismatch: " + month)
        source = json.loads(path.read_text())
        _merge_hashes(reviewed, _manifest_hashes(source), "reviewed Q1 baseline")
    return reviewed


def _reviewed_hash(path, reviewed, label):
    resolved = str(Path(path).resolve())
    if resolved not in reviewed:
        raise ValueError(label + " is absent from reviewed Q1 baseline: " + resolved)
    expected = reviewed[resolved]
    actual = _sha(path) if Path(path).exists() else None
    if actual != expected:
        raise ValueError(label + " differs from reviewed Q1 baseline: " + resolved)
    return expected


def _production_context():
    """Load the reviewed source services without altering the frozen collector."""
    from contextlib import redirect_stdout
    import io
    import socket
    import sys
    import time
    from unittest.mock import patch

    import pandas as pd
    import talib

    from results.agent_layered_entry_2026_09_11.prepare_sources import LogEvidence
    from scripts.research.causal_parent_ledger import MAX_INPUT_HOURS, build_parent_ledger
    from scripts.research.engine_signal_replay import SignalEngine, run_signal_replay
    from scripts.research.lc_source_population import collect_native_lc
    from scripts.research.replay_clock import digest, validate_bars
    from scripts.research.virtual_book_replay import side_effect_guard

    reviewed = _reviewed_q1_hashes()
    _reviewed_hash(INVENTORY, reviewed, "archive inventory")
    _reviewed_hash(REFERENCE, reviewed, "parent reference")
    archive_expected = _reviewed_hash(ARCHIVE, reviewed, "permanent archive")
    if archive_expected != SOURCE_SHA256:
        raise ValueError("permanent archive differs from reviewed Q1 baseline")
    inventory = json.loads(INVENTORY.read_text())
    reference = json.loads(REFERENCE.read_text())
    _verify_archive_inventory(inventory)
    reference_manifest = reference["ledgers"][0]["manifest"]
    parent_paths = deepcopy(reference_manifest["source_paths"])
    parent_hashes = deepcopy(reference_manifest["source_hashes"])
    for key, path in parent_paths.items():
        expected = _reviewed_hash(path, reviewed, "parent source " + key)
        if expected != parent_hashes.get(key):
            raise ValueError("parent source differs from reviewed Q1 reference: " + key)
    stream = reference_manifest["data_stream_id"]
    if stream != "btc_1m_2021_2026_saved_5b8a4533f70b8ccd":
        raise ValueError("unexpected data stream identity")
    runtime = _runtime_manifest()
    if runtime["talib"] != reference["atr_contract"]["version"]:
        raise ValueError("TA-Lib runtime differs from frozen parent reference")

    code_paths = [
        Path(__file__),
        OLD_COLLECTOR,
        PRIVATE_PREPARER,
        ROOT / "scripts/research/engine_signal_replay.py",
        ROOT / "scripts/research/live_feature_replay.py",
        ROOT / "scripts/research/causal_parent_ledger.py",
        ROOT / "scripts/research/replay_clock.py",
        ROOT / "scripts/research/virtual_book_replay.py",
        ROOT / "engine/archetypes/logic.py",
    ]
    source_paths = [ARCHIVE, INVENTORY, REFERENCE]
    source_paths.extend(Path(path) for path in parent_paths.values())
    config_paths = [ROOT / "configs/champion_paper.json", LC_CONFIG]
    source_hashes = {}
    for path in source_paths:
        expected = (
            SOURCE_SHA256
            if path.resolve() == ARCHIVE.resolve()
            else _reviewed_hash(path, reviewed, "shared source")
        )
        source_hashes[str(path.resolve())] = expected
    code_hashes = {}
    for path in code_paths:
        expected = (
            _sha(path)
            if path.resolve() == Path(__file__).resolve()
            else _reviewed_hash(path, reviewed, "shared code")
        )
        code_hashes[str(path.resolve())] = expected
    config_hashes = {
        str(path.resolve()): _reviewed_hash(path, reviewed, "shared config")
        for path in config_paths
    }
    if code_hashes.get(str(OLD_COLLECTOR.resolve())) != OLD_COLLECTOR_SHA256:
        raise ValueError("frozen Q1 collector hash mismatch")
    guarded = {}
    for label, hashes in (
        ("source", source_hashes),
        ("code", code_hashes),
        ("config", config_hashes),
    ):
        _merge_hashes(guarded, hashes, label)
    _verify_hashes(guarded)

    def load_minutes(seed, end):
        minute = pd.read_parquet(
            ARCHIVE,
            columns=["open", "high", "low", "close", "vol"],
            filters=[("ts", ">=", pd.Timestamp(seed)), ("ts", "<", pd.Timestamp(end))],
        ).rename(columns={"vol": "volume"})
        return minute[["open", "high", "low", "close", "volume"]]

    def run_replay(hourly, month):
        logs = LogEvidence()
        captured = io.StringIO()
        began = time.monotonic()
        progress_output = sys.stdout
        original_update = SignalEngine.update

        def update_with_progress(engine, *args, **kwargs):
            result = original_update(engine, *args, **kwargs)
            if engine.bar_index % 120 == 0:
                print(
                    "SOURCE_PROGRESS",
                    month,
                    engine.bar_index,
                    "/",
                    len(hourly),
                    "hours",
                    round(time.monotonic() - began, 1),
                    "seconds",
                    file=progress_output,
                    flush=True,
                )
            return result

        print("SOURCE_LAUNCH", month, len(hourly), "hours", flush=True)
        try:
            with (
                redirect_stdout(captured),
                side_effect_guard(logs),
                patch.object(socket, "has_ipv6", False),
                patch.object(SignalEngine, "update", update_with_progress),
            ):
                replay = run_signal_replay(hourly, instrument="BTC", timeframe="1h")
        except BaseException:
            print(captured.getvalue()[-20000:], file=sys.stderr)
            print(json.dumps(logs, default=str), file=sys.stderr)
            raise
        if logs:
            print(json.dumps(logs, default=str), file=sys.stderr)
        return replay, list(logs), logs.warning_count

    return {
        "load_minutes": load_minutes,
        "validate_bars": validate_bars,
        "digest": digest,
        "run_replay": run_replay,
        "collect_native_lc": collect_native_lc,
        "build_parent_ledger": build_parent_ledger,
        "atr": talib.ATR,
        "max_input_hours": MAX_INPUT_HOURS,
        "stream": stream,
        "reference_atr_contract": deepcopy(reference["atr_contract"]),
        "parent_source_paths": parent_paths,
        "parent_source_hashes": parent_hashes,
        "source_hashes": source_hashes,
        "code_hashes": code_hashes,
        "config_hashes": config_hashes,
        "runtime": runtime,
    }


def _build_month_source(month, *, context=None):
    """Run the unchanged causal replay/LC/parent primitives for one source unit."""
    import pandas as pd

    start_dt, end_dt, seed_dt = month_bounds(month)
    services = _production_context() if context is None else context
    guarded = {}
    for label, hashes in (
        ("source", services["source_hashes"]),
        ("code", services["code_hashes"]),
        ("config", services["config_hashes"]),
    ):
        _merge_hashes(guarded, hashes, label)
    _verify_hashes(guarded)

    start, end, seed = map(pd.Timestamp, (start_dt, end_dt, seed_dt))
    minute = services["load_minutes"](seed, end)
    services["validate_bars"](minute, "1min")
    hourly = _hourly_from_minutes(
        minute,
        seed,
        end,
        max_input_hours=services["max_input_hours"],
    )
    hourly_rows = [
        {"timestamp": str(timestamp), **record}
        for timestamp, record in zip(hourly.index, hourly.to_dict("records"))
    ]
    hourly_input_hash = services["digest"](hourly_rows)
    replay, error_evidence, warning_count = services["run_replay"](hourly, month)
    replay_manifest = deepcopy(replay["source_manifest"])
    order = replay_manifest.get("signals", {}).get("archetype_order")
    if len(order or []) != 17 or set(order) != EXPECTED_ARCHETYPES:
        raise ValueError("native replay did not evaluate the fixed all-17 inventory")
    replay_hashes = dict(guarded)
    _nested_manifest_files(replay_manifest, replay_hashes, "source replay manifest")
    _verify_hashes(replay_hashes)
    candidates = services["collect_native_lc"](replay["rows"], str(start), str(end))
    _validate_candidates(candidates, start_dt, end_dt)

    atr_contract = {
        **deepcopy(services["reference_atr_contract"]),
        "source": (
            CONSTRUCTION_ID
            + "; same-minute-stream complete hours; independent cold start {}; "
            "exactly 30 days before {}"
        ).format(seed, start),
    }
    hourly["atr_14"] = services["atr"](
        hourly.high.to_numpy(),
        hourly.low.to_numpy(),
        hourly.close.to_numpy(),
        timeperiod=14,
    )
    if not hourly.atr_14.iloc[:14].isna().all() or not hourly.atr_14.iloc[14:].notna().all():
        raise ValueError("unexpected TA-Lib ATR warmup boundary")
    hourly["atr_available_at"] = hourly.index + pd.Timedelta("1h")
    ledgers = {}
    for anchor in ("4H", "1D"):
        ledgers[anchor + "_N3"] = services["build_parent_ledger"](
            hourly,
            instrument="BTC",
            data_stream_id=services["stream"],
            anchor_timeframe=anchor,
            pivot_n=3,
            atr_contract=atr_contract,
            source_paths=services["parent_source_paths"],
            expected_hashes=services["parent_source_hashes"],
        )
    _verify_hashes(replay_hashes)

    code_manifest = {"files": deepcopy(services["code_hashes"])}
    private_path = str(PRIVATE_PREPARER.resolve())
    if private_path in services["code_hashes"]:
        code_manifest["private_local_reproduction_dependency"] = {
            "path": str(PRIVATE_PREPARER),
            "sha256": services["code_hashes"][private_path],
            "helpers": ["LogEvidence"],
            "availability": "ignored local artifact required only for source replay",
        }
    return {
        "schema": SOURCE_SCHEMA,
        "certified": False,
        "scope": (
            "source-only native long LC population before winner/H2 filtering; "
            "no outcomes, market roles, selection, allocation, or book replay"
        ),
        "month": month,
        "source_path": str(ARCHIVE.resolve()),
        "source_sha256": SOURCE_SHA256,
        "seed": str(seed),
        "start": str(start),
        "end_exclusive": str(end),
        "seed_days": 30,
        "data_stream_id": services["stream"],
        "minute_rows": len(minute),
        "hourly_rows": len(hourly),
        "hourly_input_hash": hourly_input_hash,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "parent_ledgers": ledgers,
        "atr_contract": atr_contract,
        "replay_contract_id": replay["contract_id"],
        "replay_blockers": deepcopy(replay["blockers"]),
        "replay_error_evidence": deepcopy(error_evidence),
        "suppressed_warning_count": warning_count,
        "source_manifest": {
            "files": deepcopy(services["source_hashes"]),
            "replay": replay_manifest,
            "recovered_parent_source_paths": deepcopy(services["parent_source_paths"]),
            "recovered_parent_source_hashes": deepcopy(services["parent_source_hashes"]),
        },
        "code_manifest": code_manifest,
        "config_manifest": {
            "files": deepcopy(services["config_hashes"]),
            "effective_signal_config": deepcopy(
                replay_manifest["signals"]["effective_config"]
            ),
        },
        "runtime": deepcopy(services["runtime"]),
        "missing_input_limits": [
            "Historical candle receipt times are unavailable; bar-close availability is reconstructed.",
            "Historical macro and derivatives observations are absent and were not backfilled or defaulted.",
            "Each month independently resets native replay and parent history at its fixed 30-day seed.",
            "Reproduction requires the hash-pinned ignored private preparer named in code_manifest.",
            "This source population contains no outcome, assessment, selection, allocation, or execution result.",
        ],
    }


def _source_for_month(month):
    if month in SAVED_Q1:
        return _load_saved_q1(month)
    return _build_month_source(month)


def _json_text(value):
    from scripts.research.replay_clock import json_safe

    return json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"


def _candidate_manifest(source, source_text):
    start, end, seed = month_bounds(source["month"])
    candidates = [
        {
            "candidate_id": row["candidate_id"],
            "decision_time": _utc(row["decision_time"], "decision_time").isoformat(),
            "exposure_key": row["candidate_id"],
            "track": row.get("track"),
        }
        for row in source["candidates"]
    ]
    return {
        "schema": MANIFEST_SCHEMA,
        "wrapper": WRAPPER_ID,
        "wrapper_sha256": _sha(__file__),
        "source_payload_schema": SOURCE_SCHEMA,
        "construction": CONSTRUCTION_ID,
        "month": source["month"],
        "seed": seed.isoformat(),
        "start": start.isoformat(),
        "end_exclusive": end.isoformat(),
        "source_file": "source.json",
        "source_file_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
        "candidate_count": len(candidates),
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
        "candidates": candidates,
    }


def _publish_prepared_directory(temporary, output, expected_source_hash):
    """Reserve output without clobbering; publish completion manifest last.

    A directory without ``manifest.json`` is deliberately an incomplete attempt.
    The campaign controller must preserve/report that state rather than treating it
    as a completed source or retrying into the same path.
    """
    try:
        output.mkdir(parents=False, exist_ok=False)
    except FileExistsError as exc:
        raise ValueError("output collision; source artifacts are immutable") from exc
    os.replace(temporary / "source.json", output / "source.json")
    if _sha(output / "source.json") != expected_source_hash:
        raise ValueError("published source hash mismatch; completion withheld")
    os.replace(temporary / "manifest.json", output / "manifest.json")


def prepare_source(month: str, out: Path) -> dict:
    """Create one immutable source directory and return its full source payload."""
    month_bounds(month)
    output = Path(out)
    if output.exists():
        raise ValueError("use a new output path; source artifacts are immutable")
    source = _source_for_month(month)
    _validate_source(source, month)
    source_text = _json_text(source)
    manifest = _candidate_manifest(source, source_text)
    manifest_text = _json_text(manifest)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix="." + output.name + ".", dir=output.parent))
    try:
        with (temporary / "source.json").open("x") as handle:
            handle.write(source_text)
        with (temporary / "manifest.json").open("x") as handle:
            handle.write(manifest_text)
        _publish_prepared_directory(
            temporary,
            output,
            manifest["source_file_sha256"],
        )
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)
    return source
