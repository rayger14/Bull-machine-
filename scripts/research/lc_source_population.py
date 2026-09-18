"""Source-only native liquidity-compression population reconstruction.

The historical replay entry point deliberately has a local, ignored dependency on
``results/agent_layered_entry_2026_09_11/prepare_sources.py``.  It reuses that
proven producer's hash and log-evidence helpers, while skipping its minute event
detector, H2 selector, assessment, scoring, and outcome paths.  Heavy and private
dependencies are imported only when :func:`prepare_month` actually runs.
"""

import argparse
from copy import deepcopy
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PRIVATE_PREPARER = ROOT / "results/agent_layered_entry_2026_09_11/prepare_sources.py"
INVENTORY = ROOT / "results/research_validation_2026_09_10/minute_parent_coverage/calendar_inventory.json"
REFERENCE = ROOT / "results/research_validation_2026_09_10/h3_parent_permission/june_minute_frozen_reference.json"
LC_CONFIG = ROOT / "configs/archetypes/liquidity_compression.yaml"
SOURCE_SHA256 = "5b8a4533f70b8ccd0dc533984469e6886ec73ce315d48f150c667ccb97e84035"
MONTHS = ("2026-01", "2026-02", "2026-03")
ONE_HOUR = timedelta(hours=1)


def _utc(value, label):
    """Parse a timezone-aware clock and normalize it to UTC."""
    if not isinstance(value, (str, datetime)):
        raise ValueError(label + " must be a valid UTC timestamp")
    try:
        parsed = value if isinstance(value, datetime) else datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(label + " must be a valid UTC timestamp") from exc
    if parsed.tzinfo is None:
        raise ValueError(label + " must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _fixed_month(month):
    if month not in MONTHS:
        raise ValueError("month must be a fixed Q1 month: " + ", ".join(MONTHS))
    start = datetime.fromisoformat(month + "-01T00:00:00+00:00")
    if month == "2026-01":
        end = datetime(2026, 2, 1, tzinfo=timezone.utc)
    elif month == "2026-02":
        end = datetime(2026, 3, 1, tzinfo=timezone.utc)
    else:
        end = datetime(2026, 4, 1, tzinfo=timezone.utc)
    return start, end, start - timedelta(days=30)


def _mapping(value, label):
    if not isinstance(value, dict):
        raise ValueError(label + " must be a mapping")
    return value


def collect_native_lc(rows, start, end):
    """Return every native long LC row with current/previous features and diagnostics.

    Eligibility is intentionally limited to the captured native direction.  The
    engine's selected winner, structural/H2 diagnostic, numeric gates, and other
    archetypes never filter this source population.
    """
    window_start = _utc(start, "start")
    window_end = _utc(end, "end")
    if window_start >= window_end:
        raise ValueError("start must precede end")
    try:
        source_rows = list(rows)
    except TypeError as exc:
        raise ValueError("rows must be iterable") from exc

    candidates = []
    prior_decision = None
    prior_features = None
    for raw in source_rows:
        row = _mapping(raw, "replay row")
        decision = _utc(row.get("decision_time"), "decision_time")
        if prior_decision is not None and decision <= prior_decision:
            raise ValueError("decision clocks must be unique and strictly increasing")

        output = _mapping(row.get("output"), "row.output")
        features = output.get("features")
        if features is not None:
            features = _mapping(features, "row.output.features")
        previous = (
            prior_features
            if prior_features is not None and prior_decision + ONE_HOUR == decision
            else None
        )

        signal = output.get("engine_signal")
        diagnostic = None
        if signal is not None:
            signal = _mapping(signal, "row.output.engine_signal")
            archetypes = _mapping(signal.get("archetypes"), "engine_signal.archetypes")
            diagnostic = _mapping(
                archetypes.get("liquidity_compression"),
                "liquidity_compression diagnostic",
            )
        native = diagnostic.get("native_signal") if diagnostic is not None else None
        if native is not None:
            native = _mapping(native, "liquidity_compression.native_signal")

        if (
            window_start <= decision < window_end
            and native is not None
            and native.get("direction") == "long"
        ):
            selected = diagnostic.get("selected")
            if not isinstance(selected, bool):
                raise ValueError("liquidity_compression.selected must be boolean")
            decision_text = decision.isoformat()
            candidates.append(
                {
                    "candidate_id": "hourly-lc:" + decision_text,
                    "track": "hourly",
                    "decision_time": decision_text,
                    "setup_open": (decision - ONE_HOUR).isoformat(),
                    "features": deepcopy(features),
                    "previous_features": deepcopy(previous),
                    "native_diagnostic": deepcopy(diagnostic),
                    "native_emitted": selected,
                    "feature_available_at": deepcopy(output.get("features_available_at")),
                    "feature_observation_ids": deepcopy(output.get("feature_observation_ids", {})),
                    "feature_observation_visible_at": deepcopy(
                        output.get("feature_observation_visible_at", {})
                    ),
                }
            )

        prior_decision = decision
        prior_features = features
    return candidates


def _resolved_hashes(paths, sha):
    return {str(Path(path).resolve()): sha(path) for path in paths}


def _build_month_source(month):
    """Run the proven causal source pipeline for one month (slow, local-only)."""
    from contextlib import redirect_stdout
    import io
    import platform
    import socket
    import sys
    import time
    from unittest.mock import patch

    import numpy as np
    import pandas as pd
    import talib

    from results.agent_layered_entry_2026_09_11.prepare_sources import LogEvidence, sha
    from scripts.research.causal_parent_ledger import MAX_INPUT_HOURS, build_parent_ledger
    from scripts.research.engine_signal_replay import (
        EXPECTED_ARCHETYPES,
        SignalEngine,
        run_signal_replay,
    )
    from scripts.research.live_feature_replay_report import assert_unchanged, manifest_files
    from scripts.research.replay_clock import digest, validate_bars
    from scripts.research.virtual_book_replay import side_effect_guard

    start_dt, end_dt, seed_dt = _fixed_month(month)
    start, end, seed = map(pd.Timestamp, (start_dt, end_dt, seed_dt))
    inventory = json.loads(INVENTORY.read_text())
    reference = json.loads(REFERENCE.read_text())
    source_path = Path(inventory["source_path"])
    if sha(source_path) != inventory.get("source_sha256") or sha(source_path) != SOURCE_SHA256:
        raise ValueError("original minute archive hash mismatch")
    reference_manifest = reference["ledgers"][0]["manifest"]
    for key, path in reference_manifest["source_paths"].items():
        if sha(path) != reference_manifest["source_hashes"].get(key):
            raise ValueError("recovered parent source hash mismatch: " + key)
    stream = reference_manifest["data_stream_id"]
    if stream != "btc_1m_2021_2026_saved_5b8a4533f70b8ccd":
        raise ValueError("unexpected data stream identity")
    talib_runtime = "python-talib={};library={}".format(
        talib.__version__, talib.__ta_version__.decode()
    )
    if talib_runtime != reference["atr_contract"]["version"]:
        raise ValueError("TA-Lib runtime differs from frozen parent reference")

    code_paths = [
        Path(__file__),
        PRIVATE_PREPARER,
        ROOT / "scripts/research/engine_signal_replay.py",
        ROOT / "scripts/research/live_feature_replay.py",
        ROOT / "scripts/research/causal_parent_ledger.py",
        ROOT / "scripts/research/replay_clock.py",
        ROOT / "scripts/research/virtual_book_replay.py",
        ROOT / "engine/archetypes/logic.py",
    ]
    source_paths = [source_path, INVENTORY, REFERENCE]
    source_paths.extend(Path(path) for path in reference_manifest["source_paths"].values())
    config_paths = [ROOT / "configs/champion_paper.json", LC_CONFIG]
    code_hashes = _resolved_hashes(code_paths, sha)
    source_hashes = _resolved_hashes(source_paths, sha)
    config_hashes = _resolved_hashes(config_paths, sha)
    guarded_hashes = dict(source_hashes, **code_hashes, **config_hashes)
    assert_unchanged(guarded_hashes)

    archive = pd.read_parquet(source_path).rename(columns={"vol": "volume"})
    archive = archive[["open", "high", "low", "close", "volume"]]
    minute = archive.loc[(archive.index >= seed) & (archive.index < end)].copy()
    validate_bars(minute, "1min")
    expected_minutes = int((end - seed) / pd.Timedelta("1min"))
    if len(minute) != expected_minutes or minute.index[0] != seed:
        raise ValueError("fixed seed/month minute coverage is incomplete")
    if minute.index[-1] + pd.Timedelta("1min") != end:
        raise ValueError("fixed month tail is incomplete")
    grouped = minute.groupby(minute.index.floor("h"))
    if not (grouped.size() == 60).all():
        raise ValueError("only complete source hours may enter replay")
    hourly = grouped.agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    if len(hourly) > MAX_INPUT_HOURS:
        raise ValueError("hourly source exceeds parent-ledger input cap")
    hourly_rows = [
        {"timestamp": str(timestamp), **record}
        for timestamp, record in zip(hourly.index, hourly.to_dict("records"))
    ]
    hourly_input_hash = digest(hourly_rows)

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
    assert_unchanged(guarded_hashes)
    assert_unchanged(manifest_files(replay["source_manifest"]))
    order = replay["source_manifest"]["signals"]["archetype_order"]
    if len(order) != 17 or set(order) != EXPECTED_ARCHETYPES:
        raise ValueError("native replay did not evaluate the fixed all-17 inventory")
    candidates = collect_native_lc(replay["rows"], str(start), str(end))

    atr_contract = {
        **reference["atr_contract"],
        "source": (
            "lc-persistent-master-source-v1; same-minute-stream complete hours; "
            "independent cold start {}; exactly 30 days before {}"
        ).format(seed, start),
    }
    hourly["atr_14"] = talib.ATR(
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
        ledger = build_parent_ledger(
            hourly,
            instrument="BTC",
            data_stream_id=stream,
            anchor_timeframe=anchor,
            pivot_n=3,
            atr_contract=atr_contract,
            source_paths=reference_manifest["source_paths"],
            expected_hashes=reference_manifest["source_hashes"],
        )
        ledgers[anchor + "_N3"] = ledger
    assert_unchanged(guarded_hashes)
    replay_manifest = deepcopy(replay["source_manifest"])
    replay_contract_id = replay["contract_id"]
    replay_blockers = deepcopy(replay["blockers"])
    del replay

    return {
        "schema": "lc-source-population-v1",
        "certified": False,
        "scope": (
            "source-only native long LC population before winner/H2 filtering; "
            "no outcomes, market roles, selection, allocation, or book replay"
        ),
        "month": month,
        "source_path": str(source_path),
        "source_sha256": SOURCE_SHA256,
        "seed": str(seed),
        "start": str(start),
        "end_exclusive": str(end),
        "seed_days": 30,
        "data_stream_id": stream,
        "minute_rows": len(minute),
        "hourly_rows": len(hourly),
        "hourly_input_hash": hourly_input_hash,
        "candidate_count": len(candidates),
        "candidates": candidates,
        "parent_ledgers": ledgers,
        "atr_contract": atr_contract,
        "replay_contract_id": replay_contract_id,
        "replay_blockers": replay_blockers,
        "replay_error_evidence": list(logs),
        "suppressed_warning_count": logs.warning_count,
        "source_manifest": {
            "files": source_hashes,
            "replay": replay_manifest,
            "recovered_parent_source_paths": deepcopy(reference_manifest["source_paths"]),
            "recovered_parent_source_hashes": deepcopy(reference_manifest["source_hashes"]),
        },
        "code_manifest": {
            "files": code_hashes,
            "private_local_reproduction_dependency": {
                "path": str(PRIVATE_PREPARER),
                "sha256": code_hashes[str(PRIVATE_PREPARER.resolve())],
                "helpers": ["LogEvidence", "sha"],
                "availability": "ignored local artifact required only for source replay",
            },
        },
        "config_manifest": {
            "files": config_hashes,
            "effective_signal_config": deepcopy(
                replay_manifest["signals"]["effective_config"]
            ),
        },
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "talib": talib_runtime,
        },
        "missing_input_limits": [
            "Historical candle receipt times are unavailable; bar-close availability is reconstructed.",
            "Historical macro and derivatives observations are absent and were not backfilled or defaulted.",
            "Each month independently resets native replay and parent history at its fixed 30-day seed.",
            "Reproduction requires the hash-pinned ignored private preparer named in code_manifest.",
            "This source population contains no outcome, assessment, selection, allocation, or execution result.",
        ],
    }


def _save_immutable(path, value):
    from scripts.research.replay_clock import json_safe

    path = Path(path)
    text = json.dumps(json_safe(value), indent=2, sort_keys=True, allow_nan=False) + "\n"
    if path.exists():
        if path.read_text() != text:
            raise ValueError("Refusing unequal overwrite of frozen artifact: " + str(path))
        return
    with path.open("x") as handle:
        handle.write(text)


def prepare_month(month, out):
    """Replay one fixed Q1 month; save immutable source-only source.json."""
    _fixed_month(month)
    output = Path(out)
    if output.exists():
        raise ValueError("use a new output directory; source artifacts are immutable")
    output.mkdir(parents=True, exist_ok=False)
    source = _build_month_source(month)
    _save_immutable(output / "source.json", source)
    return source


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--month", required=True, choices=MONTHS)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args(argv)
    prepare_month(args.month, args.out)
    print("SOURCE_ARTIFACT", str(args.out / "source.json"), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
