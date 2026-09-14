import hashlib
import importlib
import json
import sqlite3
import subprocess
import sys
from copy import deepcopy
from datetime import datetime, timezone

import pytest


AS_OF = "2026-01-01T00:00:00Z"


def memory_module():
    try:
        return importlib.import_module("scripts.research.master_memory")
    except ModuleNotFoundError:
        pytest.fail("scripts.research.master_memory has not been implemented")


def doctrine_fixture(**updates):
    record = {
        "kind": "doctrine",
        "author": "source-curator",
        "content": {"rule": "Treat context as evidence, not an outcome label."},
        "source_refs": ["docs/rulecard.md#context"],
        "tags": ["lc", "methodology"],
        "available_at": None,
        "event_end": None,
        "case_ids": [],
    }
    record.update(updates)
    return record


def dated_fixture(**updates):
    record = {
        "kind": "lesson",
        "author": "case-curator",
        "content": {"finding": "The named rejection invalidated the setup."},
        "source_refs": ["results/cases.json#C01"],
        "tags": ["lc", "reviewed-case"],
        "available_at": "2025-12-20T12:00:00Z",
        "event_end": "2025-12-15T08:00:00Z",
        "case_ids": ["C01"],
    }
    record.update(updates)
    return record


def open_memory(path):
    return memory_module().ResearchMemory(path)


def approve(db, record_id, reviewer="independent-reviewer"):
    return db.review(record_id, reviewer, "approve", "Checked against named source")


def canonical_hash(value):
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def test_unreviewed_memory_does_not_enter_snapshot(tmp_path):
    db_path = tmp_path / "memory.sqlite"
    db = open_memory(db_path)
    rid = db.propose(doctrine_fixture())
    first = db.snapshot(as_of=AS_OF, training_end=AS_OF)
    assert first["records"] == []

    before_review = datetime.now(timezone.utc)
    review_id = approve(db, rid)
    after_review = datetime.now(timezone.utc)
    second = db.snapshot(as_of=AS_OF, training_end=AS_OF)

    assert [record["id"] for record in second["records"]] == [rid]
    assert second["review_ids"] == [review_id]
    with sqlite3.connect(db_path) as connection:
        recorded_at = connection.execute(
            "SELECT recorded_at FROM reviews WHERE id = ?", (review_id,)
        ).fetchone()[0]
    recorded_clock = datetime.fromisoformat(recorded_at)
    assert before_review <= recorded_clock <= after_review
    assert recorded_at != "2026-01-01T00:00:00+00:00"
    db.close()


def test_self_review_and_empty_rationale_are_refused(tmp_path):
    db = open_memory(tmp_path / "memory.sqlite")
    rid = db.propose(doctrine_fixture())

    with pytest.raises(ValueError, match="reviewer.*author"):
        db.review(rid, "source-curator", "approve", "Checked")
    with pytest.raises(ValueError, match="rationale"):
        db.review(rid, "independent-reviewer", "approve", "  ")
    with pytest.raises(ValueError, match="approve.*reject"):
        db.review(rid, "independent-reviewer", "revise", "Checked")
    db.close()


def test_snapshot_eligibility_uses_both_historical_clocks_and_exclusions(tmp_path):
    db = open_memory(tmp_path / "memory.sqlite")
    records = [
        dated_fixture(case_ids=["eligible", "shared"]),
        dated_fixture(
            content={"finding": "not available yet"},
            source_refs=["results/cases.json#future"],
            available_at="2026-01-02T00:00:00Z",
            event_end="2025-12-10T00:00:00Z",
            case_ids=["future"],
        ),
        dated_fixture(
            content={"finding": "event overlaps training"},
            source_refs=["results/cases.json#overlap"],
            available_at="2025-12-20T00:00:00Z",
            event_end="2026-01-01T00:00:00Z",
            case_ids=["overlap"],
        ),
        dated_fixture(
            content={"finding": "excluded through one overlapping label"},
            source_refs=["results/cases.json#excluded"],
            case_ids=["other", "blocked"],
        ),
    ]
    ids = [db.propose(record) for record in records]
    for record_id in ids:
        approve(db, record_id)

    snapshot = db.snapshot(
        as_of=AS_OF,
        training_end=AS_OF,
        excluded_case_ids=("blocked",),
    )

    assert [record["id"] for record in snapshot["records"]] == [ids[0]]
    assert snapshot["excluded_case_ids"] == ["blocked"]
    db.close()


def test_event_may_end_before_the_lesson_becomes_available(tmp_path):
    db = open_memory(tmp_path / "memory.sqlite")
    rid = db.propose(dated_fixture())
    approve(db, rid)

    snapshot = db.snapshot(as_of=AS_OF, training_end=AS_OF)

    assert [record["id"] for record in snapshot["records"]] == [rid]
    db.close()


def test_case_free_methodology_may_have_dated_clocks(tmp_path):
    db = open_memory(tmp_path / "memory.sqlite")
    rid = db.propose(
        doctrine_fixture(
            available_at="2025-12-20T00:00:00Z",
            event_end="2025-12-19T00:00:00Z",
        )
    )
    approve(db, rid)

    snapshot = db.snapshot(as_of=AS_OF, training_end=AS_OF)

    assert [record["id"] for record in snapshot["records"]] == [rid]
    db.close()


def test_latest_review_controls_new_snapshots_but_old_snapshot_is_frozen(tmp_path):
    db = open_memory(tmp_path / "memory.sqlite")
    rid = db.propose(doctrine_fixture())
    approval_id = approve(db, rid)
    approved_snapshot = db.snapshot(as_of=AS_OF, training_end=AS_OF)

    rejection_id = db.review(
        rid,
        "second-independent-reviewer",
        "reject",
        "Source does not support the whole proposition",
    )
    rejected_snapshot = db.snapshot(as_of=AS_OF, training_end=AS_OF)

    assert rejected_snapshot["records"] == []
    assert rejected_snapshot["review_ids"] == []
    assert rejection_id != approval_id
    assert db.load_snapshot(approved_snapshot["id"]) == approved_snapshot
    assert [record["id"] for record in approved_snapshot["records"]] == [rid]
    db.close()


def test_reopening_preserves_record_and_snapshot_identities(tmp_path):
    path = tmp_path / "memory.sqlite"
    first = open_memory(path)
    rid = first.propose(doctrine_fixture())
    approve(first, rid)
    snapshot = first.snapshot(as_of=AS_OF, training_end=AS_OF)
    first.close()

    reopened = open_memory(path)

    assert reopened.propose(deepcopy(doctrine_fixture())) == rid
    assert reopened.snapshot(as_of=AS_OF, training_end=AS_OF)["id"] == snapshot["id"]
    assert reopened.load_snapshot(snapshot["id"]) == snapshot
    reopened.close()


def test_duplicate_proposal_is_idempotent(tmp_path):
    path = tmp_path / "memory.sqlite"
    db = open_memory(path)
    first = db.propose(doctrine_fixture())
    second = db.propose(deepcopy(doctrine_fixture()))

    with sqlite3.connect(path) as connection:
        count = connection.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    assert second == first
    assert count == 1
    db.close()


@pytest.mark.parametrize(
    ("table", "column", "replacement", "operation"),
    [
        (
            "records",
            "record_json",
            '{"kind":"doctrine"}',
            lambda db, snapshot_id: db.snapshot(as_of=AS_OF, training_end=AS_OF),
        ),
        (
            "reviews",
            "review_json",
            '{"verdict":"approve"}',
            lambda db, snapshot_id: db.load_snapshot(snapshot_id),
        ),
        (
            "snapshots",
            "snapshot_json",
            '{"records":[]}',
            lambda db, snapshot_id: db.load_snapshot(snapshot_id),
        ),
    ],
    ids=["record", "review", "snapshot"],
)
def test_tampering_with_hashed_storage_is_detected(
    tmp_path, table, column, replacement, operation
):
    path = tmp_path / "memory.sqlite"
    db = open_memory(path)
    rid = db.propose(doctrine_fixture())
    approve(db, rid)
    snapshot = db.snapshot(as_of=AS_OF, training_end=AS_OF)
    with sqlite3.connect(path) as connection:
        connection.execute(f"UPDATE {table} SET {column} = ?", (replacement,))
        connection.commit()

    with pytest.raises(ValueError, match="integrity"):
        operation(db, snapshot["id"])
    db.close()


@pytest.mark.parametrize(
    "record",
    [
        pytest.param(
            {key: value for key, value in doctrine_fixture().items() if key != "content"},
            id="missing-field",
        ),
        pytest.param(doctrine_fixture(extra="not allowed"), id="extra-field"),
        pytest.param(doctrine_fixture(kind="memo"), id="unknown-kind"),
        pytest.param(doctrine_fixture(kind=[]), id="non-string-kind"),
        pytest.param(doctrine_fixture(author=" "), id="empty-author"),
        pytest.param(doctrine_fixture(content=[]), id="content-not-object"),
        pytest.param(doctrine_fixture(content={"score": float("nan")}), id="nan"),
        pytest.param(doctrine_fixture(content={"score": float("inf")}), id="infinity"),
        pytest.param(doctrine_fixture(source_refs="source"), id="source-refs-not-list"),
        pytest.param(
            doctrine_fixture(source_refs=("docs/rulecard.md#context",)),
            id="source-refs-tuple",
        ),
        pytest.param(doctrine_fixture(source_refs=[]), id="missing-source-ref"),
        pytest.param(doctrine_fixture(source_refs=[""]), id="empty-source-ref"),
        pytest.param(doctrine_fixture(tags=("lc",)), id="tags-tuple"),
        pytest.param(doctrine_fixture(case_ids=()), id="case-ids-tuple"),
        pytest.param(doctrine_fixture(tags=["lc", "lc"]), id="duplicate-tags"),
        pytest.param(doctrine_fixture(case_ids=["C01", "C01"]), id="duplicate-case-ids"),
        pytest.param(doctrine_fixture(tags=[""]), id="empty-tag"),
        pytest.param(doctrine_fixture(case_ids=[""]), id="empty-case-id"),
        pytest.param(
            doctrine_fixture(case_ids=["C01"]), id="case-bearing-without-clocks"
        ),
        pytest.param(
            doctrine_fixture(available_at=AS_OF), id="only-one-clock"
        ),
        pytest.param(
            dated_fixture(kind="lesson", case_ids=[]), id="lesson-without-case"
        ),
        pytest.param(
            dated_fixture(kind="market_state", event_end=None),
            id="market-state-without-clock",
        ),
        pytest.param(
            doctrine_fixture(kind="lesson"), id="timeless-lesson"
        ),
        pytest.param(
            doctrine_fixture(available_at="2025-12-01T00:00:00", event_end=AS_OF),
            id="naive-clock",
        ),
    ],
)
def test_invalid_record_schema_is_rejected(tmp_path, record):
    db = open_memory(tmp_path / "memory.sqlite")
    with pytest.raises(ValueError):
        db.propose(record)
    db.close()


def test_tag_filter_uses_intersection_and_empty_filter_means_all(tmp_path):
    db = open_memory(tmp_path / "memory.sqlite")
    lc_id = db.propose(doctrine_fixture())
    wyckoff_id = db.propose(
        doctrine_fixture(
            content={"rule": "Read effort versus result."},
            source_refs=["docs/wyckoff.md#effort-result"],
            tags=["wyckoff"],
        )
    )
    for record_id in (lc_id, wyckoff_id):
        approve(db, record_id)

    tagged = db.snapshot(as_of=AS_OF, training_end=AS_OF, tags=("lc", "nested"))
    all_records = db.snapshot(as_of=AS_OF, training_end=AS_OF)

    assert [record["id"] for record in tagged["records"]] == [lc_id]
    assert tagged["tags"] == ["lc", "nested"]
    assert [record["id"] for record in all_records["records"]] == sorted(
        [lc_id, wyckoff_id]
    )
    db.close()


@pytest.mark.parametrize(
    ("as_of", "training_end", "match"),
    [
        ("2026-01-01T00:00:00", AS_OF, "timezone-aware"),
        (AS_OF, "2026-01-01T00:00:00", "timezone-aware"),
        (AS_OF, "2026-01-02T00:00:00Z", "training_end"),
    ],
)
def test_snapshot_rejects_invalid_cutoff_chronology(tmp_path, as_of, training_end, match):
    db = open_memory(tmp_path / "memory.sqlite")
    with pytest.raises(ValueError, match=match):
        db.snapshot(as_of=as_of, training_end=training_end)
    db.close()


def test_record_and_snapshot_ids_are_hashes_of_canonical_bodies(tmp_path):
    db = open_memory(tmp_path / "memory.sqlite")
    record = doctrine_fixture()
    rid = db.propose(record)
    approve(db, rid)
    snapshot = db.snapshot(as_of=AS_OF, training_end=AS_OF)
    snapshot_body = {key: value for key, value in snapshot.items() if key != "id"}

    assert rid == canonical_hash(record)
    assert snapshot["id"] == canonical_hash(snapshot_body)
    db.close()


def test_cli_propose_review_and_snapshot_emit_json(tmp_path):
    db_path = tmp_path / "memory.sqlite"
    record_path = tmp_path / "record.json"
    record_path.write_text(json.dumps(doctrine_fixture()))
    base = [sys.executable, "-m", "scripts.research.master_memory"]

    proposed = subprocess.run(
        base + ["propose", "--db", str(db_path), "--record", str(record_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    rid = json.loads(proposed.stdout)["id"]
    reviewed = subprocess.run(
        base
        + [
            "review",
            "--db",
            str(db_path),
            "--id",
            rid,
            "--reviewer",
            "independent-reviewer",
            "--verdict",
            "approve",
            "--rationale",
            "Checked against named source",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert len(json.loads(reviewed.stdout)["id"]) == 64
    snapshotted = subprocess.run(
        base
        + [
            "snapshot",
            "--db",
            str(db_path),
            "--as-of",
            AS_OF,
            "--training-end",
            AS_OF,
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    assert [record["id"] for record in json.loads(snapshotted.stdout)["records"]] == [
        rid
    ]


def test_cli_reports_invalid_json_without_traceback(tmp_path):
    record_path = tmp_path / "record.json"
    record_path.write_text("not-json")

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.research.master_memory",
            "propose",
            "--db",
            str(tmp_path / "memory.sqlite"),
            "--record",
            str(record_path),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 2
    assert "invalid JSON" in result.stderr
    assert "Traceback" not in result.stderr
