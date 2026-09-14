"""Durable, reviewed research memory with content-addressed snapshots."""

import argparse
import hashlib
import json
import math
import sqlite3
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path


RECORD_FIELDS = {
    "kind",
    "author",
    "content",
    "source_refs",
    "tags",
    "available_at",
    "event_end",
    "case_ids",
}
METHODOLOGY_KINDS = {"doctrine", "code_map", "hypothesis"}
DATED_KINDS = {"lesson", "market_state"}
RECORD_KINDS = METHODOLOGY_KINDS | DATED_KINDS
REVIEW_VERDICTS = {"approve", "reject"}
SNAPSHOT_FIELDS = {
    "as_of",
    "training_end",
    "excluded_case_ids",
    "tags",
    "records",
    "review_ids",
}


def _canonical(value):
    try:
        return json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("value must be finite JSON") from exc


def _digest(value):
    return hashlib.sha256(_canonical(value).encode("ascii")).hexdigest()


def _now():
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def _clock(value, field):
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a timezone-aware ISO timestamp")
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"{field} must be a timezone-aware ISO timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError(f"{field} must be timezone-aware")
    return parsed.astimezone(timezone.utc)


def _string_list(value, field, *, require_value=False, allow_tuple=False):
    expected = (list, tuple) if allow_tuple else (list,)
    if not isinstance(value, expected) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field} must be a list of nonempty strings")
    result = list(value)
    if require_value and not result:
        raise ValueError(f"{field} must contain at least one value")
    if any(not isinstance(item, str) or not item.strip() for item in result):
        raise ValueError(f"{field} must be a list of nonempty strings")
    if len(set(result)) != len(result):
        raise ValueError(f"{field} values must be unique")
    return result


def _validate_json(value, path="content"):
    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{path} must not contain NaN or Infinity")
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_json(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} object keys must be strings")
            _validate_json(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} must contain only JSON values")


def _validate_record(record):
    if not isinstance(record, dict) or set(record) != RECORD_FIELDS:
        raise ValueError("record must contain exactly the required fields")
    if not isinstance(record["kind"], str) or record["kind"] not in RECORD_KINDS:
        raise ValueError("kind must be doctrine, code_map, hypothesis, lesson, or market_state")
    if not isinstance(record["author"], str) or not record["author"].strip():
        raise ValueError("author must be a nonempty string")
    if not isinstance(record["content"], dict):
        raise ValueError("content must be a JSON object")
    _validate_json(record["content"])
    _string_list(record["source_refs"], "source_refs", require_value=True)
    _string_list(record["tags"], "tags")
    case_ids = _string_list(record["case_ids"], "case_ids")

    available_at = record["available_at"]
    event_end = record["event_end"]
    timeless = available_at is None and event_end is None
    if (available_at is None) != (event_end is None):
        raise ValueError("available_at and event_end must both be dated or both be null")
    if timeless:
        if record["kind"] not in METHODOLOGY_KINDS or case_ids:
            raise ValueError("only case-free methodology may be timeless")
    else:
        _clock(available_at, "available_at")
        _clock(event_end, "event_end")
    if record["kind"] in DATED_KINDS and (timeless or not case_ids):
        raise ValueError("lesson and market_state require dated clocks and case_ids")
    _canonical(record)


def _decode(text, name):
    try:
        value = json.loads(text)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{name} integrity check failed: invalid JSON") from exc
    _validate_json(value, name)
    return value


class ResearchMemory:
    def __init__(self, path):
        self.path = Path(path)
        self._connection = sqlite3.connect(str(self.path), isolation_level=None)
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._connection.executescript(
            """
            CREATE TABLE IF NOT EXISTS records (
                id TEXT PRIMARY KEY,
                record_json TEXT NOT NULL,
                proposed_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reviews (
                id TEXT PRIMARY KEY,
                record_id TEXT NOT NULL REFERENCES records(id),
                reviewer TEXT NOT NULL,
                verdict TEXT NOT NULL CHECK (verdict IN ('approve', 'reject')),
                rationale TEXT NOT NULL,
                recorded_at TEXT NOT NULL,
                review_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS reviews_record_order
                ON reviews(record_id, recorded_at);
            CREATE TABLE IF NOT EXISTS snapshots (
                id TEXT PRIMARY KEY,
                snapshot_json TEXT NOT NULL,
                created_at TEXT NOT NULL
            );
            """
        )

    @contextmanager
    def _transaction(self):
        self._connection.execute("BEGIN IMMEDIATE")
        try:
            yield
        except BaseException:
            self._connection.rollback()
            raise
        else:
            self._connection.commit()

    def _record(self, record_id):
        row = self._connection.execute(
            "SELECT record_json FROM records WHERE id = ?", (record_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown record id: {record_id}")
        record = _decode(row[0], "record")
        if _digest(record) != record_id:
            raise ValueError("record integrity check failed")
        _validate_record(record)
        return record

    def _review(self, review_id):
        row = self._connection.execute(
            """SELECT record_id, reviewer, verdict, rationale, recorded_at, review_json
               FROM reviews WHERE id = ?""",
            (review_id,),
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown review id: {review_id}")
        review = _decode(row[5], "review")
        if _digest(review) != review_id:
            raise ValueError("review integrity check failed")
        expected = {
            "record_id": row[0],
            "reviewer": row[1],
            "verdict": row[2],
            "rationale": row[3],
            "recorded_at": row[4],
        }
        if review != expected:
            raise ValueError("review integrity check failed")
        _clock(review["recorded_at"], "recorded_at")
        return review

    def propose(self, record):
        record = deepcopy(record)
        _validate_record(record)
        record_text = _canonical(record)
        record_id = hashlib.sha256(record_text.encode("ascii")).hexdigest()
        with self._transaction():
            existing = self._connection.execute(
                "SELECT record_json FROM records WHERE id = ?", (record_id,)
            ).fetchone()
            if existing is not None:
                stored = _decode(existing[0], "record")
                if _digest(stored) != record_id or _canonical(stored) != record_text:
                    raise ValueError("record identity integrity check failed")
                return record_id
            self._connection.execute(
                "INSERT INTO records(id, record_json, proposed_at) VALUES (?, ?, ?)",
                (record_id, record_text, _now()),
            )
        return record_id

    def review(self, record_id, reviewer, verdict, rationale):
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError("reviewer must be a nonempty string")
        if verdict not in REVIEW_VERDICTS:
            raise ValueError("verdict must be approve or reject")
        if not isinstance(rationale, str) or not rationale.strip():
            raise ValueError("rationale must be nonempty")
        with self._transaction():
            record = self._record(record_id)
            if reviewer.strip() == record["author"].strip():
                raise ValueError("reviewer must differ from record author")
            review = {
                "record_id": record_id,
                "reviewer": reviewer,
                "verdict": verdict,
                "rationale": rationale,
                "recorded_at": _now(),
            }
            review_id = _digest(review)
            review_text = _canonical(review)
            existing = self._connection.execute(
                "SELECT review_json FROM reviews WHERE id = ?", (review_id,)
            ).fetchone()
            if existing is not None:
                if existing[0] != review_text:
                    raise ValueError("review identity integrity check failed")
                return review_id
            self._connection.execute(
                """INSERT INTO reviews(
                       id, record_id, reviewer, verdict, rationale, recorded_at, review_json
                   ) VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    review_id,
                    record_id,
                    reviewer,
                    verdict,
                    rationale,
                    review["recorded_at"],
                    review_text,
                ),
            )
        return review_id

    def snapshot(self, *, as_of, training_end, excluded_case_ids=(), tags=()):
        as_of_clock = _clock(as_of, "as_of")
        training_clock = _clock(training_end, "training_end")
        if training_clock > as_of_clock:
            raise ValueError("training_end must be less than or equal to as_of")
        exclusions = sorted(
            _string_list(excluded_case_ids, "excluded_case_ids", allow_tuple=True)
        )
        requested_tags = sorted(_string_list(tags, "tags", allow_tuple=True))
        excluded = set(exclusions)
        tag_filter = set(requested_tags)

        with self._transaction():
            selected = []
            review_ids = []
            rows = self._connection.execute(
                "SELECT id FROM records ORDER BY id"
            ).fetchall()
            for (record_id,) in rows:
                record = self._record(record_id)
                latest = self._connection.execute(
                    """SELECT id FROM reviews
                       WHERE record_id = ?
                       ORDER BY recorded_at DESC, rowid DESC LIMIT 1""",
                    (record_id,),
                ).fetchone()
                if latest is None:
                    continue
                review = self._review(latest[0])
                if review["record_id"] != record_id or review["verdict"] != "approve":
                    continue
                if excluded.intersection(record["case_ids"]):
                    continue
                if tag_filter and not tag_filter.intersection(record["tags"]):
                    continue
                if record["available_at"] is not None:
                    available = _clock(record["available_at"], "available_at")
                    event_end = _clock(record["event_end"], "event_end")
                    if available > as_of_clock or event_end > as_of_clock:
                        continue
                    if event_end >= training_clock:
                        continue
                selected.append({"id": record_id, **deepcopy(record)})
                review_ids.append(latest[0])

            body = {
                "as_of": as_of_clock.isoformat(),
                "training_end": training_clock.isoformat(),
                "excluded_case_ids": exclusions,
                "tags": requested_tags,
                "records": selected,
                "review_ids": review_ids,
            }
            snapshot_id = _digest(body)
            snapshot_text = _canonical(body)
            existing = self._connection.execute(
                "SELECT snapshot_json FROM snapshots WHERE id = ?", (snapshot_id,)
            ).fetchone()
            if existing is not None:
                stored = _decode(existing[0], "snapshot")
                if _digest(stored) != snapshot_id or _canonical(stored) != snapshot_text:
                    raise ValueError("snapshot identity integrity check failed")
            else:
                self._connection.execute(
                    "INSERT INTO snapshots(id, snapshot_json, created_at) VALUES (?, ?, ?)",
                    (snapshot_id, snapshot_text, _now()),
                )
        return {"id": snapshot_id, **body}

    def load_snapshot(self, snapshot_id):
        row = self._connection.execute(
            "SELECT snapshot_json FROM snapshots WHERE id = ?", (snapshot_id,)
        ).fetchone()
        if row is None:
            raise ValueError(f"unknown snapshot id: {snapshot_id}")
        body = _decode(row[0], "snapshot")
        if not isinstance(body, dict) or set(body) != SNAPSHOT_FIELDS:
            raise ValueError("snapshot integrity check failed")
        if _digest(body) != snapshot_id:
            raise ValueError("snapshot integrity check failed")
        records = body["records"]
        review_ids = body["review_ids"]
        if not isinstance(records, list) or not isinstance(review_ids, list):
            raise ValueError("snapshot integrity check failed")
        if len(records) != len(review_ids):
            raise ValueError("snapshot integrity check failed")
        for stored_record, review_id in zip(records, review_ids):
            if not isinstance(stored_record, dict) or "id" not in stored_record:
                raise ValueError("snapshot record integrity check failed")
            record_id = stored_record["id"]
            record = {key: value for key, value in stored_record.items() if key != "id"}
            _validate_record(record)
            if _digest(record) != record_id:
                raise ValueError("snapshot record integrity check failed")
            review = self._review(review_id)
            if review["record_id"] != record_id or review["verdict"] != "approve":
                raise ValueError("snapshot review integrity check failed")
        return {"id": snapshot_id, **deepcopy(body)}

    def close(self):
        self._connection.close()


def _parser():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    propose = commands.add_parser("propose", help="store an immutable proposal")
    propose.add_argument("--db", required=True, type=Path)
    propose.add_argument("--record", required=True, type=Path)

    review = commands.add_parser("review", help="append an independent review")
    review.add_argument("--db", required=True, type=Path)
    review.add_argument("--id", required=True)
    review.add_argument("--reviewer", required=True)
    review.add_argument("--verdict", required=True, choices=sorted(REVIEW_VERDICTS))
    review.add_argument("--rationale", required=True)

    snapshot = commands.add_parser("snapshot", help="freeze approved memory")
    snapshot.add_argument("--db", required=True, type=Path)
    snapshot.add_argument("--as-of", required=True)
    snapshot.add_argument("--training-end", required=True)
    return parser


def main(argv=None):
    parser = _parser()
    args = parser.parse_args(argv)
    db = None
    try:
        db = ResearchMemory(args.db)
        if args.command == "propose":
            try:
                record = json.loads(args.record.read_text())
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON in {args.record}: {exc.msg}") from exc
            result = {"id": db.propose(record)}
        elif args.command == "review":
            result = {
                "id": db.review(
                    args.id, args.reviewer, args.verdict, args.rationale
                )
            }
        else:
            result = db.snapshot(as_of=args.as_of, training_end=args.training_end)
    except (OSError, sqlite3.Error, ValueError) as exc:
        parser.error(str(exc))
    finally:
        if db is not None:
            db.close()
    print(json.dumps(result, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
