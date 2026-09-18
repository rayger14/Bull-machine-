"""Checkpoint policy tests: never invent billing or expand the paid sample."""
import importlib.util
from pathlib import Path

import pytest


def checkpoint_class():
    spec = importlib.util.find_spec("scripts.research.lc_judgment_checkpoint")
    assert spec is not None, "bounded checkpoint implementation required"
    from scripts.research.lc_judgment_checkpoint import JudgmentCheckpoint
    return JudgmentCheckpoint


def test_unknown_billing_stays_unknown(tmp_path):
    cls = checkpoint_class()
    with cls(tmp_path / "runtime", preparation_dir=tmp_path / "prep") as runner:
        snapshot = {"balance_credits": None, "observed_at": None,
                    "source": "user_authorized_unmetered_two_case_checkpoint"}
        assert runner._budget_snapshot(snapshot, start=True) == snapshot
        assert runner._check_reservation_budget(snapshot, {}) == (snapshot, None)
        with pytest.raises(ValueError):
            runner._budget_snapshot(dict(snapshot, balance_credits=725))


def test_third_case_cannot_consume_attempt_even_before_preparation(tmp_path):
    cls = checkpoint_class()
    with cls(tmp_path / "runtime", preparation_dir=tmp_path / "prep") as runner:
        with pytest.raises(ValueError, match="two-case"):
            runner.reserve_role("hourly-lc:2026-01-29T16:00:00+00:00", "specialist", {})
        assert not (tmp_path / "runtime" / "reservations").exists()


def test_unmetered_checkpoint_rejects_claimed_credit_ceiling(tmp_path):
    cls = checkpoint_class()
    with cls(tmp_path / "runtime", preparation_dir=tmp_path / "prep") as runner:
        with pytest.raises(ValueError, match="unavailable"):
            runner.authorize_start({}, {}, max_observed_balance_drop=300)


def test_checkpoint_code_is_pinned(tmp_path):
    cls = checkpoint_class()
    with cls(tmp_path / "runtime", preparation_dir=tmp_path / "prep") as runner:
        assert Path("scripts/research/lc_judgment_checkpoint.py").resolve() in runner._runtime_dependencies()
