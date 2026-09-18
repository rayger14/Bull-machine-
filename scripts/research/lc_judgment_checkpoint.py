"""User-authorized two-case checkpoint with explicitly unavailable billing.

Preserves the frozen execution machinery; never fabricates a balance reading.
The two allowed cases and existing one-attempt-per-role ledger bound launches
to four. This is a call ceiling, NOT a credit ceiling.
"""
from copy import deepcopy
from pathlib import Path

from scripts.research.lc_judgment_execution import JudgmentExecution


ALLOWED_CASES = (
    "hourly-lc:2026-01-20T06:00:00+00:00",
    "hourly-lc:2026-01-25T09:00:00+00:00",
)
UNKNOWN_BUDGET = {
    "balance_credits": None,
    "observed_at": None,
    "source": "user_authorized_unmetered_two_case_checkpoint",
}


class JudgmentCheckpoint(JudgmentExecution):
    def _runtime_dependencies(self):
        return super()._runtime_dependencies() + [Path(__file__).resolve()]

    def _budget_snapshot(self, snapshot, *, start=False):
        if snapshot != UNKNOWN_BUDGET:
            raise ValueError("checkpoint requires explicitly unknown billing")
        return deepcopy(snapshot)

    def _check_reservation_budget(self, snapshot, authorization):
        return self._budget_snapshot(snapshot), None

    def authorize_start(self, snapshot, readiness, *, max_observed_balance_drop=None):
        if max_observed_balance_drop is not None:
            raise ValueError("billing unavailable; cannot enforce a credit ceiling")
        return super().authorize_start(snapshot, readiness)

    def reserve_role(self, case_id, role, budget_snapshot, *, authorization_token=None):
        if case_id not in ALLOWED_CASES:
            raise ValueError("outside user-authorized two-case checkpoint")
        return super().reserve_role(case_id, role, budget_snapshot,
                                    authorization_token=authorization_token)
