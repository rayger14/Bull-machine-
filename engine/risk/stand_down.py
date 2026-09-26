"""Loss-triggered book-level stand-down — the regime feedback limb.

Why (docs/knowledge/trader_knowledge_standdown_sweep_2026_08_31.md,
fresh_data_battery_2026_08_30.md): regime PREDICTION is 0-for-everything
(fear/greed, 200d, ADX, day-type, dominance, OI quadrants, absorption tape all
buried); reacting to realized losses is 2-for-2 (live cohort +$4,977, fresh
2025-26 sim +$6,165). Mancini's react-don't-predict, made mechanical.

Rule (pre-registered class, K=2/48h -> 24h): when K positions close net-negative
within `window_hours` of each other, refuse ALL new entries until `pause_hours`
after the K-th loss. Book-level: losses cluster ACROSS archetypes when the tape
turns (per-archetype streams are too sparse to trigger — measured Aug 31).

Causality: a position registers at its FINAL exit timestamp with its NET pnl
summed over scale-out legs (a stopped runner whose early legs banked profit is
only a loss if the position lost money overall). Nothing here reads features —
by design. K=3 tested NEGATIVE; the effect is specific, do not "tune" K upward.

Live rollout: mode 'shadow' logs would-block decisions without blocking;
'enforce' blocks. Pause state is in-memory (not persisted across restarts) —
acceptable in shadow; revisit before enforce promotion.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import pandas as pd


class LossStandDown:
    def __init__(self, k: int = 2, window_hours: float = 48.0,
                 pause_hours: float = 24.0):
        self.k = int(k)
        self.window = pd.Timedelta(hours=float(window_hours))
        self.pause = pd.Timedelta(hours=float(pause_hours))
        self._open_pnl: Dict[str, float] = {}
        self._loss_times: List[pd.Timestamp] = []
        self._pause_until: Optional[pd.Timestamp] = None
        # observability
        self.losses_registered = 0
        self.pauses_triggered = 0

    def record_leg(self, position_id: str, pnl: float) -> None:
        """Accumulate realized pnl of one exit leg for a position."""
        self._open_pnl[position_id] = self._open_pnl.get(position_id, 0.0) + float(pnl)

    def finalize(self, position_id: str, exit_ts: pd.Timestamp) -> None:
        """Position fully closed: register a loss event if net pnl < 0."""
        net = self._open_pnl.pop(position_id, 0.0)
        if net >= 0:
            return
        exit_ts = pd.Timestamp(exit_ts)
        self.losses_registered += 1
        self._loss_times.append(exit_ts)
        # prune outside window (list stays tiny)
        cutoff = exit_ts - self.window
        self._loss_times = [t for t in self._loss_times if t >= cutoff]
        if len(self._loss_times) >= self.k:
            until = exit_ts + self.pause
            if self._pause_until is None or until > self._pause_until:
                self._pause_until = until
                self.pauses_triggered += 1

    def blocked(self, now: pd.Timestamp) -> bool:
        return self._pause_until is not None and pd.Timestamp(now) < self._pause_until

    def status(self, now: pd.Timestamp) -> str:
        if self.blocked(now):
            return f"STAND_DOWN until {self._pause_until} ({len(self._loss_times)} recent losses)"
        return f"active ({len(self._loss_times)} losses in window, {self.pauses_triggered} pauses total)"
