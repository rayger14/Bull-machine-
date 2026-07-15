"""Maker-first shadow fill ledger — measurement only, no behavior change.

For every REAL entry the paper runner takes (at market/taker), also record the
counterfactual: would a resting limit order at the signal price have filled?

Fill rule (conservative): a resting limit only fills if price trades THROUGH
it on a later bar — long: bar low < limit; short: bar high > limit. The entry
bar itself never counts (the order would have been posted at its close).
If unfilled after HORIZON_BARS, the entry is a MISS and we record the chase
cost: how far price ran from the limit by then (signed so that positive =
you'd have to pay up to chase).

Output: <output_dir>/maker_shadow.jsonl, one record per resolved entry:
  {position_id, archetype, direction, limit_price, entry_ts, resolved_ts,
   filled, bars_waited, chase_bps}
Pending (unresolved) entries persist across restarts in
<output_dir>/maker_shadow_pending.json.

Consumed by bin/live_evidence.py section 4 (execution costs).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

HORIZON_BARS = 3  # 1h bars: a wick_trap retest either happens fast or not


class MakerShadowLedger:
    def __init__(self, output_dir: Path):
        self.dir = Path(output_dir)
        self.ledger_path = self.dir / "maker_shadow.jsonl"
        self.pending_path = self.dir / "maker_shadow_pending.json"
        self.pending: list[dict] = []
        if self.pending_path.exists():
            try:
                self.pending = json.loads(self.pending_path.read_text())
            except Exception:
                logger.warning("[MAKER_SHADOW] pending state unreadable, resetting")
                self.pending = []

    def record_entry(self, position_id: str, archetype: str, direction: str,
                     limit_price: float, ts) -> None:
        """Call when a real position opens. Limit = signal price (pre-slippage)."""
        if not limit_price or limit_price <= 0:
            return
        self.pending.append({
            "position_id": position_id,
            "archetype": archetype,
            "direction": direction,
            "limit_price": float(limit_price),
            "entry_ts": str(ts),
            "bars_waited": 0,
        })
        self._save_pending()

    def on_bar(self, high, low, close, ts) -> None:
        """Call once per bar AFTER exits, BEFORE new entries are recorded."""
        if not self.pending:
            return
        try:
            high, low, close = float(high), float(low), float(close)
        except (TypeError, ValueError):
            return
        if not (high > 0 and low > 0 and close > 0):
            return
        still = []
        for p in self.pending:
            p["bars_waited"] += 1
            lim = p["limit_price"]
            filled = (low < lim) if p["direction"] == "long" else (high > lim)
            if filled:
                self._resolve(p, ts, filled=True, chase_bps=0.0)
            elif p["bars_waited"] >= HORIZON_BARS:
                # chase cost: positive = price ran away, you'd pay up to enter
                sign = 1.0 if p["direction"] == "long" else -1.0
                chase_bps = sign * (close - lim) / lim * 10_000
                self._resolve(p, ts, filled=False, chase_bps=chase_bps)
            else:
                still.append(p)
        self.pending = still
        self._save_pending()

    def _resolve(self, p: dict, ts, filled: bool, chase_bps: float) -> None:
        rec = {**p, "resolved_ts": str(ts), "filled": filled,
               "chase_bps": round(chase_bps, 2)}
        try:
            with self.ledger_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
        except Exception as e:
            logger.warning("[MAKER_SHADOW] ledger write failed: %s", e)
        logger.info("[MAKER_SHADOW] %s %s %s after %d bars%s",
                    p["archetype"], p["direction"],
                    "FILLED" if filled else "MISSED", p["bars_waited"],
                    "" if filled else f" (chase {chase_bps:+.0f}bp)")

    def _save_pending(self) -> None:
        try:
            self.pending_path.write_text(json.dumps(self.pending))
        except Exception as e:
            logger.warning("[MAKER_SHADOW] pending save failed: %s", e)
