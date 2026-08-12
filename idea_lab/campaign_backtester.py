"""
CAMPAIGN BACKTESTER  (STUDY ONLY -- extends idea_lab/backtester.py for multi-entry
                      WI-topology campaigns; wyckoff_audit add.57 -> add.58)
================================================================================
The idea-lab backtester (backtester.py) is ONE-POSITION, STATIC-PLAN: once a trade
opens, its stop/targets are frozen. WI's CAMPAIGN-v2 topology (add.57 verbatim)
needs THREE things the static engine cannot express:

  1. A STRUCTURE-EVENT TRAILING STOP: after TP1, the stop moves UP under each
     NEWLY CREATED LPS (a confirmed higher swing low). NOT a pivot-ATR trail
     (add.56, harmful) -- it updates ONLY when a new LPS forms.
  2. A DERISK-TO-RUNNER ladder: TP1 40% at structure -> TP2 bank down to ~1/10 at
     the next structure target -> a 10% runner.
  3. A campaign-death RUNNER EXIT: the 10% runner rides until the campaign dies
     (eye_dir loses bull, or a body close breaks the campaign structure floor) or
     the trailing stop is hit -- NO fixed max_hold (sanity cap only).

This engine is a strict SUPERSET of run_backtest: with a single full-size target,
no trail, and no death signal it reduces EXACTLY to run_backtest (verified by the
same 10-spec referee self-test to < 0.01%). COST ACCOUNTING is IMPORTED from
backtester.py (single source of truth) -- slip once in the fill, commission both
sides, stop-first pessimism, target-needs-a-close. Nothing here re-derives costs.

STRATEGY INTERFACE (the strat OWNS all campaign state; the engine is a thin executor):
  strat.observe(i)          -> called EVERY bar (flat or positioned) BEFORE anything
                               else; updates the campaign floor (rising LPS), detects
                               campaign birth/death. Must be causal (reads <= i only).
  strat.entry(df, i)        -> None | CampaignEntryPlan   (called only when FLAT)
  strat.campaign_floor      -> float: running-highest confirmed LPS low of the LIVE
                               campaign (the structure-event trail reference). NaN if none.
  strat.campaign_alive      -> bool: is the current campaign still alive at bar i.

The engine reports, per trade, which campaign it belonged to (plan.meta['campaign_id'])
and the entry ordinal within the campaign (plan.meta['entry_ord']).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

# single source of truth for costs / account (add.26 discipline)
from backtester import (
    _slip_fill, COMMISSION_RATE, SLIPPAGE_BPS, INITIAL_CASH, RISK_PCT,
    compute_stats, _referee_textbook, EntryPlan,
)

SANITY_CAP_BARS = 400   # OURS: campaign structure IS the time limit; this is only a runaway guard


@dataclass
class CampaignEntryPlan:
    """A campaign entry. targets is the derisk ladder [(level, frac_of_ORIGINAL), ...]
    taken in order (TP1 40% at structure, TP2 50% at next structure target). Whatever
    remains after the ladder is the RUNNER (e.g. 10%), which rides on the structure-event
    trailing stop until campaign death. stop0 = created-LPS-low - buf*ATR (the initial
    hard stop). trail_buf_atr and atr_entry parametrise the post-TP1 structure-event trail:
    after TP1, cur_stop = max(cur_stop, campaign_floor - trail_buf_atr*atr_entry)."""
    direction: str                                  # "long" (campaign door is long-only)
    stop0: float
    targets: List[Tuple[float, float]] = field(default_factory=list)
    atr_entry: float = 0.0
    trail_buf_atr: float = 0.25
    sanity_cap_bars: int = SANITY_CAP_BARS
    risk_mult: float = 1.0
    meta: dict = field(default_factory=dict)


def run_campaign_backtest(df: pd.DataFrame, strat, risk_pct: float = RISK_PCT,
                          initial_cash: float = INITIAL_CASH,
                          label: str = "campaign") -> dict:
    """Drive a campaign strategy bar-by-bar, one position at a time (re-enter only when
    FLAT, per WI). Structure-event trailing stop + campaign-death runner exit are applied
    while positioned. Returns {trades, equity_curve, stats}."""
    highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float)
    closes = df["close"].to_numpy(float)
    idx = df.index
    n = len(df)

    equity = initial_cash
    equity_curve = [equity]
    trades: List[dict] = []

    i = 0
    while i < n:
        strat.observe(i)                       # causal campaign-state update (birth/death/floor)
        plan = strat.entry(df, i)
        if plan is None:
            equity_curve.append(equity)
            i += 1
            continue

        is_long = plan.direction == "long"
        entry_raw = closes[i]
        entry_fill = _slip_fill(entry_raw, side_is_buy=is_long)
        stop = float(plan.stop0)
        stop_dist = abs(entry_fill - stop)
        if stop_dist <= 0 or not np.isfinite(stop_dist):
            equity_curve.append(equity); i += 1; continue

        risk_mult = float(getattr(plan, "risk_mult", 1.0) or 1.0)
        risk_dollars = equity * risk_pct * risk_mult
        orig_qty = risk_dollars / stop_dist
        entry_comm = orig_qty * entry_fill * COMMISSION_RATE

        remaining = orig_qty
        realized_gross = 0.0
        exit_comm_total = 0.0
        cur_stop = stop
        targets = list(plan.targets)
        first_tp_done = False
        atr_entry = float(plan.atr_entry)
        trail_buf = float(plan.trail_buf_atr)
        cap = int(plan.sanity_cap_bars)
        entry_idx = i
        exit_reason = "open"
        last_exit_j = None

        def _take(px_raw, frac_qty):
            nonlocal realized_gross, exit_comm_total
            fill = _slip_fill(px_raw, side_is_buy=(not is_long))
            g = (fill - entry_fill) * frac_qty if is_long else (entry_fill - fill) * frac_qty
            realized_gross += g
            exit_comm_total += abs(frac_qty) * fill * COMMISSION_RATE
            return fill

        j = entry_idx + 1
        end = min(entry_idx + 1 + cap, n)
        while j < end and remaining > 1e-12:
            strat.observe(j)                    # keep campaign floor / alive current (causal)
            # (1) structure-event trailing stop: only AFTER TP1, only moves UP, only on a
            #     newly-created (higher) LPS floor. Never loosens (OURS: monotone-up).
            if first_tp_done and np.isfinite(strat.campaign_floor):
                cand = strat.campaign_floor - trail_buf * atr_entry
                if cand > cur_stop:
                    cur_stop = cand
            lo, hi, cl = lows[j], highs[j], closes[j]
            # (2) STOP FIRST (wick, fill at stop level) -- stop-first pessimism
            stop_hit = (lo <= cur_stop) if is_long else (hi >= cur_stop)
            if stop_hit:
                _take(cur_stop, remaining)
                remaining = 0.0
                exit_reason = "stop" if not first_tp_done else "stop_after_tp1"
                last_exit_j = j
                break
            # (3) DERISK LADDER (close-confirmed, fill at close)
            fired = True
            while fired and targets and remaining > 1e-12:
                fired = False
                lvl, frac = targets[0]
                tgt_hit = (cl >= lvl) if is_long else (cl <= lvl)
                if tgt_hit:
                    q = min(frac * orig_qty, remaining)
                    _take(cl, q)
                    remaining -= q
                    targets.pop(0)
                    fired = True
                    if not first_tp_done:
                        first_tp_done = True     # TP1 -> engage the structure-event trail
                    last_exit_j = j
                    if remaining <= 1e-12:
                        exit_reason = "take_profit"
            # (4) RUNNER: rides until CAMPAIGN DEATH (eye_dir loses bull / body break of floor)
            if not targets and remaining > 1e-12 and not strat.campaign_alive:
                _take(cl, remaining)
                remaining = 0.0
                exit_reason = "campaign_death"
                last_exit_j = j
                break
            j += 1

        if remaining > 1e-12:
            jn = min(end - 1, n - 1)
            _take(closes[jn], remaining)
            exit_reason = ("sanity_cap" if jn == min(entry_idx + cap, n - 1)
                           else "backtest_end")
            last_exit_j = jn
            remaining = 0.0

        pnl = realized_gross - entry_comm - exit_comm_total
        equity += pnl
        R = pnl / risk_dollars if risk_dollars > 0 else 0.0
        trades.append({
            "entry_time": idx[entry_idx], "exit_time": idx[last_exit_j],
            "direction": plan.direction, "entry_fill": entry_fill,
            "stop0": stop, "qty": orig_qty, "risk_dollars": risk_dollars,
            "gross": realized_gross, "entry_comm": entry_comm,
            "exit_comm": exit_comm_total, "pnl": pnl, "R": R,
            "exit_reason": exit_reason, "bars_held": last_exit_j - entry_idx,
            "meta": plan.meta,
        })
        equity_curve.append(equity)
        i = last_exit_j + 1

    stats = compute_stats(trades, equity_curve, initial_cash, label)
    return {"trades": trades, "equity_curve": equity_curve, "stats": stats}


# ===========================================================================
# REFEREE SELF-TEST: the campaign engine, driven with a SINGLE full-size target,
# no trail, no death, must reproduce the backtester referee's textbook PnL to
# < 0.01% -- proving the multi-entry extension did not disturb cost/fill parity.
# ===========================================================================
class _StaticFixture:
    """A minimal strat that fires ONE fixed-SL/TP trade at _idx, exposing the
    campaign-engine hooks as inert (floor NaN, always-alive, single target)."""
    def __init__(self, idx, direction, sl, tp, horizon):
        self._idx, self._dir, self._sl, self._tp = idx, direction, sl, tp
        self._hz = horizon
        self.campaign_floor = np.nan
        self.campaign_alive = True
        self._fired = False

    def observe(self, i):
        pass

    def entry(self, df, i):
        if i != self._idx or self._fired:
            return None
        self._fired = True
        return CampaignEntryPlan(
            direction=self._dir, stop0=self._sl,
            targets=[(self._tp, 1.0)], atr_entry=0.0, trail_buf_atr=0.0,
            sanity_cap_bars=self._hz, meta={"selftest": True})


def run_selftest(df: pd.DataFrame) -> bool:
    # Referee parity is timeframe-independent (it is about cost/fill mechanics), so we
    # exercise it on the full available daily frame with spread-out entry indices.
    win = df
    closes = win["close"].to_numpy(float)
    highs = win["high"].to_numpy(float)
    lows = win["low"].to_numpy(float)
    n = len(win)
    HORIZON = 60
    base = [(0.05, "long"), (0.12, "long"), (0.22, "short"), (0.33, "long"),
            (0.44, "short"), (0.55, "long"), (0.63, "short"), (0.74, "long"),
            (0.83, "short"), (0.92, "long")]
    specs = [(int(f * n), d) for f, d in base]
    print("\n=== CAMPAIGN-ENGINE SELF-TEST vs bt_audit referee (textbook-clean) ===")
    print(f"{'idx':>5} {'dir':>5} {'lab_pnl':>12} {'ref_pnl':>12} {'rel_%':>10} {'reason':>14}")
    max_rel = 0.0
    for idx, direction in specs:
        ep = float(closes[idx])
        sl, tp = (ep * 0.98, ep * 1.04) if direction == "long" else (ep * 1.02, ep * 0.96)
        strat = _StaticFixture(idx, direction, sl, tp, HORIZON)
        res = run_campaign_backtest(win, strat, risk_pct=RISK_PCT, label=f"st_{idx}")
        lab_pnl = res["trades"][0]["pnl"] if res["trades"] else 0.0
        ref = _referee_textbook(highs, lows, closes, idx, direction, sl, tp, HORIZON)
        ref_pnl = ref["pnl_textbook"]
        rel = abs(lab_pnl - ref_pnl) / (abs(ref_pnl) + 1e-9) * 100
        max_rel = max(max_rel, rel)
        rsn = res["trades"][0]["exit_reason"] if res["trades"] else "none"
        print(f"{idx:>5} {direction:>5} {lab_pnl:>12.4f} {ref_pnl:>12.4f} {rel:>10.2e} {rsn:>14}")
    ok = max_rel < 0.01
    print(f"\nMAX per-trade relative discrepancy: {max_rel:.2e} %  [target < 0.01%] -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    SC = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
          "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/one_strategy")
    import sys, os
    sys.path.insert(0, os.path.dirname(__file__))
    from xasset_spx_port import load_spx
    df = load_spx(os.path.join(SC, "BTC-USD.parquet"))
    run_selftest(df)
