"""
IDEA-LAB BACKTESTER  (STUDY ONLY — no production code, no engine, no deploy)

A clean, single-strategy, ONE-POSITION-AT-A-TIME backtester built to be a fast
idea-screener. By construction it makes the engine artifacts that repeatedly
contaminated idea tests (wyckoff_audit addenda 22-26) STRUCTURALLY IMPOSSIBLE:
    * NO fusion            (no domain-score competition)
    * NO dedup             (no bar routed to another archetype)
    * NO margin competition / crowd-out (non-binding wallet, addendum-22 method rule)
    * NO 16-archetype book  (exactly one strategy at a time)

COST ACCOUNTING (fixes the addendum-26 bug from the start):
    * slippage: applied ONCE, in the fill price only
        long  entry fill = px*(1+slip);  long  exit fill = px*(1-slip)
        short entry fill = px*(1-slip);  short exit fill = px*(1+slip)
    * commission: charged BOTH sides, ONCE each, on the traded notional
        pnl = gross - entry_comm - exit_comm
    This is the "textbook-clean single-count" convention the referee
    (scratchpad/bt_audit/differential_test.py) computes as `pnl_textbook`.
    The production engine instead (a) omits entry commission from trade.pnl and
    (b) double-charges entry slippage; we deliberately do NOT replicate that.

INTRABAR CONVENTION (matches the engine's conservative pessimism, addendum 26):
    * STOP fires on the wick: for a long, stop hits when bar low <= stop, and it
      FILLS AT THE STOP LEVEL (then slipped).
    * TARGET needs a CLOSE: a target hits when the bar CLOSES through it, and it
      FILLS AT THE CLOSE (then slipped).  No target-first intrabar inflation.
    * On a bar where BOTH the stop wick and a target close would trigger, the
      STOP wins (stop-first pessimism).
    * Exits are checked from entry_idx+1 (the entry bar itself cannot exit).

SIZING: fixed risk per trade. risk_$ = equity * risk_pct; qty = risk_$ / |entry_fill - stop|.
    Non-binding wallet: no margin cap, no leverage cap (this is signal
    measurement, per the addendum-22 permanent method rule). One position at a
    time so there is never capital competition to model.

STRATEGY INTERFACE:
    entry_fn(hist_df, i) -> None | EntryPlan
        hist_df : the full frame; the fn may ONLY read rows with index <= i
                  (causal; enforced by convention — helpers below slice [:i+1]).
        i       : integer position of the current (decision) bar.
        returns None to stay flat, or an EntryPlan to open a long/short at this
        bar's close.
    EntryPlan carries the stop, the scale-out target ladder, an optional
    post-TP1 stop move, a runner target, and max_hold — enough to express BOTH
    Wyckoff-Insider banked-and-derisked geometry AND a naive R-ladder.

OUTPUT: per-trade log (list of dict), equity curve, and a stats dict
    (n, WR, PF, avgR, PnL, MaxDD).
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Callable, Optional, List, Tuple

# ---------------------------------------------------------------------------
# Cost / account constants (single source of truth)
# ---------------------------------------------------------------------------
COMMISSION_RATE = 0.0002     # per side, on notional
SLIPPAGE_BPS    = 3.0        # per side, applied ONCE in the fill price
INITIAL_CASH    = 100_000.0
RISK_PCT        = 0.01       # 1% of equity risked per trade (fixed-fractional)


@dataclass
class EntryPlan:
    """A fully-specified trade plan produced by a strategy at the entry bar.

    targets: list of (price_level, fraction_of_ORIGINAL_qty) scale-outs, in the
             order they should be taken. Fractions need not sum to 1.0; whatever
             remains after the ladder + any runner is closed at max_hold/stop.
    move_stop_to_after_first_tp: if not None, once the FIRST target fills the
             stop is moved to this price (WI derisk). None = fixed stop.
    runner_target: final target price for the last remaining fraction. None =
             no distinct runner (the ladder + time/stop governs the tail).
    max_hold_bars: hard time exit (close-fill) counted from the entry bar.
    """
    direction: str                              # "long" or "short"
    stop: float
    targets: List[Tuple[float, float]] = field(default_factory=list)
    move_stop_to_after_first_tp: Optional[float] = None
    runner_target: Optional[float] = None
    max_hold_bars: int = 168
    risk_mult: float = 1.0                       # per-trade conviction sizing multiplier
    meta: dict = field(default_factory=dict)


def _slip_fill(price: float, side_is_buy: bool) -> float:
    """Apply slippage ONCE, in the fill price. Buys fill higher, sells lower."""
    s = SLIPPAGE_BPS / 10_000.0
    return price * (1 + s) if side_is_buy else price * (1 - s)


def run_backtest(df: pd.DataFrame,
                 entry_fn: Callable[[pd.DataFrame, int], Optional[EntryPlan]],
                 risk_pct: float = RISK_PCT,
                 initial_cash: float = INITIAL_CASH,
                 label: str = "strategy") -> dict:
    """Drive `entry_fn` bar-by-bar, one position at a time, and record trades.

    Returns {trades, equity_curve, stats}.
    """
    highs  = df["high"].to_numpy(dtype=float)
    lows   = df["low"].to_numpy(dtype=float)
    closes = df["close"].to_numpy(dtype=float)
    idx    = df.index

    equity = initial_cash
    equity_curve = [equity]
    trades: List[dict] = []

    n = len(df)
    i = 0
    while i < n:
        plan = entry_fn(df, i)
        if plan is None:
            equity_curve.append(equity)
            i += 1
            continue

        # ---- open at THIS bar's close ----
        direction = plan.direction
        is_long = direction == "long"
        entry_raw = closes[i]
        entry_fill = _slip_fill(entry_raw, side_is_buy=is_long)
        stop = float(plan.stop)
        stop_dist = abs(entry_fill - stop)
        if stop_dist <= 0 or not np.isfinite(stop_dist):
            equity_curve.append(equity); i += 1; continue

        risk_mult = float(getattr(plan, "risk_mult", 1.0) or 1.0)
        risk_dollars = equity * risk_pct * risk_mult   # conviction sizing (default 1.0x)
        qty = risk_dollars / stop_dist            # position sized off entry-stop distance
        orig_qty = qty
        notional_entry = qty * entry_fill
        entry_comm = notional_entry * COMMISSION_RATE

        # exit-management state
        remaining = orig_qty
        realized_gross = 0.0
        exit_comm_total = 0.0
        cur_stop = stop
        targets = list(plan.targets)
        first_tp_done = False
        runner_target = plan.runner_target
        max_hold = int(plan.max_hold_bars)
        entry_idx = i
        exit_reason = "open"
        last_exit_j = None

        def _take(px_raw, frac_qty, is_long_):
            """Close frac_qty at raw price px_raw; slip once, accrue gross+comm."""
            nonlocal realized_gross, exit_comm_total
            fill = _slip_fill(px_raw, side_is_buy=(not is_long_))  # exit = opposite side
            g = (fill - entry_fill) * frac_qty if is_long_ else (entry_fill - fill) * frac_qty
            realized_gross += g
            exit_comm_total += abs(frac_qty) * fill * COMMISSION_RATE
            return fill

        # ---- walk bars from entry+1 ----
        j = entry_idx + 1
        end = min(entry_idx + 1 + max_hold, n)
        while j < end and remaining > 1e-12:
            lo, hi, cl = lows[j], highs[j], closes[j]
            # 1) STOP FIRST (wick, fill at stop level)
            stop_hit = (lo <= cur_stop) if is_long else (hi >= cur_stop)
            if stop_hit:
                _take(cur_stop, remaining, is_long)
                remaining = 0.0
                exit_reason = "stop" if not first_tp_done else "stop_after_tp1"
                last_exit_j = j
                break
            # 2) TARGET LADDER (close-confirmed, fill at close)
            fired = True
            while fired and targets and remaining > 1e-12:
                fired = False
                lvl, frac = targets[0]
                tgt_hit = (cl >= lvl) if is_long else (cl <= lvl)
                if tgt_hit:
                    q = min(frac * orig_qty, remaining)
                    _take(cl, q, is_long)
                    remaining -= q
                    targets.pop(0)
                    fired = True
                    if not first_tp_done:
                        first_tp_done = True
                        if plan.move_stop_to_after_first_tp is not None:
                            cur_stop = float(plan.move_stop_to_after_first_tp)
                    last_exit_j = j
                    if remaining <= 1e-12:
                        exit_reason = "take_profit"   # ladder fully closed the position
            # 3) RUNNER TARGET (close-confirmed) for the tail
            if runner_target is not None and remaining > 1e-12 and not targets:
                rt_hit = (cl >= runner_target) if is_long else (cl <= runner_target)
                if rt_hit:
                    _take(cl, remaining, is_long)
                    remaining = 0.0
                    exit_reason = "runner_target"
                    last_exit_j = j
                    break
            j += 1

        # ---- time / end exit for anything left ----
        if remaining > 1e-12:
            jn = min(end - 1, n - 1)
            _take(closes[jn], remaining, is_long)
            exit_reason = "time_exit" if jn == min(entry_idx + max_hold, n - 1) else "backtest_end"
            last_exit_j = jn
            remaining = 0.0

        pnl = realized_gross - entry_comm - exit_comm_total
        equity += pnl
        R = pnl / risk_dollars if risk_dollars > 0 else 0.0
        trades.append({
            "entry_time": idx[entry_idx], "exit_time": idx[last_exit_j],
            "direction": direction, "entry_fill": entry_fill,
            "stop0": stop, "qty": orig_qty, "risk_dollars": risk_dollars,
            "gross": realized_gross, "entry_comm": entry_comm,
            "exit_comm": exit_comm_total, "pnl": pnl, "R": R,
            "exit_reason": exit_reason, "bars_held": last_exit_j - entry_idx,
            "meta": plan.meta,
        })
        equity_curve.append(equity)

        # one position at a time: resume scanning AFTER the exit bar
        i = last_exit_j + 1

    stats = compute_stats(trades, equity_curve, initial_cash, label)
    return {"trades": trades, "equity_curve": equity_curve, "stats": stats}


def compute_stats(trades, equity_curve, initial_cash, label="strategy") -> dict:
    n = len(trades)
    if n == 0:
        return {"label": label, "n": 0, "WR": 0.0, "PF": 0.0, "avgR": 0.0,
                "PnL": 0.0, "MaxDD_pct": 0.0}
    pnls = np.array([t["pnl"] for t in trades])
    Rs = np.array([t["R"] for t in trades])
    wins = pnls[pnls > 0].sum()
    losses = -pnls[pnls < 0].sum()
    wr = float((pnls > 0).mean())
    pf = float(wins / losses) if losses > 1e-9 else float("inf")
    eq = np.array(equity_curve)
    peak = np.maximum.accumulate(eq)
    dd = (eq - peak) / peak
    maxdd = float(dd.min() * 100.0)
    return {
        "label": label, "n": n, "WR": round(wr, 4),
        "PF": round(pf, 3) if np.isfinite(pf) else float("inf"),
        "avgR": round(float(Rs.mean()), 4),
        "PnL": round(float(pnls.sum()), 2),
        "MaxDD_pct": round(maxdd, 2),
        "gross_win": round(float(wins), 2), "gross_loss": round(float(losses), 2),
    }


# ===========================================================================
# SELF-TEST: reproduce the bt_audit fixed-SL/TP scenario and confirm per-trade
# PnL matches the REFEREE's textbook-clean single-count PnL to < 0.01%.
# (We match textbook-clean, NOT engine-parity, because the lab fixes the
#  addendum-26 entry-cost bug by design; the only expected delta vs the engine
#  recorded pnl is exactly entry_comm, which is the bug.)
# ===========================================================================
def _referee_textbook(highs, lows, closes, entry_idx, direction, sl, tp, horizon):
    """Independent numpy referee (ported from differential_test.py) — textbook
    single-count PnL, sized by RISK (not the engine's notional/leverage path)
    so it is apples-to-apples with the idea-lab's fixed-risk sizing."""
    slip = SLIPPAGE_BPS / 10_000.0
    entry_price = float(closes[entry_idx])
    is_long = direction == "long"
    fill_entry = entry_price * (1 + slip) if is_long else entry_price * (1 - slip)

    stop_dist = abs(fill_entry - sl)
    risk_dollars = INITIAL_CASH * RISK_PCT
    qty = risk_dollars / stop_dist

    exit_price = None; reason = None
    for j in range(entry_idx + 1, min(entry_idx + 1 + horizon, len(closes))):
        lo, hi, cl = float(lows[j]), float(highs[j]), float(closes[j])
        if is_long:
            if lo <= sl:  exit_price, reason = sl, "stop"; break
            if cl >= tp:  exit_price, reason = cl, "take_profit"; break
        else:
            if hi >= sl:  exit_price, reason = sl, "stop"; break
            if cl <= tp:  exit_price, reason = cl, "take_profit"; break
    if exit_price is None:
        jn = min(entry_idx + horizon, len(closes) - 1)
        exit_price, reason = float(closes[jn]), "time_exit"

    fill_exit = exit_price * (1 - slip) if is_long else exit_price * (1 + slip)
    gross = (fill_exit - fill_entry) * qty if is_long else (fill_entry - fill_exit) * qty
    entry_comm = qty * fill_entry * COMMISSION_RATE
    exit_comm = qty * fill_exit * COMMISSION_RATE
    pnl_textbook = gross - entry_comm - exit_comm
    return {"pnl_textbook": pnl_textbook, "reason": reason, "qty": qty}


def run_selftest(df: pd.DataFrame) -> bool:
    """Reproduce the 10-spec fixed-SL/TP scenario from differential_test.py."""
    win = df[(df.index >= "2024-01-01") & (df.index <= "2024-03-01")]
    closes = win["close"].to_numpy(float)
    highs  = win["high"].to_numpy(float)
    lows   = win["low"].to_numpy(float)
    HORIZON = 240
    specs = [(10, "long"), (60, "long"), (120, "short"), (200, "long"),
             (300, "short"), (450, "long"), (600, "short"), (800, "long"),
             (950, "short"), (1100, "long")]

    print("\n=== IDEA-LAB SELF-TEST vs bt_audit referee (textbook-clean) ===")
    print(f"{'idx':>5} {'dir':>5} {'lab_pnl':>12} {'ref_pnl':>12} {'rel_%':>10} {'reason':>12}")
    max_rel = 0.0
    for idx, direction in specs:
        ep = float(closes[idx])
        if direction == "long":
            sl, tp = ep * 0.98, ep * 1.04
        else:
            sl, tp = ep * 1.02, ep * 0.96

        # Drive the SAME single fixed-SL/TP trade through the idea-lab core.
        def _fixed_entry(_df, k, _idx=idx, _dir=direction, _sl=sl, _tp=tp):
            if k != _idx:
                return None
            return EntryPlan(direction=_dir, stop=_sl,
                             targets=[(_tp, 1.0)], move_stop_to_after_first_tp=None,
                             runner_target=None, max_hold_bars=HORIZON)
        res = run_backtest(win, _fixed_entry, risk_pct=RISK_PCT,
                           label=f"selftest_{idx}")
        lab_pnl = res["trades"][0]["pnl"] if res["trades"] else 0.0
        ref = _referee_textbook(highs, lows, closes, idx, direction, sl, tp, HORIZON)
        ref_pnl = ref["pnl_textbook"]
        rel = abs(lab_pnl - ref_pnl) / (abs(ref_pnl) + 1e-9) * 100
        max_rel = max(max_rel, rel)
        print(f"{idx:>5} {direction:>5} {lab_pnl:>12.4f} {ref_pnl:>12.4f} "
              f"{rel:>10.2e} {res['trades'][0]['exit_reason']:>12}")
    ok = max_rel < 0.01
    print(f"\nMAX per-trade relative discrepancy: {max_rel:.2e} %   "
          f"[target < 0.01%]  ->  {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    REPO = "/Users/rayghandchi/Bull Machine/Bull-machine-"
    df = pd.read_parquet(f"{REPO}/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet")
    run_selftest(df)
