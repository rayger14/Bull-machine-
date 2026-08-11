"""
FRACTAL EXECUTION + WI/MONEYTAUR EXIT TOOLKIT — study library (STUDY ONLY)
==========================================================================
wyckoff_audit addendum 56. A pre-registered 2x2 factorial on the SAME validated
trend-continuation door fires (add.48/54). The door's ENTRIES never change (same
signals); only EXECUTION (entry timing) and EXIT (target/trail craft) vary.

DESIGN — paired per-trade control (each fire its own control):
  1. The canonical fire set = the door's Arm-A one-position-at-a-time trades
     (run via the vetted backtester.run_backtest, identical to add.54).
  2. Each arm re-simulates each fire INDEPENDENTLY (per-trade R is equity-invariant:
     R = pnl/risk$ and pnl scales linearly with risk$, so R is scale-free). This
     isolates execution quality holding the fire set fixed — it is NOT a re-run
     sequential portfolio (that would confound fire SELECTION with EXECUTION).
  3. Parity: sim_trade_daily(spec='A') reproduces run_backtest's per-trade R to
     0.00% on every asset (self-test) — validating the walker as the referee for
     the dynamic-trail arms that run_backtest cannot express.

All OURS params fixed BEFORE measuring (see FractalExecConfig). No grids, no tuning.
"""
from __future__ import annotations
import os, sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester import (run_backtest, EntryPlan, COMMISSION_RATE, SLIPPAGE_BPS,
                        INITIAL_CASH, RISK_PCT)
from trend_continuation_door import build_daily_sensors, TrendContinuationDoor
from xasset_spx_port import load_spx, fractal_swings
from unified_archetype_v2 import (M2_SOS_WIN, LPS_LOOKBACK, STOP_BUF_ATR, MAX_HOLD,
                                  MIN_TP1_R, DEDUP_K)
from engine.features.eye_state import CONFIRMED_BREAK

SLIP = SLIPPAGE_BPS / 10_000.0
FIXED_RISK = 0.01 * INITIAL_CASH   # $1000 nominal risk unit (for PnL reporting only)


# ===========================================================================
# PRE-REGISTERED CONFIG (OURS; fixed before measurement; NO sweeps)
# ===========================================================================
@dataclass(frozen=True)
class FractalExecConfig:
    # --- entry half (Arm B sniper) ---
    K_DAILY: int = 5                 # LTF work window = 5 daily bars = 120 1H bars
    GP_LO: float = 0.618             # golden-pocket shallow edge (retrace from high)
    GP_HI: float = 0.786             # golden-pocket deep edge (recorded; limit uses shallow)
    # --- exit half (Arm C toolkit) ---
    WICK_RATIO_MIN: float = 1.5      # wick_magnets.py default min_wick_ratio
    MAGNET_DECAY_BARS: int = 7       # 168h / 24 = 7 daily bars (7-day decay)
    NEGFIB_PRIMARY: float = 0.272    # runner target: rhigh + 0.272*(rhigh-rlow)
    NEGFIB_STRETCH: float = 0.618    # stretch fallback
    MIN_TP1_R: float = MIN_TP1_R     # 0.5R floor for any TP1 candidate
    TRAIL_ARM_R: float = 1.0         # Moneytaur trail arms once trade >= +1R
    TRAIL_BE_PLUS_R: float = 0.5     # trail floor = entry + 0.5R
    TRAIL_ATR_MULT: float = 1.0      # pivot_low - 1*ATR
    SWING_N: int = 10                # daily fractal pivot lag (matches store convention)
    MAX_HOLD: int = MAX_HOLD         # 168 daily bars

CFG = FractalExecConfig()


# ===========================================================================
# ASSET LOADING
# ===========================================================================
CRYPTO_DAILY = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
                "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/one_strategy")
XASSET_DAILY = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
                "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/xasset")
H1_DIR = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
          "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/fractal_execution/h1")
BTC_1H_STORE = ("/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/"
                "BTC_1H_FEATURES_V22_CTX.parquet")

CRYPTO = ["ETH-USD", "SOL-USD", "LTC-USD", "XRP-USD", "ADA-USD",
          "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD", "BTC-USD"]
XASSET = ["SPX", "NDX", "GOLD"]
FAMILY = {**{s: "crypto" for s in CRYPTO}, **{s: "equity" for s in ["SPX", "NDX"]},
          "GOLD": "gold"}


def load_daily(sym: str):
    """Return (df, sr, bj, eye) daily sensors for any of the 13 markets."""
    if sym in CRYPTO:
        raw = load_spx(os.path.join(CRYPTO_DAILY, f"{sym}.parquet"))
    elif sym in XASSET:
        raw = load_spx(os.path.join(XASSET_DAILY, f"{sym}_1D.parquet"))
    else:
        raise ValueError(sym)
    return build_daily_sensors(raw)


def load_h1(sym: str) -> pd.DataFrame:
    """1H OHLCV for the entry-half assets. BTC from the V22 store; others from
    the Coinbase 1H cache. Returns a DatetimeIndex OHLCV frame."""
    if sym == "BTC-USD":
        df = pd.read_parquet(BTC_1H_STORE)[["open", "high", "low", "close", "volume"]].copy()
        return df.sort_index()
    p = os.path.join(H1_DIR, f"{sym}_1H.parquet")
    df = pd.read_parquet(p)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.set_index("timestamp").sort_index()
    df.index.name = None
    return df[["open", "high", "low", "close", "volume"]].astype(float)


# ===========================================================================
# FIRE EXTRACTION — canonical Arm-A door fires + full per-fire context
# ===========================================================================
def extract_fires(sym, df, sr, bj, eye):
    """Run the validated door (Arm A) once; return (fires, armA_trades).
    Each fire dict carries everything the alternative arms need, computed causally."""
    strat = TrendContinuationDoor(df, sr, bj, eye, variant="struct", conviction=False)
    res = run_backtest(df, strat, risk_pct=RISK_PCT, label=f"{sym}_A")
    trades = res["trades"]
    elog = strat.entries_log

    idx = df.index
    O = df["open"].to_numpy(float); H = df["high"].to_numpy(float)
    L = df["low"].to_numpy(float);  C = df["close"].to_numpy(float)
    ATR = df["atr_14"].to_numpy(float)
    EMA = df["ema_200"].to_numpy(float)
    rhigh = sr["struct_range_high"].to_numpy(float)
    rlow = sr["struct_range_low"].to_numpy(float)
    swh = df["swing_high_50"].to_numpy(float)
    eye_state = eye["eye_state"].to_numpy(object)
    eye_dir = eye["eye_dir"].to_numpy(object)
    range_upper = eye["range_upper_1d"].to_numpy(float)
    # daily fractal pivots for the Moneytaur trail + magnets (recompute daily, causal)
    dsh, dsl = fractal_swings(df, CFG.SWING_N)   # daily pivot high/low, ffilled

    # wick-magnet upper levels: per-bar upper-wick magnet price (bar high) if
    # upper_wick/body >= WICK_RATIO_MIN, else NaN. Causal (bar's own OHLC).
    body = np.abs(C - O)
    upper_wick = H - np.maximum(O, C)
    with np.errstate(divide="ignore", invalid="ignore"):
        uw_ratio = np.where(body > 0, upper_wick / body, 0.0)
    magnet_up = np.where(uw_ratio >= CFG.WICK_RATIO_MIN, H, np.nan)
    # active bojan-high zone bottom (nearest-above edge for a TP anchor; WI: HIGHS=TP)
    bh_active = bj["bojan_high_active"].to_numpy(np.int8)
    bh_bottom_raw = bj["bojan_high_zone_bottom"].to_numpy(float)
    bojan_high_bottom = np.where(bh_active == 1, bh_bottom_raw, np.nan)

    pos = {t: k for k, t in enumerate(idx)}
    fires = []
    for tr, el in zip(trades, elog):
        i = pos[tr["entry_time"]]
        break_level = el["break_level"]
        # recompute the CONFIRMED_BREAK bar to anchor the golden pocket high
        lo = max(0, i - M2_SOS_WIN)
        bbar = None
        for c in range(i - 1, lo - 1, -1):
            if eye_state[c] == CONFIRMED_BREAK and eye_dir[c] == "bull":
                bbar = c; break
        high_anchor = float(np.nanmax(H[bbar:i + 1])) if bbar is not None else float(H[i])
        created_low = float(np.nanmin(L[max(0, i - LPS_LOOKBACK + 1):i + 1]))
        stop = created_low - STOP_BUF_ATR * ATR[i]
        fires.append({
            "sym": sym, "family": FAMILY[sym], "i": i, "entry_time": tr["entry_time"],
            "entry_raw": float(C[i]), "atr": float(ATR[i]),
            "stop": float(stop), "created_low": created_low,
            "break_level": float(break_level) if np.isfinite(break_level) else np.nan,
            "high_anchor": high_anchor, "bbar": bbar,
            "rhigh": float(rhigh[i]) if np.isfinite(rhigh[i]) else np.nan,
            "rlow": float(rlow[i]) if np.isfinite(rlow[i]) else np.nan,
            "swh": float(swh[i]) if np.isfinite(swh[i]) else np.nan,
            "below_ema200": bool(el["below_ema200"]),
            "armA_R": float(tr["R"]), "armA_reason": tr["exit_reason"],
        })
    arrays = {"O": O, "H": H, "L": L, "C": C, "ATR": ATR, "idx": idx,
              "dsl": dsl, "magnet_up": magnet_up, "bojan_high_bottom": bojan_high_bottom,
              "n": len(df)}
    return fires, trades, arrays


# ===========================================================================
# PER-TRADE EXIT WALKER (replicates backtester.py cost/intrabar conventions
# EXACTLY, plus a dynamic Moneytaur trail the stock backtester cannot express).
# Returns R (equity-invariant) + diagnostics. entry_idx immune; exits from +1.
# ===========================================================================
def sim_trade_daily(arrays, entry_idx, entry_fill, stop, plan, spec, fire):
    """Simulate ONE fire's exits on the daily frame.
    plan/spec select geometry: 'A' struct (add.54), 'C' WI toolkit, 'C_trail' ablation.
    entry_fill = actual (slipped) fill price; stop = initial hard stop price.
    Returns dict(R, pnl, reason, max_R, bars_held, tp1_level, runner_level)."""
    H = arrays["H"]; L = arrays["L"]; C = arrays["C"]; ATR = arrays["ATR"]
    dsl = arrays["dsl"]; n = arrays["n"]
    is_long = True
    stop_dist = abs(entry_fill - stop)
    if stop_dist <= 0 or not np.isfinite(stop_dist):
        return None
    risk_d = FIXED_RISK
    qty = risk_d / stop_dist
    orig_qty = qty
    entry_comm = qty * entry_fill * COMMISSION_RATE

    def R_at(price):
        return (price - entry_fill) / stop_dist   # unrealized R for a long

    # ---- build the plan (targets ladder + runner + optional trail) ----
    tp1 = plan["tp1"]; tp1_frac = plan["tp1_frac"]
    be_after_tp1 = plan.get("be_after_tp1", True)
    be_level = plan.get("be_level", entry_fill)   # nominal-entry BE (backtester parity)
    runner_target = plan.get("runner_target", None)
    use_trail = plan.get("use_trail", False)

    remaining = orig_qty
    realized_gross = 0.0
    exit_comm_total = 0.0
    cur_stop = stop
    first_tp_done = False
    max_R = R_at(entry_fill)
    reason = "open"
    last_j = entry_idx

    def take(px_raw, q):
        nonlocal realized_gross, exit_comm_total
        fill = px_raw * (1 - SLIP)   # long exit sells lower
        realized_gross += (fill - entry_fill) * q
        exit_comm_total += abs(q) * fill * COMMISSION_RATE

    end = min(entry_idx + 1 + CFG.MAX_HOLD, n)
    j = entry_idx + 1
    while j < end and remaining > 1e-12:
        lo, hi, cl = L[j], H[j], C[j]
        max_R = max(max_R, R_at(hi))
        # 1) STOP FIRST (wick, fill at stop level)
        if lo <= cur_stop:
            take(cur_stop, remaining)
            remaining = 0.0
            reason = "stop" if not first_tp_done else "stop_after_tp1"
            last_j = j; break
        # 2) TP1 (close-confirmed)
        if not first_tp_done and tp1 is not None and cl >= tp1:
            q = min(tp1_frac * orig_qty, remaining)
            take(cl, q); remaining -= q
            first_tp_done = True
            last_j = j
            if be_after_tp1:
                cur_stop = be_level   # backtester moves stop to the nominal (unslipped) entry
            if remaining <= 1e-12:
                reason = "take_profit"
        # 3) RUNNER management (after TP1)
        if first_tp_done and remaining > 1e-12:
            # Moneytaur trail (arms once >= +1R): raise stop to max(entry+0.5R, pivot_low - 1*ATR)
            if use_trail and max_R >= CFG.TRAIL_ARM_R:
                be_plus = be_level + CFG.TRAIL_BE_PLUS_R * stop_dist
                piv = dsl[j]
                cand = be_plus
                if np.isfinite(piv):
                    cand = max(be_plus, piv - CFG.TRAIL_ATR_MULT * ATR[j])
                cur_stop = max(cur_stop, cand)   # trail only up
            # runner target (close-confirmed)
            if runner_target is not None and cl >= runner_target:
                take(cl, remaining); remaining = 0.0
                reason = "runner_target"; last_j = j; break
        j += 1

    # time / end exit for the tail
    if remaining > 1e-12:
        jn = min(end - 1, n - 1)
        take(C[jn], remaining)
        reason = "time_exit" if jn == min(entry_idx + CFG.MAX_HOLD, n - 1) else "backtest_end"
        last_j = jn
    pnl = realized_gross - entry_comm - exit_comm_total
    R = pnl / risk_d
    return {"R": R, "pnl": pnl, "reason": reason, "max_R": max_R,
            "bars_held": last_j - entry_idx, "tp1": tp1, "runner_target": runner_target}


# ===========================================================================
# PLAN BUILDERS per arm (all off the SAME fire context)
# ===========================================================================
def plan_A(fire, arrays):
    """Arm A struct geometry (add.54 headline): TP1 40% at struct_range_high (else
    swing_high_50 else entry+1R, clearing >=0.5R) -> BE -> runner 60% to measured
    move rhigh+(rhigh-rlow), floored entry+2R."""
    e = fire["entry_raw"]; R = e - fire["stop"]
    rhigh, rlow, swh = fire["rhigh"], fire["rlow"], fire["swh"]
    tp1 = None
    for cand in (rhigh, swh):
        if np.isfinite(cand) and cand >= e + CFG.MIN_TP1_R * R:
            tp1 = cand; break
    if tp1 is None:
        tp1 = e + 1 * R
    measured = rhigh + (rhigh - rlow) if (np.isfinite(rhigh) and np.isfinite(rlow) and rhigh > rlow) else -np.inf
    tt = max(e + 2 * R, measured)
    return {"tp1": tp1, "tp1_frac": 0.40, "be_after_tp1": True, "be_level": e,
            "runner_target": tt, "use_trail": False}


def _tp1_wi(fire, arrays):
    """Arm C TP1 candidate cascade: nearest-above of {active bojan_high zone,
    unswept wick-magnet (7-day decay), struct_range_high} that clears >=0.5R."""
    e = fire["entry_raw"]; R = e - fire["stop"]; i = fire["i"]
    floor = e + CFG.MIN_TP1_R * R
    cands = []
    # bojan_high zone bottom (nearest-above edge) — from the daily bojan sensor
    bhb = arrays.get("bojan_high_bottom")
    if bhb is not None and np.isfinite(bhb[i]) and bhb[i] >= floor:
        cands.append(bhb[i])
    # unswept upper wick magnet formed within last MAGNET_DECAY_BARS, above entry,
    # not exceeded since formation
    mup = arrays["magnet_up"]; H = arrays["H"]
    w0 = max(0, i - CFG.MAGNET_DECAY_BARS + 1)
    for f in range(i, w0 - 1, -1):
        lvl = mup[f]
        if np.isfinite(lvl) and lvl >= floor:
            # unswept: no bar in (f, i] exceeded lvl
            if f == i or np.nanmax(H[f + 1:i + 1]) < lvl:
                cands.append(lvl)
    # struct_range_high
    if np.isfinite(fire["rhigh"]) and fire["rhigh"] >= floor:
        cands.append(fire["rhigh"])
    if not cands:
        return e + 1 * R
    return float(min(cands))   # nearest-above


def _runner_negfib(fire):
    """Runner target = negative-fib extension of the RANGE: rhigh + 0.272*(rhigh-rlow)
    primary; if that is not above entry, try 0.618 stretch; else floor entry+2R."""
    e = fire["entry_raw"]; R = e - fire["stop"]
    rhigh, rlow = fire["rhigh"], fire["rlow"]
    if np.isfinite(rhigh) and np.isfinite(rlow) and rhigh > rlow:
        rng = rhigh - rlow
        prim = rhigh + CFG.NEGFIB_PRIMARY * rng
        if prim > e + CFG.MIN_TP1_R * R:
            return prim
        stretch = rhigh + CFG.NEGFIB_STRETCH * rng
        if stretch > e + CFG.MIN_TP1_R * R:
            return stretch
    return e + 2 * R


def plan_C(fire, arrays):
    """Arm C composite WI/Moneytaur exit engine."""
    return {"tp1": _tp1_wi(fire, arrays), "tp1_frac": 0.40, "be_after_tp1": True,
            "be_level": fire["entry_raw"],
            "runner_target": _runner_negfib(fire), "use_trail": True}


def plan_C_trail(fire, arrays):
    """C-trail ablation: baseline Arm-A TP1 target (struct_range_high) + BE, but the
    runner uses the Moneytaur trail INSTEAD of the measured-move runner target.
    Attributes whether Arm-C's edge comes from the new TARGETS or the TRAIL."""
    a = plan_A(fire, arrays)
    return {"tp1": a["tp1"], "tp1_frac": 0.40, "be_after_tp1": True,
            "be_level": fire["entry_raw"], "runner_target": None, "use_trail": True}
