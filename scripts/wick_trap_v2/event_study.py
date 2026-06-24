#!/usr/bin/env python3
"""
Wick Trap v2 (failed breakdown) — fast event-study harness.

Research harness ONLY (not the production backtester). Tests the level-anchored
sweep -> reclaim -> acceptance design from:
  docs/knowledge/trader_knowledge_failed_breakdown_es_transfer_2026_06_11.md
  docs/knowledge/industry_study_wick_trap_detection_2026_06_11.md

Pre-registered grid (54 combos, NO post-hoc extensions):
  A (acceptance bars)        in {1, 2, 3}
  C (no-chase, x ATR14)      in {1.0, 1.5, 2.0}
  level_quality_low minimum  in {0.0 (off), 0.3, 0.5}
  day_type counter-trend sizing x0.25: ON / OFF

Splits: train 2018-2022, OOS-1 2023-2024, OOS-2 (pristine) 2025-2026-06.

Hypotheses (pre-registered):
  H1: any combo PF>=1.3 with n>=30 in BOTH OOS windows?
  H2: level_quality_low terciles monotonic with avg trade PnL?
  H3: day_type x0.25 sizing improves 2022 AND 2025-26 without damaging 2021/2023-24?
  H4: structural stop vs 4.9*ATR fixed stop on the SAME entries.

Interface contract: prefers engine.features.level_features.compute_level_features
if present; otherwise uses the clearly-marked FALLBACK implementations below.
"""

import os
import sys
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd

REPO = "/Users/rayghandchi/Bull Machine/Bull-machine-"
DATA = os.path.join(REPO, "data/cache/binance_vision/klines/BTCUSDT_1h.parquet")
OUTDIR = os.path.join(REPO, "results/wick_trap_v2")

# ----------------------------- sim constants (fixed, not gridded) ----------
CAPITAL = 100_000.0
RISK_PCT = 0.02                  # 2% risk per trade (static base, no compounding)
COST_PER_SIDE = 0.0002 + 0.0003  # 2bps commission + 3bps slippage, per side
STOP_ATR_BUFFER = 0.25           # structural stop = sweep-bar low - 0.25*ATR14
TP1_R, TP1_PCT = 1.0, 0.50       # scale 50% at +1R
TP2_R, TP2_PCT = 2.0, 0.30       # scale 30% at +2R (runner 20% remains)
TRAIL_ATR = 2.0                  # runner trails 2*ATR after TP2
TIME_STOP_BARS = 168             # 168h
COOLDOWN_BARS = 18
H4_FIXED_ATR_STOP = 4.9          # old wick_trap atr_stop_mult, for H4 comparison

# fallback feature params (fixed, clearly NOT part of the grid)
ATR_N = 14
PIVOT_K = 10                     # pivot low = min of +/-10 bars, confirmed K bars later
MAX_PIVOTS = 5                   # keep last 5 confirmed pivot lows as active levels
PIVOT_MAX_AGE = 500              # bars
QUALITY_TOUCH_WINDOW = 200       # bars looked back for touch count
QUALITY_TOUCH_TOL_ATR = 0.25

TRAIN = ("2018-01-01", "2023-01-01")
OOS1 = ("2023-01-01", "2025-01-01")
OOS2 = ("2025-01-01", "2026-06-11")
SPLITS = {"train_2018_2022": TRAIN, "oos1_2023_2024": OOS1, "oos2_2025_2026": OOS2}


# ============================================================================
# Feature computation
# ============================================================================

def atr14(df: pd.DataFrame) -> np.ndarray:
    h, l, c = df["high"].values, df["low"].values, df["close"].values
    pc = np.roll(c, 1)
    pc[0] = c[0]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1.0 / ATR_N, adjust=False).mean().values


def round_grid(price: float) -> float:
    """Round-number grid spacing ~ 1% order of magnitude (Osler round numbers)."""
    if price <= 0:
        return 1.0
    return 10.0 ** (np.floor(np.log10(price)) - 1.0)


# ---------------------------------------------------------------------------
# FALLBACK implementations (used when engine/features/level_features.py absent)
# ---------------------------------------------------------------------------

def fallback_day_type(df: pd.DataFrame) -> np.ndarray:
    """FALLBACK day_type: classify TODAY from the PRIOR completed UTC day
    (no lookahead). trend if prior day had directional ratio>0.5 and range
    above 1.1x its 20-day median. 0 range, 1 trend_up, -1 trend_down."""
    daily = df.resample("1D").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}
    ).dropna()
    rng = (daily["high"] - daily["low"]).replace(0, np.nan)
    dr = (daily["close"] - daily["open"]) / rng
    range_mult = rng / rng.rolling(20, min_periods=5).median()
    dt = np.where((dr.abs() > 0.5) & (range_mult > 1.1), np.sign(dr), 0.0)
    dt_prior = pd.Series(dt, index=daily.index).shift(1).fillna(0.0)  # prior day -> today
    return dt_prior.reindex(df.index.normalize()).fillna(0.0).values


def fallback_pivot_lows(low: np.ndarray, k: int = PIVOT_K):
    """FALLBACK confirmed swing lows: low[i] strictly the window min of +/-k bars,
    usable only from bar i+k (confirmation lag, no repaint).
    Returns list of (confirm_idx, pivot_idx, level)."""
    n = len(low)
    s = pd.Series(low)
    win_min = s.rolling(2 * k + 1, center=True).min().values
    out = []
    for i in range(k, n - k):
        if low[i] == win_min[i]:
            out.append((i + k, i, low[i]))
    return out


def fallback_sweep_events(df: pd.DataFrame, atr: np.ndarray,
                          prior_day_low: np.ndarray) -> list:
    """FALLBACK sweep_low_event detection.
    A sweep = bar trades BELOW a pre-existing qualified STRUCTURAL support level
    intrabar AND closes back ABOVE it (LuxAlgo reclaim criterion).
    Qualified levels: last MAX_PIVOTS confirmed pivot lows (age<=PIVOT_MAX_AGE)
    and the prior UTC-day low. Round numbers contribute to QUALITY only (Osler
    stop clustering), they are NOT standalone levels — a fine round-number grid
    made every green bar a 'sweep' (diagnosed: 63% junk events).
    A level is CONSUMED once swept (field standard: a raided liquidity pool is
    mitigated) — pivots are removed; prior-day low fires once per UTC day.
    Returns list of dicts: {idx, level, sweep_low, quality}.
    """
    low, close = df["low"].values, df["close"].values
    days = df.index.normalize()
    n = len(low)
    pivots = fallback_pivot_lows(low)
    piv_ptr = 0
    active = []  # (pivot_idx, level)
    events = []
    pdl_consumed_day = None
    for t in range(n):
        # activate pivots confirmed by bar t
        while piv_ptr < len(pivots) and pivots[piv_ptr][0] <= t:
            _, pidx, lvl = pivots[piv_ptr]
            active.append((pidx, lvl))
            if len(active) > MAX_PIVOTS:
                active.pop(0)
            piv_ptr += 1
        if np.isnan(atr[t]) or atr[t] <= 0:
            continue
        cands = []
        for j, (pidx, lvl) in enumerate(active):
            if t - pidx <= PIVOT_MAX_AGE and low[t] < lvl < close[t]:
                cands.append(("pivot", lvl, j))
        pdl = prior_day_low[t]
        if (not np.isnan(pdl) and low[t] < pdl < close[t]
                and days[t] != pdl_consumed_day):
            cands.append(("prior_day_low", pdl, -1))
        if not cands:
            continue
        # take the HIGHEST swept level (closest to the reclaim close);
        # consume ALL swept levels (the raid took them out)
        kind, lvl, _ = max(cands, key=lambda x: x[1])
        swept_piv = {j for k, _, j in cands if k == "pivot"}
        active = [p for j, p in enumerate(active) if j not in swept_piv]
        if any(k == "prior_day_low" for k, _, _ in cands):
            pdl_consumed_day = days[t]
        g = round_grid(close[t])
        q = fallback_level_quality(df, t, lvl, atr[t], kind, pdl, g)
        events.append({"idx": t, "level": lvl, "sweep_low": low[t], "quality": q})
    return events


def fallback_level_quality(df, t, lvl, atr_t, kind, pdl, grid):
    """FALLBACK level_quality_low in [0,1]:
      touches: bars in last 200 whose low came within 0.25*ATR of the level
               (multi-touch shelf -> 'levels are earned'), 0.2/touch capped 0.6
      +0.25 if level is the prior UTC-day low (session extreme)
      +0.15 if level sits on a round number (Osler stop clustering)."""
    lo = df["low"].values
    a = max(0, t - QUALITY_TOUCH_WINDOW)
    touches = int(np.sum(np.abs(lo[a:t] - lvl) <= QUALITY_TOUCH_TOL_ATR * atr_t))
    q = min(0.6, 0.2 * touches)
    if not np.isnan(pdl) and abs(lvl - pdl) <= QUALITY_TOUCH_TOL_ATR * atr_t:
        q += 0.25
    if abs(lvl - np.round(lvl / grid) * grid) <= 1e-9 or \
       abs(lvl / grid - np.round(lvl / grid)) < 0.02:
        q += 0.15
    return min(1.0, q)


# ---------------------------------------------------------------------------
# Real-module adapter (interface contract with the parallel level-features agent)
# ---------------------------------------------------------------------------

def try_real_features(df: pd.DataFrame, atr: np.ndarray):
    """Prefer engine.features.level_features.compute_level_features if importable.

    Swept-level reconstruction: the module defines sweep_low_event on
    support = fmax(prior_day_low, swing_low_50) (its line 183), so the swept
    level is the highest of those two inside the sweep bar's low..close range.
    Round numbers are NOT part of the module's sweep trigger and are excluded.

    day_type for SIZING: the module's day_type is CONCURRENT (-1 requires the
    current close below prior_day_low) — at any reclaim entry close > support
    >= prior_day_low, so concurrent day_type is PROVABLY never -1 at entry.
    Per the trader doc ('classify the day before trading it') sizing uses the
    prior UTC day's CLOSING day_type state, carried over the current day
    (point-in-time correct: prior day is complete)."""
    try:
        sys.path.insert(0, REPO)
        from engine.features.level_features import compute_level_features  # noqa
    except Exception:
        return None
    try:
        feats = compute_level_features(df)
        req = ["sweep_low_event", "level_quality_low", "day_type",
               "prior_day_low", "swing_low_50"]
        if any(c not in feats.columns for c in req):
            return None
        low, close = df["low"].values, df["close"].values
        events = []
        sw = feats["sweep_low_event"].values.astype(bool)
        qual = feats["level_quality_low"].values
        for t in np.where(sw)[0]:
            cands = [feats[c].values[t] for c in ("prior_day_low", "swing_low_50")]
            cands = [x for x in cands if np.isfinite(x) and low[t] < x < close[t]]
            if not cands:
                continue
            q = qual[t]
            events.append({"idx": t, "level": max(cands), "sweep_low": low[t],
                           "quality": float(q) if np.isfinite(q) else 0.0})
        # day-start day_type = prior day's closing concurrent state
        s = pd.Series(feats["day_type"].values.astype(float), index=df.index)
        day = df.index.normalize()
        day_start = s.groupby(day).last().shift(1)
        day_type = day_start.reindex(day).fillna(0.0).values
        return events, day_type, "engine.features.level_features (day-start day_type)"
    except Exception as e:
        print(f"[warn] real level_features failed ({e}); using fallback", file=sys.stderr)
        return None


# ============================================================================
# Trade simulation
# ============================================================================

@dataclass
class Trade:
    entry_idx: int
    exit_idx: int = 0
    entry_time: pd.Timestamp = None
    level: float = 0.0
    quality: float = 0.0
    day_type: float = 0.0
    size_mult: float = 1.0
    entry: float = 0.0
    stop: float = 0.0
    pnl: float = 0.0
    hold_bars: int = 0
    stopped_out: bool = False
    exits: list = field(default_factory=list)


def simulate_trade(i0, level, sweep_low, quality, day_type_i, size_mult,
                   o, h, l, c, atr, stop_mode="structural"):
    """Simulate one long trade entered at close of bar i0.
    stop_mode: 'structural' (sweep_low - 0.25*ATR) or 'fixed49' (entry - 4.9*ATR).
    Conservative intrabar ordering: stop checked BEFORE targets each bar.
    Returns Trade or None."""
    n = len(c)
    entry = c[i0]
    a = atr[i0]
    if stop_mode == "structural":
        stop = sweep_low - STOP_ATR_BUFFER * a
    else:
        stop = entry - H4_FIXED_ATR_STOP * a
    risk_dist = entry - stop
    if risk_dist <= 0 or not np.isfinite(risk_dist):
        return None
    qty0 = (CAPITAL * RISK_PCT * size_mult) / risk_dist
    tp1 = entry + TP1_R * risk_dist
    tp2 = entry + TP2_R * risk_dist

    tr = Trade(entry_idx=i0, level=level, quality=quality, day_type=day_type_i,
               size_mult=size_mult, entry=entry, stop=stop)
    remaining = qty0
    hit1 = hit2 = False
    trail = -np.inf
    max_high = entry
    pnl = -qty0 * entry * COST_PER_SIDE  # entry cost

    def exit_part(qty, price, why, i):
        nonlocal pnl, remaining
        pnl += qty * (price - entry) - qty * price * COST_PER_SIDE
        remaining -= qty
        tr.exits.append((why, i))

    i = i0
    for i in range(i0 + 1, min(i0 + TIME_STOP_BARS + 1, n)):
        # 1) hard stop first (conservative); gap-aware fill at open if gapped
        if l[i] <= stop:
            exit_part(remaining, min(stop, o[i]), "stop", i)
            tr.stopped_out = True
            break
        # 2) scale-outs (gap-aware: fill at open if it gapped past the target)
        if not hit1 and h[i] >= tp1:
            exit_part(qty0 * TP1_PCT, max(tp1, o[i]), "tp1", i)
            hit1 = True
        if not hit2 and h[i] >= tp2:
            exit_part(qty0 * TP2_PCT, max(tp2, o[i]), "tp2", i)
            hit2 = True
            trail = max(trail, h[i] - TRAIL_ATR * atr[i])
        # 3) runner trail (active after tp2)
        if hit2:
            max_high = max(max_high, h[i])
            trail = max(trail, max_high - TRAIL_ATR * atr[i])
            if l[i] <= trail and remaining > 1e-12:
                exit_part(remaining, min(max(trail, stop), o[i]), "trail", i)
                break
        if remaining <= 1e-12:
            break
    else:
        i = min(i0 + TIME_STOP_BARS, n - 1)
    if remaining > 1e-12:  # time stop / end of data
        exit_part(remaining, c[i], "time", i)
    tr.pnl = pnl
    tr.exit_idx = i
    tr.hold_bars = i - i0
    return tr


def build_entries(events, A, C, lq_min, closes, atr, n):
    """Entry candidates for one combo: sweep at s -> A consecutive closes above
    level (acceptance) -> enter at close of bar s+A, if close within C*ATR of
    level (no-chase) and quality >= lq_min."""
    out = []
    for ev in events:
        s, lvl, q = ev["idx"], ev["level"], ev["quality"]
        if q < lq_min:
            continue
        e = s + A
        if e >= n:
            continue
        if A > 0 and not np.all(closes[s + 1:e + 1] > lvl):
            continue
        if not np.isfinite(atr[e]) or atr[e] <= 0:
            continue
        if closes[e] - lvl > C * atr[e]:  # no-chase guard
            continue
        out.append((e, lvl, ev["sweep_low"], q))
    return out


def run_combo(entries, day_type, o, h, l, c, atr, idx, daytype_sizing,
              stop_mode="structural"):
    """Sequential sim: one position at a time, 18-bar cooldown after exit."""
    trades = []
    next_ok = 0
    for (e, lvl, swl, q) in entries:
        if e < next_ok:
            continue
        dt = day_type[e]
        mult = 0.25 if (daytype_sizing and dt == -1) else 1.0
        tr = simulate_trade(e, lvl, swl, q, dt, mult, o, h, l, c, atr, stop_mode)
        if tr is None:
            continue
        tr.entry_time = idx[e]
        trades.append(tr)
        next_ok = tr.exit_idx + COOLDOWN_BARS
    return trades


# ============================================================================
# Metrics
# ============================================================================

def metrics(trades):
    if not trades:
        return dict(n=0, pf=np.nan, pnl=0.0, win_rate=np.nan,
                    maxdd_pct=0.0, avg_hold_h=np.nan)
    pnl = np.array([t.pnl for t in trades])
    gp, gl = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    eq = CAPITAL + np.cumsum(pnl)
    peak = np.maximum.accumulate(np.concatenate([[CAPITAL], eq]))[1:]
    dd = ((eq - peak) / peak).min() * 100 if len(eq) else 0.0
    return dict(
        n=len(trades),
        pf=(gp / gl) if gl > 0 else np.inf,
        pnl=float(pnl.sum()),
        win_rate=float((pnl > 0).mean() * 100),
        maxdd_pct=float(dd),
        avg_hold_h=float(np.mean([t.hold_bars for t in trades])),
    )


def split_trades(trades, start, end):
    s, e = pd.Timestamp(start), pd.Timestamp(end)
    return [t for t in trades if s <= t.entry_time < e]


# ============================================================================
# Main
# ============================================================================

def main():
    os.makedirs(OUTDIR, exist_ok=True)
    df = pd.read_parquet(DATA)
    df.index = df.index.tz_localize(None)
    df = df.sort_index()
    o, h, l, c = (df[k].values for k in ("open", "high", "low", "close"))
    n = len(df)
    atr = atr14(df)

    # prior UTC-day low (known at day open — no lookahead)
    dlow = df["low"].resample("1D").min().shift(1)
    prior_day_low = dlow.reindex(df.index.normalize()).values

    real = try_real_features(df, atr)
    if real is not None:
        events, day_type, feat_src = real
    else:
        feat_src = "FALLBACK (engine/features/level_features.py not available)"
        events = fallback_sweep_events(df, atr, prior_day_low)
        day_type = fallback_day_type(df)
    print(f"feature source: {feat_src}")
    print(f"sweep_low events detected: {len(events)}")

    grid = list(product([1, 2, 3], [1.0, 1.5, 2.0], [0.0, 0.3, 0.5], [False, True]))
    rows, combo_trades = [], {}
    for A, C, lq, dts in grid:
        entries = build_entries(events, A, C, lq, c, atr, n)
        trades = run_combo(entries, day_type, o, h, l, c, atr, df.index, dts)
        trades = [t for t in trades if t.entry_time >= pd.Timestamp("2018-01-01")]
        combo_trades[(A, C, lq, dts)] = trades
        row = dict(A=A, C=C, lq_min=lq, daytype_sizing=dts)
        for name, (s, e) in SPLITS.items():
            m = metrics(split_trades(trades, s, e))
            for k, v in m.items():
                row[f"{name}.{k}"] = v
        rows.append(row)
        print(f"A={A} C={C} lq={lq} dts={int(dts)}: "
              + " | ".join(f"{nm}: n={row[f'{nm}.n']} pf={row[f'{nm}.pf']:.2f} "
                           f"pnl=${row[f'{nm}.pnl']:,.0f}" for nm in SPLITS))

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUTDIR, "grid_results.csv"), index=False)

    # ---------------- best train combo (pre-registered selection rule:
    # highest train PF among combos with train n>=30; tie-break train PnL)
    ok = res[res["train_2018_2022.n"] >= 30].copy()
    pick_pool = ok if len(ok) else res
    best = pick_pool.sort_values(
        ["train_2018_2022.pf", "train_2018_2022.pnl"], ascending=False).iloc[0]
    bkey = (int(best["A"]), float(best["C"]), float(best["lq_min"]),
            bool(best["daytype_sizing"]))
    btr = combo_trades[bkey]

    # per-year for best train combo
    per_year = []
    for y in range(2018, 2027):
        yt = split_trades(btr, f"{y}-01-01", f"{y + 1}-01-01")
        m = metrics(yt)
        m["year"] = y
        per_year.append(m)
    py = pd.DataFrame(per_year)[["year", "n", "pf", "pnl", "win_rate",
                                 "maxdd_pct", "avg_hold_h"]]

    # ---------------- H1
    h1 = res[(res["oos1_2023_2024.pf"] >= 1.3) & (res["oos1_2023_2024.n"] >= 30) &
             (res["oos2_2025_2026.pf"] >= 1.3) & (res["oos2_2025_2026.n"] >= 30)]

    # ---------------- H2: quality terciles on the lq=0 variant of best combo
    # (lq_min=0 so quality is untruncated), pooled and per macro-split
    qkey = (bkey[0], bkey[1], 0.0, bkey[3])
    qtr = combo_trades[qkey]
    h2_rows = []
    for scope, (s, e) in [("ALL", ("2018-01-01", "2026-06-11"))] + list(SPLITS.items()):
        tt = split_trades(qtr, s, e)
        if len(tt) < 9:
            continue
        qs = np.array([t.quality for t in tt])
        ps = np.array([t.pnl for t in tt])
        # rank-based terciles (stable ties) -> three near-equal groups
        order = np.argsort(qs, kind="stable")
        ranks = np.empty(len(qs), dtype=int)
        ranks[order] = np.arange(len(qs))
        bins = ranks * 3 // len(qs)
        for b, name in enumerate(("low", "mid", "high")):
            mask = bins == b
            h2_rows.append(dict(scope=scope, tercile=name, n=int(mask.sum()),
                                avg_pnl=float(ps[mask].mean()) if mask.any() else np.nan,
                                avg_quality=float(qs[mask].mean()) if mask.any() else np.nan))
    h2 = pd.DataFrame(h2_rows)

    # ---------------- H3: daytype ON vs OFF, per year, same (A,C,lq) as best
    h3_rows = []
    for dts in (False, True):
        k = (bkey[0], bkey[1], bkey[2], dts)
        for y in range(2021, 2027):
            m = metrics(split_trades(combo_trades[k], f"{y}-01-01", f"{y + 1}-01-01"))
            h3_rows.append(dict(daytype_sizing=dts, year=y, n=m["n"],
                                pf=m["pf"], pnl=m["pnl"]))
    h3 = pd.DataFrame(h3_rows)

    # ---------------- H4: structural vs 4.9*ATR fixed stop, same entries
    entries_b = build_entries(events, bkey[0], bkey[1], bkey[2], c, atr, n)
    h4_rows = []
    for mode in ("structural", "fixed49"):
        tr = run_combo(entries_b, day_type, o, h, l, c, atr, df.index, bkey[3],
                       stop_mode=mode)
        tr = [t for t in tr if t.entry_time >= pd.Timestamp("2018-01-01")]
        for name, (s, e) in SPLITS.items():
            tt = split_trades(tr, s, e)
            m = metrics(tt)
            so = float(np.mean([t.stopped_out for t in tt]) * 100) if tt else np.nan
            h4_rows.append(dict(stop=mode, split=name, n=m["n"], pf=m["pf"],
                                pnl=m["pnl"], win_rate=m["win_rate"],
                                stopout_rate_pct=so, maxdd_pct=m["maxdd_pct"]))
    h4 = pd.DataFrame(h4_rows)

    # ---------------- write summary
    write_summary(res, best, bkey, py, h1, h2, h3, h4, feat_src, len(events))
    print(f"\nwrote {OUTDIR}/grid_results.csv and {OUTDIR}/SUMMARY.md")
    print(f"\nBest train combo: A={bkey[0]} C={bkey[1]} lq>={bkey[2]} "
          f"daytype_sizing={bkey[3]}")
    print(py.to_string(index=False))
    print("\nH1 qualifying combos:", len(h1))
    print("\nH2 terciles:\n", h2.to_string(index=False))
    print("\nH3:\n", h3.to_string(index=False))
    print("\nH4:\n", h4.to_string(index=False))


def fmt_df(d):
    return d.to_markdown(index=False, floatfmt=".2f")


def write_summary(res, best, bkey, py, h1, h2, h3, h4, feat_src, n_events):
    lines = []
    w = lines.append
    w("# Wick Trap v2 (Failed Breakdown) — Event Study Results")
    w("")
    w(f"**Date**: {pd.Timestamp.now():%Y-%m-%d} | **Feature source**: {feat_src}")
    w(f"**Data**: BTCUSDT 1h, 2017-10 -> 2026-06 (trades counted from 2018-01-01)")
    w(f"**Sweep events detected**: {n_events}")
    w(f"**Sim**: $100K base, 2% static risk/trade, 5bps/side total cost, "
      f"structural stop (sweep low - 0.25*ATR), scale 50%@1R / 30%@2R / "
      f"20% runner 2*ATR trail, 168h time stop, 18-bar cooldown, "
      f"conservative intrabar ordering (stop before targets).")
    w("")
    w("Pre-registered grid: A in {1,2,3}, C in {1.0,1.5,2.0}, "
      "lq_min in {0,0.3,0.5}, day_type x0.25 sizing ON/OFF = 54 combos. "
      "All 54 reported in grid_results.csv — no cherry-picking, no post-hoc "
      "extensions. Combos with split n<30 are NOISE and flagged as such.")
    w("")
    w("## Full grid (per-split PF / n / PnL)")
    w("")
    g = res.copy()
    for nm in SPLITS:
        g[f"{nm}"] = g.apply(
            lambda r: f"n={int(r[f'{nm}.n'])}{'*' if r[f'{nm}.n'] < 30 else ''} "
                      f"PF={r[f'{nm}.pf']:.2f} ${r[f'{nm}.pnl']/1000:,.1f}K", axis=1)
    w(g[["A", "C", "lq_min", "daytype_sizing"] + list(SPLITS)].to_markdown(index=False))
    w("")
    w("`*` = n<30 in that split: treat as noise.")
    w("")
    w(f"## Best train combo (pre-registered rule: max train PF, train n>=30)")
    w("")
    w(f"A={bkey[0]}, C={bkey[1]}, lq_min={bkey[2]}, daytype_sizing={bkey[3]}")
    w("")
    w("### Per-year")
    w("")
    w(fmt_df(py))
    w("")
    w("## H1 — any combo PF>=1.3 AND n>=30 in BOTH OOS windows?")
    w("")
    if len(h1):
        w(f"**YES — {len(h1)} combo(s) qualify:**")
        w("")
        cols = ["A", "C", "lq_min", "daytype_sizing",
                "oos1_2023_2024.n", "oos1_2023_2024.pf", "oos1_2023_2024.pnl",
                "oos2_2025_2026.n", "oos2_2025_2026.pf", "oos2_2025_2026.pnl"]
        w(fmt_df(h1[cols]))
    else:
        w("**NO.** No combo achieved PF>=1.3 with n>=30 in both OOS windows. "
          "Negative result, reported plainly.")
    w("")
    w("## H2 — level_quality terciles vs avg trade PnL (lq_min=0 variant of best combo)")
    w("")
    w(fmt_df(h2))
    w("")
    w("## H3 — day_type x0.25 counter-trend sizing, per year (best combo's A/C/lq)")
    w("")
    w(fmt_df(h3))
    w("")
    w("## H4 — structural stop vs 4.9*ATR fixed stop (same entries, best combo)")
    w("")
    w(fmt_df(h4))
    w("")
    w("## Caveats")
    w("- Research harness, NOT the production backtester: static (non-compounding)"
      " sizing, long-only, no portfolio interaction with other archetypes.")
    w("- day_type for sizing = prior UTC day's CLOSING day_type state (the "
      "module's concurrent day_type is provably never -1 at a reclaim entry: "
      "entry requires close > support >= prior_day_low, while concurrent -1 "
      "requires close < prior_day_low). 'Classify the day before trading it.'")
    w("- Swept level reconstructed as max(prior_day_low, swing_low_50) inside "
      "the sweep bar's low..close range — matches the module's sweep "
      "definition; round numbers excluded (they are quality context, not "
      "sweep triggers).")
    w("- Stops fill at min(stop, next open) (gap-aware); intrabar ordering is "
      "conservative (stop checked before targets in the same bar).")
    w("- Splits cross sim state (cooldown/one-position) at boundaries; effect is "
      "negligible at 18-bar cooldown.")
    with open(os.path.join(OUTDIR, "SUMMARY.md"), "w") as f:
        f.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
