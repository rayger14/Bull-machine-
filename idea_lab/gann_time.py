"""
GANN TIME LAYER  (STUDY ONLY -- WI batch-6 verbatim mechanics, wyckoff_audit add.57)
====================================================================================
Harmonic-count time engine. "Gann timed. Wyckoff mapped." Time says WHEN turns are
likely; never trade Gann alone -- react only when a Gann window and a Wyckoff/structure
confirmation align. This module produces ONLY the time windows; the strategy layer ANDs
them with structure.

FRESH IMPLEMENTATION (stated per the task): the founding-era engine/temporal/gann.py
(Fibonacci 21/34/55/89 with a 30-day projection cap) and gann_cycles.py (ACF 30/60/90 +
Square-of-9 PRICE levels) do NOT implement WI's calendar-day harmonic-count-from-anchor
scheme. The halving date there (2024-04-20) also differs from WI's verbatim 2024-04-19.
So this is built fresh from the add.57 verbatim spec; nothing is reused from those files.

VERBATIM SPEC (add.57):
  - Anchors = major swing highs/lows (weekly N=5 pivots) + the crypto HALVINGS
    (2016-07-09, 2020-05-11, 2024-04-19).
  - Counts = 90 / 180 / 360 / 540 / 720 / 1080 / 1440 CALENDAR DAYS from each anchor,
    plus 144 (the recurring hidden number) as 144 DAYS and 144 WEEKS (=1008 days).
  - A bar is IN-WINDOW if within +-3 days of any count from any anchor ("+-1-3 bars").
  - Halving vibrations: +180d minor, +360d expansion, +720d 2-yr crest / DISTRIBUTION,
    +1440d next-halving reset.

CAUSALITY (verified by the self-test at the bottom):
  Anchors are CONFIRMED pivots only -- a weekly N=5 pivot is confirmed 5 weeks after it
  prints (symmetric fractal), and a bar t may only "see" an anchor whose confirm-date
  <= t. Every count is >= 90 days, and the pivot confirmation lag is ~35 days, so all
  count windows open (>= 87 days after the anchor) strictly AFTER the anchor is already
  confirmed -- causal by construction. Halvings are fixed past dates (no lag). A 3-point
  truncation check confirms in-window flags computed on truncated history match the
  full-history flags up to the truncation point (NO repaint).

TWO PRE-REGISTERED WINDOW TYPES:
  entry_window[t]  : within +-TOL_D of ANY count from ANY anchor (highs, lows, halvings)
                     -> the G-ENTRY confluence tier (measurement first; sizing only).
  danger_window[t] : within +-TOL_D of the "final/danger" counts (OURS-interpretation,
                     flagged): halving + {720,1080,1440}  OR  a MAJOR SWING HIGH +
                     {360,540,720,1080,1440}. Matches WI "720d+ from halving, or 360d+
                     counts from a MAJOR high" -> the G-EXIT stand-down tier (ANDed with
                     structure weakness by the strategy).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

# --- VERBATIM counts (add.57) ---
COUNTS_DAYS = [90, 180, 360, 540, 720, 1080, 1440]
COUNT_144_DAYS = 144
COUNT_144_WEEKS_DAYS = 144 * 7        # 1008 days
ALL_COUNTS = sorted(set(COUNTS_DAYS + [COUNT_144_DAYS, COUNT_144_WEEKS_DAYS]))

# --- VERBATIM crypto halvings (add.57) ---
HALVINGS = [pd.Timestamp("2016-07-09"), pd.Timestamp("2020-05-11"),
            pd.Timestamp("2024-04-19")]

# --- OURS (flagged): tolerance and the danger-count definition ---
TOL_D = 3                              # "+-1-3 bars" -> +-3 calendar days (widest verbatim)
WEEKLY_N = 5                           # weekly swing-pivot fractal half-width (add.57 "weekly N=5")
DANGER_HALVING_COUNTS = [720, 1080, 1440]
DANGER_HIGH_COUNTS = [360, 540, 720, 1080, 1440]


def _weekly_pivots(df_daily: pd.DataFrame, n: int = WEEKLY_N):
    """Confirmed weekly N=n symmetric fractal pivots. Returns list of dicts:
    {date: pivot week-start date, confirm: date confirmed (pivot_week + n weeks),
     kind: 'high'|'low', price}. Causal: a pivot at week p is confirmed only after
     n right-side weekly bars close, i.e. at week p+n."""
    o = df_daily["open"].resample("1W").first()
    h = df_daily["high"].resample("1W").max()
    l = df_daily["low"].resample("1W").min()
    c = df_daily["close"].resample("1W").last()
    wk = pd.DataFrame({"open": o, "high": h, "low": l, "close": c}).dropna()
    hi = wk["high"].to_numpy(float)
    lo = wk["low"].to_numpy(float)
    widx = wk.index
    piv = []
    m = len(wk)
    for p in range(n, m - n):
        if hi[p] > hi[p - n:p].max() and hi[p] > hi[p + 1:p + 1 + n].max():
            piv.append({"date": widx[p], "confirm": widx[p + n], "kind": "high",
                        "price": float(hi[p])})
        if lo[p] < lo[p - n:p].min() and lo[p] < lo[p + 1:p + 1 + n].min():
            piv.append({"date": widx[p], "confirm": widx[p + n], "kind": "low",
                        "price": float(lo[p])})
    return piv


def _anchors(df_daily: pd.DataFrame, use_halvings: bool):
    """Assemble the causal anchor list. Each anchor: {date, confirm, kind, source}.
    Halvings are always-confirmed (confirm == date). Weekly pivots confirm at +n weeks.
    Halvings are only included if they fall inside the data span (use_halvings=crypto)."""
    anchors = []
    for pv in _weekly_pivots(df_daily):
        anchors.append({"date": pv["date"], "confirm": pv["confirm"],
                        "kind": pv["kind"], "source": "pivot"})
    if use_halvings:
        lo, hi = df_daily.index.min(), df_daily.index.max()
        for hv in HALVINGS:
            if lo <= hv <= hi:
                anchors.append({"date": hv, "confirm": hv, "kind": "halving",
                                "source": "halving"})
    return anchors


def _mark(index: pd.DatetimeIndex, anchor_date, confirm_date, counts, tol_d):
    """Return a boolean array over `index`: True where the bar date is within +-tol_d of
    (anchor_date + count) for any count, AND the anchor is already confirmed at that bar
    (bar_date >= confirm_date). Causal."""
    days = index.values.astype("datetime64[D]")
    out = np.zeros(len(index), dtype=bool)
    confirmed = index >= confirm_date
    for cnt in counts:
        center = np.datetime64(pd.Timestamp(anchor_date).date()) + np.timedelta64(cnt, "D")
        delta = (days - center).astype("timedelta64[D]").astype(int)
        out |= (np.abs(delta) <= tol_d) & confirmed
    return out


def compute_gann_windows(df_daily: pd.DataFrame, use_halvings: bool = True,
                         tol_d: int = TOL_D):
    """Causal per-bar Gann windows on a daily frame.
    Returns dict:
      entry_window : bool[n]  (any count from any anchor)
      danger_window: bool[n]  (final/danger counts: halving 720+ or MAJOR-high 360+)
      n_anchors, anchors (list), by_count (diagnostic counts per horizon).
    """
    index = df_daily.index
    n = len(index)
    entry_w = np.zeros(n, dtype=bool)
    danger_w = np.zeros(n, dtype=bool)
    anchors = _anchors(df_daily, use_halvings)
    for a in anchors:
        entry_w |= _mark(index, a["date"], a["confirm"], ALL_COUNTS, tol_d)
        # danger tier
        if a["source"] == "halving":
            danger_w |= _mark(index, a["date"], a["confirm"], DANGER_HALVING_COUNTS, tol_d)
        elif a["kind"] == "high":
            danger_w |= _mark(index, a["date"], a["confirm"], DANGER_HIGH_COUNTS, tol_d)
    return {"entry_window": entry_w, "danger_window": danger_w,
            "n_anchors": len(anchors), "anchors": anchors}


# ===========================================================================
# SELF-TEST: causality + 3-point truncation (no repaint) + coverage sanity.
# ===========================================================================
def run_selftest(df_daily: pd.DataFrame, use_halvings: bool = True) -> bool:
    full = compute_gann_windows(df_daily, use_halvings)
    n = len(df_daily)
    ew = full["entry_window"]; dw = full["danger_window"]
    print("\n=== GANN TIME LAYER SELF-TEST ===")
    print(f"  bars={n}  anchors={full['n_anchors']} "
          f"(halvings {'ON' if use_halvings else 'OFF'})")
    print(f"  entry_window coverage : {100*ew.mean():.1f}% of bars")
    print(f"  danger_window coverage: {100*dw.mean():.1f}% of bars")

    # (1) Causality invariant: every in-window bar must have an anchor confirmed on/before it.
    idx = df_daily.index
    causal_ok = True
    confirm_dates = [a["confirm"] for a in full["anchors"]]
    earliest_confirm = min(confirm_dates) if confirm_dates else idx.max()
    # a stricter check: no window may open before the FIRST possible count after the FIRST
    # confirmed anchor (>=90d - tol). We just assert no in-window bar precedes any confirm.
    first_win = np.where(ew)[0]
    if len(first_win):
        if idx[first_win[0]] < earliest_confirm:
            causal_ok = False
    print(f"  causality (no window before any anchor confirm): "
          f"{'PASS' if causal_ok else 'FAIL'}")

    # (2) 3-point truncation / no-repaint: compute on truncated history at 3 cut points;
    #     flags up to the cut must be IDENTICAL to the full-history flags (no future data
    #     ever changes a past flag).
    repaint_bad = 0
    cuts = [int(0.4 * n), int(0.6 * n), int(0.8 * n)]
    for cut in cuts:
        sub = df_daily.iloc[:cut]
        r = compute_gann_windows(sub, use_halvings)
        # align on shared index
        m_full_e = ew[:cut]
        m_sub_e = r["entry_window"]
        m_full_d = dw[:cut]
        m_sub_d = r["danger_window"]
        de = int(np.sum(m_full_e != m_sub_e))
        dd = int(np.sum(m_full_d != m_sub_d))
        repaint_bad += de + dd
        print(f"  truncation@{cut:>5} ({idx[cut-1].date()}): "
              f"entry mismatches={de}  danger mismatches={dd}")
    repaint_ok = repaint_bad == 0
    print(f"  3-point truncation (no-repaint): total mismatches={repaint_bad} -> "
          f"{'PASS' if repaint_ok else 'FAIL'}")

    ok = causal_ok and repaint_ok
    print(f"  GANN SELF-TEST -> {'PASS' if ok else 'FAIL'}")
    return ok


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(__file__))
    from xasset_spx_port import load_spx
    SC = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
          "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/one_strategy")
    df = load_spx(os.path.join(SC, "BTC-USD.parquet"))
    run_selftest(df, use_halvings=True)
