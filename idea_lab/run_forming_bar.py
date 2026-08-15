"""
DELIVERABLE 2 — FORMING-BAR GAP comparison (wyckoff_audit add.66; STUDY ONLY)
============================================================================
ONE pre-registered comparison (no grid):
  A = status-quo door (retest-hold on completed daily close)  [TrendContinuationDoor]
  B = forming-bar door (retest-hold may confirm on week-to-date evidence) [FormingRetestDoor]
on the 25-market wide basket + a pre-registered 50-stock random sample (seed 66).

Reports: fire counts, ADDED fires (B fires A didn't), entry-timing shift (days earlier for
matched break_level setups), and paired outcome delta. Referee parity on a 8-name sample.
"""
from __future__ import annotations
import os, sys, json, random, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))

from xasset_spx_port import load_spx
from trend_continuation_door import build_daily_sensors, TrendContinuationDoor
from forming_bar_sensor import FormingRetestDoor, truncation_check_wtd
from backtester import run_backtest
from run_xasset_spx import selftest_on
from run_wi_batch7 import breadth300_universe, wide_universe, RECENT, pf_of

SEED = 66


def run_door(DoorCls, df, sr, bj, eye, key):
    strat = DoorCls(df, sr, bj, eye, variant="struct", conviction=False)
    res = run_backtest(df, strat, label=key)
    rows = []
    for t, e in zip(res["trades"], strat.entries_log):
        rows.append(dict(key=key, entry_time=pd.Timestamp(t["entry_time"]),
                         break_level=e.get("break_level", np.nan), R=float(t["R"]),
                         below=bool(e.get("below_ema200", False))))
    return rows, res["stats"]


def analyze(key, path, fam, recent_only, do_selftest=False):
    df = load_spx(path)
    dfs, sr, bj, eye = build_daily_sensors(df)
    st = selftest_on(dfs, key) if do_selftest else None
    tc = truncation_check_wtd(dfs.index, dfs["high"].to_numpy(float))
    A, sA = run_door(TrendContinuationDoor, dfs, sr, bj, eye, key)
    B, sB = run_door(FormingRetestDoor, dfs, sr, bj, eye, key)
    if recent_only:
        A = [r for r in A if r["entry_time"] >= RECENT]
        B = [r for r in B if r["entry_time"] >= RECENT]
    return A, B, st, tc


def match_and_compare(A, B):
    """Match B fires to A fires by (key, break_level) within 60 trading days; report
    timing shift and paired R for matched setups; count added/removed."""
    a_by = {}
    for r in A:
        a_by.setdefault((r["key"], round(r["break_level"], 6)), []).append(r)
    matched = []; added = []
    used = set()
    for rb in B:
        kkey = (rb["key"], round(rb["break_level"], 6))
        cands = a_by.get(kkey, [])
        best = None
        for j, ra in enumerate(cands):
            if (rb["key"], id(ra)) in used:
                continue
            dt = abs((rb["entry_time"] - ra["entry_time"]).days)
            if dt <= 60 and (best is None or dt < best[1]):
                best = (ra, dt, j)
        if best is not None:
            ra, dt, j = best
            used.add((rb["key"], id(ra)))
            shift = (ra["entry_time"] - rb["entry_time"]).days  # >0 => B earlier
            matched.append(dict(key=rb["key"], shift_days=shift, R_A=ra["R"], R_B=rb["R"]))
        else:
            added.append(rb)
    return matched, added


def report(tag, A, B, matched, added):
    RA = np.array([r["R"] for r in A]); RB = np.array([r["R"] for r in B])
    print(f"\n{'='*90}\n{tag}\n{'='*90}")
    print(f"  A status-quo : n={len(A):4d}  PF {pf_of(RA):.2f}  meanR {RA.mean():+.3f}  sumR {RA.sum():+.1f}")
    print(f"  B forming    : n={len(B):4d}  PF {pf_of(RB):.2f}  meanR {RB.mean():+.3f}  sumR {RB.sum():+.1f}")
    print(f"  matched setups (same break_level): {len(matched)}   B-only ADDED fires: {len(added)}")
    if matched:
        shifts = np.array([m["shift_days"] for m in matched])
        earlier = int((shifts > 0).sum()); same = int((shifts == 0).sum())
        dR = np.array([m["R_B"] - m["R_A"] for m in matched])
        print(f"  timing: B earlier on {earlier}/{len(matched)} matched (same-day {same}); "
              f"median shift {np.median(shifts):+.0f}d  max {shifts.max():+.0f}d")
        print(f"  matched paired ΔR (B-A): mean {dR.mean():+.4f}  sum {dR.sum():+.2f}  "
              f"(n_nonzero {int((np.abs(dR)>1e-9).sum())})")
    if added:
        RAdd = np.array([r["R"] for r in added])
        print(f"  ADDED-fire outcomes: meanR {RAdd.mean():+.3f}  sumR {RAdd.sum():+.1f}  "
              f"PF {pf_of(RAdd):.2f}  WR {(RAdd>0).mean():.0%}")
    return dict(nA=len(A), nB=len(B), pfA=pf_of(RA), pfB=pf_of(RB),
                meanRA=float(RA.mean()), meanRB=float(RB.mean()),
                n_matched=len(matched), n_added=len(added),
                added_meanR=(float(np.mean([r["R"] for r in added])) if added else None),
                added_sumR=(float(np.sum([r["R"] for r in added])) if added else None))


def main():
    print("="*90)
    print("DELIVERABLE 2 — FORMING-BAR GAP (add.66): status-quo vs forming-week retest-hold")
    print("="*90)

    # ---- stock sample (pre-registered 50, seed 66) ----
    uni = breadth300_universe()
    random.seed(SEED)
    stock_sample = random.sample(uni, min(50, len(uni)))
    st_sample = set(k for k, _, _ in random.sample(stock_sample, 8))
    print(f"stock universe {len(uni)}; sample 50 (seed {SEED}); referee self-test on 8: {sorted(st_sample)}")

    stA, stB = [], []; st_fail = []; trunc_checked = trunc_mism = 0
    for key, path, sec in stock_sample:
        try:
            A, B, st, tc = analyze(key, path, sec, recent_only=True, do_selftest=(key in st_sample))
        except Exception as ex:
            print(f"  ERR {key}: {ex}"); continue
        stA += A; stB += B
        trunc_checked += tc[0]; trunc_mism += tc[1]
        if st is False: st_fail.append(key)
    print(f"REFEREE PARITY (stocks): {'ALL 0.00% PASS' if not st_fail else 'FAIL '+str(st_fail)}")
    m_st, a_st = match_and_compare(stA, stB)
    rep_st = report("STOCK SAMPLE (50 names, recent 2018+)", stA, stB, m_st, a_st)

    # ---- wide basket (25 trending pass families + shadow) ----
    wu = wide_universe()
    wbA, wbB = [], []
    for key, path, fam in wu:
        if not os.path.exists(path):
            continue
        if fam in ("fx", "energy"):   # trending pass-basket only, per add.62
            continue
        try:
            A, B, _, tc = analyze(key, path, fam, recent_only=False, do_selftest=False)
        except Exception as ex:
            print(f"  ERR wide {key}: {ex}"); continue
        wbA += A; wbB += B
        trunc_checked += tc[0]; trunc_mism += tc[1]
    m_wb, a_wb = match_and_compare(wbA, wbB)
    rep_wb = report("WIDE BASKET (trending pass families, full history)", wbA, wbB, m_wb, a_wb)

    print(f"\nWTD truncation no-repaint check: {trunc_checked} points, {trunc_mism} mismatches "
          f"-> {'CAUSAL/no-repaint' if trunc_mism == 0 else 'REPAINT DETECTED'}")

    dump = dict(stock=rep_st, wide=rep_wb, trunc_checked=trunc_checked, trunc_mism=trunc_mism,
                st_fail=st_fail)
    outp = os.path.join(os.path.dirname(__file__), "forming_bar_results.json")
    json.dump(dump, open(outp, "w"), indent=2, default=str)
    print(f"\nresults -> {outp}")


if __name__ == "__main__":
    main()
