"""
WI BATCH-7 CONFLUENCE TEST  (wyckoff_audit add.65; STUDY ONLY; nothing ships)
============================================================================
The first POWERED test of Wyckoff-Insider's confluence-stack critique of THE DOOR.
WI reviewed our exact mechanized rulebook and said the door "fires on structure
alone -- I almost never take the pure retest without additional confluence." His
core claim ON TRIAL: "the naked retest underperforms the confluent retest."

Until now untestable (n~106). Now decidable with:
  * PRIMARY   = 476 S&P-500 names, recent 2018-2026  (n ~ 2,622 fires)
  * SECONDARY = (a) full-history stock pool (n ~ 8,601)  +  (b) 25-market wide basket
  * the add.60-verified PAIRED size-multiplier ΔR machinery (MDE 0.02-0.05R at this n)

THE DOOR IS FROZEN. Every batch-7 spec is tested as an OVERLAY on the door's
EXISTING fires. NO door param/logic change. All six flags are computed in a
POST-HOC pass from the SAME causal sensor arrays the door used (OHLCV, atr_14,
eye_state/eye_dir/range_upper_1d) + the door's own logged break_level. The break
BAR index is recomputed deterministically from the eye arrays (verified to
reproduce the door's break_level exactly, 26/26 on AAPL) -> zero door mutation.

CAUSALITY: every flag reads only bars <= entry bar, EXCEPT N3 (acceptance) which
by design reads bars i+1,i+2 -- it is a DELAYED-ENTRY diagnostic, NOT a sizing
overlay, and is EXCLUDED from the causal combined score {N1,N2a,N2b,N4}.

-----------------------------------------------------------------------------
THE SIX SPECS  (pre-registered EXACT definitions; WI numbers verbatim where given,
                flagged OURS where interpolated; NO grids beyond these values)
-----------------------------------------------------------------------------
N1 FRESHNESS      : bars-from-confirmed-break to retest entry <= F. F=10
                    (WI "8-12", midpoint, OURS). Also report full bars-to-retest
                    distribution + outcome-vs-staleness diagnostic.
N2a SWEEP-RETEST  : on entry bar i OR bar i-1, the LOW undercuts break_level by
                    0.1-0.4 ATR  (0.1*ATR <= break_level-low <= 0.4*ATR)  AND that
                    bar CLOSES back above break_level. WI "spring-style/liquidity
                    grab retest". ATR = atr_14 at that bar.
N2b CANDLE ANATOMY: entry-bar lower wick >= 0.55 of total range  AND  close in top
                    40% of range ((close-low)/(high-low) >= 0.60). WI verbatim
                    "55-60%", use 0.55. Flagged: adjacent to closed Bojan/quality
                    axes but this EXACT anatomy rule was never tested.
N3 ACCEPTANCE     : within 2 bars after the retest bar a higher low forms
                    (exists j in {i+1,i+2} with low[j] > low[i]). FORWARD info ->
                    reported as (a) descriptive flag and (b) an entry-shift variant
                    (enter at bar i+2 close when flag true), NOT as a sizing overlay.
N4 ORIGIN LOCATION: at the break bar, break_level sits in the LOWER 60% of the
                    trailing 365-CALENDAR-DAY high-low span
                    ((break_level-lo365)/(hi365-lo365) <= 0.60). WI "breakout from
                    discount/EQ of the HTF range, not extreme premium"; 0.60 OURS.
N5 CHOP-COUNT VETO: count FAILED breakouts of the SAME instrument in the trailing
                    90 CALENDAR days. PRE-REGISTERED single failed-breakout def:
                    a bull CONFIRMED_BREAK bar b (eye_state==CONFIRMED_BREAK &
                    eye_dir=='bull') is FAILED iff, within 15 bars after b, some
                    close drops back below its break_level (=range_upper_1d[b]);
                    the failure is COUNTED for a fire at bar i only if it RESOLVED
                    (resolution bar) <= i  (strict causal, no repaint). Flag =
                    failed_count >= 3 -> "chop regime". Tested as a VETO (removal)
                    AND as inverse sizing (0.75x). Report bear/chop-window effect.
N6 CLIMAX-ATR VETO: atr_14 at the break bar > 1.8x its own trailing-90-bar median
                    (blow-off). 1.8 OURS. Tested as a VETO (removal).

COMBINED CONFLUENCE SCORE = sum of passing specs among {N1,N2a,N2b,N4} (the causal
  entry-quality ones). Report mean R by score level 0-4 (MONOTONICITY is the test)
  and a sized portfolio (1.0R base, 1.25R at score>=2, 1.5R at score>=3) vs flat.

-----------------------------------------------------------------------------
PASS RULES  (pre-registered, fixed BEFORE measuring)
-----------------------------------------------------------------------------
A sizing-tier spec ADOPTS iff:
  (P1) paired ΔR bootstrap 95% CI-lo > 0 on the PRIMARY set  (the task rule), AND
  (P2) the SIGN holds on the SECONDARY set, AND
  (P3) coverage non-degenerate (10-90%).
  [We ALSO report C2 = meanR_flag >= meanR_unflag per the add.61 binding lesson:
   on a net-positive book C1/P1 is a WEAK bar (almost any positive-catching flag
   clears it); C2 + family/monotonicity consistency is what actually decides. Any
   P1-pass with C2-fail is flagged a "false pass" per add.61.]
A veto (N5/N6) ADOPTS iff its removal-delta (meanR improvement) bootstrap CI-lo>0
  on PRIMARY AND the sign holds on SECONDARY AND coverage non-degenerate.
The combined score ADOPTS iff mean R is MONOTONIC in score AND the sized portfolio
  beats flat with paired CI-lo > 0.
Any ADOPT = prototype-grade; forward proof still required. Failures CLOSE per the
standing one-round discipline (round 1 for N1-N6).
"""
from __future__ import annotations
import os, sys, json, random, warnings
from collections import defaultdict, Counter

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from xasset_spx_port import load_spx
from trend_continuation_door import build_daily_sensors, TrendContinuationDoor
from backtester import run_backtest, INITIAL_CASH
from run_xasset_spx import selftest_on
from engine.features.eye_state import CONFIRMED_BREAK

SCROOT = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
          "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad")
B3 = os.path.join(SCROOT, "breadth300")
WB = os.path.join(SCROOT, "wide_basket")
OS_ = os.path.join(SCROOT, "one_strategy")
XA = os.path.join(SCROOT, "xasset")

FIXED_RISK = 0.01 * INITIAL_CASH           # $1000/trade, matches add.62/64 pooling
RECENT = pd.Timestamp("2018-01-01")
MIN_YEARS = 8.0
M2_SOS_WIN = 360                            # door constant (for break-bar recompute)

# ---- pre-registered spec constants ----
N1_F = 10                                   # freshness bars (OURS, WI 8-12 midpoint)
N2A_LO, N2A_HI = 0.1, 0.4                   # sweep undercut in ATR (WI verbatim)
N2B_WICK = 0.55                             # lower-wick fraction (WI 55-60, use 0.55)
N2B_CLOSEPOS = 0.60                         # close in top 40% of range
N4_DISCOUNT = 0.60                          # origin in lower 60% of 365d span (OURS)
N4_LOOKBACK_DAYS = 365
N5_WINDOW_DAYS = 90                         # trailing chop-count window
N5_FAIL_BARS = 15                           # break->close-back-below window
N5_THRESH = 3                               # failed_count >= 3 = chop
N6_ATR_MULT = 1.8                           # climax multiple (OURS)
N6_MED_BARS = 90
BOOT = 10_000
SEED = 20260815
MULT = 1.25                                 # sizing multiplier -> paired ΔR = 0.25*R*flag
INV_MULT = 0.75                             # inverse-size for N5 -> ΔR = -0.25*R*flag

EQUITY_BEARS = [
    ("2018Q4", "2018-10-01", "2018-12-24"),
    ("COVID",  "2020-02-19", "2020-04-07"),
    ("2022",   "2022-01-03", "2022-10-13"),
    ("2026YTD","2026-01-01", "2026-12-31"),
]


# =========================================================================== flags
def confirmed_break_events(es, ed, ru):
    """All bull CONFIRMED_BREAK bars -> (break_idx, break_level)."""
    out = []
    for b in range(len(es)):
        if es[b] == CONFIRMED_BREAK and ed[b] == "bull" and np.isfinite(ru[b]):
            out.append((b, float(ru[b])))
    return out


def precompute_failed_breaks(closes, breaks, fail_bars):
    """For each confirmed bull break (b, level), determine if it FAILED (a close
    drops back below level within fail_bars) and its RESOLUTION bar (strict causal:
    first close<level, else b+fail_bars if it never recovers=not failed).
    Returns list of (break_idx, resolution_idx) for FAILED breaks only."""
    n = len(closes)
    failed = []
    for (b, lvl) in breaks:
        end = min(b + fail_bars, n - 1)
        res = None
        for j in range(b + 1, end + 1):
            if closes[j] < lvl:
                res = j
                break
        if res is not None:
            failed.append((b, res))       # resolved-failed at bar res
    return failed


def analyze_asset(key, path, family, do_selftest=False):
    """Run the FROZEN door; compute the six batch-7 flags per fire (post-hoc,
    causal). Returns (records, selftest_ok, n_break_fires)."""
    df_raw = load_spx(path)
    df, sr, bj, eye = build_daily_sensors(df_raw)
    st = selftest_on(df, key) if do_selftest else None

    strat = TrendContinuationDoor(df, sr, bj, eye, variant="struct", conviction=False)
    res = run_backtest(df, strat, label=key)
    trades = res["trades"]; elog = strat.entries_log
    assert len(trades) == len(elog), f"{key}: trade/log length mismatch"

    o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
    atr = df["atr_14"].to_numpy(float)
    es = eye["eye_state"].to_numpy(dtype=object); ed = eye["eye_dir"].to_numpy(dtype=object)
    ru = eye["range_upper_1d"].to_numpy(float)
    idx = df.index
    ts = np.asarray(idx.values, dtype="datetime64[ns]")
    n = len(df)

    # N5 sensor: all failed breaks with resolution bars (causal)
    breaks = confirmed_break_events(es, ed, ru)
    failed = precompute_failed_breaks(c, breaks, N5_FAIL_BARS)   # [(b_idx, res_idx)]
    fail_bidx = np.array([f[0] for f in failed], dtype=int)
    fail_res = np.array([f[1] for f in failed], dtype=int)

    records = []
    for t, e in zip(trades, elog):
        i = idx.get_loc(t["entry_time"])
        if isinstance(i, slice):           # dup timestamp guard (should not happen)
            i = i.start
        bl = e.get("break_level", np.nan)
        below = bool(e.get("below_ema200", False))
        R = float(t["R"])
        rec = dict(key=key, fam=family, entry_time=str(idx[i]), entry_idx=int(i),
                   R=R, below_ema200=below, break_level=(float(bl) if np.isfinite(bl) else None))

        if not np.isfinite(bl):
            # M1 fire (no break_level) -> door here is M2-broad-only so this is ~never;
            # still record with all flags False so counts are honest.
            rec.update(bars_to_retest=None, n1=False, n2a=False, n2b=False, n3=False,
                       n4=False, n5_count=0, n5=False, n6=False, score=0)
            records.append(rec)
            continue

        # break bar (recompute exactly as the door did)
        lo = max(0, i - M2_SOS_WIN); bidx = None
        for cc in range(i - 1, lo - 1, -1):
            if es[cc] == CONFIRMED_BREAK and ed[cc] == "bull":
                bidx = cc; break
        # (bidx must be finite & ru match -- verified; guard anyway)
        atr_i = atr[i]

        # ---- N1 freshness ----
        bars_to_retest = (i - bidx) if bidx is not None else None
        n1 = bool(bidx is not None and (i - bidx) <= N1_F)

        # ---- N2a sweep-retest (bar i or i-1) ----
        n2a = False
        for b in (i, i - 1):
            if b < 0 or not np.isfinite(atr[b]) or atr[b] <= 0:
                continue
            undercut = bl - l[b]
            if (N2A_LO * atr[b] <= undercut <= N2A_HI * atr[b]) and (c[b] >= bl):
                n2a = True; break

        # ---- N2b candle anatomy (entry bar i) ----
        rng = h[i] - l[i]
        if rng > 0:
            lower_wick = min(o[i], c[i]) - l[i]
            close_pos = (c[i] - l[i]) / rng
            n2b = bool((lower_wick / rng >= N2B_WICK) and (close_pos >= N2B_CLOSEPOS))
        else:
            n2b = False

        # ---- N3 acceptance (FORWARD: bars i+1,i+2) ----
        n3 = False
        for j in (i + 1, i + 2):
            if j < n and l[j] > l[i]:
                n3 = True; break

        # ---- N4 origin location (at break bar, trailing 365d) ----
        n4 = False
        if bidx is not None:
            t0 = ts[bidx] - np.timedelta64(N4_LOOKBACK_DAYS, "D")
            w = (ts <= ts[bidx]) & (ts >= t0)
            if w.sum() >= 2:
                hi365 = np.nanmax(h[w]); lo365 = np.nanmin(l[w])
                if hi365 > lo365:
                    pos = (bl - lo365) / (hi365 - lo365)
                    n4 = bool(pos <= N4_DISCOUNT)

        # ---- N5 chop-count (trailing 90d, causal: resolution<=i) ----
        if len(fail_bidx):
            t0 = ts[i] - np.timedelta64(N5_WINDOW_DAYS, "D")
            in_win = (ts[fail_bidx] >= t0) & (ts[fail_bidx] < ts[i]) & (fail_res <= i)
            n5_count = int(in_win.sum())
        else:
            n5_count = 0
        n5 = bool(n5_count >= N5_THRESH)

        # ---- N6 climax ATR (at break bar) ----
        n6 = False
        if bidx is not None and np.isfinite(atr[bidx]):
            m0 = max(0, bidx - N6_MED_BARS)
            med = np.nanmedian(atr[m0:bidx]) if bidx > m0 else np.nan
            if np.isfinite(med) and med > 0:
                n6 = bool(atr[bidx] > N6_ATR_MULT * med)

        score = int(n1) + int(n2a) + int(n2b) + int(n4)
        rec.update(bars_to_retest=(int(bars_to_retest) if bars_to_retest is not None else None),
                   n1=n1, n2a=n2a, n2b=n2b, n3=n3, n4=n4,
                   n5_count=n5_count, n5=n5, n6=n6, score=score)
        records.append(rec)

    return records, st, len(breaks)


# =========================================================================== stats
def pf_of(Rs):
    Rs = np.asarray(Rs, float)
    w = Rs[Rs > 0].sum(); ll = -Rs[Rs < 0].sum()
    return (w / ll) if ll > 1e-9 else (float("inf") if w > 0 else 0.0)


def maxdd_ordered(recs):
    """MaxDD% on fixed-risk pnl ordered by entry_time (proxy; one-position frame
    is not modeled cross-asset, so this is an equal-weight pooled-curve proxy)."""
    order = sorted(range(len(recs)), key=lambda k: recs[k]["entry_time"])
    pnls = np.array([recs[k]["R"] * FIXED_RISK for k in order])
    eq = INITIAL_CASH + np.cumsum(pnls)
    eq = np.concatenate([[INITIAL_CASH], eq])
    peak = np.maximum.accumulate(eq)
    return float(((eq - peak) / peak).min() * 100)


def paired_dR_ci(R, flag, delta, B=BOOT, seed=SEED):
    R = np.asarray(R, float); flag = np.asarray(flag, bool)
    dR = np.where(flag, delta * R, 0.0)
    rng = np.random.default_rng(seed)
    nn = len(R)
    idxb = rng.integers(0, nn, size=(B, nn))
    means = dR[idxb].mean(1)
    return dict(mean_dR=float(dR.mean()), ci_lo=float(np.percentile(means, 2.5)),
                ci_hi=float(np.percentile(means, 97.5)), total_dPnL=float(dR.sum() * FIXED_RISK))


def removal_delta_ci(R, flag, B=BOOT, seed=SEED):
    """Veto removal test: delta = meanR(kept=~flag) - meanR(all). Bootstrap CI.
    CI-lo>0 => removing flagged reliably improves the book's mean R."""
    R = np.asarray(R, float); flag = np.asarray(flag, bool)
    keep = ~flag
    if keep.sum() == 0 or flag.sum() == 0:
        return dict(delta=float("nan"), ci_lo=float("nan"), ci_hi=float("nan"),
                    meanR_all=float(R.mean()), meanR_kept=float("nan"))
    rng = np.random.default_rng(seed)
    nn = len(R); idxb = rng.integers(0, nn, size=(B, nn))
    deltas = np.empty(B)
    for k in range(B):
        rs = R[idxb[k]]; fs = flag[idxb[k]]
        ks = ~fs
        m_all = rs.mean()
        m_keep = rs[ks].mean() if ks.sum() else m_all
        deltas[k] = m_keep - m_all
    return dict(delta=float(R[keep].mean() - R.mean()),
                ci_lo=float(np.percentile(deltas, 2.5)), ci_hi=float(np.percentile(deltas, 97.5)),
                meanR_all=float(R.mean()), meanR_kept=float(R[keep].mean()))


def decompose(R, flag):
    R = np.asarray(R, float); flag = np.asarray(flag, bool)
    fl = R[flag]; un = R[~flag]
    return dict(n=len(R), n_flag=int(flag.sum()), cover=float(flag.mean()),
                mean_flag=float(fl.mean()) if len(fl) else float("nan"),
                mean_unflag=float(un.mean()) if len(un) else float("nan"),
                wr_flag=float((fl > 0).mean()) if len(fl) else float("nan"),
                wr_unflag=float((un > 0).mean()) if len(un) else float("nan"),
                pf_flag=pf_of(fl) if len(fl) else float("nan"),
                pf_unflag=pf_of(un) if len(un) else float("nan"))


# =========================================================================== universes
def breadth300_universe():
    files = sorted(f[:-8] for f in os.listdir(B3) if f.endswith(".parquet"))
    RAW = json.load(open(os.path.join(SCROOT, "sp500_raw.json")))
    sectors = RAW["sectors"]
    universe = []
    for key in files:
        try:
            dfp = pd.read_parquet(os.path.join(B3, f"{key}.parquet"))
        except Exception:
            continue
        ts = pd.DatetimeIndex(pd.to_datetime(dfp["ts"]))
        span = (ts.max() - ts.min()).days / 365.25
        gaps = int((ts.to_series().diff().dt.days.dropna() > 7).sum())
        nan_ohlc = int(dfp[["open", "high", "low", "close"]].isna().any(axis=1).sum())
        if span < MIN_YEARS or gaps > 3 or nan_ohlc > 0:
            continue
        universe.append((key, os.path.join(B3, f"{key}.parquet"), sectors.get(key, "?")))
    return universe


def wide_universe():
    U = []
    for s in ["BTC-USD", "ETH-USD", "SOL-USD", "LTC-USD", "XRP-USD", "ADA-USD",
              "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD"]:
        U.append((s, os.path.join(OS_, f"{s}.parquet"), "crypto"))
    U.append(("GOLD", os.path.join(XA, "GOLD_1D.parquet"), "metal"))
    U.append(("NDX", os.path.join(XA, "NDX_1D.parquet"), "equity-index"))
    NEW = {"SPX": "equity-index", "DJI": "equity-index", "RUT": "equity-index",
           "N225": "equity-index", "DAX": "equity-index", "FTSE": "equity-index",
           "SILVER": "metal", "COPPER": "metal", "PLATINUM": "metal",
           "OIL": "energy", "EURUSD": "fx", "GBPUSD": "fx", "USDJPY": "fx",
           "AUDUSD": "fx", "AAPL": "single-stock", "MSFT": "single-stock", "NVDA": "single-stock"}
    for k, fam in NEW.items():
        U.append((k, os.path.join(WB, f"{k}.parquet"), fam))
    return U


# =========================================================================== report
SPECS_SIZING = [("N1 freshness<=10", "n1"), ("N2a sweep-retest", "n2a"),
                ("N2b candle anatomy", "n2b"), ("N4 origin discount", "n4")]
SPECS_VETO = [("N5 chop-count>=3", "n5"), ("N6 climax-ATR", "n6")]


def spec_table(title, recs):
    R = np.array([r["R"] for r in recs], float)
    print(f"\n{'='*100}\n{title}   (n={len(recs)}, pooled PF {pf_of(R):.2f}, meanR {R.mean():+.3f})\n{'='*100}")
    print(f"{'spec':<22}{'cover':>7}{'meanR_flag':>11}{'meanR_unfl':>11}{'PF_flag':>9}"
          f"{'PF_unfl':>9}{'meanDR':>9}{'DR_CIlo':>9}{'DR_CIhi':>9}")
    out = {}
    for label, kk in SPECS_SIZING:
        flag = np.array([r[kk] for r in recs], bool)
        dec = decompose(R, flag); ci = paired_dR_ci(R, flag, MULT - 1.0)
        out[kk] = dict(dec=dec, ci=ci)
        print(f"{label:<22}{dec['cover']:>6.1%}{dec['mean_flag']:>11.3f}{dec['mean_unflag']:>11.3f}"
              f"{dec['pf_flag']:>9.2f}{dec['pf_unflag']:>9.2f}{ci['mean_dR']:>9.4f}"
              f"{ci['ci_lo']:>9.4f}{ci['ci_hi']:>9.4f}")
    # vetoes
    print(f"\n{'veto':<22}{'cover':>7}{'meanR_flag':>11}{'meanR_kept':>11}{'PF_flag':>9}"
          f"{'rmDelta':>9}{'rm_CIlo':>9}{'rm_CIhi':>9}")
    for label, kk in SPECS_VETO:
        flag = np.array([r[kk] for r in recs], bool)
        dec = decompose(R, flag); rm = removal_delta_ci(R, flag)
        inv = paired_dR_ci(R, flag, INV_MULT - 1.0)   # inverse sizing (N5)
        out[kk] = dict(dec=dec, rm=rm, inv=inv)
        print(f"{label:<22}{dec['cover']:>6.1%}{dec['mean_flag']:>11.3f}{rm['meanR_kept']:>11.3f}"
              f"{dec['pf_flag']:>9.2f}{rm['delta']:>9.4f}{rm['ci_lo']:>9.4f}{rm['ci_hi']:>9.4f}")
    return out


def combined_score(title, recs):
    R = np.array([r["R"] for r in recs], float)
    sc = np.array([r["score"] for r in recs], int)
    print(f"\n{'-'*70}\nCOMBINED CONFLUENCE SCORE (sum of N1,N2a,N2b,N4)  |  {title}\n{'-'*70}")
    print(f"{'score':>6}{'n':>7}{'cover%':>8}{'meanR':>9}{'PF':>8}{'WR':>7}")
    prev = None; monotonic = True; rows = []
    for s in range(5):
        m = sc == s
        if m.sum() == 0:
            print(f"{s:>6}{0:>7}{0:>8.1f}{'--':>9}{'--':>8}{'--':>7}"); rows.append(None); continue
        mr = R[m].mean()
        print(f"{s:>6}{m.sum():>7}{100*m.mean():>8.1f}{mr:>9.3f}{pf_of(R[m]):>8.2f}{(R[m]>0).mean():>7.1%}")
        rows.append(mr)
        if prev is not None and mr < prev - 1e-9:
            monotonic = False
        prev = mr
    present = [r for r in rows if r is not None]
    mono = all(present[k] <= present[k+1] + 1e-9 for k in range(len(present)-1))
    # sized portfolio: 1.0 base, 1.25 at >=2, 1.5 at >=3
    mult = np.where(sc >= 3, 1.5, np.where(sc >= 2, 1.25, 1.0))
    dR = (mult - 1.0) * R
    rng = np.random.default_rng(SEED); nn = len(R)
    idxb = rng.integers(0, nn, size=(BOOT, nn)); means = dR[idxb].mean(1)
    ci_lo = float(np.percentile(means, 2.5)); ci_hi = float(np.percentile(means, 97.5))
    print(f"  MONOTONIC in score: {'YES' if mono else 'NO'}")
    print(f"  sized(1/1.25/1.5) vs flat: meanΔR {dR.mean():+.4f}  CI[{ci_lo:+.4f},{ci_hi:+.4f}]  "
          f"ΔPnL ${dR.sum()*FIXED_RISK:,.0f}  -> {'beats flat' if ci_lo>0 else 'no'}")
    return dict(monotonic=bool(mono), rows=[None if r is None else float(r) for r in rows],
                sized_ci_lo=ci_lo, sized_ci_hi=ci_hi, sized_dPnL=float(dR.sum()*FIXED_RISK))


def freshness_diagnostic(recs):
    vals = np.array([r["bars_to_retest"] for r in recs if r["bars_to_retest"] is not None])
    R = np.array([r["R"] for r in recs if r["bars_to_retest"] is not None])
    print(f"\n{'-'*70}\nN1 FRESHNESS DISTRIBUTION DIAGNOSTIC (bars break->retest)\n{'-'*70}")
    if len(vals) == 0:
        print("  no break-based fires"); return {}
    qs = [0, 10, 25, 50, 75, 90, 100]
    pcts = {q: float(np.percentile(vals, q)) for q in qs}
    print("  percentiles: " + "  ".join(f"p{q}={pcts[q]:.0f}" for q in qs))
    print(f"  mean {vals.mean():.1f}  max {vals.max()}   (door validity window = {M2_SOS_WIN} bars)")
    bins = [(0, 10), (11, 30), (31, 90), (91, 180), (181, 360), (361, 100000)]
    print(f"\n  {'staleness bucket':<18}{'n':>6}{'cover%':>8}{'meanR':>9}{'PF':>8}{'WR':>7}")
    buck = {}
    for a, b in bins:
        m = (vals >= a) & (vals <= b)
        lab = f"{a}-{b if b<100000 else 'max'}"
        if m.sum() == 0:
            print(f"  {lab:<18}{0:>6}"); buck[lab] = None; continue
        print(f"  {lab:<18}{m.sum():>6}{100*m.mean():>8.1f}{R[m].mean():>9.3f}{pf_of(R[m]):>8.2f}{(R[m]>0).mean():>7.1%}")
        buck[lab] = dict(n=int(m.sum()), meanR=float(R[m].mean()), pf=pf_of(R[m]))
    return dict(pcts=pcts, buckets=buck, mean=float(vals.mean()), max=int(vals.max()))


def n5_bearwindow(recs):
    print(f"\n{'-'*70}\nN5 BEAR/CHOP-WINDOW EFFECT (chop-flag coverage + flagged vs unflagged meanR)\n{'-'*70}")
    R = np.array([r["R"] for r in recs], float)
    et = pd.DatetimeIndex([r["entry_time"] for r in recs])
    n5 = np.array([r["n5"] for r in recs], bool)
    print(f"  {'window':<10}{'fires':>7}{'chop%':>8}{'meanR_all':>10}{'meanR_chop':>11}{'meanR_calm':>11}{'PF_all':>8}")
    res = {}
    for name, a, b in EQUITY_BEARS:
        m = (et >= pd.Timestamp(a)) & (et <= pd.Timestamp(b))
        if m.sum() == 0:
            print(f"  {name:<10}{0:>7}"); continue
        Rw = R[m]; fw = n5[m]
        mr_c = Rw[fw].mean() if fw.sum() else float("nan")
        mr_k = Rw[~fw].mean() if (~fw).sum() else float("nan")
        print(f"  {name:<10}{m.sum():>7}{100*fw.mean():>8.1f}{Rw.mean():>10.3f}{mr_c:>11.3f}{mr_k:>11.3f}{pf_of(Rw):>8.2f}")
        res[name] = dict(fires=int(m.sum()), chop_cover=float(fw.mean()),
                         meanR_all=float(Rw.mean()), pf_all=pf_of(Rw))
    return res


def subset(recs, recent_only=False, fams=None):
    out = recs
    if recent_only:
        out = [r for r in out if pd.Timestamp(r["entry_time"]) >= RECENT]
    if fams is not None:
        out = [r for r in out if r["fam"] in fams]
    return out


def main():
    print("=" * 100)
    print("WI BATCH-7 CONFLUENCE TEST (add.65) — six confluence specs as OVERLAYS on the FROZEN door")
    print("PRIMARY: 476 S&P-500 names recent 2018-26 | SECONDARY: full-history stocks + 25-mkt wide basket")
    print("=" * 100)

    # ---------------- PRIMARY + full-history: breadth300 ----------------
    uni = breadth300_universe()
    print(f"\nBREADTH-300 universe: {len(uni)} names pass screens (>= {MIN_YEARS}yr + integrity)")
    random.seed(7)
    sample = set(k for k, _, _ in random.sample(uni, min(10, len(uni))))
    print(f"REFEREE PARITY self-test sample (10): {sorted(sample)}")

    stock_recs = []; st_results = {}; nbreaks_total = 0
    for j, (key, path, sec) in enumerate(uni):
        try:
            recs, st, nbk = analyze_asset(key, path, sec, do_selftest=(key in sample))
        except Exception as ex:
            print(f"  ERR {key}: {ex}"); continue
        stock_recs.extend(recs); nbreaks_total += nbk
        if st is not None:
            st_results[key] = st
        if (j + 1) % 50 == 0:
            print(f"  ...{j+1}/{len(uni)} assets, {len(stock_recs)} fires", flush=True)
    st_fail = [k for k, v in st_results.items() if v is False]
    print(f"\nREFEREE PARITY: {len(st_results)} tested -> "
          f"{'ALL 0.00% PASS' if not st_fail else 'FAIL: '+','.join(st_fail)}")
    print(f"total confirmed-break events across universe: {nbreaks_total}")

    full = stock_recs
    primary = subset(stock_recs, recent_only=True)
    print(f"\nfires: full-history {len(full)}  |  recent-2018+ (PRIMARY) {len(primary)}")

    # ---------------- SECONDARY: wide basket ----------------
    wu = wide_universe()
    wide_recs = []
    for key, path, fam in wu:
        if not os.path.exists(path):
            print(f"  (missing {key} {path})"); continue
        try:
            recs, _, _ = analyze_asset(key, path, fam, do_selftest=False)
        except Exception as ex:
            print(f"  ERR wide {key}: {ex}"); continue
        wide_recs.extend(recs)
    trending = subset(wide_recs, fams={"crypto", "equity-index", "metal", "single-stock"})
    fxenergy = subset(wide_recs, fams={"fx", "energy"})
    print(f"\nWIDE BASKET fires: all {len(wide_recs)}  trending(pass) {len(trending)}  fx+energy {len(fxenergy)}")

    # =============== spec tables ===============
    prim = spec_table("PRIMARY — 476 stocks, recent 2018-2026", primary)
    fullo = spec_table("SECONDARY(a) — 476 stocks, FULL history", full)
    wideo = spec_table("SECONDARY(b) — wide basket, TRENDING (pass) families", trending)
    if fxenergy:
        _ = spec_table("SHADOW — wide basket fx+energy (domain boundary; not a pass unit)", fxenergy)

    # =============== combined score ===============
    cs_prim = combined_score("PRIMARY", primary)
    cs_full = combined_score("SECONDARY(a) full-history", full)
    cs_wide = combined_score("SECONDARY(b) wide trending", trending)

    # =============== freshness diagnostic ===============
    fr = freshness_diagnostic(primary)

    # =============== N5 bear/chop-window ===============
    n5bw = n5_bearwindow(full)   # full-history stock pool = has all bear windows

    # =============== consistency verdict table ===============
    print("\n" + "=" * 100)
    print("CROSS-SET CONSISTENCY & PRE-REGISTERED VERDICTS")
    print("=" * 100)
    print(f"{'spec':<22}{'P_cover':>8}{'P_dRlo':>9}{'P_C2':>6}{'S_full_sign':>12}{'S_wide_sign':>12}{'verdict':>10}")
    verdicts = {}
    for label, kk in SPECS_SIZING:
        p = prim[kk]; f = fullo[kk]; w = wideo[kk]
        cover_ok = 0.10 <= p["dec"]["cover"] <= 0.90
        c1 = p["ci"]["ci_lo"] > 0
        c2 = p["dec"]["mean_flag"] >= p["dec"]["mean_unflag"]
        s_full = f["ci"]["mean_dR"] > 0
        s_wide = w["ci"]["mean_dR"] > 0
        adopt = c1 and cover_ok and s_full and s_wide
        v = ("ADOPT*" if (adopt and c2) else ("FALSE-PASS" if adopt and not c2 else "CLOSE"))
        verdicts[kk] = dict(cover=p["dec"]["cover"], p_dRlo=p["ci"]["ci_lo"], c1=c1, c2=c2,
                            s_full=s_full, s_wide=s_wide, verdict=v)
        print(f"{label:<22}{p['dec']['cover']:>8.1%}{p['ci']['ci_lo']:>9.4f}{str(c2):>6}"
              f"{('+' if s_full else '-'):>12}{('+' if s_wide else '-'):>12}{v:>10}")
    for label, kk in SPECS_VETO:
        p = prim[kk]; f = fullo[kk]; w = wideo[kk]
        cover_ok = 0.10 <= p["dec"]["cover"] <= 0.90
        rmlo = p["rm"]["ci_lo"]
        s_full = f["rm"]["delta"] > 0
        s_wide = w["rm"]["delta"] > 0
        adopt = (rmlo > 0) and cover_ok and s_full and s_wide
        v = "ADOPT-veto*" if adopt else "CLOSE"
        verdicts[kk] = dict(cover=p["dec"]["cover"], rm_lo=rmlo, s_full=s_full,
                            s_wide=s_wide, verdict=v)
        print(f"{label:<22}{p['dec']['cover']:>8.1%}{rmlo:>9.4f}{'--':>6}"
              f"{('+' if s_full else '-'):>12}{('+' if s_wide else '-'):>12}{v:>10}")

    # =============== N3 entry-shift variant ===============
    print("\n" + "-" * 70)
    print("N3 ACCEPTANCE — descriptive flag (forward info; not a sizing overlay)")
    print("-" * 70)
    for tag, rr in [("PRIMARY", primary), ("full", full), ("wide-trending", trending)]:
        R = np.array([r["R"] for r in rr], float); f3 = np.array([r["n3"] for r in rr], bool)
        d = decompose(R, f3)
        print(f"  {tag:<14} cover {d['cover']:.1%}  meanR_flag {d['mean_flag']:+.3f}  "
              f"meanR_unflag {d['mean_unflag']:+.3f}  (entry-shift variant deferred: requires re-sim "
              f"from i+2 -> NOT cheap on the frozen backtester; flag reported only)")

    # =============== dump ===============
    dump = dict(
        n_primary=len(primary), n_full=len(full), n_wide_trending=len(trending),
        n_fxenergy=len(fxenergy), referee_fail=st_fail,
        primary_specs={k: {"cover": v["dec"]["cover"], "meanR_flag": v["dec"]["mean_flag"],
                           "meanR_unflag": v["dec"]["mean_unflag"],
                           "dR_ci_lo": v.get("ci", {}).get("ci_lo"),
                           "rm_ci_lo": v.get("rm", {}).get("ci_lo")}
                       for k, v in prim.items()},
        verdicts=verdicts,
        combined={"primary": cs_prim, "full": cs_full, "wide": cs_wide},
        freshness=fr, n5_bearwindow=n5bw,
    )
    outp = os.path.join(B3, "wi_batch7_results.json")
    with open(outp, "w") as fh:
        json.dump(dump, fh, indent=2, default=str)
    print(f"\nresults dumped -> {outp}")


if __name__ == "__main__":
    main()
