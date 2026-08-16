"""
THE TIGHTENED-ENTRY TEST  (wyckoff_audit add.67; STUDY ONLY; nothing ships)
==========================================================================
Definitive Layer-3/7 fidelity experiment. Does WI's STRICT entry
("LPS + Bojan. Market structure break on LPS. That is the trade.") beat our LOOSE
door entry ("any daily close that holds above the broken level after dipping into
the retest zone") — per-trade AND as a book?

Spec is FROZEN in idea_lab/tightened_entry_PREREGISTRATION.txt. This harness runs
the four pre-registered strict variants (V0={S1+S3}, Va={S1+S2a+S3}, Vb={S1+S2b+S3},
Vc={S1+S2c+S3}) vs the loose door baseline, with:
  * fire-set census (shrinkage + MSS-search termination reasons)
  * per-trade meanR + bootstrap CI, WR, PF   (isolated single-position frame)
  * BOOK comparison (total R, PnL at 1% risk)
  * PAIRED per-setup ΔR + bootstrap CI (same setup, two entries) — isolates entry
    quality from setup selection
  * skipped-winners opportunity cost + entry-delay distribution
  * S2c inverse (unfinished-low-below) diagnostic + weekly-upper-wick magnet split
  * referee parity 0.00% vs the independent walker (run_backtest one-shot at MSS)

Reuses the FROZEN add.48 door + the add.66 parity-clean single-trade simulator.
Production untouched. No door param/logic change.
"""
from __future__ import annotations
import os, sys, json, warnings
from collections import Counter, defaultdict

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, ".."))

from xasset_spx_port import load_spx
from trend_continuation_door import build_daily_sensors, TrendContinuationDoor
from backtester import (run_backtest, EntryPlan, _slip_fill, INITIAL_CASH,
                        RISK_PCT, COMMISSION_RATE)
from unified_archetype_v2 import (RTZ_ATR, M2_SOS_WIN, LPS_LOOKBACK, STOP_BUF_ATR,
                                  MAX_HOLD, MIN_TP1_R)
from engine.features.eye_state import CONFIRMED_BREAK
from run_wi_batch7 import (breadth300_universe, wide_universe, RECENT, pf_of,
                           N2B_WICK, N2B_CLOSEPOS, SCROOT)
from run_near_miss import sim_one
from run_xasset_spx import selftest_on

FIXED_RISK = RISK_PCT * INITIAL_CASH          # $1000 / trade
BOOT = 10_000
SEED = 20260867

# ---- pre-registered strict constants (all OURS; see PREREGISTRATION) ----
MSS_CAP = 60                 # bars after LPS to find the MSS (else strict skips)
S2C_OPEN_WICK = 0.10         # S2c(ii): open-low >= 0.10*range = a real lower wick
S2C_DAYRANK = 3              # S2c(i): weekly low printed on day-rank >= 3 = finished


# =========================================================================== helpers
def boot_ci_mean(x, B=BOOT, seed=SEED):
    x = np.asarray(x, float)
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idxb = rng.integers(0, len(x), size=(B, len(x)))
    m = x[idxb].mean(1)
    return float(x.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def boot_ci_paired(dR, B=BOOT, seed=SEED):
    """Bootstrap CI on the mean of a paired-difference vector."""
    dR = np.asarray(dR, float)
    if len(dR) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idxb = rng.integers(0, len(dR), size=(B, len(dR)))
    m = dR[idxb].mean(1)
    return float(dR.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def book_stats(Rs):
    Rs = np.asarray(Rs, float)
    if len(Rs) == 0:
        return dict(n=0, totR=0.0, pnl=0.0, pf=float("nan"), wr=float("nan"), meanR=float("nan"))
    return dict(n=len(Rs), totR=float(Rs.sum()), pnl=float(Rs.sum() * FIXED_RISK),
                pf=pf_of(Rs), wr=float((Rs > 0).mean()), meanR=float(Rs.mean()))


def strict_geo(low_lps, atr_mss, entry_raw, rhigh, rlow, swh):
    """WI strict-entry plan geometry at the MSS bar: stop under the LPS low, same
    TP1/BE/runner formula as the door (evaluated at the shifted entry)."""
    stop = low_lps - STOP_BUF_ATR * atr_mss
    R = entry_raw - stop
    if not np.isfinite(R) or R <= 0:
        return None
    tp1 = None
    for cand in (rhigh, swh):
        if np.isfinite(cand) and cand >= entry_raw + MIN_TP1_R * R:
            tp1 = cand; break
    if tp1 is None:
        tp1 = entry_raw + 1.0 * R
    measured = rhigh + (rhigh - rlow) if (np.isfinite(rhigh) and np.isfinite(rlow) and rhigh > rlow) else -np.inf
    tt = max(entry_raw + 2.0 * R, measured)
    return dict(stop=float(stop), tp1=float(tp1), runner=float(tt), be=float(entry_raw))


def independent_R(df, mss_idx, geo):
    """INDEPENDENT reference walker: drive run_backtest with a one-shot EntryPlan that
    fires ONLY at the MSS bar. Structurally separate from sim_one -> referee parity."""
    def _fn(_df, k, _mss=mss_idx, _g=geo):
        if k != _mss:
            return None
        return EntryPlan(direction="long", stop=_g["stop"], targets=[(_g["tp1"], 0.40)],
                         move_stop_to_after_first_tp=_g["be"], runner_target=_g["runner"],
                         max_hold_bars=MAX_HOLD)
    res = run_backtest(df, _fn, label="strict_one_shot")
    return res["trades"][0]["R"] if res["trades"] else None


# =========================================================================== per-asset
def analyze_asset(key, path, family, parity_budget=0):
    """Run the frozen door -> loose fires; for each, construct the WI strict entry
    (S1, S2a/b/c, S3) and re-simulate exits from the shifted MSS bar. Returns a list
    of per-setup records + parity results."""
    df = load_spx(path)
    dfs, sr, bj, eye = build_daily_sensors(df)

    strat = TrendContinuationDoor(dfs, sr, bj, eye, variant="struct", conviction=False)
    res = run_backtest(dfs, strat, label=key)
    trades = res["trades"]; elog = strat.entries_log
    assert len(trades) == len(elog), f"{key}: trade/log mismatch"

    o = dfs["open"].to_numpy(float); h = dfs["high"].to_numpy(float)
    l = dfs["low"].to_numpy(float); c = dfs["close"].to_numpy(float)
    atr = dfs["atr_14"].to_numpy(float)
    rhigh = sr["struct_range_high"].to_numpy(float)
    rlow = sr["struct_range_low"].to_numpy(float)
    swh = dfs["swing_high_50"].to_numpy(float)
    es = eye["eye_state"].to_numpy(object); ed = eye["eye_dir"].to_numpy(object)
    ru = eye["range_upper_1d"].to_numpy(float)
    bl_act = bj["bojan_low_active"].to_numpy(np.int8)
    low_zones = bj.attrs.get("low_zones", [])          # full roster (formed_i / invalidated_i)
    idx = dfs.index
    n = len(dfs)
    # weekly bin id per daily bar (freq='W'); monotonic non-decreasing -> "closed" test
    week_id = pd.PeriodIndex(idx, freq="W").astype("int64").to_numpy()

    recs = []
    parity = []   # (sim_R, indep_R)
    for t, e in zip(trades, elog):
        i = idx.get_loc(t["entry_time"])
        if isinstance(i, slice):
            i = i.start
        bl = e.get("break_level", np.nan)
        if not np.isfinite(bl):
            continue                                   # M1 fire (no break) -> door is M2-only, ~never
        loose_R = float(t["R"])

        # --- recompute break bar (exactly as the door) ---
        lo = max(0, i - M2_SOS_WIN); bidx = None
        for cc in range(i - 1, lo - 1, -1):
            if es[cc] == CONFIRMED_BREAK and ed[cc] == "bull":
                bidx = cc; break
        if bidx is None:
            continue

        # --- follow-through peak + LPS (S1) ---
        fp = bidx + int(np.argmax(h[bidx:i + 1]))       # follow-through peak in [break..fire]
        lps = fp + int(np.argmin(l[fp:i + 1]))          # pullback low after the peak
        atr_lps = atr[lps]
        s1 = bool(np.isfinite(atr_lps) and l[lps] <= bl + RTZ_ATR * atr_lps)

        # --- minor swing high of the pullback leg (for S3) ---
        if lps - fp >= 2:                               # >=1 bar strictly between fp and lps
            msh = float(np.nanmax(h[fp + 1:lps]))
            mono = False
        else:                                           # monotonic pullback -> fallback last 5 bars
            a0 = max(0, lps - 5)
            msh = float(np.nanmax(h[a0:lps])) if lps > a0 else float(h[lps])
            mono = True

        # --- S3: MSS search forward from lps+1 (causal; invalidation + cap) ---
        mss = None; term = "cap"
        jend = min(lps + 1 + MSS_CAP, n)
        for j in range(lps + 1, jend):
            if c[j] < l[lps]:                           # LPS broke (body close) -> no MSS
                term = "lps_broke"; break
            if c[j] > msh:
                mss = j; term = "mss"; break
        else:
            term = "cap"

        # --- S2a one-candle anatomy at LPS or LPS-1 ---
        def _anat(b):
            rng = h[b] - l[b]
            if rng <= 0:
                return False
            lw = min(o[b], c[b]) - l[b]
            cp = (c[b] - l[b]) / rng
            return bool((lw / rng >= N2B_WICK) and (cp >= N2B_CLOSEPOS))
        s2a = bool(_anat(lps) or (lps - 1 >= 0 and _anat(lps - 1)))

        # --- S2b persistent-zone overlap at LPS (roster; any active low zone) ---
        band_lo = bl - RTZ_ATR * atr_lps
        band_hi = bl + RTZ_ATR * atr_lps
        s2b = False
        if bl_act[lps] == 1:
            for z in low_zones:
                fi = z["formed_i"]; iv = z["invalidated_i"]
                active = (fi is not None and fi <= lps) and (iv is None or iv > lps)
                if active and (z["bottom"] <= band_hi and z["top"] >= band_lo):
                    s2b = True; break

        # --- S2c finished-low (closed-bar; needs the MSS decision bar for closedness) ---
        s2c_i = s2c_ii = s2c = False; day_rank = None
        rng_lps = h[lps] - l[lps]
        s2c_ii = bool(rng_lps > 0 and (o[lps] - l[lps]) >= S2C_OPEN_WICK * rng_lps)
        unfinished_wick = bool(rng_lps > 0 and (o[lps] - l[lps]) < S2C_OPEN_WICK * rng_lps)
        early_week = None
        if mss is not None:
            wk_lps = week_id[lps]
            closed = week_id[mss] > wk_lps               # LPS week closed as of decision bar
            if closed:
                # trading bars of the LPS week (all <= mss since closed)
                wk_bars = [k for k in range(max(0, lps - 10), mss + 1) if week_id[k] == wk_lps]
                if wk_bars:
                    low_bar = min(wk_bars, key=lambda k: l[k])
                    day_rank = wk_bars.index(low_bar) + 1   # 1-indexed within the week
                    s2c_i = bool(day_rank >= S2C_DAYRANK)
                    early_week = bool(day_rank <= 2)
        s2c = bool(s2c_i or s2c_ii)
        unfinished_below = bool((early_week is True) or unfinished_wick)   # inverse diagnostic

        # --- strict re-sim from the MSS bar ---
        strict_R = None; delay = None; price_delta = None; geo = None
        if mss is not None and s1:
            entry_raw = c[mss]
            geo = strict_geo(l[lps], atr[mss], entry_raw, rhigh[mss], rlow[mss], swh[mss])
            if geo is not None:
                strict_R = sim_one(h, l, c, mss, geo["stop"], geo["tp1"], geo["runner"],
                                   geo["be"], MAX_HOLD)
                delay = int(mss - i)
                price_delta = float((c[mss] - c[i]) / c[i])   # >0 = entering higher (worse fill)
                # referee parity (independent walker) on a per-asset budget
                if strict_R is not None and len(parity) < parity_budget:
                    iR = independent_R(dfs, mss, geo)
                    if iR is not None:
                        parity.append((float(strict_R), float(iR)))

        # --- weekly-upper-wick magnet overhead (diagnostic; unfilled HTF supply in TT path) ---
        magnet = None
        if mss is not None and geo is not None:
            entry_px = c[mss]; tgt = geo["runner"]
            # scan completed weekly-anchored swing highs (swh reused as HTF supply proxy) between
            # entry and target that price has NOT yet traded through since formation
            m = False
            hh = swh[mss]
            if np.isfinite(hh) and entry_px < hh < tgt:
                m = True
            magnet = bool(m)

        recs.append(dict(
            key=key, fam=family, entry_time=str(idx[i]), door_idx=int(i), break_idx=int(bidx),
            lps_idx=int(lps), fp_idx=int(fp), monotonic=bool(mono), mss_term=term,
            mss_idx=(int(mss) if mss is not None else None),
            loose_R=loose_R, strict_R=(float(strict_R) if strict_R is not None else None),
            delay=delay, price_delta=price_delta,
            s1=s1, s2a=s2a, s2b=s2b, s2c=s2c, s2c_i=s2c_i, s2c_ii=s2c_ii, day_rank=day_rank,
            unfinished_below=unfinished_below, magnet=magnet,
            below_ema200=bool(e.get("below_ema200", False)),
        ))
    return recs, parity


# =========================================================================== reporting
VARIANTS = [("V0 {S1+S3}", None), ("Va {S1+S2a+S3}", "s2a"),
            ("Vb {S1+S2b+S3}", "s2b"), ("Vc {S1+S2c+S3}", "s2c")]


def strict_fires(recs, s2key):
    """Setups where the strict entry fires for a variant: S1 + (S2 flag) + MSS found."""
    out = []
    for r in recs:
        if not r["s1"]:
            continue
        if r["strict_R"] is None:      # no MSS / bad geo
            continue
        if s2key is not None and not r[s2key]:
            continue
        out.append(r)
    return out


def variant_block(title, recs):
    loose_R = np.array([r["loose_R"] for r in recs], float)
    loose = book_stats(loose_R)
    print(f"\n{'='*104}\n{title}   (loose door fires n={loose['n']}, "
          f"book PF {loose['pf']:.2f}, meanR {loose['meanR']:+.3f}, totR {loose['totR']:+.1f}, "
          f"PnL ${loose['pnl']:,.0f})\n{'='*104}")
    # census of MSS termination on S1-passing setups
    s1recs = [r for r in recs if r["s1"]]
    term = Counter(r["mss_term"] for r in s1recs)
    print(f"S1 coverage: {sum(r['s1'] for r in recs)}/{len(recs)} setups.  "
          f"MSS-search termination (S1 setups): {dict(term)}")
    print(f"\n{'variant':<16}{'n_strict':>9}{'shrink':>8}{'meanR':>9}{'CI-lo':>8}{'CI-hi':>8}"
          f"{'WR':>7}{'PF':>7}{'bookR':>9}{'book%':>8}{'PnL$':>11}"
          f"{'pairN':>7}{'pairΔR':>9}{'Δ CIlo':>8}{'Δ CIhi':>8}")
    out = {}
    for label, s2key in VARIANTS:
        sf = strict_fires(recs, s2key)
        sR = np.array([r["strict_R"] for r in sf], float)
        sb = book_stats(sR)
        m, lo, hi = boot_ci_mean(sR) if len(sR) else (float("nan"),)*3
        shrink = (sb["n"] / loose["n"]) if loose["n"] else float("nan")
        bookpct = (sb["totR"] / loose["totR"]) if abs(loose["totR"]) > 1e-9 else float("nan")
        # paired: setups where BOTH loose and strict fire (all strict setups are door fires)
        dR = np.array([r["strict_R"] - r["loose_R"] for r in sf], float)
        pm, plo, phi = boot_ci_paired(dR) if len(dR) else (float("nan"),)*3
        out[label] = dict(n=sb["n"], shrink=shrink, meanR=m, ci_lo=lo, ci_hi=hi, wr=sb["wr"],
                          pf=sb["pf"], bookR=sb["totR"], bookpct=bookpct, pnl=sb["pnl"],
                          pairN=len(dR), pairdR=pm, pair_lo=plo, pair_hi=phi)
        pf_s = f"{sb['pf']:.2f}" if np.isfinite(sb["pf"]) else "inf"
        print(f"{label:<16}{sb['n']:>9}{shrink:>8.2f}{m:>9.3f}{lo:>8.3f}{hi:>8.3f}"
              f"{sb['wr']:>7.1%}{pf_s:>7}{sb['totR']:>9.1f}{bookpct:>8.1%}{sb['pnl']:>11,.0f}"
              f"{len(dR):>7}{pm:>9.3f}{plo:>8.3f}{phi:>8.3f}")
    # skipped-winners opportunity cost (relative to V0 = the maximal strict set)
    v0 = set(id(r) for r in strict_fires(recs, None))
    skipped = [r for r in recs if id(r) not in v0]
    sk_R = np.array([r["loose_R"] for r in skipped], float)
    if len(sk_R):
        nw = int((sk_R > 0).sum())
        print(f"\nSKIPPED by V0 (no MSS / S1 fail): n={len(sk_R)}  winners={nw} "
              f"({nw/len(sk_R):.0%})  summed loose R of skipped={sk_R.sum():+.1f}  "
              f"(meanR skipped {sk_R.mean():+.3f})")
    return out, loose


def delay_report(recs):
    d = np.array([r["delay"] for r in recs if r["delay"] is not None], int)
    pd_ = np.array([r["price_delta"] for r in recs if r["price_delta"] is not None], float)
    if len(d) == 0:
        print("\n(no shifted entries)"); return {}
    print(f"\nENTRY-DELAY (MSS_bar - door_bar), bars: p10={np.percentile(d,10):.0f} "
          f"p50={np.percentile(d,50):.0f} p90={np.percentile(d,90):.0f} max={d.max()} "
          f"mean={d.mean():.1f}  |  neg/zero/pos delay = "
          f"{int((d<0).sum())}/{int((d==0).sum())}/{int((d>0).sum())}")
    print(f"ENTRY-PRICE shift (strict close vs door close): median {np.median(pd_)*100:+.2f}%  "
          f"worse-fill(higher) {int((pd_>0).sum())}/{len(pd_)} ({(pd_>0).mean():.0%})")
    return dict(delay_p50=float(np.percentile(d, 50)), delay_p90=float(np.percentile(d, 90)),
                delay_mean=float(d.mean()), price_med=float(np.median(pd_)))


def s2c_inverse(recs):
    """Inverse (unfinished-low-below) diagnostic on the LOOSE fires: does an
    unfinished LPS predict worse loose R / more stops? (Not an entry gate.)"""
    R = np.array([r["loose_R"] for r in recs], float)
    unf = np.array([bool(r["unfinished_below"]) for r in recs], bool)
    if unf.sum() == 0 or (~unf).sum() == 0:
        print("\nS2c INVERSE: degenerate coverage"); return {}
    mu_u = R[unf].mean(); mu_f = R[~unf].mean()
    print(f"\nS2c INVERSE (unfinished-low-below vs finished), LOOSE fires: "
          f"unfinished n={int(unf.sum())} meanR {mu_u:+.3f} PF {pf_of(R[unf]):.2f}  |  "
          f"finished n={int((~unf).sum())} meanR {mu_f:+.3f} PF {pf_of(R[~unf]):.2f}  "
          f"(Δ finished-unfinished {mu_f-mu_u:+.3f})")
    return dict(unf_meanR=float(mu_u), fin_meanR=float(mu_f), n_unf=int(unf.sum()))


def magnet_report(recs):
    sf = [r for r in recs if r["strict_R"] is not None and r["magnet"] is not None]
    if not sf:
        return
    R = np.array([r["strict_R"] for r in sf], float)
    mg = np.array([r["magnet"] for r in sf], bool)
    if mg.sum() == 0 or (~mg).sum() == 0:
        print("MAGNET (weekly upper-wick overhead): degenerate coverage"); return
    print(f"MAGNET overhead-supply in TT path (diagnostic): with n={int(mg.sum())} "
          f"meanR {R[mg].mean():+.3f}  |  without n={int((~mg).sum())} meanR {R[~mg].mean():+.3f}")


def subset(recs, recent=False, fams=None):
    out = recs
    if recent:
        out = [r for r in out if pd.Timestamp(r["entry_time"]) >= RECENT]
    if fams is not None:
        out = [r for r in out if r["fam"] in fams]
    return out


def main():
    print("=" * 104)
    print("THE TIGHTENED-ENTRY TEST (add.67) — WI's strict LPS+Bojan+MSS entry vs the loose door")
    print("PRIMARY: 476 S&P-500 recent 2018-26 | SECONDARY: full-history stocks + 25-mkt wide basket")
    print("=" * 104)

    # ---------------- PRIMARY + full-history: breadth300 ----------------
    uni = breadth300_universe()
    print(f"\nBREADTH-300 universe: {len(uni)} names pass screens")
    stock_recs = []; parity_all = []
    import random as _r; _r.seed(7)
    sample = set(k for k, _, _ in _r.sample(uni, min(10, len(uni))))
    for j, (key, path, sec) in enumerate(uni):
        try:
            recs, par = analyze_asset(key, path, sec, parity_budget=(20 if key in sample else 0))
        except Exception as ex:
            print(f"  ERR {key}: {ex}"); continue
        stock_recs.extend(recs); parity_all.extend(par)
        if (j + 1) % 100 == 0:
            print(f"  ...{j+1}/{len(uni)}  door fires {len(stock_recs)}", flush=True)

    # referee parity
    if parity_all:
        diffs = np.array([abs(a - b) for a, b in parity_all])
        nfail = int((diffs > 1e-9).sum())
        print(f"\nREFEREE PARITY (sim_one vs independent run_backtest walker): "
              f"{len(parity_all)} strict entries checked, {nfail} mismatches "
              f"(max |Δ| {diffs.max():.2e}) -> {'0.00% PASS' if nfail==0 else 'FAIL'}")

    full = stock_recs
    primary = subset(stock_recs, recent=True)
    print(f"\ndoor fires: full-history {len(full)}  |  recent-2018+ (PRIMARY) {len(primary)}")

    # ---------------- SECONDARY: wide basket ----------------
    wu = wide_universe(); wide_recs = []
    for key, path, fam in wu:
        if not os.path.exists(path):
            print(f"  (missing {key})"); continue
        try:
            recs, _ = analyze_asset(key, path, fam, parity_budget=0)
        except Exception as ex:
            print(f"  ERR wide {key}: {ex}"); continue
        wide_recs.extend(recs)
    trending = subset(wide_recs, fams={"crypto", "equity-index", "metal", "single-stock"})
    print(f"WIDE BASKET door fires: all {len(wide_recs)}  trending(pass) {len(trending)}")

    # =============== variant blocks ===============
    prim_out, prim_loose = variant_block("PRIMARY — 476 stocks, recent 2018-2026", primary)
    dpr = delay_report([r for r in primary if r["strict_R"] is not None])
    s2c_inverse(primary); magnet_report(primary)

    full_out, full_loose = variant_block("SECONDARY(a) — 476 stocks, FULL history", full)
    delay_report([r for r in full if r["strict_R"] is not None])

    wide_out, wide_loose = variant_block("SECONDARY(b) — wide basket, TRENDING families", trending)
    delay_report([r for r in trending if r["strict_R"] is not None])

    # =============== verdict vs pre-registered pass rule ===============
    print("\n" + "=" * 104)
    print("PRE-REGISTERED VERDICT  (VALIDATE iff paired ΔR CI-lo>0 on PRIMARY, sign holds on a")
    print("secondary, AND book R >= 90% of loose book R). Else the LOOSE door is VINDICATED.")
    print("=" * 104)
    print(f"{'variant':<16}{'P_pairΔR':>10}{'P_ΔCIlo':>9}{'P_book%':>9}{'full_ΔR':>9}"
          f"{'wide_ΔR':>9}{'VERDICT':>26}")
    verdicts = {}
    for label, _ in VARIANTS:
        p = prim_out[label]; f = full_out[label]; w = wide_out[label]
        ci_pos = np.isfinite(p["pair_lo"]) and p["pair_lo"] > 0
        book_ok = np.isfinite(p["bookpct"]) and p["bookpct"] >= 0.90
        sign_sec = ((np.isfinite(f["pairdR"]) and f["pairdR"] > 0) or
                    (np.isfinite(w["pairdR"]) and w["pairdR"] > 0))
        validate = bool(ci_pos and book_ok and sign_sec)
        reason = []
        if not ci_pos: reason.append("paired ΔR CI-lo<=0")
        if not book_ok: reason.append(f"book {p['bookpct']:.0%}<90%")
        if not sign_sec: reason.append("no secondary sign")
        v = "VALIDATE (prototype)" if validate else "REJECT: " + "; ".join(reason)
        verdicts[label] = dict(validate=validate, primary=p, full=f, wide=w)
        print(f"{label:<16}{p['pairdR']:>10.3f}{p['pair_lo']:>9.3f}{p['bookpct']:>9.1%}"
              f"{f['pairdR']:>9.3f}{w['pairdR']:>9.3f}{v:>26}")

    any_val = any(verdicts[l]["validate"] for l, _ in VARIANTS)
    print("\n" + ("-> AT LEAST ONE STRICT VARIANT VALIDATES — see block above (prototype-grade)."
                  if any_val else
                  "-> NO strict variant clears the rule. The LOOSE DOOR is VINDICATED: our looser\n"
                  "   translation is not a bug — it is an improvement on the teacher (Layer 3 closes)."))

    dump = dict(n_primary=len(primary), n_full=len(full), n_wide=len(trending),
                parity_checked=len(parity_all),
                primary=prim_out, full=full_out, wide=wide_out,
                loose={"primary": prim_loose, "full": full_loose, "wide": wide_loose},
                delay_primary=dpr, verdicts={k: v["validate"] for k, v in verdicts.items()},
                any_validate=any_val)
    outp = os.path.join(SCROOT, "breadth300", "tightened_entry_results.json")
    json.dump(dump, open(outp, "w"), indent=2, default=str)
    print(f"\nresults -> {outp}")


if __name__ == "__main__":
    main()
