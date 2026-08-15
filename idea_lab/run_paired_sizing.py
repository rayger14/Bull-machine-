"""
PAIRED SIZE-MULTIPLIER ΔR RE-TEST  (wyckoff_audit add.61)
=========================================================
Re-measure the THREE reopened sizing hypotheses (nulled by the in/out-split channel,
which add.60 proved APPARATUS-BLIND: MDE=∞ at n=106) with the audit-verified PAIRED
ΔR machinery (add.60 M1a: MDE 0.02R deterministic / 0.50R noisy). STUDY ONLY.

This is NOT fishing: add.60's explicit recommendation (#1) is to re-run any conviction-
sizing/time-window hypothesis previously judged by the in/out SPLIT as a PAIRED size-
multiplier ΔR test. Folds are spent for NEW round-3 tests; this is the corrected FINAL
read of these three, and it is final (no round 3).

THE THREE HYPOTHESES (ONLY these three):
  H1  GANN in-window conviction sizing — entries inside the add.59 MAJOR-anchor window
      (trailing-365d-extreme weekly-N5 pivots + halvings, 9 counts, ±3d) sized 1.25x.
  H2  ob_quality top-half sizing — top-half (by pooled-basket median) of a faithful
      daily-frame port of the 5-component HOB quality score, sized 1.25x.
  H3  eq_magnet sizing — eq_magnet-proximate entries (add.53 gem#3 faithful price port)
      sized 1.25x.
POSITIVE CONTROL: fib-time tier (the door's OWN fib_time_confluence flag, already sized
  ×1.25, validated separately). If the paired machinery cannot see the known-positive
  fib-time effect on THIS basket, the machinery is flagged.

PAIRED ΔR MECHANICS (deterministic given the flag):
  A 1.25x size multiplier on a fixed-1%-risk trade with realized R-multiple R contributes
  1.25·R·FIXED_RISK dollars instead of R·FIXED_RISK. Per-trade portfolio ΔR = 0.25·R for
  a FLAGGED trade, 0 otherwise (sizing does not change the realized R-multiple, only the
  dollar scale). The 1.25x overlay profits iff Σ(flagged R) > 0 ⇔ flagged-subset mean R>0.

PASS RULE (pre-registered, fixed BEFORE measuring):
  ADOPT-candidate iff BOTH:
    (1) paired portfolio mean-ΔR bootstrap 95% CI-lower > 0  (10k resamples), AND
    (2) flagged-subset mean R  ≥  unflagged-subset mean R
        (the multiplier must lever the BETTER trades, not merely any positive trades).
  Otherwise CLOSE. Coverage guard: flag on >80% or <10% of trades is degenerate → note it.
  Any ADOPT is prototype-grade pending FORWARD proof (standing rule).

STUDY ONLY. No production code/config touched. Reuses audit_common (exact add.59/60 basket).
"""
from __future__ import annotations
import os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from audit_common import load_basket, basket_R, basket_stats_from_R, FIXED_RISK, CRYPTO
from gann_time import compute_gann_windows
from unified_archetype_v2 import RTZ_ATR

BOOT = 10_000
SEED = 20260813
MULT = 1.25                       # the pre-registered conviction multiplier
DELTA = MULT - 1.0               # 0.25 : the extra size fraction -> per-trade ΔR = DELTA·R·flag

# ---- eq_magnet (add.53 gem#3 faithful price-only port; verbatim from run_one_strategy) ----
EQ_PIV_N = 10        # tail(10) confirmed pivots
EQ_TOL = 0.001       # 0.1% equal-level tolerance
EQ_CNT = 3           # cluster count >= 3
SWING_N = 10         # causal fractal confirm lag


def compute_eq_magnet(df) -> np.ndarray:
    """Causal per-bar boolean: an equal-level magnet (>=EQ_CNT pivots within EQ_TOL among
    the last EQ_PIV_N confirmed pivots, highs OR lows) sits within RTZ_ATR*ATR of close.
    VERBATIM copy of idea_lab/run_one_strategy.compute_eq_magnet (add.54)."""
    high = df["high"].to_numpy(float); low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float); atr = df["atr_14"].to_numpy(float)
    n = len(df)
    piv_conf_idx, piv_level = [], []
    for p in range(SWING_N, n - SWING_N):
        hp, lp = high[p], low[p]
        if hp > high[p - SWING_N:p].max() and hp > high[p + 1:p + 1 + SWING_N].max():
            piv_conf_idx.append(p + SWING_N); piv_level.append(hp)
        if lp < low[p - SWING_N:p].min() and lp < low[p + 1:p + 1 + SWING_N].min():
            piv_conf_idx.append(p + SWING_N); piv_level.append(lp)
    order = np.argsort(piv_conf_idx, kind="stable")
    piv_conf_idx = [piv_conf_idx[k] for k in order]
    piv_level = [piv_level[k] for k in order]
    out = np.zeros(n, dtype=bool); levels_so_far = []; ptr = 0
    for i in range(n):
        while ptr < len(piv_conf_idx) and piv_conf_idx[ptr] <= i:
            levels_so_far.append(piv_level[ptr]); ptr += 1
        if len(levels_so_far) < EQ_CNT or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        recent = levels_so_far[-EQ_PIV_N:]; band = RTZ_ATR * atr[i]; hit = False
        for lvl in set(np.round(recent, 6)):
            cluster = [x for x in recent if abs(x - lvl) <= EQ_TOL * lvl]
            if len(cluster) >= EQ_CNT:
                mag = float(np.mean(cluster))
                if abs(close[i] - mag) <= band:
                    hit = True; break
        out[i] = hit
    return out


# ---- ob_quality daily-frame proxy (faithful port of engine/liquidity/hob.py 5-comp) ----
# Weights (hob.py defaults): consolidation .25, volume .30, reaction .20, wick .15, confl .10.
# APPROXIMATIONS (documented): (a) MTF confluence component = 0.0 (no 4H/1D dict on the daily
# frame; weight only .10); (b) wick volume-z multiplier = 1.0 (daily window<84 bars -> the
# detector's own z=0 fallback fires); (c) reaction "pips" (×10000) saturates on daily crypto
# moves — a FAITHFUL reproduction of the store proxy's formula; (d) we do NOT apply the
# detector's consolidation<0.3 HOB REJECTION (that would drop door entries and change the
# basket — the intent is a top-half sizing split of the SAME 106 fires, not a re-gate);
# (e) level.price = the door's break_level (the reclaimed structure being retested).
OBW = (0.25, 0.30, 0.20, 0.15, 0.10)
OB_RECENT = 20; OB_VOLTHR = 1.5; OB_MINPIPS = 50.0


def ob_quality_proxy(o, h, l, c, v, i, level_price):
    lo = max(0, i - OB_RECENT + 1)
    wh = h[lo:i + 1]; wl = l[lo:i + 1]; wv = v[lo:i + 1]
    if level_price is None or not np.isfinite(level_price) or level_price <= 0:
        level_price = c[i]
    # 1 consolidation
    rng_pct = (np.nanmax(wh) - np.nanmin(wl)) / level_price
    consol = min(1.0, max(0.0, 1.0 - rng_pct / 0.05))
    # 2 volume surge
    if len(wv) >= 1 and np.isfinite(wv).any():
        recent_vol = np.nanmean(wv[-5:]); avg_vol = np.nanmean(wv)
        vr = recent_vol / avg_vol if avg_vol and avg_vol > 0 else 1.0
    else:
        vr = 1.0
    vol_s = min(1.0, vr / 3.0) if vr >= OB_VOLTHR else vr / OB_VOLTHR * 0.5
    # 3 reaction (long)
    rp = (c[i] - level_price) / level_price * 10000.0
    react = min(1.0, rp / (OB_MINPIPS * 2)) if rp >= OB_MINPIPS else max(0.0, rp / OB_MINPIPS)
    # 4 wick (lower wick at support; vol-z multiplier = 1.0 on daily)
    body = abs(c[i] - o[i]); lower_wick = (o[i] - l[i]) if c[i] > o[i] else (c[i] - l[i])
    wick_ratio = lower_wick / body if body > 0 else 0.0
    wick = min(1.0, wick_ratio / 2.0)
    # 5 confluence (MTF) -> 0.0 (documented)
    confl = 0.0
    return (consol * OBW[0] + vol_s * OBW[1] + react * OBW[2] + wick * OBW[3] + confl * OBW[4])


# --------------------------------------------------------------------------- machinery
def paired_dR_ci(R, flag, delta=DELTA, B=BOOT, seed=SEED):
    """Per-trade portfolio ΔR = delta·R·flag. Bootstrap mean-ΔR 95% CI over the n trades."""
    R = np.asarray(R, float); flag = np.asarray(flag, bool)
    dR = np.where(flag, delta * R, 0.0)
    rng = np.random.default_rng(seed)
    n = len(R)
    idx = rng.integers(0, n, size=(B, n))
    means = dR[idx].mean(1)
    return dict(mean_dR=float(dR.mean()),
                ci_lo=float(np.percentile(means, 2.5)),
                ci_hi=float(np.percentile(means, 97.5)),
                total_dPnL=float(dR.sum() * FIXED_RISK))


def decompose(R, flag):
    R = np.asarray(R, float); flag = np.asarray(flag, bool)
    fl = R[flag]; un = R[~flag]
    return dict(n=len(R), n_flag=int(flag.sum()), cover=float(flag.mean()),
                mean_flag=float(fl.mean()) if len(fl) else float("nan"),
                mean_unflag=float(un.mean()) if len(un) else float("nan"),
                wr_flag=float((fl > 0).mean()) if len(fl) else float("nan"),
                wr_unflag=float((un > 0).mean()) if len(un) else float("nan"))


def boosted_portfolio(R, flag, mult=MULT):
    R = np.asarray(R, float); flag = np.asarray(flag, bool)
    flat = basket_stats_from_R(list(R))
    boosted = basket_stats_from_R(list(np.where(flag, mult * R, R)))
    return flat, boosted


def family_of(name):
    if name in CRYPTO:
        return "crypto"
    if name == "GOLD":
        return "gold"
    if name == "NDX":
        return "equity"
    return "other"


def verdict(dec, ci):
    c1 = ci["ci_lo"] > 0
    c2 = dec["mean_flag"] >= dec["mean_unflag"]
    degen = dec["cover"] > 0.80 or dec["cover"] < 0.10
    passed = c1 and c2 and not degen
    return c1, c2, degen, passed


# --------------------------------------------------------------------------- main
def build_rows():
    # Rebuild the door per asset (deterministic -> reproduces load_basket's v1 exactly) so
    # we can read the strategy's entries_log (break_level + the door's own fib flag).
    from trend_continuation_door import TrendContinuationDoor
    from backtester import run_backtest
    assets = load_basket(with_campaigns=False, verbose=False)
    rows = []   # dict per trade
    census = {}  # per-asset flag coverage on the door's own entries
    for name, a in assets.items():
        df = a["df"]; hv = a["hv"]
        strat = TrendContinuationDoor(df, a["sr"], a["bj"], a["eye"],
                                      variant="struct", conviction=False)
        res = run_backtest(df, strat, label="v1")
        # parity assert vs load_basket's v1
        assert res["stats"]["n"] == a["v1"]["stats"]["n"], f"{name} re-run mismatch"
        o = df["open"].to_numpy(float); h = df["high"].to_numpy(float)
        l = df["low"].to_numpy(float); c = df["close"].to_numpy(float)
        v = df["volume"].to_numpy(float)
        gann_major = compute_gann_windows(df, use_halvings=hv, major_only=True)["entry_window"]
        eqm = compute_eq_magnet(df)
        # entry_time -> break_level, fib flag from the door's own entries_log
        elog = {e["entry_time"]: e for e in strat.entries_log}
        idxmap = {ts: k for k, ts in enumerate(df.index)}
        c_h1 = c_eq = c_fib = 0; ntr = 0
        for t in res["trades"]:
            et = t["entry_time"]; i = idxmap.get(et)
            if i is None:
                # nearest by searchsorted (should not happen; entries are on df bars)
                i = int(df.index.get_indexer([et], method="nearest")[0])
            e = elog.get(et, {})
            bl = e.get("break_level", np.nan)
            fibf = bool(e.get("time_present", False))
            obq = ob_quality_proxy(o, h, l, c, v, i, bl)
            h1 = bool(gann_major[i]); h3 = bool(eqm[i])
            rows.append(dict(name=name, fam=family_of(name), R=float(t["R"]),
                             h1=h1, h3=h3, obq=obq, fib=fibf))
            c_h1 += h1; c_eq += h3; c_fib += fibf; ntr += 1
        census[name] = dict(n=ntr, h1=c_h1, eq=c_eq, fib=c_fib)
    return rows, census


def main():
    print("=" * 96)
    print("PAIRED SIZE-MULTIPLIER ΔR RE-TEST  (add.61)  |  1.25x on flagged; paired ΔR=0.25·R·flag")
    print("=" * 96)
    rows, census = build_rows()
    R = np.array([r["R"] for r in rows], float)
    b = basket_stats_from_R(list(R))
    print(f"\nBASKET PARITY CHECK: n={b['n']}  PF={b['PF']:.2f}  PnL=${b['PnL']:,.0f}  "
          f"MaxDD={b['MaxDD']:.2f}%  meanR={R.mean():+.3f}  sd={R.std(ddof=1):.3f}  "
          f"WR={(R>0).mean():.1%}   (expect n=106 PF2.61 $53,182 add.59/60)")

    # ob_quality flag = top-half by pooled-basket median
    obq = np.array([r["obq"] for r in rows], float)
    med = float(np.median(obq))
    h2 = obq >= med
    print(f"\nob_quality proxy: median={med:.4f}  range[{obq.min():.4f},{obq.max():.4f}]  "
          f"top-half(>=med) coverage={h2.mean():.1%}")

    hyps = [
        ("H1 GANN major-anchor in-window", np.array([r["h1"] for r in rows], bool)),
        ("H2 ob_quality top-half",         h2),
        ("H3 eq_magnet proximity",         np.array([r["h3"] for r in rows], bool)),
        ("POSITIVE CONTROL fib-time",      np.array([r["fib"] for r in rows], bool)),
    ]

    print("\n" + "-" * 96)
    print(f"{'hypothesis':<34}{'cover':>7}{'meanR_flag':>11}{'meanR_unfl':>11}"
          f"{'meanΔR':>9}{'ΔR_CI_lo':>10}{'ΔR_CI_hi':>10}{'ΔPnL$':>9}")
    print("-" * 96)
    results = {}
    for label, flag in hyps:
        dec = decompose(R, flag); ci = paired_dR_ci(R, flag)
        results[label] = (dec, ci, flag)
        print(f"{label:<34}{dec['cover']:>6.1%}{dec['mean_flag']:>11.3f}"
              f"{dec['mean_unflag']:>11.3f}{ci['mean_dR']:>9.4f}{ci['ci_lo']:>10.4f}"
              f"{ci['ci_hi']:>10.4f}{ci['total_dPnL']:>9,.0f}")

    print("\n" + "=" * 96)
    print("VERDICTS (pre-registered: CI-lo>0 AND meanR_flag>=meanR_unflag AND coverage in [10%,80%])")
    print("=" * 96)
    for label, flag in hyps:
        dec, ci, _ = results[label]
        c1, c2, degen, passed = verdict(dec, ci)
        flat, boosted = boosted_portfolio(R, flag)
        pf_b = boosted["PF"]; pf_b_s = "inf" if pf_b == float("inf") else f"{pf_b:.2f}"
        tag = ("POSITIVE-CONTROL" if "CONTROL" in label else
               ("ADOPT-candidate" if passed else "CLOSE"))
        print(f"\n{label}")
        print(f"  coverage {dec['cover']:.1%} (n_flag={dec['n_flag']}/{dec['n']})   "
              f"WR flag {dec['wr_flag']:.0%} vs unflag {dec['wr_unflag']:.0%}")
        print(f"  C1 paired ΔR CI-lo>0 : {ci['ci_lo']:+.4f}R  -> {'PASS' if c1 else 'FAIL'}   "
              f"(meanΔR {ci['mean_dR']:+.4f}R, CI[{ci['ci_lo']:+.4f},{ci['ci_hi']:+.4f}], "
              f"ΔPnL ${ci['total_dPnL']:,.0f})")
        print(f"  C2 meanR_flag>=unflag: {dec['mean_flag']:+.3f} vs {dec['mean_unflag']:+.3f} "
              f"-> {'PASS' if c2 else 'FAIL'}")
        print(f"  coverage guard [10%,80%]: {'DEGENERATE' if degen else 'ok'}")
        print(f"  boosted portfolio: PF {flat['PF']:.2f}->{pf_b_s}  "
              f"PnL ${flat['PnL']:,.0f}->${boosted['PnL']:,.0f}  "
              f"MaxDD {flat['MaxDD']:.2f}%->{boosted['MaxDD']:.2f}%")
        print(f"  ==> {tag}")

    # per-family consistency
    print("\n" + "=" * 96)
    print("PER-FAMILY CONSISTENCY (meanR flag vs unflag, per family)")
    print("=" * 96)
    fams = ["crypto", "equity", "gold"]
    for label, flag in hyps:
        print(f"\n{label}")
        for fam in fams:
            m = np.array([r["fam"] == fam for r in rows], bool)
            Rf = R[m]; fl = flag[m]
            if m.sum() == 0:
                continue
            nf = int(fl.sum())
            mf = Rf[fl].mean() if nf > 0 else float("nan")
            mu = Rf[~fl].mean() if (m.sum() - nf) > 0 else float("nan")
            print(f"  {fam:<7} n={m.sum():>3}  n_flag={nf:>3}  "
                  f"meanR_flag={mf:+.3f}  meanR_unflag={mu:+.3f}  "
                  f"{'flag>unflag' if (nf>0 and (m.sum()-nf)>0 and mf>=mu) else ''}")

    # census
    print("\n" + "=" * 96)
    print("PER-ASSET FLAG CENSUS (door's own entries)")
    print("=" * 96)
    print(f"{'asset':<10}{'n':>4}{'H1_gann':>9}{'H3_eq':>7}{'fib':>6}")
    for name, cz in census.items():
        print(f"{name:<10}{cz['n']:>4}{cz['h1']:>9}{cz['eq']:>7}{cz['fib']:>6}")


if __name__ == "__main__":
    main()
