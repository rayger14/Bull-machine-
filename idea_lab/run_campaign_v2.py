"""
CAMPAIGN-v2 + GANN TIME LAYER STUDY  (STUDY ONLY -- wyckoff_audit add.57 -> add.58)
==================================================================================
Pre-registered comparison of the add.54 v1 door management (single entry, 60% runner,
168d hold) vs WI's CAMPAIGN-v2 topology (add.57), on all available markets, plus the
Gann time layer (G-ENTRY split + G-EXIT stand-down). NO grids. All interpretation choices
are pre-registered in campaign_strategy.py / gann_time.py headers. Nothing ships.

PASS RULES (pre-registered, fixed before running):
  CAMPAIGN-v2 passes iff  basket PF >= v1 PF - 0.3  AND  basket PnL >= v1 PnL
                          AND cadence >= 1.5x v1     AND  basket MaxDD not worse than 1.5x v1.
  G-ENTRY passes iff in-window mean R >= out-window mean R with CI-backed direction.
  G-EXIT  passes iff it improves basket MaxDD or PnL WITHOUT runner-tail collapse >30%.
"""
from __future__ import annotations
import os, sys, itertools
from collections import defaultdict, Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from backtester import run_backtest, compute_stats, INITIAL_CASH, RISK_PCT
from campaign_backtester import run_campaign_backtest
from trend_continuation_door import build_daily_sensors, TrendContinuationDoor
from campaign_strategy import CampaignV2Door
from gann_time import compute_gann_windows
from xasset_spx_port import load_spx
from run_xasset_spx import selftest_on

SC = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
      "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad")
CRYPTO = ["BTC-USD", "ETH-USD", "SOL-USD", "LTC-USD", "XRP-USD", "ADA-USD",
          "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD"]
XASSET = [("GOLD", f"{SC}/xasset/GOLD_1D.parquet"),
          ("NDX",  f"{SC}/xasset/NDX_1D.parquet")]
FIXED_RISK = 0.01 * INITIAL_CASH
RNG = np.random.default_rng(20260812)


def pf_of(pnls):
    p = np.asarray(pnls, float)
    w = p[p > 0].sum(); l = -p[p < 0].sum()
    return (w / l) if l > 1e-9 else (float("inf") if w > 0 else 0.0)


def _walk_v1(closes, highs, lows, entry_idx, plan, created_low, danger, n):
    """Re-simulate ONE v1 door trade's per-trade R (equity-invariant, risk_dollars fixed),
    optionally with the G-EXIT overlay: while positioned, if bar is in a Gann DANGER window
    AND structure confirms weakness (close < the entry's created LPS low) -> FULL exit at
    close. Returns (R, exit_idx). Faithful to backtester cost mechanics (slip once, comm
    both sides, stop-first, target-needs-close)."""
    from backtester import _slip_fill, COMMISSION_RATE
    entry_fill = _slip_fill(closes[entry_idx], side_is_buy=True)
    stop = float(plan.stop); stop_dist = abs(entry_fill - stop)
    if stop_dist <= 0:
        return 0.0, entry_idx
    risk_d = INITIAL_CASH * RISK_PCT
    orig_qty = risk_d / stop_dist
    entry_comm = orig_qty * entry_fill * COMMISSION_RATE
    remaining = orig_qty; gross = 0.0; exit_comm = 0.0
    cur_stop = stop; targets = list(plan.targets); first_tp = False
    runner = plan.runner_target; max_hold = int(plan.max_hold_bars)

    def take(px, q):
        nonlocal gross, exit_comm
        fill = _slip_fill(px, side_is_buy=False)
        gross += (fill - entry_fill) * q
        exit_comm += abs(q) * fill * COMMISSION_RATE

    j = entry_idx + 1; end = min(entry_idx + 1 + max_hold, n); exit_idx = end - 1
    while j < end and remaining > 1e-12:
        lo, hi, cl = lows[j], highs[j], closes[j]
        if lo <= cur_stop:                                   # stop-first
            take(cur_stop, remaining); remaining = 0.0; exit_idx = j; break
        if danger is not None and danger[j] and np.isfinite(created_low) and cl < created_low:
            take(cl, remaining); remaining = 0.0; exit_idx = j; break   # G-EXIT full exit
        fired = True
        while fired and targets and remaining > 1e-12:
            fired = False; lvl, frac = targets[0]
            if cl >= lvl:
                q = min(frac * orig_qty, remaining); take(cl, q); remaining -= q
                targets.pop(0); fired = True; exit_idx = j
                if not first_tp:
                    first_tp = True
                    if plan.move_stop_to_after_first_tp is not None:
                        cur_stop = float(plan.move_stop_to_after_first_tp)
        if runner is not None and remaining > 1e-12 and not targets and cl >= runner:
            take(cl, remaining); remaining = 0.0; exit_idx = j; break
        j += 1
    if remaining > 1e-12:
        jn = min(end - 1, n - 1); take(closes[jn], remaining); exit_idx = jn
    pnl = gross - entry_comm - exit_comm
    return pnl / risk_d, exit_idx


def v1_gann_exit_resim(df, sr, bj, eye, danger):
    """Paired per-trade ΔR of the G-EXIT overlay on the v1 door (the Part-1 winner).
    Returns list of (R_base, R_gexit) over the exact v1 fire set."""
    strat = TrendContinuationDoor(df, sr, bj, eye, variant="struct", conviction=False)
    closes = df["close"].to_numpy(float); highs = df["high"].to_numpy(float)
    lows = df["low"].to_numpy(float); n = len(df)
    pairs = []; i = 0
    while i < n:
        plan = strat(df, i)
        if plan is None:
            i += 1; continue
        sig = strat._u._m2(i)
        created_low = sig["created_low"] if sig else np.nan
        Rb, xb = _walk_v1(closes, highs, lows, i, plan, created_low, None, n)
        Rg, _ = _walk_v1(closes, highs, lows, i, plan, created_low, danger, n)
        pairs.append((Rb, Rg))
        i = xb + 1
    return pairs


def basket_stats(trades_R):
    """Fixed-1%-risk basket over a list of (exit_time, R). Ordered by exit; returns
    PF, PnL$, MaxDD%, n."""
    if not trades_R:
        return {"PF": 0.0, "PnL": 0.0, "MaxDD": 0.0, "n": 0}
    tr = sorted(trades_R, key=lambda x: x[0])
    eq = [INITIAL_CASH]; e = INITIAL_CASH
    pnls = []
    for _, R in tr:
        d = R * FIXED_RISK; e += d; eq.append(e); pnls.append(d)
    eq = np.array(eq); peak = np.maximum.accumulate(eq)
    dd = ((eq - peak) / peak).min() * 100
    return {"PF": pf_of(pnls), "PnL": float(np.sum(pnls)), "MaxDD": float(dd), "n": len(tr)}


def boot_ci(vals, B=10000, f=np.mean):
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    v = np.asarray(vals, float)
    idx = RNG.integers(0, len(v), size=(B, len(v)))
    stat = f(v[idx], axis=1)
    return (float(np.percentile(stat, 2.5)), float(np.percentile(stat, 97.5)))


def run_asset(name, df, sr, bj, eye, use_halvings):
    selftest_on(df, name)
    gw = compute_gann_windows(df, use_halvings=use_halvings)
    entry_w = gw["entry_window"]; danger_w = gw["danger_window"]
    span_yrs = (df.index.max() - df.index.min()).days / 365.25

    # v1 baseline (add.54 headline: single entry, 60% runner, 168d)
    v1 = run_backtest(df, TrendContinuationDoor(df, sr, bj, eye, variant="struct",
                                                conviction=False), label="v1")
    # CAMPAIGN-v2
    v2strat = CampaignV2Door(df, sr, bj, eye)
    v2 = run_campaign_backtest(df, v2strat, label="v2")
    # CAMPAIGN-v2 + G-EXIT
    v2g_strat = CampaignV2Door(df, sr, bj, eye, gann_danger=danger_w, enable_gexit=True)
    v2g = run_campaign_backtest(df, v2g_strat, label="v2+gexit")

    # index map for entry-time -> bar position (for Gann window tagging)
    pos = {t: k for k, t in enumerate(df.index)}
    for t in v2["trades"]:
        k = pos[t["entry_time"]]
        t["in_gann_entry"] = bool(entry_w[k])
        t["below_ema200"] = bool(df["close"].iloc[k] < df["ema_200"].iloc[k])
    for t in v1["trades"]:
        k = pos[t["entry_time"]]
        t["in_gann_entry"] = bool(entry_w[k])
        t["below_ema200"] = bool(df["close"].iloc[k] < df["ema_200"].iloc[k])
    for t in v2g["trades"]:
        k = pos[t["entry_time"]]
        t["below_ema200"] = bool(df["close"].iloc[k] < df["ema_200"].iloc[k])

    # G-EXIT paired per-trade re-sim on the v1 door (Part-1 winner)
    gexit_pairs = v1_gann_exit_resim(df, sr, bj, eye, danger_w)

    # campaign segmentation from v2 meta
    camps = defaultdict(list)
    for t in v2["trades"]:
        camps[t["meta"]["campaign_id"]].append(t)
    n_camp = len(camps)
    n_ent = len(v2["trades"])
    return {"name": name, "span_yrs": span_yrs, "v1": v1, "v2": v2, "v2g": v2g,
            "n_camp": n_camp, "n_ent": n_ent, "camps": camps, "gexit_pairs": gexit_pairs,
            "entry_w_cov": float(entry_w.mean()), "danger_w_cov": float(danger_w.mean())}


def main():
    print("=" * 100)
    print("CAMPAIGN-v2 + GANN STUDY -- v1 (add.54) vs WI campaign topology (add.57). "
          "Costs 2bps+3bps, risk 1%, $100k. STUDY ONLY.")
    print("=" * 100)

    results = {}
    # crypto
    for sym in CRYPTO:
        p = os.path.join(SC, "one_strategy", f"{sym}.parquet")
        if not os.path.exists(p):
            print(f"  MISSING {sym}"); continue
        df, sr, bj, eye = build_daily_sensors(load_spx(p))
        results[sym] = run_asset(sym, df, sr, bj, eye, use_halvings=True)
    # xasset (no halvings)
    for name, p in XASSET:
        if not os.path.exists(p):
            print(f"  MISSING {name}"); continue
        df, sr, bj, eye = build_daily_sensors(load_spx(p))
        results[name] = run_asset(name, df, sr, bj, eye, use_halvings=False)
    print("\n  NOTE: SPX unavailable this run (Yahoo v8 429 + Stooq JS-gated, per add.54). "
          "Non-crypto independent evidence = GOLD (uncorrelated) + NDX.")

    # ---------------- PART 1: per-asset v1 vs v2 ----------------
    print("\n" + "=" * 100)
    print("PART 1 -- v1 vs CAMPAIGN-v2  (per-asset, per-asset-compounded equity)")
    print("=" * 100)
    hdr = (f"{'asset':<9}{'yrs':>5} | {'v1_n':>5}{'v1_PF':>7}{'v1_PnL':>10}{'v1_DD':>7} | "
           f"{'v2_n':>5}{'camp':>5}{'e/cmp':>6}{'v2_PF':>7}{'v2_PnL':>10}{'v2_DD':>7}"
           f"{'v1t/y':>7}{'v2t/y':>7}")
    print(hdr)
    for name in results:
        r = results[name]
        s1, s2 = r["v1"]["stats"], r["v2"]["stats"]
        epc = r["n_ent"] / r["n_camp"] if r["n_camp"] else 0.0
        v1ty = s1["n"] / r["span_yrs"]; v2ty = r["n_ent"] / r["span_yrs"]
        pf1 = "inf" if s1["PF"] == float("inf") else f"{s1['PF']:.2f}"
        pf2 = "inf" if s2["PF"] == float("inf") else f"{s2['PF']:.2f}"
        print(f"{name:<9}{r['span_yrs']:>5.1f} | {s1['n']:>5}{pf1:>7}{s1['PnL']:>10,.0f}"
              f"{s1['MaxDD_pct']:>7.1f} | {s2['n']:>5}{r['n_camp']:>5}{epc:>6.2f}{pf2:>7}"
              f"{s2['PnL']:>10,.0f}{s2['MaxDD_pct']:>7.1f}{v1ty:>7.2f}{v2ty:>7.2f}")

    # ---------------- BASKET (fixed 1% risk, cross-asset additive) ----------------
    v1_R = []; v2_R = []
    v1_raw = 0; v2_raw = 0
    span_lo = min(r["v1"]["trades"][0]["entry_time"] for r in results.values()
                  if r["v1"]["trades"]) if any(r["v1"]["trades"] for r in results.values()) else None
    for name, r in results.items():
        for t in r["v1"]["trades"]:
            v1_R.append((t["exit_time"], t["R"])); v1_raw += 1
        for t in r["v2"]["trades"]:
            v2_R.append((t["exit_time"], t["R"])); v2_raw += 1
    b1 = basket_stats(v1_R); b2 = basket_stats(v2_R)
    # basket calendar span
    all_exits = [x[0] for x in v1_R + v2_R]
    all_entries = [t["entry_time"] for r in results.values() for t in r["v1"]["trades"] + r["v2"]["trades"]]
    span = (max(all_exits) - min(all_entries)).days / 365.25
    print("\n" + "-" * 100)
    print(f"BASKET (fixed 1% risk, additive):")
    print(f"  v1: PF {b1['PF']:.2f}  PnL ${b1['PnL']:,.0f}  MaxDD {b1['MaxDD']:.2f}%  "
          f"n={b1['n']}  cadence {v1_raw/span:.2f} trades/yr")
    print(f"  v2: PF {b2['PF']:.2f}  PnL ${b2['PnL']:,.0f}  MaxDD {b2['MaxDD']:.2f}%  "
          f"n={b2['n']}  cadence {v2_raw/span:.2f} trades/yr")
    cadence_mult = (v2_raw / v1_raw) if v1_raw else float("nan")
    print(f"  CADENCE MULTIPLIER v2/v1 = {cadence_mult:.2f}x  (calendar span {span:.1f}yr)")

    # ---------------- PART 1b: paired campaign-level comparison + bootstrap ----------------
    print("\n" + "=" * 100)
    print("PART 1b -- PAIRED campaign comparison (same campaigns, two managements)")
    print("=" * 100)
    per_camp_delta = []   # ($v2 - $v1) per campaign, fixed 1% risk
    per_camp_rows = []
    for name, r in results.items():
        v1_by_time = sorted(r["v1"]["trades"], key=lambda t: t["entry_time"])
        for cid, ts in r["camps"].items():
            c_start = min(t["entry_time"] for t in ts)
            c_end = max(t["exit_time"] for t in ts)
            v2_R = sum(t["R"] for t in ts)
            v1_R = sum(t["R"] for t in v1_by_time
                       if c_start <= t["entry_time"] <= c_end)
            d = (v2_R - v1_R) * FIXED_RISK
            per_camp_delta.append(d)
            per_camp_rows.append((name, cid, len(ts), v1_R, v2_R, d))
    n_c = len(per_camp_delta)
    mean_d = float(np.mean(per_camp_delta)) if per_camp_delta else 0.0
    ci = boot_ci(per_camp_delta)
    wins = sum(1 for d in per_camp_delta if d > 0)
    print(f"  campaigns compared: {n_c}   v2>v1: {wins}/{n_c}")
    print(f"  mean ΔPnL/campaign (v2-v1, fixed 1% risk): ${mean_d:,.0f}   "
          f"bootstrap 95% CI [${ci[0]:,.0f}, ${ci[1]:,.0f}]")
    print(f"  total ΔPnL across campaigns: ${sum(per_camp_delta):,.0f}")

    # ---------------- PART 1 VERDICT ----------------
    print("\n" + "-" * 100)
    v1PF = b1["PF"] if np.isfinite(b1["PF"]) else 99
    v2PF = b2["PF"] if np.isfinite(b2["PF"]) else 99
    c1 = v2PF >= v1PF - 0.3
    c2 = b2["PnL"] >= b1["PnL"]
    c3 = cadence_mult >= 1.5
    c4 = b2["MaxDD"] >= 1.5 * b1["MaxDD"]      # MaxDD negative; not worse than 1.5x
    print("PART 1 PRE-REGISTERED VERDICT:")
    print(f"  C1 basket PF >= v1PF-0.3 ({v1PF-0.3:.2f}): v2PF {v2PF:.2f} -> {'PASS' if c1 else 'FAIL'}")
    print(f"  C2 basket PnL >= v1 (${b1['PnL']:,.0f}): v2 ${b2['PnL']:,.0f} -> {'PASS' if c2 else 'FAIL'}")
    print(f"  C3 cadence >= 1.5x: {cadence_mult:.2f}x -> {'PASS' if c3 else 'FAIL'}")
    print(f"  C4 MaxDD not worse than 1.5x v1 ({1.5*b1['MaxDD']:.2f}%): v2 {b2['MaxDD']:.2f}% -> "
          f"{'PASS' if c4 else 'FAIL'}")
    part1_pass = c1 and c2 and c3 and c4
    print(f"  => CAMPAIGN-v2 {'PASSES' if part1_pass else 'FAILS'} the pre-registered rule.")

    # ---------------- PART 2: GANN ----------------
    print("\n" + "=" * 100)
    print("PART 2 -- GANN TIME LAYER")
    print("=" * 100)
    print("  entry-window / danger-window coverage per asset:")
    for name, r in results.items():
        print(f"    {name:<9} entry_w {100*r['entry_w_cov']:>5.1f}%   danger_w {100*r['danger_w_cov']:>5.1f}%")

    # G-ENTRY split on the Part-1 winner's entries (default v2; both if ambiguous)
    winner_key = "v2" if part1_pass else "v1"
    print(f"\n  G-ENTRY split -- measured on {winner_key.upper()} entries "
          f"({'Part-1 winner' if part1_pass else 'Part-1 winner=v1 (v2 failed); reporting v1 (independent edge)'}):")
    for wk in (["v1", "v2"] if not part1_pass else ["v2"]):
        inR = []; outR = []
        for r in results.values():
            for t in r[wk]["trades"]:
                (inR if t["in_gann_entry"] else outR).append(t["R"])
        mi = np.mean(inR) if inR else float("nan")
        mo = np.mean(outR) if outR else float("nan")
        # bootstrap difference of means
        if inR and outR:
            bi = RNG.integers(0, len(inR), size=(10000, len(inR)))
            bo = RNG.integers(0, len(outR), size=(10000, len(outR)))
            dstat = np.asarray(inR)[bi].mean(1) - np.asarray(outR)[bo].mean(1)
            dci = (float(np.percentile(dstat, 2.5)), float(np.percentile(dstat, 97.5)))
        else:
            dci = (float("nan"), float("nan"))
        print(f"    [{wk}] in-window n={len(inR)} meanR={mi:+.3f}  |  out-window n={len(outR)} "
              f"meanR={mo:+.3f}  |  Δ(in-out)={mi-mo:+.3f} CI[{dci[0]:+.3f},{dci[1]:+.3f}]")
        gentry_pass = (mi >= mo) and (dci[0] > 0)
        print(f"       G-ENTRY {'PASS' if gentry_pass else 'FAIL'} "
              f"(needs in>=out with CI lower bound > 0)")

    # G-EXIT (a) on CAMPAIGN-v2: note redundancy with the campaign's own body-break death.
    print(f"\n  G-EXIT (a) on CAMPAIGN-v2 -- REDUNDANCY CHECK:")
    v2_R = []; v2g_R = []
    for r in results.values():
        for t in r["v2"]["trades"]:
            v2_R.append((t["exit_time"], t["R"]))
        for t in r["v2g"]["trades"]:
            v2g_R.append((t["exit_time"], t["R"]))
    bb2 = basket_stats(v2_R); bb2g = basket_stats(v2g_R)
    print(f"    v2      : PF {bb2['PF']:.2f}  PnL ${bb2['PnL']:,.0f}  MaxDD {bb2['MaxDD']:.2f}%  n={bb2['n']}")
    print(f"    v2+gexit: PF {bb2g['PF']:.2f}  PnL ${bb2g['PnL']:,.0f}  MaxDD {bb2g['MaxDD']:.2f}%  n={bb2g['n']}")
    identical = abs(bb2['PnL'] - bb2g['PnL']) < 1e-6
    print(f"    => {'IDENTICAL (G-EXIT INERT on v2: its structure clause close<floor IS the campaign D2 death)' if identical else 'differs'}")

    # G-EXIT (b) on the v1 door (Part-1 WINNER) -- paired per-trade ΔR, add.56 method.
    print(f"\n  G-EXIT (b) on the v1 door (Part-1 winner) -- paired per-trade ΔR:")
    allpairs = [p for r in results.values() for p in r["gexit_pairs"]]
    dR = [g - b for (b, g) in allpairs]
    Rb = [b for (b, g) in allpairs]; Rg = [g for (b, g) in allpairs]
    changed = [(b, g) for (b, g) in allpairs if abs(g - b) > 1e-9]
    meanΔ = float(np.mean(dR)) if dR else 0.0
    ciΔ = boot_ci(dR)
    maxRb = max(Rb) if Rb else 0; maxRg = max(Rg) if Rg else 0
    tailb = sum(x for x in Rb if x >= 3); tailg = sum(x for x in Rg if x >= 3)
    print(f"    n={len(allpairs)}  trades altered by G-EXIT: {len(changed)}")
    print(f"    mean ΔR (gexit-base) = {meanΔ:+.3f}  bootstrap 95% CI [{ciΔ[0]:+.3f}, {ciΔ[1]:+.3f}]")
    print(f"    sumR base {sum(Rb):+.1f} -> gexit {sum(Rg):+.1f}   "
          f"maxR {maxRb:.2f}->{maxRg:.2f}   runner-tail sumR(>=3) {tailb:.1f}->{tailg:.1f}")
    tail_collapse = (tailb > 0 and (tailb - tailg) / abs(tailb) > 0.30)
    pnl_better = sum(Rg) >= sum(Rb)
    gexit_pass = pnl_better and not tail_collapse
    print(f"    G-EXIT(v1) {'PASS' if gexit_pass else 'FAIL'} "
          f"(improve PnL/DD without runner-tail collapse >30%; tail Δ "
          f"{100*(tailg-tailb)/(abs(tailb)+1e-9):+.0f}%)")
    if len(changed) == 0:
        print("    NOTE: G-EXIT altered ZERO v1 trades -- the danger-window + (close<created_low) "
              "conjunction never co-occurs while positioned (v1 exits before/without it).")

    # soft spots (correctly tagged now): bear-rally leak (below-EMA200) + campaign-death give-back
    print("\n  Soft spots (v2 basket):")
    trs2 = [t for r in results.values() for t in r["v2"]["trades"]]
    leak = [t["R"] for t in trs2 if t.get("below_ema200")]
    deaths = [t["R"] for t in trs2 if t["exit_reason"] == "campaign_death"]
    print(f"    below-EMA200 (bear-rally leak): n={len(leak)} sumR={sum(leak):+.1f}")
    print(f"    campaign_death give-back exits: n={len(deaths)} sumR={sum(deaths):+.1f}")

    print("\n" + "=" * 100)
    print("EXIT REASONS (v2 basket):",
          dict(Counter(t["exit_reason"] for r in results.values() for t in r["v2"]["trades"])))
    print("=" * 100)


if __name__ == "__main__":
    main()
