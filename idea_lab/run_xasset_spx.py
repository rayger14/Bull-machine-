"""
CROSS-ASSET DIAGNOSTIC RUNNER: LONG unified (v2) -> SPX. STUDY ONLY.
See xasset_spx_port.py for the pre-registered port spec. Nothing ships.

Run 1: SPX HOURLY 2023-2026 (1H exec + 1D N=5 struct range) -- DIRECT 1:1 port.
       ONE bull regime -> FLATTERS a dip-buyer; a pass here is WEAK evidence.
Run 2: SPX DAILY 1990-2026 (1D exec + WEEKLY N=5 struct range) -- THE REAL TEST.
       Per named regime: n/WR/PF/avgR/PnL/MaxDD + below/above-EMA200 split.
"""
from __future__ import annotations
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backtester import (
    run_backtest, compute_stats, _referee_textbook, EntryPlan,
    INITIAL_CASH, RISK_PCT, SLIPPAGE_BPS, COMMISSION_RATE,
)
from unified_archetype_v2 import UnifiedArchetypeV2
from xasset_spx_port import load_spx, build_base_sensors, build_sensors_spx

XA = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
      "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/xasset")
SPX_1D = f"{XA}/SPX_1D.parquet"
SPX_1H = f"{XA}/SPX_1H.parquet"

# named SPX regimes (by entry date). bear windows are the DISCRIMINATOR.
REGIMES = [
    ("1990s bull",     "1990-01-01", "2000-03-23", "bull"),
    ("dotcom bear",    "2000-03-24", "2002-10-09", "BEAR"),
    ("2003-07 bull",   "2002-10-10", "2007-10-09", "bull"),
    ("GFC bear",       "2007-10-10", "2009-03-09", "BEAR"),
    ("2009-2020 bull", "2009-03-10", "2020-02-19", "bull"),
    ("COVID crash",    "2020-02-20", "2020-04-30", "BEAR"),
    ("2020-21 bull",   "2020-05-01", "2021-12-31", "bull"),
    ("2022 bear",      "2022-01-01", "2022-10-12", "BEAR"),
    ("2023-2026 bull", "2022-10-13", "2026-12-31", "bull"),
]


def fmt(s):
    pf = s["PF"]; pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (f"n={s['n']:>3}  WR={s['WR']*100:5.1f}%  PF={pf_s:>5}  "
            f"avgR={s['avgR']:+.3f}  PnL=${s['PnL']:>11,.0f}  MaxDD={s['MaxDD_pct']:6.2f}%")


def exits(trades):
    from collections import Counter
    return dict(Counter(t["exit_reason"] for t in trades))


def pathway_split(res):
    out = {}
    for p in ("M1", "M2"):
        tp = [t for t in res["trades"] if t["meta"].get("pathway") == p]
        eq = [INITIAL_CASH]; e = INITIAL_CASH
        for t in tp:
            e += t["pnl"]; eq.append(e)
        out[p] = compute_stats(tp, eq, INITIAL_CASH, label=p)
    return out


def selftest_on(df, label):
    """Reuse the backtester's textbook referee on an SPX window with valid indices.
    Confirms the idea-lab core math matches the referee to <0.01% on THIS asset."""
    win = df.iloc[-500:] if len(df) >= 500 else df
    closes = win["close"].to_numpy(float); highs = win["high"].to_numpy(float)
    lows = win["low"].to_numpy(float)
    HORIZON = 120
    step = max(1, (len(win) - 20) // 10)
    specs = [(k, "long" if j % 2 == 0 else "short")
             for j, k in enumerate(range(10, len(win) - HORIZON - 2, step))][:10]
    print(f"\n=== IDEA-LAB SELF-TEST vs referee (textbook-clean) on {label} ===")
    print(f"{'idx':>5} {'dir':>5} {'lab_pnl':>12} {'ref_pnl':>12} {'rel_%':>10} {'reason':>12}")
    max_rel = 0.0
    for idx, direction in specs:
        ep = float(closes[idx])
        sl, tp = (ep * 0.98, ep * 1.04) if direction == "long" else (ep * 1.02, ep * 0.96)

        def _fixed(_df, k, _idx=idx, _dir=direction, _sl=sl, _tp=tp):
            if k != _idx:
                return None
            return EntryPlan(direction=_dir, stop=_sl, targets=[(_tp, 1.0)],
                             move_stop_to_after_first_tp=None, runner_target=None,
                             max_hold_bars=HORIZON)
        res = run_backtest(win, _fixed, risk_pct=RISK_PCT, label=f"st_{idx}")
        lab = res["trades"][0]["pnl"] if res["trades"] else 0.0
        ref = _referee_textbook(highs, lows, closes, idx, direction, sl, tp, HORIZON)["pnl_textbook"]
        rel = abs(lab - ref) / (abs(ref) + 1e-9) * 100
        max_rel = max(max_rel, rel)
        rn = res["trades"][0]["exit_reason"] if res["trades"] else "none"
        print(f"{idx:>5} {direction:>5} {lab:>12.4f} {ref:>12.4f} {rel:>10.2e} {rn:>12}")
    ok = max_rel < 0.01
    print(f"MAX per-trade relative discrepancy: {max_rel:.2e} %  [target <0.01%] -> "
          f"{'PASS (0.00%)' if ok else 'FAIL'}")
    return ok


def run_variants(df, sr, bj, eye, tag):
    """struct/flat (headline), struct/conv, naive/flat, naive/conv + m2-broad diag."""
    out = {}
    for variant in ("struct", "naive"):
        for conv in (False, True):
            strat = UnifiedArchetypeV2(df, sr, bj, eye, variant=variant,
                                       conviction=conv, m2_mode="aligned")
            res = run_backtest(df, strat, risk_pct=RISK_PCT, label=f"{tag}|{variant}|{conv}")
            res["entries_log"] = strat.entries_log
            out[(variant, conv)] = res
    sb = UnifiedArchetypeV2(df, sr, bj, eye, variant="struct", conviction=False, m2_mode="broad")
    rb = run_backtest(df, sb, risk_pct=RISK_PCT, label=f"{tag}|broad")
    rb["entries_log"] = sb.entries_log
    out[("broad", False)] = rb
    return out


def ema200_pooled(res):
    bl = [0, 0.0]; ab = [0, 0.0]
    for t, e in zip(res["trades"], res["entries_log"]):
        k = bl if e["below_ema200"] else ab
        k[0] += 1; k[1] += t["pnl"]
    return bl, ab


# ============================ RUN 1: HOURLY ============================
def run_hourly():
    print("\n" + "#" * 96)
    print("# RUN 1 -- SPX HOURLY 2023-2026 (1H exec + 1D N=5 struct range) : DIRECT 1:1 PORT")
    print("#   WEAK EVIDENCE: single bull regime -> structurally FLATTERS a dip-buyer.")
    print("#" * 96)
    raw = load_spx(SPX_1H)
    df = build_base_sensors(raw)
    print(f"bars={len(df)}  {df.index[0]} -> {df.index[-1]}  "
          f"vol_zero%={100*(df['volume']==0).mean():.1f}%")
    selftest_on(df, "SPX_1H")
    sr, bj, eye = build_sensors_spx(df, struct_tf="1D")
    v = run_variants(df, sr, bj, eye, "H")
    print("\n-- aggregate (long-only, own trade list) --")
    print(f"  struct flat (HEADLINE): {fmt(v[('struct', False)]['stats'])}")
    print(f"                          exits {exits(v[('struct', False)]['trades'])}")
    print(f"  struct conv           : {fmt(v[('struct', True)]['stats'])}")
    print(f"  naive  flat           : {fmt(v[('naive', False)]['stats'])}")
    print(f"  naive  conv           : {fmt(v[('naive', True)]['stats'])}")
    ps = pathway_split(v[('struct', False)])
    print(f"  M1 (RTZ spring)       : {fmt(ps['M1'])}")
    print(f"  M2 (aligned, inert)   : {fmt(ps['M2'])}")
    rb = v[('broad', False)]
    print(f"  M2-broad (diag)       : {fmt(pathway_split(rb)['M2'])}   agg {fmt(rb['stats'])}")
    bl, ab = ema200_pooled(v[('struct', False)])
    tot = bl[0] + ab[0]
    sh = f"{100*ab[0]/tot:.0f}%" if tot else "n/a"
    print(f"  EMA200 split (headline): below={bl[0]} (${bl[1]:+,.0f})  "
          f"above={ab[0]} (${ab[1]:+,.0f})  ABOVE-share={sh}")
    return v


# ============================ RUN 2: DAILY ============================
def run_daily():
    print("\n" + "#" * 96)
    print("# RUN 2 -- SPX DAILY 1990-2026 (1D exec + WEEKLY N=5 struct range) : THE REAL TEST")
    print("#   Params UNCHANGED in bar units (168 daily-bar max_hold ~ 8mo; SPRING_WIN=72 bars).")
    print("#" * 96)
    raw = load_spx(SPX_1D)
    df = build_base_sensors(raw)
    print(f"bars={len(df)}  {df.index[0]} -> {df.index[-1]}  "
          f"vol_zero%={100*(df['volume']==0).mean():.2f}%")
    selftest_on(df, "SPX_1D")
    sr, bj, eye = build_sensors_spx(df, struct_tf="1W")
    v = run_variants(df, sr, bj, eye, "D")

    hd = v[('struct', False)]
    print("\n-- FULL-SAMPLE aggregate (CONFOUNDED by SPX secular uptrend; NOT the evidence) --")
    print(f"  struct flat (HEADLINE): {fmt(hd['stats'])}")
    print(f"                          exits {exits(hd['trades'])}")
    print(f"  struct conv           : {fmt(v[('struct', True)]['stats'])}")
    print(f"  naive  flat           : {fmt(v[('naive', False)]['stats'])}")
    ps = pathway_split(hd)
    print(f"  M1 (RTZ spring)       : {fmt(ps['M1'])}")
    print(f"  M2 (aligned, inert)   : {fmt(ps['M2'])}")
    rb = v[('broad', False)]
    print(f"  M2-broad (diag)       : {fmt(pathway_split(rb)['M2'])}   agg {fmt(rb['stats'])}")

    # ---- PER-REGIME table (the core deliverable) ----
    trades = hd["trades"]; elog = hd["entries_log"]
    ent_times = pd.DatetimeIndex([e["entry_time"] for e in elog])
    print("\n" + "=" * 96)
    print("PER-REGIME BREAKDOWN (headline struct/flat) -- BEAR windows are the DISCRIMINATOR")
    print("  n / WR / PF / avgR / PnL / MaxDD  +  below/above-EMA200 split (n, PnL)")
    print("=" * 96)
    print(f"{'regime':<16}{'kind':<6}{'n':>4}{'WR':>7}{'PF':>7}{'avgR':>8}"
          f"{'PnL':>12}{'MaxDD':>8}  {'below-EMA200':>22}  {'above-EMA200':>22}")
    bear_agg = []
    for name, a, b, kind in REGIMES:
        m = (ent_times >= a) & (ent_times <= b)
        idxs = np.where(m)[0]
        rtr = [trades[i] for i in idxs]
        rlog = [elog[i] for i in idxs]
        eq = [INITIAL_CASH]; e = INITIAL_CASH
        for t in rtr:
            e += t["pnl"]; eq.append(e)
        s = compute_stats(rtr, eq, INITIAL_CASH, label=name)
        bl = [0, 0.0]; ab = [0, 0.0]
        for t, el in zip(rtr, rlog):
            k = bl if el["below_ema200"] else ab
            k[0] += 1; k[1] += t["pnl"]
        pf = s["PF"]; pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        below_s = f"n={bl[0]:<2} ${bl[1]:>+9,.0f}"
        above_s = f"n={ab[0]:<2} ${ab[1]:>+9,.0f}"
        print(f"{name:<16}{kind:<6}{s['n']:>4}{s['WR']*100:>6.1f}%{pf_s:>7}"
              f"{s['avgR']:>+8.3f}{s['PnL']:>12,.0f}{s['MaxDD_pct']:>7.1f}%  "
              f"{below_s:>22}  {above_s:>22}")
        if kind == "BEAR":
            bear_agg.append((name, s, bl, ab, rtr))

    # ---- DISCRIMINATOR: pooled BEAR-only vs BTC markdown ----
    print("\n" + "=" * 96)
    print("DISCRIMINATOR -- pooled SPX BEAR windows (dotcom + GFC + COVID + 2022)")
    print("  BTC died in its markdown: OOS-B PF 0.29, 5/5 stops, avgR -0.53. Does SPX bleed too?")
    print("=" * 96)
    all_bear = [t for (_, _, _, _, rtr) in bear_agg for t in rtr]
    eq = [INITIAL_CASH]; e = INITIAL_CASH
    for t in all_bear:
        e += t["pnl"]; eq.append(e)
    sb = compute_stats(all_bear, eq, INITIAL_CASH, label="ALL BEARS")
    print(f"  ALL SPX BEARS pooled  : {fmt(sb)}")
    print(f"                          exits {exits(all_bear)}")
    for name, s, bl, ab, _ in bear_agg:
        tot = bl[0] + ab[0]
        shb = f"{100*bl[0]/tot:.0f}%" if tot else "n/a"
        print(f"    {name:<14}: {fmt(s)}   below-EMA200-share={shb}")
    return v


def main():
    print("COSTS: commission 2bps/side (both in PnL), slippage 3bps/side (once in fill). "
          f"Base risk {RISK_PCT*100:.0f}%/trade, start ${INITIAL_CASH:,.0f}. One position at a time.")
    print("HEADLINE = struct/flat (rmult=1.0 -> immune to fib/bojan conviction sensors).")
    run_hourly()
    run_daily()


if __name__ == "__main__":
    main()
