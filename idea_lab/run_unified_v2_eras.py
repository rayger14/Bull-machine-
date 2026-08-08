"""
Run THE UNIFIED ARCHETYPE v2 across the 3 pre-registered eras. STUDY ONLY.

Mirrors run_unified.py exactly (same eras, costs, self-test gate, sensors built
ONCE on the full causal history then sliced per era). ADDS:
  * per-PATHWAY attribution (M1 vs M2 separately): n/WR/PF/avgR/PnL/MaxDD
  * THE ACID TEST: below/above-EMA200 split per pathway per era
  * M2-broad diagnostic (relaxed past strict ALIGNED_FORMING)
  * champion-long overlap (if logs available) and conviction coverage
Both exit variants (struct/flat headline + naive) as v1 did.
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from backtester import run_backtest, run_selftest, compute_stats, INITIAL_CASH, RISK_PCT
from unified_archetype_v2 import UnifiedArchetypeV2, build_sensors_v2

REPO = "/Users/rayghandchi/Bull Machine/Bull-machine-"
STORE = f"{REPO}/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
CHAMP = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
         "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/wyk_measure")

ERAS = {
    "TRAIN 2018-2022":   ("2018-01-01", "2022-12-31"),
    "OOS-A 2023-2024":   ("2023-01-01", "2024-12-31"),
    "OOS-B 2025-2026H1": ("2025-01-01", "2026-06-10"),
}
CHAMP_LOGS = {
    "TRAIN 2018-2022":   "champ_V22on_train",
    "OOS-A 2023-2024":   "champ_V22_oosA",
    "OOS-B 2025-2026H1": "champ_V22on_hold",
}
OVERLAP_WIN_H = 48


def fmt(s):
    pf = s["PF"]; pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (f"n={s['n']:>3}  WR={s['WR']*100:5.1f}%  PF={pf_s:>5}  "
            f"avgR={s['avgR']:+.3f}  PnL=${s['PnL']:>10,.0f}  MaxDD={s['MaxDD_pct']:6.2f}%")


def exits(trades):
    from collections import Counter
    return dict(Counter(t["exit_reason"] for t in trades))


def pathway_split(res, initial_cash=INITIAL_CASH):
    """Split a run's trades by pathway (from trade meta) and compute per-pathway stats
    on a per-pathway equity curve (start = initial_cash, sequential)."""
    out = {}
    for p in ("M1", "M2"):
        tp = [t for t in res["trades"] if t["meta"].get("pathway") == p]
        eq = [initial_cash]
        e = initial_cash
        for t in tp:
            e += t["pnl"]; eq.append(e)
        out[p] = compute_stats(tp, eq, initial_cash, label=p)
    return out


def ema_split(trades, entries_log):
    """below/above EMA200 split (the acid test), overall and per pathway."""
    by = {"M1": {"below": [0, 0.0], "above": [0, 0.0]},
          "M2": {"below": [0, 0.0], "above": [0, 0.0]}}
    # entries_log aligns 1:1 with trades in order
    for t, e in zip(trades, entries_log):
        p = e["pathway"]
        k = "below" if e["below_ema200"] else "above"
        by[p][k][0] += 1
        by[p][k][1] += t["pnl"]
    return by


def load_champ_longs(era):
    p = os.path.join(CHAMP, CHAMP_LOGS[era], "trade_log.csv")
    if not os.path.exists(p):
        return None
    t = pd.read_csv(p, parse_dates=["timestamp"])
    longs = t[t.direction == "long"].drop_duplicates("position_id")
    return pd.DatetimeIndex(longs["timestamp"]).sort_values()


def overlap_count(entry_times, champ_times, win_h):
    if champ_times is None or len(entry_times) == 0 or len(champ_times) == 0:
        return None
    ct = champ_times.values.astype("datetime64[ns]")
    win = np.timedelta64(win_h, "h")
    n = 0
    for et in pd.DatetimeIndex(entry_times).values.astype("datetime64[ns]"):
        if np.abs(ct - et).min() <= win:
            n += 1
    return n


def main():
    df = pd.read_parquet(STORE)
    if not run_selftest(df):
        raise SystemExit("SELF-TEST FAILED")

    print("\nBuilding v2 sensors (1D N=5 struct range + Bojan-low + HTF eye state) "
          "on full causal history...")
    sr, bj, eye = build_sensors_v2(df)

    results = {}
    entlog = {}          # headline struct/flat entries_log per era
    entlog_broad = {}    # m2-broad struct/flat
    for era, (a, b) in ERAS.items():
        mask = (df.index >= a) & (df.index <= b)
        sub, sr_s, bj_s, eye_s = df.loc[mask], sr.loc[mask], bj.loc[mask], eye.loc[mask]
        variants = {}
        for variant in ("struct", "naive"):
            for conv in (False, True):
                strat = UnifiedArchetypeV2(sub, sr_s, bj_s, eye_s, variant=variant,
                                           conviction=conv, m2_mode="aligned")
                res = run_backtest(sub, strat, risk_pct=RISK_PCT,
                                   label=f"{era}|{variant}|{'conv' if conv else 'flat'}")
                res["entries_log"] = strat.entries_log
                variants[(variant, conv)] = res
                if variant == "struct" and not conv:
                    entlog[era] = strat.entries_log
        # m2-broad diagnostic (struct/flat only)
        strat_b = UnifiedArchetypeV2(sub, sr_s, bj_s, eye_s, variant="struct",
                                     conviction=False, m2_mode="broad")
        res_b = run_backtest(sub, strat_b, risk_pct=RISK_PCT, label=f"{era}|broad")
        res_b["entries_log"] = strat_b.entries_log
        variants[("broad", False)] = res_b
        entlog_broad[era] = strat_b.entries_log
        results[era] = variants

    # ================= per-era aggregate results =================
    print("\n" + "=" * 94)
    print("UNIFIED v2 — per-era AGGREGATE (both doors combined; long-only, own trade list)")
    print("=" * 94)
    print(f"Costs: commission 2bps/side (both in PnL), slippage 3bps/side (once in fill). "
          f"Base risk {RISK_PCT*100:.0f}%/trade, start ${INITIAL_CASH:,.0f}. One position at a time.")
    print("Conviction = 1.5x when fib_time_confluence>0 OR Bojan-low active. Headline = struct/flat.")
    for era in ERAS:
        v = results[era]
        print(f"\n### {era}")
        print(f"  struct  flat : {fmt(v[('struct', False)]['stats'])}")
        print(f"                 exits {exits(v[('struct', False)]['trades'])}")
        print(f"  struct  conv : {fmt(v[('struct', True)]['stats'])}")
        print(f"  naive   flat : {fmt(v[('naive', False)]['stats'])}")
        print(f"  naive   conv : {fmt(v[('naive', True)]['stats'])}")

    # ================= per-PATHWAY attribution (headline struct/flat) =================
    print("\n" + "=" * 94)
    print("PER-PATHWAY ATTRIBUTION (headline struct/flat) — which DOOR produced what")
    print("=" * 94)
    for era in ERAS:
        ps = pathway_split(results[era][("struct", False)])
        print(f"\n### {era}")
        print(f"  M1 (RTZ spring)   : {fmt(ps['M1'])}")
        print(f"  M2 (LPS/SOS align): {fmt(ps['M2'])}")

    # ================= THE ACID TEST: below/above EMA200 =================
    print("\n" + "=" * 94)
    print("ACID TEST — below/above EMA200 split per pathway per era (headline struct/flat)")
    print("  v1 was ~76% below-EMA200 multi-era / 100% below live. Did M2 break the DNA?")
    print("=" * 94)
    pooled = {"M1": {"below": [0, 0.0], "above": [0, 0.0]},
              "M2": {"below": [0, 0.0], "above": [0, 0.0]}}
    for era in ERAS:
        res = results[era][("struct", False)]
        by = ema_split(res["trades"], entlog[era])
        print(f"\n### {era}")
        for p in ("M1", "M2"):
            bl, blp = by[p]["below"]; ab, abp = by[p]["above"]
            tot = bl + ab
            share = f"{100*ab/tot:.0f}%" if tot else "n/a"
            print(f"  {p}: n={tot:>3}  below={bl:>3} (PnL ${blp:>+9,.0f})  "
                  f"above={ab:>3} (PnL ${abp:>+9,.0f})  ABOVE-share={share}")
            for k in ("below", "above"):
                pooled[p][k][0] += by[p][k][0]; pooled[p][k][1] += by[p][k][1]
    print("\n### POOLED (all eras)")
    for p in ("M1", "M2"):
        bl, blp = pooled[p]["below"]; ab, abp = pooled[p]["above"]
        tot = bl + ab
        share = f"{100*ab/tot:.0f}%" if tot else "n/a"
        print(f"  {p}: n={tot:>3}  below={bl:>3} (PnL ${blp:>+9,.0f})  "
              f"above={ab:>3} (PnL ${abp:>+9,.0f})  ABOVE-share={share}")

    # ================= verdict vs 3/3 bar =================
    def passes(s):
        return (s["PnL"] > 0) and (s["PF"] > 1.0) and s["n"] > 0
    print("\n" + "=" * 94)
    print("VERDICT vs PRE-REGISTERED 3/3 BAR (positive PnL AND PF>1 in ALL THREE eras)")
    print("=" * 94)
    for variant in ("struct", "naive"):
        for conv in (False, True):
            pv = {era: passes(results[era][(variant, conv)]["stats"]) for era in ERAS}
            tag = f"{variant}/{'conv' if conv else 'flat'}"
            npass = sum(pv.values())
            print(f"  {tag:14s} {pv}  -> {npass}/3 "
                  + ("PASS 3/3" if npass == 3 else f"({npass}/3)"))

    # ================= M2-broad diagnostic =================
    print("\n" + "=" * 94)
    print("M2-BROAD DIAGNOSTIC (relaxed past strict ALIGNED_FORMING; struct/flat) — NOT headline")
    print("=" * 94)
    for era in ERAS:
        res_b = results[era][("broad", False)]
        by = ema_split(res_b["trades"], entlog_broad[era])
        ps = pathway_split(res_b)
        print(f"\n### {era}")
        print(f"  aggregate   : {fmt(res_b['stats'])}")
        print(f"  M2 (broad)  : {fmt(ps['M2'])}")
        bl = by["M2"]["below"][0]; ab = by["M2"]["above"][0]; tot = bl + ab
        share = f"{100*ab/tot:.0f}%" if tot else "n/a"
        print(f"  M2 EMA200   : below={bl} above={ab}  ABOVE-share={share}")

    # ================= champion-long overlap =================
    print("\n" + "=" * 94)
    print(f"CHAMPION-LONG OVERLAP  (unified entry within +/-{OVERLAP_WIN_H}h of any champion long)")
    print("=" * 94)
    for era in ERAS:
        ent = [e["entry_time"] for e in entlog[era]]
        champ = load_champ_longs(era)
        ov = overlap_count(ent, champ, OVERLAP_WIN_H)
        if ov is None:
            print(f"  {era:<20} unified_entries={len(ent):>3}  (champ log unavailable)")
        else:
            pct = 100 * ov / len(ent) if ent else 0.0
            print(f"  {era:<20} unified_entries={len(ent):>3}  overlap={ov:>3} ({pct:.0f}%)  "
                  f"distinct={len(ent)-ov} ({100-pct:.0f}%)")

    # ================= conviction coverage =================
    print("\n" + "=" * 94)
    print("CONVICTION-TIER COVERAGE (R-SIZE): how often does time/Bojan fire? (non-selective = amplifier)")
    print("=" * 94)
    for era in ERAS:
        log = entlog[era]; n = len(log)
        tp = sum(1 for e in log if e["time_present"])
        bp = sum(1 for e in log if e["bojan_present"])
        any_ = sum(1 for e in log if e["time_present"] or e["bojan_present"])
        print(f"  {era:<20} n={n:>3}  time>0:{tp:>3}  bojan:{bp:>3}  "
              f"either(=1.5x):{any_:>3} ({100*any_/max(n,1):.0f}%)")

    # pooled n honesty
    pooled_n = sum(results[era][("struct", False)]["stats"]["n"] for era in ERAS)
    m2_n = sum(pathway_split(results[era][("struct", False)])["M2"]["n"] for era in ERAS)
    print(f"\n  POOLED n (struct/flat) = {pooled_n}  |  M2-only pooled n = {m2_n}")
    for era in ERAS:
        ps = pathway_split(results[era][("struct", False)])
        if ps["M2"]["n"] < 30:
            print(f"    [{era}] M2 n={ps['M2']['n']} < 30 -> DIRECTIONAL ONLY, not statistically separable")


if __name__ == "__main__":
    main()
