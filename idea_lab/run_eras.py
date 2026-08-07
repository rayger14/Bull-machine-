"""
Run the integrated all-seeing-eye strategy across the 3 pre-registered eras.
STUDY ONLY. Reports WI-geometry vs naive-ladder per era, spring-overlap, verdict.
"""
from __future__ import annotations
import sys
import numpy as np
import pandas as pd
from backtester import run_backtest, run_selftest, INITIAL_CASH, RISK_PCT
from strategy_eye import EyeStrategy

REPO = "/Users/rayghandchi/Bull Machine/Bull-machine-"
STORE = f"{REPO}/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"

ERAS = {
    "TRAIN 2018-2022":  ("2018-01-01", "2022-12-31"),
    "OOS-A 2023-2024":  ("2023-01-01", "2024-12-31"),
    "OOS-B 2025-2026H1": ("2025-01-01", "2026-06-10"),
}


def fmt_stats(s):
    pf = s["PF"]
    pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (f"n={s['n']:>3}  WR={s['WR']*100:5.1f}%  PF={pf_s:>5}  "
            f"avgR={s['avgR']:+.3f}  PnL=${s['PnL']:>11,.0f}  MaxDD={s['MaxDD_pct']:6.2f}%")


def exit_breakdown(trades):
    from collections import Counter
    c = Counter(t["exit_reason"] for t in trades)
    return dict(c)


def main():
    df = pd.read_parquet(STORE)

    # ---- gate: self-test parity ----
    ok = run_selftest(df)
    if not ok:
        print("SELF-TEST FAILED — aborting.")
        sys.exit(1)

    all_entries = []   # (era, entries_log) for overlap
    era_results = {}

    for era, (a, b) in ERAS.items():
        sub = df[(df.index >= a) & (df.index <= b)].copy()
        wi = EyeStrategy(sub, variant="wi")
        res_wi = run_backtest(sub, wi, risk_pct=RISK_PCT, label=f"{era}|WI")
        naive = EyeStrategy(sub, variant="naive")
        res_nv = run_backtest(sub, naive, risk_pct=RISK_PCT, label=f"{era}|naive")
        era_results[era] = {"wi": res_wi, "nv": res_nv,
                            "entries": wi.entries_log}
        all_entries.append((era, wi.entries_log))

    # =====================================================================
    print("\n" + "=" * 90)
    print("PER-ERA RESULTS — INTEGRATED ALL-SEEING-EYE (long-only)")
    print("=" * 90)
    print(f"Costs: commission {2e-4*1e4:.0f}bps/side (both sides in PnL), "
          f"slippage 3bps/side (once in fill). Risk {RISK_PCT*100:.0f}%/trade, "
          f"start ${INITIAL_CASH:,.0f}. One position at a time.")
    for era in ERAS:
        r = era_results[era]
        print(f"\n### {era}")
        print(f"  WI-geometry : {fmt_stats(r['wi']['stats'])}")
        print(f"                exits {exit_breakdown(r['wi']['trades'])}")
        print(f"  naive-ladder: {fmt_stats(r['nv']['stats'])}")
        print(f"                exits {exit_breakdown(r['nv']['trades'])}")

    # =====================================================================
    # spring-overlap / orthogonality: production spring archetype (A) fires on
    # tf1h_pti_trap_type in ('spring','bear_trap'). Our entry is orthogonal when
    # its CONFIRMATION was NOT a pti-spring (i.e. driven by wyckoff_spring_a/b),
    # AND the entry bar's pti_trap is not 'spring'/'bear_trap'.
    print("\n" + "=" * 90)
    print("SPRING-OVERLAP / ORTHOGONALITY vs production `spring` archetype (A)")
    print("  production A identity gate: tf1h_pti_trap_type in {spring, bear_trap}")
    print("=" * 90)
    total_entries = 0
    total_conf_pti = 0
    total_entry_pti_A = 0
    total_orthogonal = 0
    for era, log in all_entries:
        n = len(log)
        conf_pti = sum(1 for e in log if e["conf_pti_spring"])
        entry_pti_A = sum(1 for e in log
                          if e["pti_trap_at_entry"] in ("spring", "bear_trap"))
        orthogonal = sum(1 for e in log
                         if (not e["conf_pti_spring"])
                         and e["pti_trap_at_entry"] not in ("spring", "bear_trap"))
        total_entries += n; total_conf_pti += conf_pti
        total_entry_pti_A += entry_pti_A; total_orthogonal += orthogonal
        print(f"  {era:<20} entries={n:>3}  conf=pti_spring:{conf_pti:>3}  "
              f"entry_bar_in_A_gate:{entry_pti_A:>3}  orthogonal:{orthogonal:>3}")
    print(f"  {'POOLED':<20} entries={total_entries:>3}  conf=pti_spring:{total_conf_pti:>3}  "
          f"entry_bar_in_A_gate:{total_entry_pti_A:>3}  orthogonal:{total_orthogonal:>3}")
    if total_entries:
        print(f"  -> {total_orthogonal}/{total_entries} "
              f"({100*total_orthogonal/total_entries:.0f}%) of eye entries are ORTHOGONAL "
              f"to what production spring already takes.")

    # =====================================================================
    # verdict vs the pre-registered 3/3 bar
    print("\n" + "=" * 90)
    print("VERDICT vs 3/3 ACCEPTANCE BAR (positive PnL AND PF>1 in ALL THREE eras)")
    print("=" * 90)
    def passes(stats):
        pf = stats["PF"]
        return (stats["PnL"] > 0) and (pf > 1.0) and stats["n"] > 0
    wi_pass = {era: passes(era_results[era]["wi"]["stats"]) for era in ERAS}
    nv_pass = {era: passes(era_results[era]["nv"]["stats"]) for era in ERAS}
    print("  WI-geometry pass-per-era :", wi_pass,
          "->", "PASS 3/3" if all(wi_pass.values()) else "FAIL")
    print("  naive-ladder pass-per-era:", nv_pass,
          "->", "PASS 3/3" if all(nv_pass.values()) else "FAIL")

    # WI-vs-ladder head-to-head (PnL and PF per era)
    print("\n  WI-exits vs naive-ladder (same entries) — does geometry help?")
    for era in ERAS:
        w = era_results[era]["wi"]["stats"]; v = era_results[era]["nv"]["stats"]
        dpnl = w["PnL"] - v["PnL"]
        print(f"    {era:<20} WI PnL ${w['PnL']:>10,.0f} vs naive ${v['PnL']:>10,.0f}  "
              f"(WI-naive = ${dpnl:>+10,.0f})")

    # pooled sample-size honesty
    pooled_n = sum(era_results[era]["wi"]["stats"]["n"] for era in ERAS)
    print(f"\n  POOLED n (WI) = {pooled_n}  "
          f"{'[UNDERPOWERED: pooled < 30 -> directional only, not statistically separable]' if pooled_n < 30 else '[n>=30]'}")


if __name__ == "__main__":
    main()
