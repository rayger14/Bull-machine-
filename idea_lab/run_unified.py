"""
Run THE UNIFIED ARCHETYPE across the 3 pre-registered eras. STUDY ONLY.

Reports: per-era n/WR/PF/avgR/PnL/MaxDD (flat + conviction, struct + naive),
champion-long overlap, comparison vs the add.27 integrated-eye result, and the
verdict vs the pre-registered 3/3 acceptance bar.

Sensors are built ONCE on the FULL causal history then sliced per era (strictly
causal: a value at t depends only on data <= t, so slicing after a full forward
pass == building per-slice, WITHOUT the artificial range reset at era boundaries).
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd
from backtester import run_backtest, run_selftest, INITIAL_CASH, RISK_PCT
from unified_archetype import UnifiedArchetype, build_sensors

REPO = "/Users/rayghandchi/Bull Machine/Bull-machine-"
STORE = f"{REPO}/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
CHAMP = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
         "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/wyk_measure")

ERAS = {
    "TRAIN 2018-2022":   ("2018-01-01", "2022-12-31"),
    "OOS-A 2023-2024":   ("2023-01-01", "2024-12-31"),
    "OOS-B 2025-2026H1": ("2025-01-01", "2026-06-10"),
}
# champion long-entry logs per era (deployed 'on' config = temporal gates on)
CHAMP_LOGS = {
    "TRAIN 2018-2022":   "champ_V22on_train",
    "OOS-A 2023-2024":   "champ_V22_oosA",
    "OOS-B 2025-2026H1": "champ_V22on_hold",
}
OVERLAP_WIN_H = 48   # a unified entry "overlaps" a champion long if within +/- this many hours


def fmt(s):
    pf = s["PF"]; pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (f"n={s['n']:>3}  WR={s['WR']*100:5.1f}%  PF={pf_s:>5}  "
            f"avgR={s['avgR']:+.3f}  PnL=${s['PnL']:>10,.0f}  MaxDD={s['MaxDD_pct']:6.2f}%")


def exits(trades):
    from collections import Counter
    return dict(Counter(t["exit_reason"] for t in trades))


def load_champ_longs(era):
    p = os.path.join(CHAMP, CHAMP_LOGS[era], "trade_log.csv")
    t = pd.read_csv(p, parse_dates=["timestamp"])
    longs = t[t.direction == "long"].drop_duplicates("position_id")
    return pd.DatetimeIndex(longs["timestamp"]).sort_values()


def overlap_count(entry_times, champ_times, win_h):
    if len(entry_times) == 0 or len(champ_times) == 0:
        return 0
    ct = champ_times.values.astype("datetime64[ns]")
    win = np.timedelta64(win_h, "h")
    n = 0
    for et in pd.DatetimeIndex(entry_times).values.astype("datetime64[ns]"):
        d = np.abs(ct - et)
        if d.min() <= win:
            n += 1
    return n


def main():
    df = pd.read_parquet(STORE)
    if not run_selftest(df):
        raise SystemExit("SELF-TEST FAILED")

    print("\nBuilding fixed sensors (1D N=5 structural range + Bojan-low) on full causal history...")
    sr, bj = build_sensors(df)

    results = {}
    all_entries = {}
    for era, (a, b) in ERAS.items():
        mask = (df.index >= a) & (df.index <= b)
        sub = df.loc[mask]
        sr_s = sr.loc[mask]
        bj_s = bj.loc[mask]
        variants = {}
        for variant in ("struct", "naive"):
            for conv in (False, True):
                strat = UnifiedArchetype(sub, sr_s, bj_s, variant=variant, conviction=conv)
                res = run_backtest(sub, strat, risk_pct=RISK_PCT,
                                   label=f"{era}|{variant}|{'conv' if conv else 'flat'}")
                variants[(variant, conv)] = res
                if variant == "struct" and not conv:
                    all_entries[era] = strat.entries_log
        results[era] = variants

    # ================= per-era results =================
    print("\n" + "=" * 92)
    print("THE UNIFIED ARCHETYPE — per-era results (long-only, own trade list)")
    print("=" * 92)
    print(f"Costs: commission 2bps/side (both in PnL), slippage 3bps/side (once in fill). "
          f"Base risk {RISK_PCT*100:.0f}%/trade, start ${INITIAL_CASH:,.0f}. One position at a time.")
    print("Conviction = 1.5x risk when fib_time_confluence>0 OR a Bojan-low zone is active.")
    for era in ERAS:
        v = results[era]
        print(f"\n### {era}")
        print(f"  struct  flat : {fmt(v[('struct', False)]['stats'])}")
        print(f"                 exits {exits(v[('struct', False)]['trades'])}")
        print(f"  struct  conv : {fmt(v[('struct', True)]['stats'])}")
        print(f"  naive   flat : {fmt(v[('naive', False)]['stats'])}")
        print(f"  naive   conv : {fmt(v[('naive', True)]['stats'])}")

    # ================= verdict vs 3/3 bar =================
    def passes(s):
        return (s["PnL"] > 0) and (s["PF"] > 1.0) and s["n"] > 0
    print("\n" + "=" * 92)
    print("VERDICT vs PRE-REGISTERED 3/3 BAR (positive PnL AND PF>1 in ALL THREE eras)")
    print("=" * 92)
    for variant in ("struct", "naive"):
        for conv in (False, True):
            pv = {era: passes(results[era][(variant, conv)]["stats"]) for era in ERAS}
            tag = f"{variant}/{'conv' if conv else 'flat'}"
            npass = sum(pv.values())
            print(f"  {tag:14s} {pv}  -> {npass}/3 " +
                  ("PASS 3/3" if npass == 3 else f"({npass}/3)"))

    # ================= champion-long overlap =================
    print("\n" + "=" * 92)
    print(f"CHAMPION-LONG OVERLAP  (unified entry within +/-{OVERLAP_WIN_H}h of any champion long entry)")
    print("  Question: is this a GENUINELY DIFFERENT trade set, or does it re-take champion longs?")
    print("=" * 92)
    tot_e = tot_o = 0
    for era in ERAS:
        ent = [e["entry_time"] for e in all_entries[era]]
        champ = load_champ_longs(era)
        ov = overlap_count(ent, champ, OVERLAP_WIN_H)
        tot_e += len(ent); tot_o += ov
        pct = 100 * ov / len(ent) if ent else 0.0
        print(f"  {era:<20} unified_entries={len(ent):>3}  champ_longs={len(champ):>4}  "
              f"overlap={ov:>3} ({pct:.0f}%)  distinct={len(ent)-ov} ({100-pct:.0f}%)")
    pct = 100 * tot_o / tot_e if tot_e else 0.0
    print(f"  {'POOLED':<20} unified_entries={tot_e:>3}  "
          f"overlap={tot_o:>3} ({pct:.0f}%)  distinct={tot_e-tot_o} ({100-pct:.0f}%)")

    # ================= conviction coverage =================
    print("\n" + "=" * 92)
    print("CONVICTION-TIER COVERAGE (R4/R5): how often does time/Bojan fire on unified entries?")
    print("=" * 92)
    for era in ERAS:
        log = all_entries[era]
        n = len(log)
        tp = sum(1 for e in log if e["time_present"])
        bp = sum(1 for e in log if e["bojan_present"])
        any_ = sum(1 for e in log if e["time_present"] or e["bojan_present"])
        print(f"  {era:<20} n={n:>3}  time>0:{tp:>3}  bojan_active:{bp:>3}  "
              f"either(=1.5x):{any_:>3} ({100*any_/max(n,1):.0f}%)")

    # ================= add.27 integrated-eye comparison =================
    print("\n" + "=" * 92)
    print("COMPARISON vs add.27 INTEGRATED-EYE (ran on BROKEN sensors; FAILED, only bull era passed)")
    print("=" * 92)
    print("  add.27 eye (broken sensors) : TRAIN PF 0.72 (NEG) | OOS-A PF 1.56 (bull, PASS) |")
    print("                                OOS-B PF 0.58 (NEG)  -> 1/3, regime tailwind not edge")
    print("  unified (fixed sensors)     : see struct/flat row per era above.")
    sv = {era: results[era][("struct", False)]["stats"] for era in ERAS}
    print(f"    TRAIN PF {sv['TRAIN 2018-2022']['PF']:.2f} | "
          f"OOS-A PF {sv['OOS-A 2023-2024']['PF']:.2f} | "
          f"OOS-B PF {sv['OOS-B 2025-2026H1']['PF']:.2f}")


if __name__ == "__main__":
    main()
