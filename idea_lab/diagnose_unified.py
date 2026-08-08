"""Diagnostic: WHY the unified archetype fails OOS. Falling-knife check + trend context
at entry. STUDY ONLY — characterization, NOT threshold-fishing (no re-tune)."""
from __future__ import annotations
import numpy as np
import pandas as pd
from backtester import run_backtest, RISK_PCT
from unified_archetype import UnifiedArchetype, build_sensors

STORE = ("/Users/rayghandchi/Bull Machine/Bull-machine-/data/features_mtf/"
         "BTC_1H_FEATURES_V22_CTX.parquet")
ERAS = {"TRAIN 2018-2022": ("2018-01-01", "2022-12-31"),
        "OOS-A 2023-2024": ("2023-01-01", "2024-12-31"),
        "OOS-B 2025-2026H1": ("2025-01-01", "2026-06-10")}


def main():
    df = pd.read_parquet(STORE)
    sr, bj = build_sensors(df)
    for era, (a, b) in ERAS.items():
        mask = (df.index >= a) & (df.index <= b)
        sub, sr_s, bj_s = df.loc[mask], sr.loc[mask], bj.loc[mask]
        strat = UnifiedArchetype(sub, sr_s, bj_s, variant="struct", conviction=False)
        res = run_backtest(sub, strat, risk_pct=RISK_PCT, label=era)
        trades = res["trades"]
        ema = sub["ema_200"]
        below = above = 0
        below_win = above_win = 0
        for t in trades:
            et = t["entry_time"]
            e_ema = ema.loc[et] if et in ema.index else np.nan
            e_px = t["entry_fill"]
            if np.isfinite(e_ema):
                if e_px < e_ema:
                    below += 1
                    if t["pnl"] > 0: below_win += 1
                else:
                    above += 1
                    if t["pnl"] > 0: above_win += 1
        n = len(trades)
        pnl_below = sum(t["pnl"] for t, et in [(t, t["entry_time"]) for t in trades]
                        if np.isfinite(ema.loc[et]) and t["entry_fill"] < ema.loc[et])
        pnl_above = sum(t["pnl"] for t in trades
                        if np.isfinite(ema.loc[t["entry_time"]]) and t["entry_fill"] >= ema.loc[t["entry_time"]])
        print(f"\n### {era}  (n={n})")
        print(f"  entries BELOW ema_200 (falling-knife risk): {below:>2}  "
              f"wins {below_win}  PnL ${pnl_below:>10,.0f}")
        print(f"  entries ABOVE ema_200 (trend-aligned)     : {above:>2}  "
              f"wins {above_win}  PnL ${pnl_above:>10,.0f}")


if __name__ == "__main__":
    main()
