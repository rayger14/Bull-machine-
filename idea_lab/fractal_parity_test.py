"""Parity self-test: sim_trade_daily(spec='A') must reproduce run_backtest's
per-trade R to 0.00% on every asset — validating the walker as the referee for
the dynamic-trail arms (C, C-trail, B) that run_backtest cannot express."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from fractal_exec_lib import (load_daily, extract_fires, sim_trade_daily, plan_A,
                              CRYPTO, XASSET, SLIP)

ALL = [s for s in CRYPTO] + XASSET


def main():
    print("=== ARM-A PARITY SELF-TEST (walker vs run_backtest) ===")
    print(f"{'asset':<9}{'n':>4}{'max_rel_%':>14}{'verdict':>9}")
    worst = 0.0
    for sym in ALL:
        try:
            df, sr, bj, eye = load_daily(sym)
        except Exception as e:
            print(f"{sym:<9}  LOAD FAIL: {e}"); continue
        fires, trades, arrays = extract_fires(sym, df, sr, bj, eye)
        max_rel = 0.0
        for fire, tr in zip(fires, trades):
            entry_fill = fire["entry_raw"] * (1 + SLIP)   # long entry slipped
            plan = plan_A(fire, arrays)
            out = sim_trade_daily(arrays, fire["i"], entry_fill, fire["stop"], plan, "A", fire)
            if out is None:
                continue
            rel = abs(out["R"] - tr["R"]) / (abs(tr["R"]) + 1e-9) * 100
            max_rel = max(max_rel, rel)
        worst = max(worst, max_rel)
        v = "PASS" if max_rel < 0.01 else "FAIL"
        print(f"{sym:<9}{len(fires):>4}{max_rel:>14.2e}{v:>9}")
    print(f"\nWORST rel discrepancy across all assets: {worst:.2e}%  "
          f"[target < 0.01%] -> {'PASS' if worst < 0.01 else 'FAIL'}")


if __name__ == "__main__":
    main()
