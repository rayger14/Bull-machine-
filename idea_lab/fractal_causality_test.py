"""3-point truncation / no-repaint check for the NEW exit sensors
(daily fractal pivot lows, upper wick magnets, active bojan-high zone bottom).
A causal sensor's value at bar i must be IDENTICAL whether computed on the full
frame or on the frame truncated at bar i. STUDY ONLY."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np

from fractal_exec_lib import load_daily, CFG, FAMILY
from xasset_spx_port import fractal_swings
from structural_range import build_structural_range
from bojan_detector import build_bojan
from trend_continuation_door import build_daily_sensors
from xasset_spx_port import reanchor_frame_weekly
from unified_archetype_v2 import BOJAN_W, HTF_N


def upper_magnet(df):
    O = df["open"].to_numpy(float); H = df["high"].to_numpy(float); C = df["close"].to_numpy(float)
    body = np.abs(C - O); uw = H - np.maximum(O, C)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(body > 0, uw / body, 0.0)
    return np.where(r >= CFG.WICK_RATIO_MIN, H, np.nan)


def bojan_high_bottom(df):
    reanch = reanchor_frame_weekly(df, HTF_N)
    sr = build_structural_range(reanch)
    bj = build_bojan(reanch, sr, BOJAN_W)
    act = bj["bojan_high_active"].to_numpy(np.int8)
    bot = bj["bojan_high_zone_bottom"].to_numpy(float)
    return np.where(act == 1, bot, np.nan)


def main():
    print("=== 3-POINT TRUNCATION / NO-REPAINT CHECK (new exit sensors) ===")
    ok_all = True
    for sym in ["BTC-USD", "SPX", "GOLD"]:
        df, sr, bj, eye = load_daily(sym)
        n = len(df)
        pts = [int(n * 0.55), int(n * 0.72), int(n * 0.90)]
        # full-frame sensors
        _, dsl_full = fractal_swings(df, CFG.SWING_N)
        mag_full = upper_magnet(df)
        bhb_full = bojan_high_bottom(df)
        for i in pts:
            sub = df.iloc[:i + 1].copy()
            _, dsl_t = fractal_swings(sub, CFG.SWING_N)
            mag_t = upper_magnet(sub)
            bhb_t = bojan_high_bottom(sub)
            def eq(a, b):
                if np.isnan(a) and np.isnan(b):
                    return True
                return np.isfinite(a) and np.isfinite(b) and abs(a - b) < 1e-6
            checks = {
                "daily_pivot_low": eq(dsl_full[i], dsl_t[i]),
                "wick_magnet_up": eq(mag_full[i], mag_t[i]),
                "bojan_high_bottom": eq(bhb_full[i], bhb_t[i]),
            }
            bad = [k for k, v in checks.items() if not v]
            status = "OK" if not bad else f"REPAINT: {bad}"
            if bad:
                ok_all = False
                print(f"  {sym:<9} i={i:<5} {status}  "
                      f"full(dsl={dsl_full[i]},mag={mag_full[i]},bhb={bhb_full[i]}) "
                      f"trunc(dsl={dsl_t[i]},mag={mag_t[i]},bhb={bhb_t[i]})")
            else:
                print(f"  {sym:<9} i={i:<5} {status}")
    print(f"\n=> {'PASS (all sensors causal, no repaint)' if ok_all else 'FAIL — repaint detected'}")


if __name__ == "__main__":
    main()
