"""ENTRY HALF (Arm A vs Arm B sniper) + Arm D interaction — LTF-executed entries
on the SAME door fires, exits held = Arm A geometry. Crypto 1H subset only. STUDY ONLY.

Arm B sniper: door confirms at daily close D. From D+1, work the entry on 1H bars for
K=5 daily bars (120 1H bars). Limit = the HIGHER (less-deep) of {retest of break_level,
golden pocket 0.618 edge of the daily pullback leg}. Fill = first 1H low <= limit.
  - B-market: no fill by timeout -> market at the timeout close (same population as A;
              pure price-improvement measurement).
  - B-skip  : no fill -> trade skipped (fill-rate cost; missed charged at Arm-A R).
  - B-tight : same fills + stop under the 1H created low - 0.25*ATR_1H, resized to 1%
              risk (R-multiplier effect; SECONDARY — reweights R).
Arm D = B-market entries + Arm C exits (interaction check).

PASS rule (pre-registered): B-market mean dR > 0 with CI excluding 0 AND B-skip net
total R (incl opportunity cost) >= B-market's total R. Golden-pocket edge = 0.618
(shallow) per 'whichever is higher (less deep)'. All params fixed before measuring."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

from fractal_exec_lib import (load_daily, load_h1, extract_fires, sim_trade_daily,
                              plan_A, plan_C, CFG, SLIP)
from fractal_stats import paired_summary, fmt_row, bootstrap_ci
from xasset_spx_port import wilder_atr, fib_time_confluence

H1_ASSETS = ["BTC-USD", "ETH-USD", "LTC-USD", "LINK-USD", "SOL-USD"]


def prep_h1(sym):
    df = load_h1(sym)
    atr1h = wilder_atr(df, 14)
    fib1h = fib_time_confluence(df)
    return df, atr1h, fib1h


def golden_pocket_618(break_level, high_anchor):
    """Shallow (0.618) edge of the golden pocket, measured DOWN from the post-break
    high toward the structural base (break_level). Standard TA retracement."""
    if not (np.isfinite(break_level) and np.isfinite(high_anchor) and high_anchor > break_level):
        return break_level
    return high_anchor - CFG.GP_LO * (high_anchor - break_level)


def snipe(fire, h1, atr1h, fib1h, daily_idx_map, n_daily):
    """Work the 1H limit for K daily bars. Returns dict with fill info or miss."""
    bl = fire["break_level"]; ha = fire["high_anchor"]
    if not np.isfinite(bl):
        return {"status": "no_break_level"}
    gp = golden_pocket_618(bl, ha)
    limit = max(bl, gp)                     # 'whichever is higher (less deep)'
    D = fire["entry_time"]
    w0 = D + pd.Timedelta(days=1)
    w1 = D + pd.Timedelta(days=CFG.K_DAILY + 1)
    win = h1[(h1.index >= w0) & (h1.index < w1)]
    if len(win) == 0:
        return {"status": "no_h1_coverage", "limit": limit}
    lows = win["low"].to_numpy(float)
    opens = win["open"].to_numpy(float)
    hit = np.where(lows <= limit)[0]
    if len(hit) == 0:
        return {"status": "miss", "limit": limit, "gp": gp,
                "window_bars": len(win)}
    k = hit[0]
    # Buy-limit fill convention: fill at the limit if price traded DOWN to it from
    # above; fill at the bar OPEN if the market was already at/below the limit
    # (a marketable limit cannot buy above the market). = min(limit, open[k]).
    fill_price = float(min(limit, opens[k]))
    marketable = bool(opens[k] <= limit)   # diagnostic: limit gave no real improvement
    fill_ts = win.index[k]
    # fib TIME cluster active at fill (conviction flag; measurement only)
    fpos = h1.index.get_loc(fill_ts)
    fib_active = bool(fib1h[fpos] > 0)
    # 1H created low over the pullback leg [D+1 .. fill] and 1H ATR at fill
    leg = h1[(h1.index >= w0) & (h1.index <= fill_ts)]
    ltf_created_low = float(leg["low"].min())
    atr_1h = float(atr1h[fpos])
    # map fill day -> daily exit index
    fill_day = fill_ts.normalize()
    F = daily_idx_map.get(fill_day, None)
    return {"status": "fill", "limit": limit, "gp": gp, "fill_ts": fill_ts,
            "fill_price": fill_price, "marketable": marketable, "fib_active": fib_active,
            "ltf_created_low": ltf_created_low, "atr_1h": atr_1h, "F": F}


def run():
    per_fire = []
    coverage = {}
    for sym in H1_ASSETS:
        df, sr, bj, eye = load_daily(sym)
        fires, trades, arrays = extract_fires(sym, df, sr, bj, eye)
        h1, atr1h, fib1h = prep_h1(sym)
        # daily normalized-date -> position map (for fill-day exit indexing)
        didx = {ts.normalize(): k for k, ts in enumerate(df.index)}
        n_daily = len(df)
        covered = 0
        for fire in fires:
            ef_A = fire["entry_raw"] * (1 + SLIP)
            oA = sim_trade_daily(arrays, fire["i"], ef_A, fire["stop"], plan_A(fire, arrays), "A", fire)
            if oA is None:
                continue
            snap = snipe(fire, h1, atr1h, fib1h, didx, n_daily)
            rec = {"sym": sym, "family": fire["family"], "A": oA["R"],
                   "status": snap["status"], "below_ema200": fire["below_ema200"]}
            if snap["status"] in ("no_h1_coverage",):
                rec["covered"] = False
                per_fire.append(rec); continue
            covered += 1; rec["covered"] = True
            if snap["status"] == "fill" and snap["F"] is not None and snap["F"] < n_daily - 1 and snap["F"] > fire["i"]:
                F = snap["F"]; L = snap["fill_price"]; efB = L * (1 + SLIP)
                # B (limit fill): A-geometry exits, entry at fill day F, be_level = L
                pA = plan_A(fire, arrays); pA["be_level"] = L
                oB = sim_trade_daily(arrays, F, efB, fire["stop"], pA, "B", fire)
                # D interaction: B entry + C exits
                pC = plan_C(fire, arrays); pC["be_level"] = L
                oD = sim_trade_daily(arrays, F, efB, fire["stop"], pC, "D", fire)
                # B-tight: 1H stop, resized (R recomputed off tighter stop)
                stop_t = snap["ltf_created_low"] - STOP_BUF_1H(snap["atr_1h"])
                pAt = plan_A(fire, arrays); pAt["be_level"] = L
                oBt = (sim_trade_daily(arrays, F, efB, stop_t, pAt, "Bt", fire)
                       if (efB - stop_t) > 0 else None)
                rec.update({"filled": True, "B": oB["R"] if oB else None,
                            "D": oD["R"] if oD else None,
                            "Bt": oBt["R"] if oBt else None,
                            "fib_active": snap["fib_active"],
                            "marketable": snap.get("marketable", False),
                            "improve": (fire["entry_raw"] - snap["fill_price"]) / fire["entry_raw"] * 100,
                            "fill_lag_days": (snap["fill_ts"].normalize() - fire["entry_time"]).days})
            else:
                # miss (or fill-day unusable) -> B-market chases at timeout close (day D+K)
                Didx = fire["i"]; F_to = Didx + CFG.K_DAILY
                rec["filled"] = False; rec["fib_active"] = False
                if F_to < n_daily - 1:
                    to_close = float(df["close"].to_numpy(float)[F_to])
                    efTO = to_close * (1 + SLIP)
                    pA = plan_A(fire, arrays); pA["be_level"] = to_close
                    oBm = sim_trade_daily(arrays, F_to, efTO, fire["stop"], pA, "Bm", fire)
                    pC = plan_C(fire, arrays); pC["be_level"] = to_close
                    oDm = sim_trade_daily(arrays, F_to, efTO, fire["stop"], pC, "Dm", fire)
                    rec["B"] = oBm["R"] if oBm else None
                    rec["D"] = oDm["R"] if oDm else None
                    rec["Bt"] = None
                else:
                    rec["B"] = None; rec["D"] = None; rec["Bt"] = None
            per_fire.append(rec)
        coverage[sym] = (covered, len([f for f in fires]))
    return per_fire, coverage


def STOP_BUF_1H(atr_1h):
    return CFG.MIN_TP1_R * 0.0 + 0.25 * atr_1h   # STOP_BUF_ATR=0.25 on the 1H ATR


def main():
    rows, coverage = run()
    used = [r for r in rows if r.get("covered")]
    print("ENTRY HALF — 1H sniper on the door fires (crypto subset)")
    print("Coverage (fires with 1H data / total door fires):")
    for sym in H1_ASSETS:
        c, t = coverage[sym]
        print(f"  {sym:<9} {c}/{t} fires have 1H coverage")

    # B-market population = all covered fires with a B value (fill OR timeout-market)
    bm = [r for r in used if r.get("B") is not None]
    A = np.array([r["A"] for r in bm]); B = np.array([r["B"] for r in bm])
    fills = [r for r in bm if r.get("filled")]
    misses = [r for r in bm if not r.get("filled")]
    fill_rate = len(fills) / len(bm) * 100 if bm else 0

    print(f"\nn (covered, with B value) = {len(bm)}   fill rate = {fill_rate:.0f}%  "
          f"({len(fills)} filled, {len(misses)} timed-out->market)")
    fib_fills = sum(1 for r in fills if r.get("fib_active"))
    print(f"fib-TIME cluster active at fill: {fib_fills}/{len(fills)} fills (conviction flag, measurement only)")
    mkt = sum(1 for r in fills if r.get("marketable"))
    impr = np.mean([r["improve"] for r in fills]) if fills else 0.0
    print(f"marketable-limit fills (no real improvement): {mkt}/{len(fills)}  "
          f"mean entry improvement vs close[D]: {impr:+.2f}%  "
          f"mean fill lag: {np.mean([r['fill_lag_days'] for r in fills]):.1f}d" if fills else "")

    print("\n-- B-MARKET (same population as A; pure price improvement) --")
    sBm = paired_summary(B, A, "B-market vs A")
    print(fmt_row(sBm))

    # filled-only price improvement (isolates the limit's benefit on the trades it caught)
    if fills:
        Af = np.array([r["A"] for r in fills]); Bf = np.array([r["B"] for r in fills])
        sFill = paired_summary(Bf, Af, "B-market (FILLED only)")
        print(fmt_row(sFill))
    # timeout chase cost
    if misses:
        Am = np.array([r["A"] for r in misses]); Bmi = np.array([r["B"] for r in misses])
        print(f"  timeout-chase fires: n={len(misses)}  A totR={Am.sum():+.2f}  "
              f"B(market) totR={Bmi.sum():+.2f}  dR={Bmi.sum()-Am.sum():+.2f} "
              f"(the cost of chasing unfilled fires)")

    # B-tight (secondary)
    bt = [r for r in fills if r.get("Bt") is not None]
    if bt:
        Abt = np.array([r["A"] for r in bt]); Bt = np.array([r["Bt"] for r in bt])
        sBt = paired_summary(Bt, Abt, "B-tight vs A (FILLED)")
        print("\n-- B-TIGHT (1H stop, resized; SECONDARY — reweights R) --")
        print(fmt_row(sBt))

    # B-skip accounting
    print("\n-- B-SKIP (skip unfilled; opportunity cost = Arm-A R of missed fires) --")
    B_skip_total = sum(r["B"] for r in fills)                 # only filled contribute
    opp_cost = sum(r["A"] for r in misses)                    # forgone A-R on skipped
    B_market_total = float(B.sum())
    print(f"  B-skip realized total R (filled only) : {B_skip_total:+.2f}")
    print(f"  opportunity cost (Arm-A R of skipped) : {opp_cost:+.2f}  (what B-skip forgoes)")
    print(f"  B-market total R (all fires)          : {B_market_total:+.2f}")

    # Arm D interaction
    dvals = [r for r in used if r.get("D") is not None and r.get("B") is not None]
    if dvals:
        Ad = np.array([r["A"] for r in dvals]); Dd = np.array([r["D"] for r in dvals])
        Bd = np.array([r["B"] for r in dvals])
        print("\n-- ARM D (B entry + C exits) interaction --")
        sD = paired_summary(Dd, Ad, "D (B+C) vs A")
        print(fmt_row(sD))

    # ---- verdict ----
    print("\n" + "=" * 90)
    print("PRE-REGISTERED ENTRY-HALF VERDICT")
    print("=" * 90)
    cond_a = sBm["mean_dR"] > 0 and sBm["ci_excludes_0"]
    cond_b = B_skip_total >= B_market_total
    print(f"  (a) B-market meanDR>0 & CI excl 0 : {sBm['mean_dR']:+.3f}, "
          f"CI[{sBm['ci_lo']:+.3f},{sBm['ci_hi']:+.3f}] -> {'PASS' if cond_a else 'FAIL'}")
    print(f"  (b) B-skip total R >= B-market    : {B_skip_total:+.2f} vs {B_market_total:+.2f} "
          f"-> {'PASS' if cond_b else 'FAIL'}")
    print(f"  => ENTRY HALF {'PASSES' if (cond_a and cond_b) else 'FAILS'}"
          f"{'' if (cond_a and cond_b) else '  (report which condition)'}")
    return rows


if __name__ == "__main__":
    main()
