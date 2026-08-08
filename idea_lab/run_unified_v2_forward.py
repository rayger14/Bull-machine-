"""
THE UNIFIED ARCHETYPE v2 — LIVE FORWARD-TEST (no look-ahead).  STUDY ONLY.

Mirrors run_unified_forward.py EXACTLY (same window, same no-look-ahead data
construction, same champion comparison), but drives UnifiedArchetypeV2 (M1 RTZ +
M2 LPS-after-SOS) and reports per-pathway + below/above-EMA200 split.

EXPECTED/CORRECT outcome (pre-registered): M2 takes few/zero trades because this
window is a markdown->chop with no clean accumulation-SOS setups -> that PROVES the
stand-down property. "Took nothing, lost nothing" proves STAND-DOWN, NOT edge; edge
validation needs a real accumulation era.

DATA PROVENANCE (identical to run_unified_forward.py) + eye additions:
  OHLC/volume/atr_14 : STORE (V22_CTX) <= 2026-06-10; Binance.US klines for the tail.
  ema_200, swing_50  : STORE <= 2026-06-10; causal recompute over combined for TAIL only.
  fib_time_confluence: STORE <= 2026-06-10; server live logs for tail; 06-11..06-17 gap->0.
  stables_rot_rising : STORE <= 2026-06-10; not logged live -> tail defaults 0 (sensitivity run).
  wyckoff_phase_dir/sos (NEW, for the eye): STORE <= 2026-06-10 ONLY. NOT available on
     the tail (not in klines) -> tail phase=None, sos=0. FLAGGED: M2's ALIGNED_FORMING
     (needs C_accum) therefore cannot fire on the 06-11..08-06 tail; the price-driven eye
     states (CONFIRMED_BREAK/TRENDING/IN_RANGE/MANIPULATION + bull/bear dir) still compute.
"""
from __future__ import annotations
import json
from collections import Counter

import numpy as np
import pandas as pd

from backtester import run_backtest, run_selftest, compute_stats, INITIAL_CASH, RISK_PCT
from unified_archetype_v2 import UnifiedArchetypeV2, build_sensors_v2

REPO = "/Users/rayghandchi/Bull Machine/Bull-machine-"
STORE = f"{REPO}/data/features_mtf/BTC_1H_FEATURES_V22_CTX.parquet"
SCRATCH = ("/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
           "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/poc_live_forensics")
KLINES = f"{SCRATCH}/klines_1h.parquet"
SERVER_TAIL = f"{SCRATCH}/server_tail_features.json"
CHAMP_TRADES = f"{SCRATCH}/live_trades.json"

STORE_END = pd.Timestamp("2026-06-10 23:00:00")
LIVE_START = pd.Timestamp("2026-02-15 00:00:00")
LIVE_END = pd.Timestamp("2026-08-06 23:00:00")
FEAT_COLS = ["open", "high", "low", "close", "volume", "atr_14",
             "swing_high_50", "swing_low_50", "ema_200",
             "stables_rot_rising", "fib_time_confluence",
             "wyckoff_phase_dir", "wyckoff_sos"]


def build_combined_frame():
    store = pd.read_parquet(STORE)
    store = store.loc[store.index <= STORE_END, FEAT_COLS].copy()

    kl = pd.read_parquet(KLINES)
    kl.index = kl.index.tz_convert("UTC").tz_localize(None)
    kl = kl[kl.index > STORE_END].copy()

    comb = pd.DataFrame(index=store.index.union(kl.index))
    for c in FEAT_COLS:
        comb[c] = np.nan
    comb.loc[store.index, FEAT_COLS] = store.values
    for c in ["open", "high", "low", "close", "volume", "atr_14"]:
        comb.loc[kl.index, c] = kl[c].values

    tail = comb.index > STORE_END
    ema_full = comb["close"].ewm(span=200, adjust=False).mean()
    comb.loc[tail, "ema_200"] = ema_full[tail]
    sh = comb["high"].rolling(50, min_periods=10).max()
    sl = comb["low"].rolling(50, min_periods=10).min()
    comb.loc[tail, "swing_high_50"] = sh[tail]
    comb.loc[tail, "swing_low_50"] = sl[tail]

    srv = pd.DataFrame(json.load(open(SERVER_TAIL)))
    srv["ts"] = pd.to_datetime(srv["timestamp"]).dt.tz_convert("UTC").dt.tz_localize(None)
    srv = srv.dropna(subset=["ts"]).drop_duplicates("ts").set_index("ts").sort_index()
    fib_srv = srv["fib_time_confluence"].reindex(comb.index)
    comb.loc[tail, "fib_time_confluence"] = fib_srv[tail]
    fib_gap = tail & comb["fib_time_confluence"].isna()
    comb.loc[fib_gap, "fib_time_confluence"] = 0.0
    comb.loc[tail, "stables_rot_rising"] = 0.0
    # eye inputs on the tail: wyckoff phase/sos not available live -> permissive defaults
    comb["wyckoff_phase_dir"] = comb["wyckoff_phase_dir"].astype(object)
    comb.loc[tail, "wyckoff_phase_dir"] = "neutral"
    comb.loc[tail, "wyckoff_sos"] = 0.0

    n_tail = int(tail.sum()); n_fibgap = int(fib_gap.sum())
    print(f"[coverage] combined bars={len(comb)}  store<= {STORE_END.date()}  "
          f"tail(klines)={n_tail}  fib-gap->0={n_fibgap}  "
          f"tail wyckoff_phase/sos -> neutral/0 (M2 ALIGNED_FORMING cannot fire on tail)")
    return comb


def fmt(s):
    pf = s["PF"]; pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (f"n={s['n']:>3}  WR={s['WR']*100:5.1f}%  PF={pf_s:>5}  "
            f"avgR={s['avgR']:+.3f}  PnL=${s['PnL']:>10,.0f}  MaxDD={s['MaxDD_pct']:6.2f}%")


def run_variant(sub, sr_s, bj_s, eye_s, variant, conv, m2_mode="aligned"):
    strat = UnifiedArchetypeV2(sub, sr_s, bj_s, eye_s, variant=variant,
                               conviction=conv, m2_mode=m2_mode)
    res = run_backtest(sub, strat, risk_pct=RISK_PCT,
                       label=f"{variant}|{'conv' if conv else 'flat'}|{m2_mode}")
    res["entries_log"] = strat.entries_log
    return res


def pathway_split(res):
    out = {}
    for p in ("M1", "M2"):
        tp = [t for t in res["trades"] if t["meta"].get("pathway") == p]
        eq = [INITIAL_CASH]; e = INITIAL_CASH
        for t in tp:
            e += t["pnl"]; eq.append(e)
        out[p] = compute_stats(tp, eq, INITIAL_CASH, label=p)
    return out


def falling_knife_check(trades, sub, entries_log):
    closes = sub["close"]; highs = sub["high"]; lows = sub["low"]; ema = sub["ema_200"]
    rows = []
    for t, e in zip(trades, entries_log):
        et = t["entry_time"]; ef = t["entry_fill"]; stop = t["stop0"]
        R = ef - stop
        pos = sub.index.get_loc(et)
        below = bool(closes.iloc[pos] < ema.iloc[pos])
        fwd = slice(pos + 1, min(pos + 1 + 168, len(sub)))
        fh = highs.iloc[fwd]; fl = lows.iloc[fwd]
        mfe_R = (fh.max() - ef) / R if len(fh) and R > 0 else np.nan
        mae_R = (ef - fl.min()) / R if len(fl) and R > 0 else np.nan
        reclaimed = bool(len(fh) and fh.max() >= ef + 1.0 * R)
        rows.append({"entry_time": et, "pathway": e["pathway"], "below_ema200": below,
                     "mfe_R": mfe_R, "mae_R": mae_R, "vrev_1R_168h": reclaimed,
                     "exit_reason": t["exit_reason"], "R": t["R"], "pnl": t["pnl"]})
    return pd.DataFrame(rows)


def champion_actual(a, b):
    d = json.load(open(CHAMP_TRADES))
    df = pd.DataFrame(d)
    df["te"] = pd.to_datetime(df["timestamp_entry"]).dt.tz_localize(None)
    df["tx"] = pd.to_datetime(df["timestamp_exit"]).dt.tz_localize(None)
    df = df[(df["te"] >= a) & (df["te"] <= b)].copy()
    df["pid"] = df["position_id"].astype(str)
    blank = df["pid"].str.len() == 0
    df.loc[blank, "pid"] = (df.loc[blank, "archetype"] + "|" +
                            df.loc[blank, "te"].astype(str) + "|" + df.loc[blank, "direction"])
    grp = df.groupby("pid").agg(pnl=("pnl", "sum"), dirn=("direction", "first")).reset_index()
    n = len(grp); pnls = grp["pnl"].to_numpy()
    wins = pnls[pnls > 0].sum(); losses = -pnls[pnls < 0].sum()
    wr = float((pnls > 0).mean()) if n else 0.0
    pf = float(wins / losses) if losses > 1e-9 else float("inf")
    return {"n_positions": n, "WR": wr, "PF": pf, "PnL": float(pnls.sum()),
            "longs": int((grp.dirn == "long").sum()), "shorts": int((grp.dirn == "short").sum())}


def main():
    comb = build_combined_frame()
    if not run_selftest(comb):
        raise SystemExit("SELF-TEST FAILED")

    print("\nBuilding v2 sensors (struct range + Bojan + eye) on full causal history...")
    sr, bj, eye = build_sensors_v2(comb)

    mask = (comb.index >= LIVE_START) & (comb.index <= LIVE_END)
    sub, sr_s, bj_s, eye_s = comb.loc[mask], sr.loc[mask], bj.loc[mask], eye.loc[mask]

    print("\n" + "=" * 96)
    print(f"UNIFIED v2 — LIVE FORWARD-TEST  [{LIVE_START.date()} -> {LIVE_END.date()}]")
    print("=" * 96)
    print(f"Costs: commission 2bps/side, slippage 3bps/side (once in fill). Base risk "
          f"{RISK_PCT*100:.0f}%/trade, start ${INITIAL_CASH:,.0f}. One position at a time.")

    variants = {}
    for variant in ("struct", "naive"):
        for conv in (False, True):
            variants[(variant, conv)] = run_variant(sub, sr_s, bj_s, eye_s, variant, conv)
    broad = run_variant(sub, sr_s, bj_s, eye_s, "struct", False, m2_mode="broad")

    print("\n### Live-window aggregates (all four spec variants; headline = struct/flat)")
    for (variant, conv), res in variants.items():
        print(f"  {variant:6s} {'conv' if conv else 'flat'}: {fmt(res['stats'])}")
        print(f"           exits {dict(Counter(t['exit_reason'] for t in res['trades']))}")
    print(f"  struct flat [M2-broad]: {fmt(broad['stats'])}")

    head = variants[("struct", False)]

    # ---- per-pathway ----
    ps = pathway_split(head)
    print("\n" + "=" * 96)
    print("PER-PATHWAY (headline struct/flat)")
    print("=" * 96)
    print(f"  M1 (RTZ spring)   : {fmt(ps['M1'])}")
    print(f"  M2 (LPS/SOS align): {fmt(ps['M2'])}")
    psb = pathway_split(broad)
    print(f"  M2 (broad diag)   : {fmt(psb['M2'])}")

    # ---- trade-by-trade ----
    print("\n" + "=" * 96)
    print("TRADE-BY-TRADE LIST (headline struct/flat)")
    print("=" * 96)
    hdr = (f"{'#':>2} {'entry_time':<16} {'path':>4} {'entry':>8} {'stop':>8} "
           f"{'exit_time':<16} {'exit_reason':>14} {'R':>6} {'pnl$':>9} {'<ema200':>7} "
           f"{'fib>0':>5} {'boj':>3}")
    print(hdr)
    for k, (t, e) in enumerate(zip(head["trades"], head["entries_log"]), 1):
        below = "yes" if e["below_ema200"] else "no"
        print(f"{k:>2} {str(t['entry_time']):<16} {e['pathway']:>4} {t['entry_fill']:>8.0f} "
              f"{t['stop0']:>8.0f} {str(t['exit_time']):<16} {t['exit_reason']:>14} "
              f"{t['R']:>6.2f} {t['pnl']:>9.0f} {below:>7} "
              f"{('Y' if e['time_present'] else '-'):>5} {('Y' if e['bojan_present'] else '-'):>3}")
    if not head["trades"]:
        print("  (no v2 entries in the live window)")

    # ---- below/above EMA200 split (acid + stand-down) ----
    print("\n" + "=" * 96)
    print("BELOW/ABOVE EMA200 + STAND-DOWN CHECK (headline struct/flat)")
    print("=" * 96)
    fk = falling_knife_check(head["trades"], sub, head["entries_log"])
    if len(fk):
        for p in ("M1", "M2"):
            sp = fk[fk.pathway == p]
            if len(sp):
                bl = int(sp.below_ema200.sum()); ab = len(sp) - bl
                print(f"  {p}: n={len(sp)}  below_ema200={bl}  above_ema200={ab}  "
                      f"avgMFE={sp.mfe_R.mean():+.2f}R avgMAE={sp.mae_R.mean():+.2f}R "
                      f"WR={(sp.R>0).mean()*100:.0f}%")
    print(f"  M2 aligned trades in this markdown window: {ps['M2']['n']} "
          f"(0 = stand-down property holds; 'took nothing, lost nothing')")
    print(f"  M2 broad   trades in this markdown window: {psb['M2']['n']}")

    # ---- champion actual ----
    champ = champion_actual(LIVE_START, LIVE_END)
    print("\n" + "=" * 96)
    print("SIDE BY SIDE vs CHAMPION (actual live paper, full 16-archetype junk book, same window)")
    print("=" * 96)
    h = head["stats"]
    print(f"  UNIFIED v2 (hypothetical, struct/flat): n={h['n']}  WR={h['WR']*100:.1f}%  "
          f"PF={h['PF']:.2f}  PnL=${h['PnL']:,.0f}  MaxDD={h['MaxDD_pct']:.1f}%")
    print(f"  CHAMPION (actual live, full book)     : n={champ['n_positions']}  "
          f"WR={champ['WR']*100:.1f}%  PF={champ['PF']:.2f}  PnL=${champ['PnL']:,.0f}  "
          f"(longs={champ['longs']} shorts={champ['shorts']})")

    # ---- stables=1 sensitivity ----
    comb2 = comb.copy()
    tailm = comb2.index > STORE_END
    comb2.loc[tailm, "stables_rot_rising"] = 1.0
    sr2, bj2, eye2 = build_sensors_v2(comb2)
    m2 = (comb2.index >= LIVE_START) & (comb2.index <= LIVE_END)
    res2 = run_variant(comb2.loc[m2], sr2.loc[m2], bj2.loc[m2], eye2.loc[m2], "struct", False)
    print(f"\n[sensitivity] TAIL stables_rot_rising=1 (blocks tail longs): "
          f"n={res2['stats']['n']} PF={res2['stats']['PF']:.2f} PnL=${res2['stats']['PnL']:,.0f} "
          f"(vs default-0 headline n={h['n']} PnL=${h['PnL']:,.0f})")


if __name__ == "__main__":
    main()
