"""
THE ONE STRATEGY -- FRESH-ASSET VALIDATION HARNESS  (STUDY ONLY; nothing ships)
===============================================================================
Runs the EXACT add.48 trend-continuation door + a PRE-REGISTERED conviction boost
stack, IDENTICAL params, on FRESH un-mined crypto daily series (Coinbase spot),
plus a Coinbase BTC-USD reference. See ONE_STRATEGY.md for the full spec.

Pre-registered verdict rule (fixed BEFORE running):
  PF >= 1.5 on a MAJORITY of fresh assets (>=5/9 with n>=5)
  AND aggregate fresh-asset (episode-additive) PF >= 1.5
  AND bear stand-down behaviour consistent with add.48 (fires concentrate above-EMA200;
      bear-window fires are the minority and are the leak, not the body)
  AND no asset catastrophically negative (PF < 0.5 with n >= 5).

Conviction stack (multiplicative, capped 2.0x; base risk 1%; NEVER gates the entry):
  fib-time confluence tier  x1.25   [validated 3/3; price-only; computed on all assets]
  eq_magnet proximity       x1.25   [add.53 gem #3 faithful price-only port; weaker]
  ob_quality top-half       x1.25   [add.53; RESERVED - needs full 5-comp HOB pipeline /
                                     V22 store proxy absent on fresh vendor series -> NOT
                                     computed here; flagged for the quality-axis study]
HEADLINE = struct/flat, rmult=1.0 (boost-immune) -> the verdict basis, exactly as add.48.
"""
from __future__ import annotations
import os, sys, itertools
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

REPO = "/Users/rayghandchi/Bull Machine/one-strategy"   # isolated worktree (study/one-strategy)
sys.path.insert(0, os.path.join(REPO, "idea_lab"))
sys.path.insert(0, REPO)

from backtester import (run_backtest, compute_stats, EntryPlan,
                        INITIAL_CASH, RISK_PCT)
from trend_continuation_door import build_daily_sensors, TrendContinuationDoor, DeadSpringDoor
from unified_archetype_v2 import (UnifiedArchetypeV2, DEDUP_K, RTZ_ATR, CONVICTION_MULT)
from run_xasset_spx import selftest_on
from xasset_spx_port import load_spx

# Fresh-asset daily parquets live in the study scratchpad (re-fetchable via
# fetch_fresh_crypto.py). Override with ONE_STRATEGY_DATA if relocating.
SC = os.environ.get(
    "ONE_STRATEGY_DATA",
    "/private/tmp/claude-501/-Users-rayghandchi-Bull-Machine-Bull-machine-/"
    "da0c7698-bb0a-4aa2-9a4e-481e62857ce4/scratchpad/one_strategy")

FRESH = ["ETH-USD", "SOL-USD", "LTC-USD", "XRP-USD", "ADA-USD",
         "DOGE-USD", "DOT-USD", "AVAX-USD", "LINK-USD"]
REF = "BTC-USD"
MIN_YEARS = 4.0

# shared crypto regime calendar (correlated market; used for bear stand-down test)
CRYPTO_REGIMES = [
    ("2018 bear",      "2018-01-01", "2018-12-15", "BEAR"),
    ("2019-21 bull",   "2018-12-16", "2021-11-09", "bull"),
    ("2022 bear",      "2021-11-10", "2022-12-31", "BEAR"),
    ("2023-24 bull",   "2023-01-01", "2024-12-31", "bull"),
    ("2025-26 mkdn",   "2025-01-01", "2026-12-31", "BEAR"),
]

# pre-registered stack multipliers
FIB_MULT = 1.25
EQ_MULT = 1.25
STACK_CAP = 2.0
# eq_magnet (add.53 gem #3 faithful port): equal-level cluster among last N_PIV
# confirmed pivots, tolerance TOL, count>=CNT; proximity = within RTZ*ATR of entry.
EQ_PIV_N = 10        # tail(10)
EQ_TOL = 0.001       # 0.1%
EQ_CNT = 3           # count >= 3
SWING_N = 10         # causal fractal pivot confirm lag (matches xasset_spx_port SWING_N)


# --------------------------------------------------------------------------- eq_magnet
def compute_eq_magnet(df: pd.DataFrame) -> np.ndarray:
    """Causal per-bar boolean: an equal-level magnet (>=EQ_CNT pivots within EQ_TOL
    among the last EQ_PIV_N confirmed pivots, highs OR lows) sits within RTZ_ATR*ATR
    of this bar's close. Faithful price-only port of add.53 gem #3 (imbalance.py:146-200:
    tail(10), tol 0.1%, count>=3). Pivots confirmed at bar+SWING_N (no look-ahead)."""
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    atr = df["atr_14"].to_numpy(float)
    n = len(df)
    # confirmed pivots: (confirm_index, level, kind)
    piv_conf_idx, piv_level = [], []
    for p in range(SWING_N, n - SWING_N):
        hp, lp = high[p], low[p]
        if hp > high[p - SWING_N:p].max() and hp > high[p + 1:p + 1 + SWING_N].max():
            piv_conf_idx.append(p + SWING_N); piv_level.append(hp)
        if lp < low[p - SWING_N:p].min() and lp < low[p + 1:p + 1 + SWING_N].min():
            piv_conf_idx.append(p + SWING_N); piv_level.append(lp)
    order = np.argsort(piv_conf_idx, kind="stable")
    piv_conf_idx = [piv_conf_idx[k] for k in order]
    piv_level = [piv_level[k] for k in order]

    out = np.zeros(n, dtype=bool)
    levels_so_far = []   # (confirm_idx, level) in confirm order
    ptr = 0
    for i in range(n):
        while ptr < len(piv_conf_idx) and piv_conf_idx[ptr] <= i:
            levels_so_far.append(piv_level[ptr]); ptr += 1
        if len(levels_so_far) < EQ_CNT or not np.isfinite(atr[i]) or atr[i] <= 0:
            continue
        recent = levels_so_far[-EQ_PIV_N:]
        band = RTZ_ATR * atr[i]
        hit = False
        for lvl in set(np.round(recent, 6)):
            cluster = [x for x in recent if abs(x - lvl) <= EQ_TOL * lvl]
            if len(cluster) >= EQ_CNT:
                mag = float(np.mean(cluster))
                if abs(close[i] - mag) <= band:
                    hit = True; break
        out[i] = hit
    return out


# --------------------------------------------------------------------------- boost stack
class BoostStackArchetype(UnifiedArchetypeV2):
    """Trend-continuation door with the pre-registered multiplicative capped stack.
    Overrides _conviction only; entries/stops/exits IDENTICAL to the headline door."""
    def set_stack(self, eq_magnet):
        self._eq = eq_magnet
        self.conviction = True
        return self

    def _conviction(self, i):
        mult = 1.0
        fib_present = self.fib[i] > 0
        eq_present = bool(self._eq[i])
        if fib_present:
            mult *= FIB_MULT
        if eq_present:
            mult *= EQ_MULT
        mult = min(mult, STACK_CAP)
        # keep the (time_present, bojan_present, rmult) contract; report fib/eq via time/bojan
        return fib_present, eq_present, mult


class BoostStackDoor:
    """Fire ONLY the M2-broad door (the trend-continuation door), sized by the stack."""
    def __init__(self, df, sr, bj, eye, eq_magnet):
        self._u = BoostStackArchetype(df, sr, bj, eye, variant="struct",
                                      conviction=True, m2_mode="broad").set_stack(eq_magnet)
        self.entries_log = self._u.entries_log

    def __call__(self, df, i):
        u = self._u
        if i - u._last_entry_i < DEDUP_K:
            return None
        if not u._regime_ok(i):
            return None
        sig = u._m2(i)
        if sig is None:
            return None
        plan = u._plan(i, sig)
        if plan is None:
            return None
        u._last_entry_i = i
        return plan


# --------------------------------------------------------------------------- helpers
def fmt(s):
    pf = s["PF"]; pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
    return (f"n={s['n']:>3}  WR={s['WR']*100:5.1f}%  PF={pf_s:>5}  avgR={s['avgR']:+.3f}  "
            f"PnL=${s['PnL']:>10,.0f}  MaxDD={s['MaxDD_pct']:6.2f}%")


def window_stats(trades, elog, a, b):
    ent = pd.DatetimeIndex([e["entry_time"] for e in elog])
    m = (ent >= a) & (ent <= b)
    idxs = np.where(m)[0]
    tr = [trades[i] for i in idxs]; lg = [elog[i] for i in idxs]
    eq = [INITIAL_CASH]; e = INITIAL_CASH
    for t in tr:
        e += t["pnl"]; eq.append(e)
    s = compute_stats(tr, eq, INITIAL_CASH)
    bl = [0, 0.0]; ab = [0, 0.0]
    for t, el in zip(tr, lg):
        k = bl if el["below_ema200"] else ab
        k[0] += 1; k[1] += t["pnl"]
    return s, bl, ab, tr


def cpcv_stability(trades, K=6, test_m=2):
    n = len(trades)
    if n < K:
        return None
    blocks = np.array_split(np.arange(n), K)
    pfs = []; ns = []
    for combo in itertools.combinations(range(K), test_m):
        idx = np.concatenate([blocks[c] for c in combo])
        tr = [trades[i] for i in idx]
        pnls = np.array([t["pnl"] for t in tr])
        w = pnls[pnls > 0].sum(); l = -pnls[pnls < 0].sum()
        pf = (w / l) if l > 1e-9 else (float("inf") if w > 0 else 0.0)
        pfs.append(pf); ns.append(len(tr))
    fin = [p for p in pfs if np.isfinite(p)]
    return {"n_folds": len(pfs), "n_per_fold": int(np.median(ns)),
            "mean_PF": float(np.mean(fin)) if fin else float("inf"),
            "frac_PF_gt1": float(np.mean([p > 1 for p in pfs])),
            "frac_PF_ge1p5": float(np.mean([p >= 1.5 for p in pfs]))}


def load_asset(sym):
    df_raw = load_spx(os.path.join(SC, f"{sym}.parquet"))
    return build_daily_sensors(df_raw)


def run_door(cls, df, sr, bj, eye, variant="struct", conviction=False):
    strat = cls(df, sr, bj, eye, variant=variant, conviction=conviction)
    res = run_backtest(df, strat, risk_pct=RISK_PCT, label=cls.__name__)
    res["entries_log"] = strat.entries_log
    return res


# --------------------------------------------------------------------------- per-asset
def run_asset(sym, df, sr, bj, eye):
    print("\n" + "#" * 96)
    print(f"# {sym}   bars={len(df)}  {df.index.min().date()} -> {df.index.max().date()}")
    print("#" * 96)
    selftest_on(df, sym)
    eqm = compute_eq_magnet(df)
    print(f"  eq_magnet active on {100*eqm.mean():.1f}% of bars   "
          f"vol_zero%={100*(df['volume']==0).mean():.2f}%")

    tc = run_door(TrendContinuationDoor, df, sr, bj, eye, "struct", False)   # HEADLINE
    tc_conv = run_door(TrendContinuationDoor, df, sr, bj, eye, "struct", True)  # add.48 fibORbojan 1.5x
    ds = run_door(DeadSpringDoor, df, sr, bj, eye, "struct", False)          # dead-spring contrast
    bs_strat = BoostStackDoor(df, sr, bj, eye, eqm)                          # pre-registered stack
    bs = run_backtest(df, bs_strat, risk_pct=RISK_PCT, label="BoostStack")
    bs["entries_log"] = bs_strat.entries_log

    el = tc["entries_log"]
    ab = sum(1 for e in el if not e["below_ema200"]); tot = len(el)
    absh = 100 * ab / tot if tot else 0.0
    print("\n-- TREND-CONTINUATION door (HEADLINE struct/flat, rmult=1.0) --")
    print(f"   {fmt(tc['stats'])}")
    print(f"   exits {dict(Counter(t['exit_reason'] for t in tc['trades']))}")
    print(f"   above-EMA200 {ab}/{tot} = {absh:.0f}%")
    print(f"-- add.48 conviction (fib OR bojan, 1.5x): {fmt(tc_conv['stats'])}")
    print(f"-- ONE-STRATEGY boost stack (fib x1.25 & eq_magnet x1.25, cap 2.0): {fmt(bs['stats'])}")
    n_boosted = sum(1 for e in bs['entries_log'] if e['risk_mult'] > 1.0)
    print(f"   boosted entries: {n_boosted}/{len(bs['entries_log'])}  "
          f"(fib-present {sum(1 for e in bs['entries_log'] if e['time_present'])}, "
          f"eq-present {sum(1 for e in bs['entries_log'] if e['bojan_present'])})")
    print(f"-- DEAD-SPRING baseline (dip-buyer contrast): {fmt(ds['stats'])}")

    # per-regime bear stand-down
    print("\n   per-regime (headline):  regime            kind   n    PF     PnL      below$    above$")
    bear_windows = bear_fires = 0; bear_trades = []; stood = 0
    for rn, a, b, kind in CRYPTO_REGIMES:
        s, blk, abk, tr = window_stats(tc["trades"], el, a, b)
        pf = s["PF"]; pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        print(f"     {rn:<17}{kind:<6}{s['n']:>4}{pf_s:>7}{s['PnL']:>10,.0f}"
              f"{blk[1]:>10,.0f}{abk[1]:>10,.0f}")
        if kind == "BEAR":
            bear_windows += 1; bear_fires += s["n"]; bear_trades += tr
            if s["n"] == 0:
                stood += 1
    if bear_trades:
        eq = [INITIAL_CASH]; e = INITIAL_CASH
        for t in bear_trades:
            e += t["pnl"]; eq.append(e)
        sb = compute_stats(bear_trades, eq, INITIAL_CASH)
        print(f"   BEAR stand-down: {stood}/{bear_windows} windows zero-fire; "
              f"pooled bear fires: {fmt(sb)}")
    else:
        print(f"   BEAR stand-down: {stood}/{bear_windows} windows zero-fire; no bear fires (clean)")

    return {"tc": tc, "tc_conv": tc_conv, "bs": bs, "ds": ds,
            "above_share": absh, "bear_fires": bear_fires,
            "bear_windows": bear_windows, "stood": stood,
            "bear_trades": bear_trades}


# --------------------------------------------------------------------------- episode clustering
def episode_cluster(all_entries, gap_days=5):
    """all_entries: list of (timestamp, asset, pnl, R). Sort by time; group entries
    whose timestamps are within +/-gap_days of the running cluster into ONE episode."""
    ev = sorted(all_entries, key=lambda x: x[0])
    episodes = []; cur = []
    for e in ev:
        if not cur:
            cur = [e]; continue
        if (e[0] - cur[-1][0]).days <= gap_days:
            cur.append(e)
        else:
            episodes.append(cur); cur = [e]
    if cur:
        episodes.append(cur)
    return episodes


# --------------------------------------------------------------------------- main
def main():
    print("THE ONE STRATEGY -- fresh-asset validation. Coinbase daily spot (Yahoo 429 -> "
          "substituted, see fetch_coinbase.py). Costs 2bps+3bps/side, risk 1%, "
          f"start ${INITIAL_CASH:,.0f}. Params IDENTICAL to add.48.")

    results = {}; excluded = []
    universe = FRESH + [REF]
    for sym in universe:
        path = os.path.join(SC, f"{sym}.parquet")
        if not os.path.exists(path):
            print(f"  MISSING {sym} -> skip"); excluded.append((sym, "no data")); continue
        df, sr, bj, eye = load_asset(sym)
        yrs = (df.index.max() - df.index.min()).days / 365.25
        if yrs < MIN_YEARS and sym != REF:
            print(f"  EXCLUDE {sym}: only {yrs:.1f}yr < {MIN_YEARS}yr history")
            excluded.append((sym, f"{yrs:.1f}yr")); continue
        results[sym] = run_asset(sym, df, sr, bj, eye)

    # ---------------- cross-asset summary + verdict ----------------
    print("\n" + "=" * 96)
    print("FRESH-ASSET SUMMARY (headline struct/flat, IDENTICAL params)")
    print("=" * 96)
    print(f"{'asset':<9}{'n':>4}{'WR':>7}{'PF':>7}{'PnL':>11}{'MaxDD':>8}"
          f"{'above200':>9}{'bearFire':>9}{'standdn':>9}{'stackPnL':>10}")
    fresh_pf_ge = 0; fresh_with_n = 0
    fresh_syms = [s for s in FRESH if s in results]
    for s in fresh_syms + [REF]:
        r = results[s]; st = r["tc"]["stats"]
        pf = st["PF"]; pf_s = "inf" if pf == float("inf") else f"{pf:.2f}"
        sd = f"{r['stood']}/{r['bear_windows']}"
        bstat = r["bs"]["stats"]
        print(f"{s:<9}{st['n']:>4}{st['WR']*100:>6.1f}%{pf_s:>7}{st['PnL']:>11,.0f}"
              f"{st['MaxDD_pct']:>7.1f}%{r['above_share']:>8.0f}%{r['bear_fires']:>9}"
              f"{sd:>9}{bstat['PnL']:>10,.0f}")
        if s in fresh_syms and st["n"] >= 5:
            fresh_with_n += 1
            if (pf == float("inf")) or (pf >= 1.5):
                fresh_pf_ge += 1

    # pooled fresh bear-fire leak + EMA200 split (the add.48 leak-watch)
    pooled_bear = []
    for s in fresh_syms:
        pooled_bear += results[s]["bear_trades"]
    if pooled_bear:
        eq = [INITIAL_CASH]; e = INITIAL_CASH
        for t in pooled_bear:
            e += t["pnl"]; eq.append(e)
        sb = compute_stats(pooled_bear, eq, INITIAL_CASH)
        print(f"\n  POOLED FRESH BEAR-WINDOW FIRES (the add.48 leak): {fmt(sb)}")
    ab_pnl = bl_pnl = 0.0; ab_n = bl_n = 0
    for s in fresh_syms:
        for t, e in zip(results[s]["tc"]["trades"], results[s]["tc"]["entries_log"]):
            if e["below_ema200"]:
                bl_pnl += t["pnl"]; bl_n += 1
            else:
                ab_pnl += t["pnl"]; ab_n += 1
    print(f"  POOLED FRESH EMA200 split: above n={ab_n} ${ab_pnl:,.0f}  |  "
          f"below n={bl_n} ${bl_pnl:,.0f}  (above-share {100*ab_n/(ab_n+bl_n):.0f}%)")

    # ---------------- episode-clustered + basket ----------------
    print("\n" + "=" * 96)
    print("EPISODE-CLUSTERED & BASKET (fresh assets only; BTC-correlation honesty)")
    print("=" * 96)
    all_entries = []      # (ts, asset, pnl, R)
    basket_trades = []    # dicts with exit_time, R
    FIXED_RISK = 0.01 * INITIAL_CASH   # additive basket risk unit (no cross-asset compounding)
    for s in fresh_syms:
        tc = results[s]["tc"]
        for t, e in zip(tc["trades"], tc["entries_log"]):
            all_entries.append((e["entry_time"], s, t["pnl"], t["R"]))
            basket_trades.append({"exit_time": t["exit_time"], "R": t["R"],
                                  "pnl_fixed": t["R"] * FIXED_RISK, "asset": s})
    raw_n = len(all_entries)
    episodes = episode_cluster(all_entries, gap_days=5)
    ep_n = len(episodes)
    # episode-additive PnL (fixed risk): each trade contributes R*FIXED_RISK
    ep_pnls = []
    for ep in episodes:
        ep_pnls.append(sum(R * FIXED_RISK for (_, _, _, R) in ep))
    ep_arr = np.array(ep_pnls)
    ep_w = ep_arr[ep_arr > 0].sum(); ep_l = -ep_arr[ep_arr < 0].sum()
    ep_pf = ep_w / ep_l if ep_l > 1e-9 else float("inf")
    # basket equity (fixed risk, ordered by exit)
    basket_trades.sort(key=lambda x: x["exit_time"])
    beq = [INITIAL_CASH]; e = INITIAL_CASH
    for t in basket_trades:
        e += t["pnl_fixed"]; beq.append(e)
    bpnls = np.array([t["pnl_fixed"] for t in basket_trades])
    bw = bpnls[bpnls > 0].sum(); bl = -bpnls[bpnls < 0].sum()
    bpf = bw / bl if bl > 1e-9 else float("inf")
    beq = np.array(beq); peak = np.maximum.accumulate(beq)
    bdd = ((beq - peak) / peak).min() * 100
    span_days = (max(t["exit_time"] for t in basket_trades) -
                 min(e0 for (e0, _, _, _) in all_entries)).days
    span_yrs = span_days / 365.25

    print(f"  RAW fresh trades         : {raw_n}")
    print(f"  INDEPENDENT episodes(+-5d): {ep_n}  (cross-asset clustered; the honest n)")
    print(f"  raw trades / basket-year : {raw_n/span_yrs:.1f}   over {span_yrs:.1f}yr calendar span")
    print(f"  episode-additive PF (fixed 1% risk) : {ep_pf:.2f}   PnL ${ep_arr.sum():,.0f}")
    print(f"  BASKET (fixed 1% risk, ordered by exit): PF {bpf:.2f}  PnL ${bpnls.sum():,.0f}  "
          f"MaxDD {bdd:.2f}%  n={len(basket_trades)}")

    # basket CPCV (entry-time-ordered, fixed-risk pnl)
    bt_entry_ordered = [{"pnl": R * FIXED_RISK}
                        for (_, _, _, R) in sorted(all_entries, key=lambda x: x[0])]
    cp = cpcv_stability(bt_entry_ordered, K=6, test_m=2)
    if cp:
        print(f"  basket CPCV (K=6,m=2): folds={cp['n_folds']} n/fold~{cp['n_per_fold']} "
              f"meanPF={cp['mean_PF']:.2f} frac>1={cp['frac_PF_gt1']:.0%} "
              f"frac>=1.5={cp['frac_PF_ge1p5']:.0%}")

    # ---------------- verdict ----------------
    print("\n" + "=" * 96)
    print("PRE-REGISTERED VERDICT")
    print("=" * 96)
    maj = fresh_pf_ge > fresh_with_n / 2.0
    catastrophic = [s for s in fresh_syms
                    if results[s]["tc"]["stats"]["n"] >= 5
                    and results[s]["tc"]["stats"]["PF"] != float("inf")
                    and results[s]["tc"]["stats"]["PF"] < 0.5]
    print(f"  C1 PF>=1.5 on majority of fresh assets: {fresh_pf_ge}/{fresh_with_n} "
          f"(n>=5 assets)  -> {'PASS' if maj else 'FAIL'}")
    print(f"  C2 aggregate fresh episode PF >= 1.5  : {ep_pf:.2f}  "
          f"-> {'PASS' if ep_pf >= 1.5 else 'FAIL'}")
    tot_bear = sum(results[s]['bear_fires'] for s in fresh_syms)
    print(f"  C3 bear stand-down consistent w/ add.48: total fresh bear fires={tot_bear} "
          f"(leak, expect minority)")
    print(f"  C4 no asset catastrophic (PF<0.5,n>=5): "
          f"{'PASS (none)' if not catastrophic else 'FAIL: '+','.join(catastrophic)}")
    if excluded:
        print(f"  excluded (history/data): {excluded}")


if __name__ == "__main__":
    main()
